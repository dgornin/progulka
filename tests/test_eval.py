"""The eval's scoring (eval/run_eval.py) on hand-made places and stops: no LLM, no network.
A live eval run spends the API key, so the arithmetic behind its tables is checked here."""

import json
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

import pytest

from app.models import GeoPoint, Interest, Place, Stop, TripBrief
from app.pipeline import Journey
from app.sources.http import http
from eval.run_eval import (
    BaselineStop,
    Point,
    _locate,
    agent_points,
    avoid_hits,
    baseline_point,
    interest_coverage,
    nominatim_kind,
    popularity_stats,
    repeats_across_days,
    score_journey,
    write_results,
)

DAY = date(2026, 10, 24)


def _place(pid: str, name: str, source: str = "osm", **tags: str) -> Place:
    return Place(id=pid, name=name, lat=59.93, lon=30.36, kind="landmark", source=source, tags=tags)


def _point(name: str, kind: str | None, wikidata: str | None = None) -> Point:
    at = datetime.combine(DAY, time(11))
    return Point(name, kind, 59.93, 30.36, None, at, at, wikidata)


def _stop(interests: list[str]) -> Stop:
    return Stop(
        place=_place("osm:node/1", "Место"),
        arrive=datetime.combine(DAY, time(11)),
        depart=datetime.combine(DAY, time(11, 30)),
        score=0.8,
        interests=interests,
        reason="подходит",
    )


def test_avoid_hits_count_stops_by_kind_or_by_name() -> None:
    points = [
        _point("Эрмитаж", "museum"),  # by kind
        _point("Музей Фаберже", None),  # by name: the baseline stop without a known kind
        _point("Эрмитаж", None),  # neither: the leniency to the baseline
        _point("Дом-коммуна инженеров и писателей", "architecture"),
    ]
    assert avoid_hits(points, ["музеи"]) == 2
    assert avoid_hits(points, []) == 0


def test_a_stop_matching_two_avoid_phrases_is_one_hit() -> None:
    points = [_point("Пивной бар «Музей»", "bar")]
    assert avoid_hits(points, ["музеи", "бары"]) == 1


def test_avoid_name_matches_whole_words_only() -> None:
    # The validator's rule: "кафе" is not in "Кафедральный собор".
    assert avoid_hits([_point("Кафедральный собор", None)], ["кафе"]) == 0


def test_interest_coverage_counts_every_interest_by_the_validators_labels() -> None:
    interests = [
        Interest(label="архитектура", weight=1.0),
        Interest(label="кофе", weight=0.7),
        Interest(label="стрит-арт", weight=0.4),  # optional, but counted
    ]
    stops = [_stop(["Архитектура "]), _stop(["кофе"]), _stop([])]
    assert interest_coverage(stops, interests) == (2, 3)
    assert interest_coverage([], interests) == (0, 3)
    assert interest_coverage(stops, []) == (0, 0)


def test_repeats_count_pairs_of_one_place_on_different_days() -> None:
    cathedral = _place("osm:way/1", "Казанский собор", wikidata="Q1")
    same_by_link = _place(
        "wikipedia:ru:2", "Казанский кафедральный собор", source="wikipedia", wikidata="Q1"
    )
    same_by_name = _place("wikivoyage:ru:3", "Казанский собор (Санкт-Петербург)", "wikivoyage")
    other = _place("osm:node/4", "Дом политкаторжан")
    days = [[cathedral, other], [same_by_link], [cathedral, same_by_name]]
    # Days 1-2: one link. Days 1-3: one id, and one name inside the other nearby.
    # Days 2-3: one link; the article and the guide entry share neither link nor name.
    assert repeats_across_days(days) == 4


def test_repeats_ignore_one_day_and_namesakes_of_one_source() -> None:
    mural, other_mural = _place("osm:node/1", "Мурал"), _place("osm:node/2", "Мурал")
    assert repeats_across_days([[mural, mural]]) == 0  # one day: not a repeat across days
    assert repeats_across_days([[mural], [other_mural]]) == 0  # two murals, both in OSM
    assert repeats_across_days([[mural], [mural]]) == 1


def test_nominatim_kind_uses_the_overpass_categories() -> None:
    assert nominatim_kind({"category": "tourism", "type": "museum", "name": "Музей"}) == "museum"
    # Unnamed in OSM, named by the model: still a place of worship.
    assert nominatim_kind({"category": "amenity", "type": "place_of_worship", "name": ""}) == (
        "church"
    )
    heritage = {
        "category": "building",
        "type": "yes",
        "name": "Дом",
        "extratags": {"heritage": "2"},
    }
    assert nominatim_kind(heritage) == "architecture"
    assert nominatim_kind({"category": "highway", "type": "pedestrian", "name": "Улица"}) is None


def test_baseline_point_drops_the_models_remarks_and_keeps_unfound_stops() -> None:
    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    stop = BaselineStop(
        name="Улица Рубинштейна — уличная еда, рядом музей", arrive="12:00", depart="12:30"
    )
    unfound = baseline_point(stop, None, brief)
    assert (unfound.name, unfound.kind, unfound.lat) == ("Улица Рубинштейна", None, None)
    assert avoid_hits([unfound], ["музеи"]) == 0  # the remark is not the place

    hit = {"lat": "59.93", "lon": "30.34", "category": "tourism", "type": "museum", "name": "X"}
    museum = baseline_point(BaselineStop(name="X", arrive="13:00", depart="14:00"), hit, brief)
    assert (museum.kind, museum.lat, museum.arrive) == ("museum", 59.93, datetime(2026, 10, 24, 13))
    assert avoid_hits([museum], ["музеи"]) == 1


def test_popularity_stats_leave_out_the_stops_that_could_not_be_scored() -> None:
    points = [
        _point("Эрмитаж", "museum", "Q132783"),
        _point("Исаакиевский собор", "church", "Q215423"),
        _point("Блокадная парикмахерская", "museum", "Q21026905"),
        _point("Мурал", "artwork", "Q135933439"),  # an id the lookup could not score
        _point("Кофейня", "cafe"),  # no id: known not to be a must-see
    ]
    popularity = {"Q132783": 0.91, "Q215423": 0.87, "Q21026905": 0.31}
    # Mean over the three scored stops; top share: two must-sees among the four known.
    assert popularity_stats(points, popularity) == (0.7, 0.5)
    # The whole lookup failed: unknown, not "no must-sees".
    assert popularity_stats(points, {}) == (None, None)
    assert popularity_stats([_point("Кофейня", "cafe")], {}) == (None, 0.0)
    assert popularity_stats([], popularity) == (None, None)


def test_both_systems_carry_the_wikidata_id_of_their_stops(
    make_journey: Callable[..., Journey],
) -> None:
    plan = make_journey().days[0].plan
    museum = plan.stops[0].place.model_copy(update={"tags": {"wikidata": "Q132783"}})
    plan.stops[0] = plan.stops[0].model_copy(update={"place": museum})
    assert [p.wikidata for p in agent_points(plan)] == ["Q132783", None]

    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    stop = BaselineStop(name="Эрмитаж", arrive="12:00", depart="13:30")
    hit = {"lat": "59.94", "lon": "30.31", "category": "tourism", "type": "museum"}
    hit["extratags"] = {"wikidata": "Q132783", "opening_hours": "Tu-Su 10:30-18:00"}
    assert baseline_point(stop, hit, brief).wikidata == "Q132783"
    assert baseline_point(stop, None, brief).wikidata is None


async def test_a_baseline_stop_found_only_by_its_address_has_no_wikidata_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The address without a house number finds the avenue, with the avenue's Wikidata id.
    avenue = {"lat": "59.958", "lon": "30.316", "category": "highway", "type": "primary"}
    avenue["extratags"] = {"wikidata": "Q4241508", "lanes": "4"}
    asked: list[str] = []

    async def get_json(url: str, params: dict[str, Any], **options: Any) -> list[dict]:
        asked.append(params["q"])
        return [avenue] if params["q"].startswith("Кронверкский") else []

    monkeypatch.setattr(http, "get_json", get_json)
    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    stop = BaselineStop(
        name="Ленфильм-маркет", address="Кронверкский пр.", arrive="12:00", depart="12:30"
    )
    start = GeoPoint(lat=59.95, lon=30.32)

    hit = await _locate(stop, "Санкт-Петербург", None, start)

    assert asked == ["Ленфильм-маркет, Санкт-Петербург", "Кронверкский пр., Санкт-Петербург"]
    assert hit is not None and hit["extratags"] == {"lanes": "4"}
    assert avenue["extratags"]["wikidata"] == "Q4241508"  # Nominatim's answer is left as it came
    point = baseline_point(stop, hit, brief)
    assert point.lat == 59.958 and point.wikidata is None

    named = BaselineStop(name="Кронверкский пр.", arrive="12:00", depart="12:30")
    assert (await _locate(named, "Санкт-Петербург", None, start))["extratags"]["wikidata"]


def test_score_journey_sums_the_days_and_finds_repeats(
    make_journey: Callable[..., Journey],
) -> None:
    journey = make_journey(days=3)
    first, second = journey.days[0].plan, journey.days[1].plan
    # Day 2 visits day 1's first place again; day 3's first stop is a museum serving
    # the brief's only interest.
    second.stops[0] = second.stops[0].model_copy(update={"place": first.stops[0].place})
    third = journey.days[2].plan
    museum = third.stops[0].place.model_copy(update={"kind": "museum"})
    third.stops[0] = third.stops[0].model_copy(
        update={"place": museum, "interests": ["архитектура"]}
    )

    third.stops[1].place.tags["wikidata"] = "Q132783"
    usage = {"llm_calls": 40, "cost_usd": 0.012345}

    m = score_journey(journey, 3, ["музеи"], usage, 95.04, {"Q132783": 0.91})

    assert (m.days_requested, m.days_planned, m.stops_per_day) == (3, 3, [2, 2, 2])
    # One impossible leg: day 3 opens 3 km from the start, at the start time.
    assert (m.closed, m.impossible_legs, m.days_over_window) == (0, 1, 0)
    assert m.repeats == 1
    assert m.avoid_hits == 1
    assert (m.interests_covered, m.interests_total) == (1, 1)
    assert (m.llm_calls, m.cost_usd, m.seconds) == (40, 0.0123, 95.0)
    assert m.walk_km > 0
    assert (m.mean_pop, m.top_share) == (0.91, 0.17)  # one must-see among six stops


def test_a_run_of_trips_keeps_the_single_walks_the_deck_reads(tmp_path) -> None:
    # A row as runs before avoid hits, interests and popularity wrote it: the .md still
    # renders it.
    old = {"stops": 9, "found": 9, "closed": 0, "impossible_legs": 5, "over_window": False}
    old |= {"walk_km": 21.6, "llm_calls": 1, "cost_usd": 0.0002, "seconds": 22.9}
    walks = [{"id": "spb-constructivism", "model": "m", "agent": old, "baseline": old}]
    first = {"commit": "abc1234", "run_at": "2026-10-20T10:00+03:00"}
    write_results(tmp_path, "m", "m", first, walks, [])

    trip = {"days_requested": 3, "days_planned": 3, "stops_per_day": [8, 9, 7], "closed": 0}
    trip |= {"impossible_legs": 0, "days_over_window": 0, "walk_km": 24.0, "repeats": 0}
    trip |= {"avoid_hits": 0, "interests_covered": 4, "interests_total": 4}
    trip |= {"mean_pop": 0.62, "top_share": 0.25}
    trip |= {"llm_calls": 50, "cost_usd": 0.02, "seconds": 180.0}
    journeys = [
        {"id": "kazan-3-days", "model": "m", "agent": trip},
        {"id": "spb-5-days", "model": "m", "error": "LLMError: timeout\nmore lines"},
    ]
    second = {"commit": "def5678+dirty", "run_at": "2026-10-21T10:00+03:00"}
    write_results(tmp_path, "m", "m", second, [], journeys)

    assert json.loads((tmp_path / "m.json").read_text()) == walks
    assert json.loads((tmp_path / "m-journeys.json").read_text()) == journeys
    meta = json.loads((tmp_path / "m.meta.json").read_text())
    assert meta["m.json"]["commit"] == "abc1234"
    assert meta["m-journeys.json"] == second | {"queries": ["kazan-3-days", "spb-5-days"]}
    md = (tmp_path / "m.md").read_text()
    assert "commit `abc1234`" in md and "commit `def5678+dirty`" in md
    assert "| spb-constructivism | baseline | 9 | 9/9 | 0 | 5 | no | — | — | — | — | 21.6 |" in md
    assert "| kazan-3-days | 3/3 | 8, 9, 7 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0.62 | 0.25 | 24.0 |" in md
    assert "| spb-5-days | error | LLMError: timeout |" in md
