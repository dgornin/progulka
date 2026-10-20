"""The eval's scoring (eval/run_eval.py) on hand-made places and stops: no LLM, no network.
A live eval run spends the API key, so the arithmetic behind its tables is checked here."""

import json
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

import pytest

from app.llm import LLM
from app.models import AgentEvent, GeoPoint, Interest, Place, Stop, TripBrief
from app.pipeline import PLANNER_FALLBACK, Journey
from app.sources import wikidata
from app.sources.http import http
from eval.run_eval import (
    HARNESS_SEARCH,
    BaselineStop,
    Point,
    _locate,
    agent_points,
    avoid_hits,
    baseline_point,
    check_route,
    harness_answer,
    interest_coverage,
    nominatim_kind,
    planner_outcome,
    popularity_stats,
    repeats_across_days,
    route_items,
    run_harness,
    score,
    score_journey,
    stored_reading,
    summary_sections,
    summary_systems,
    summary_table,
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


def test_a_trip_row_says_how_the_trip_planner_ended() -> None:
    def stage(**detail: Any) -> AgentEvent:
        data = {"name": "synthesize", "status": "done", "label": "…", "detail": detail}
        return AgentEvent(type="stage", data=data)

    started = AgentEvent(type="stage", data={"name": "synthesize", "status": "start"})
    finished = [started, stage(stop_reason="finished", steps=3, places=[5, 6])]
    gave_up = [
        started,
        stage(stop_reason="time_budget", steps=7),
        AgentEvent(type="warning", data={"message": PLANNER_FALLBACK}),
    ]

    assert planner_outcome(finished) == {"stop_reason": "finished", "steps": 3, "fallback": False}
    # Code built the days: the row must not be scored as the planner's.
    assert planner_outcome(gave_up) == {"stop_reason": "time_budget", "steps": 7, "fallback": True}
    # A single walk, TRIP_PLANNER_AGENT=0 or a pool too small to share: no planner ran.
    assert planner_outcome([AgentEvent(type="warning", data={"message": "День 2: …"})]) is None


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


def _answer(content: str, usage: dict, annotations: list | None = None) -> dict[str, Any]:
    message = {"role": "assistant", "content": content, "annotations": annotations or []}
    return {"choices": [{"message": message}], "usage": usage}


def test_harness_answer_counts_the_searches_and_each_cited_page_once() -> None:
    page = {"type": "url_citation", "url_citation": {"url": "https://sobor39.ru", "title": "Собор"}}
    other = {"type": "url_citation", "url_citation": {"url": "https://culture.ru/1"}}
    usage = {"prompt_tokens": 20478, "server_tool_use_details": {"web_search_requests": 5}}
    raw = _answer("{}", usage, [page, page, other, {"type": "file"}])
    assert harness_answer(raw) == ("{}", 5, 2)
    # OpenRouter's docs name the field server_tool_use; an engine may return no citations.
    raw = _answer("{}", {"server_tool_use": {"web_search_requests": 2}})
    assert harness_answer(raw) == ("{}", 2, 0)
    assert harness_answer({"choices": [{"message": {"content": None}}]}) == ("", 0, 0)


async def test_run_harness_counts_the_chat_answer_and_checks_its_copy_as_the_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []
    prose = "Я проверил часы работы. 11:00–12:00 Эрмитаж, 12:00–13:00 Кунсткамера."
    chat = _answer(
        prose,
        {"prompt_tokens": 20000, "completion_tokens": 700, "cost": 0.03}
        | {"server_tool_use_details": {"web_search_requests": 3}},
    )
    copy = _answer(
        '{"stops": [{"name": "Эрмитаж", "address": null, "arrive": "11:00", "depart": "12:00"},'
        ' {"name": "Кунсткамера", "address": null, "arrive": "12:00", "depart": "13:00"}]}',
        {"prompt_tokens": 500, "completion_tokens": 300, "cost": 0.001},
    )
    items = _answer(
        '{"items": [{"n": 1, "kind": "place"}, {"n": 2, "kind": "place"}]}',
        {"prompt_tokens": 200, "completion_tokens": 40, "cost": 0.0002},
    )

    async def call(self: LLM, request: dict, extra_body: dict, timeout: float) -> dict:
        sent.append(request)
        reasoning.append(extra_body.get("reasoning"))
        return [chat, copy, items][len(sent) - 1]

    async def get_json(url: str, params: dict[str, Any], **options: Any) -> list[dict]:
        if "featureType" in params:  # the city's bounding box: not needed here
            return []
        hit = {"lat": "59.93", "lon": "30.36", "category": "tourism", "type": "museum"}
        return [hit | {"name": params["q"], "extratags": {"wikidata": "Q132783"}}]

    async def scores(qids: Any, timeout_s: float | None = None) -> dict[str, float]:
        return {"Q132783": 0.91}

    monkeypatch.setattr(LLM, "_call_with_retries", call)
    monkeypatch.setattr(http, "get_json", get_json)
    monkeypatch.setattr(wikidata, "scores", scores)
    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    query = {"request": "Петербург, 24 октября, с 11 до 18", "avoid": ["музеи"]}
    base = LLM(client=None, model="anthropic/claude-haiku-5.5")

    metrics, names, answer, details = await run_harness(
        base, "anthropic/claude-opus-5.5", query, brief, GeoPoint(lat=59.93, lon=30.36), "low"
    )

    assert sent[0]["model"] == "anthropic/claude-opus-5.5"
    assert sent[0]["tools"] == [HARNESS_SEARCH]
    assert "Today is 2026-10-20" in sent[0]["messages"][0]["content"]
    # The copy and the kinds of items: the agent's model and setting.
    assert reasoning == [{"effort": "low"}, {"enabled": False}, {"enabled": False}]
    # The copy into JSON: the agent's model, without the search, given the prose alone.
    assert sent[1]["model"] == "anthropic/claude-haiku-5.5" and "tools" not in sent[1]
    assert sent[1]["messages"][1] == {"role": "user", "content": prose}
    assert (names, answer) == (["Эрмитаж", "Кунсткамера"], prose)
    assert (metrics.stops, metrics.found, metrics.avoid_hits, metrics.mean_pop) == (2, 2, 2, 0.91)
    assert (metrics.places, metrics.places_found, metrics.vague) == (2, 2, 0)
    assert details[1] == {
        "name": "Кунсткамера",
        "item": "place",
        "found": True,
        "lat": 59.93,
        "lon": 30.36,
        "arrive": "12:00",
        "depart": "13:00",
    }
    # Usage is the chat answer's alone: the copy is the eval's work.
    assert (metrics.llm_calls, metrics.prompt_tokens, metrics.completion_tokens) == (1, 20000, 700)
    assert (metrics.cost_usd, metrics.web_searches, metrics.cited_pages) == (0.03, 3, 0)
    assert base.usage.calls == 0

    sent.clear()
    chat["choices"][0]["finish_reason"] = "length"  # the answer stops mid-route
    with pytest.raises(ValueError, match="cut"):
        await run_harness(base, "m", query, brief, GeoPoint(lat=59.93, lon=30.36), "default")
    assert reasoning[-1] is None  # the model's own reasoning: no setting sent


def test_stored_reading_is_the_rows_brief_and_start_when_the_run_recorded_them() -> None:
    brief = TripBrief(
        city="Казань", date=DAY, start_time=time(10), end_time=time(17), pace="relaxed"
    )
    start = GeoPoint(lat=55.8, lon=49.1, label="Кремль")
    row = {"brief": brief.model_dump(mode="json"), "start": start.model_dump()}
    assert stored_reading(row) == (brief, start)
    assert stored_reading({"id": "kazan-history", "agent": {}}) is None  # an older run


def _route(stops: int, impossible: int, km: float, **more: Any) -> dict[str, Any]:
    m = {"stops": stops, "found": stops, "closed": 0, "impossible_legs": impossible}
    m |= {"over_window": False, "walk_km": km, "avoid_hits": 0, "mean_pop": None}
    m |= {"top_share": None, "llm_calls": 1, "cost_usd": 0.01, "seconds": 10.0}
    return m | more


def test_the_summary_compares_every_system_on_the_walks_all_of_them_answered() -> None:
    walks = [
        {"id": "a", "agent": _route(10, 0, 5.0), "baseline": _route(8, 3, 14.0, found=6)},
        {"id": "b", "agent": _route(12, 0, 4.0), "baseline": _route(9, 0, 9.0)},
    ]
    tokens = {"prompt_tokens": 20000, "completion_tokens": 1000, "mean_pop": 0.6}
    tokens |= {"transit_impossible": 0, "places": 5, "places_found": 4, "vague": 1}
    harness = [
        {"id": "a", "model": "h", "harness": _route(6, 1, 7.0, **tokens)},
        {"id": "b", "model": "h", "error": "LLMError: timeout"},
    ]
    systems = summary_systems("m", walks, harness)
    assert [label for label, _ in systems] == [
        "agent, `m`",
        "one prompt, `m`",
        "chat + web search, `h`",
    ]
    assert all(len(routes) == 1 for _, routes in systems)  # walk "b" failed for the harness
    table = summary_table(systems)
    # Transit legs, places and tokens: — for the runs before they were measured.
    assert "| agent, `m` | 1 | 0 (0/1) | — | 0 | — | — | 5.0 | — | — | 0.01 | 10.0 |" in table
    assert "| one prompt, `m` | 1 | 3 (1/1) | — | 0 | — | — | 14.0 | — | — |" in table
    assert "| chat + web search, `h` | 1 | 1 (1/1) | 0 (0/1) | 0 | 4/5 (80%) | 1 | 7.0 | 0.6 |" in (
        table
    )
    assert "| 21000 |" in table
    assert summary_systems("m", [], []) == []


def test_a_harness_model_with_fewer_walks_gets_a_summary_without_it() -> None:
    walks = [{"id": q, "agent": _route(9, 0, 5.0), "baseline": _route(8, 2, 12.0)} for q in "ab"]
    harness = [{"id": q, "model": "full", "harness": _route(7, 1, 9.0)} for q in "ab"]
    harness.append({"id": "a", "model": "cut", "harness": _route(6, 0, 8.0)})
    first, second = summary_sections("m", walks, harness)
    assert "| chat + web search, `cut` | 1 |" in first and "| agent, `m` | 1 |" in first
    assert second.startswith("Without `cut`, which answered fewer walks: over 2 walks.")
    assert "| chat + web search, `full` | 2 |" in second and "`cut`" not in second.split("\n", 1)[1]
    # Every model answered every walk, or none did: one table.
    assert len(summary_sections("m", walks, harness[:2])) == 1
    assert len(summary_sections("m", walks, harness[2:])) == 1


def test_a_harness_run_keeps_the_rows_of_the_other_models_and_queries(tmp_path) -> None:
    walks = [{"id": q, "model": "m", "agent": _route(9, 0, 5.0)} for q in ("a", "b")]
    first = {"commit": "abc1234", "run_at": "2026-10-20T20:00+03:00"}
    write_results(tmp_path, "m", "m", first, walks, [])
    searched = {"web_searches": 3, "cited_pages": 0, "prompt_tokens": 9, "completion_tokens": 1}
    haiku = [{"id": q, "model": "haiku", "harness": _route(7, 2, 9.0, **searched)} for q in "ab"]
    write_results(tmp_path, "m", "m", first, [], [], haiku)

    second = {"commit": "def5678+dirty", "run_at": "2026-10-20T21:00+03:00"}
    opus = {"id": "a", "model": "opus", "reasoning": "low"}
    opus["harness"] = _route(8, 0, 6.0, **searched)
    rerun = {"id": "b", "model": "haiku", "error": "LLMError: no choices"}
    write_results(tmp_path, "m", "m", second, [], [], [opus, rerun])

    rows = json.loads((tmp_path / "m-harness.json").read_text())
    assert [(r["model"], r["id"]) for r in rows] == [("haiku", "a"), ("opus", "a"), ("haiku", "b")]
    assert json.loads((tmp_path / "m.json").read_text()) == walks  # the agent's are untouched
    meta = json.loads((tmp_path / "m.meta.json").read_text())["m-harness.json"]
    # Haiku's row a is still the first run's: the rerun of b must not claim it.
    assert meta["haiku"] == [first | {"queries": ["a"]}, second | {"queries": ["b"]}]
    assert meta["opus"] == [second | {"reasoning": "low", "queries": ["a"]}]
    md = (tmp_path / "m.md").read_text()
    assert (
        "`haiku`: run 2026-10-20T20:00+03:00, commit `abc1234` (a); "
        "run 2026-10-20T21:00+03:00, commit `def5678+dirty` (b); "
        "`opus`: run 2026-10-20T21:00+03:00, commit `def5678+dirty`, reasoning low. "
    ) in md
    assert "| a | haiku | 7 | 7/7 | 0 | 2 | no | 0 | — | — | 9.0 | 3 | — | 10 | 0.01 | 10.0 |" in md
    assert "| b | haiku | error | LLMError: no choices |" in md
    # Walk b failed for haiku: the summary compares walk a only.
    assert "| chat + web search, `opus` | 1 | 0 (0/1) | — | 0 | — | — | 6.0 |" in md

    write_results(tmp_path, "m", "m", second, [], [], haiku)  # every walk of haiku again
    meta = json.loads((tmp_path / "m.meta.json").read_text())["m-harness.json"]
    assert meta["haiku"] == [second | {"queries": ["a", "b"]}]


def _timed(name: str, lat: float, arrive: str, depart: str) -> Point:
    at, leave = (datetime.combine(DAY, time.fromisoformat(t)) for t in (arrive, depart))
    return Point(name, None, lat, 30.36, None, at, leave)


def test_a_leg_is_too_short_by_transit_when_faster_than_walking_and_10_min_plus_20_kmh() -> None:
    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    start = GeoPoint(lat=59.93, lon=30.36)
    # About 3.3 km north each (0.03 degrees of latitude), more in the streets: an hour on
    # foot, under 25 minutes by transit. 10 minutes (+5) is too short for both, 30 for the
    # walk only; 2 minutes to a stop next door is walked, and no transport is needed.
    points = [
        _timed("Далеко", 59.96, "11:10", "11:40"),
        _timed("Ещё дальше", 59.99, "12:10", "12:30"),
        _timed("Рядом", 59.9905, "12:32", "12:40"),
    ]
    route = check_route(points, start, brief)
    assert (route.impossible_legs, route.transit_impossible) == (2, 1)


def test_found_in_osm_counts_the_specific_places_and_the_rest_are_vague() -> None:
    brief = TripBrief(city="Санкт-Петербург", date=DAY, start_time=time(11), end_time=time(18))
    start = GeoPoint(lat=59.93, lon=30.36)
    found = _timed("Эрмитаж", 59.93, "11:00", "12:00")
    lost = _timed("Нигде", 59.93, "12:00", "12:10")
    lost.lat = lost.lon = None
    transition = _timed("Переход по Невскому", 59.93, "12:10", "12:20")
    transition.lat = transition.lon = None
    items = ["place", "place", "transition"]
    m = score([found, lost, transition], start, brief, [], {}, 1.0, None, None, items)
    assert (m.found, m.stops, m.places_found, m.places, m.vague) == (1, 3, 1, 2, 1)
    assert score([found], start, brief, [], {}, 1.0).places is None  # not measured


async def test_route_items_ask_once_and_count_a_skipped_item_as_a_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    async def call(self: LLM, request: dict, extra_body: dict, timeout: float) -> dict:
        sent.append(request)
        return _answer('{"items": [{"n": 2, "kind": "endpoint"}]}', {"prompt_tokens": 9})

    monkeypatch.setattr(LLM, "_call_with_retries", call)
    base = LLM(client=None, model="anthropic/claude-haiku-5.5")
    assert await route_items(base, ["Эрмитаж", "Завершение маршрута"]) == ["place", "endpoint"]
    assert len(sent) == 1 and sent[0]["messages"][1]["content"] == (
        "1. Эрмитаж\n2. Завершение маршрута"
    )
    assert base.usage.calls == 0  # the eval's work, not counted to any system
    assert await route_items(base, []) == []
