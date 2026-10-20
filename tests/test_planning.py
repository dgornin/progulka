"""Deterministic planning: distances, opening hours, scheduling, optimizer, routing, validator.

Places are synthetic points around one start point, so every expected time can be
worked out by hand; OSRM is replaced by a recorded response.
"""

import asyncio
import json
import random
from datetime import date, datetime, time
from pathlib import Path

import pytest

from app.models import Candidate, GeoPoint, HourWeather, Interest, Place, Plan, Stop, TripBrief
from app.planning import routing
from app.planning.geo import haversine_m, street_meters, walk_limit_km, walk_minutes
from app.planning.hours import earliest_open_at_or_after, is_open, open_intervals
from app.planning.optimizer import optimize, schedule
from app.planning.validator import matches_avoid, named_kinds, names_place, validate
from app.sources.http import SourceError, http

FIXTURES = Path(__file__).parent / "fixtures"
WEDNESDAY = date(2026, 10, 21)
SATURDAY = date(2026, 10, 24)
START = GeoPoint(lat=59.9343, lon=30.3248)
WEEKDAYS = "Mo-Fr 10:00-18:00"


def _at(day: date, hhmm: str) -> datetime:
    return datetime.combine(day, time.fromisoformat(hhmm))


def _place(
    pid: str,
    kind: str = "museum",
    hours: str | None = None,
    indoor: bool | None = None,
    dlat: float = 0.0,
    dlon: float = 0.0,
    name: str | None = None,
) -> Place:
    return Place(
        id=pid,
        name=name or pid,
        lat=START.lat + dlat,
        lon=START.lon + dlon,
        kind=kind,
        source="osm",
        opening_hours=hours,
        indoor=indoor,
    )


def _cand(
    pid: str,
    score: float = 0.8,
    minutes: int = 30,
    required: bool = False,
    interests: list[str] | None = None,
    **place,
) -> Candidate:
    return Candidate(
        place=_place(pid, **place),
        score=score,
        interests=interests or [],
        reason="test",
        visit_minutes=minutes,
        required=required,
    )


def _brief(day: date = WEDNESDAY, start: str = "10:00", end: str = "18:00", **kw) -> TripBrief:
    return TripBrief(
        city="Санкт-Петербург",
        date=day,
        start_time=time.fromisoformat(start),
        end_time=time.fromisoformat(end),
        **kw,
    )


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.place.id for c in candidates]


# --- geo ---------------------------------------------------------------------


def test_distances_and_walking_time():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)  # one degree of latitude
    a, b = _place("a"), _place("b", dlat=0.009)
    assert street_meters(a, b) == pytest.approx(1.3 * 1000.7, rel=1e-3)
    assert walk_minutes(4500, "normal") == pytest.approx(60)


# --- opening hours -----------------------------------------------------------


def test_walk_limit_defaults_by_pace_and_children():
    assert walk_limit_km(_brief()) == 10
    assert walk_limit_km(_brief(pace="relaxed", with_kids=True)) == 5
    assert walk_limit_km(_brief(max_walk_km=3.5)) == 3.5


def test_weekday_hours():
    assert open_intervals(WEEKDAYS, SATURDAY) == []
    assert is_open(WEEKDAYS, _at(SATURDAY, "11:00"), _at(SATURDAY, "12:00")) is False
    assert is_open(WEEKDAYS, _at(WEDNESDAY, "11:00"), _at(WEDNESDAY, "12:00")) is True
    assert is_open(WEEKDAYS, _at(WEDNESDAY, "17:30"), _at(WEDNESDAY, "18:30")) is False
    assert is_open(WEEKDAYS, _at(WEDNESDAY, "17:30"), _at(WEDNESDAY, "18:00")) is True


def test_always_open():
    assert open_intervals("24/7", SATURDAY) == [
        (_at(SATURDAY, "00:00"), _at(date(2026, 10, 25), "00:00"))
    ]
    assert is_open("24/7", _at(SATURDAY, "23:00"), _at(SATURDAY, "23:59")) is True


@pytest.mark.parametrize("expr", [None, "", "   ", "ask the guard", "Mo-Fr 10-18"])
def test_missing_or_garbage_hours_are_unknown(expr):
    assert open_intervals(expr, WEDNESDAY) is None
    assert is_open(expr, _at(WEDNESDAY, "11:00"), _at(WEDNESDAY, "12:00")) is None
    assert earliest_open_at_or_after(expr, _at(WEDNESDAY, "09:00"), _at(WEDNESDAY, "18:00")) is None


def test_explicit_unknown_state():
    expr = "Mo-Fr 10:00-18:00; Sa unknown"
    assert is_open(expr, _at(SATURDAY, "11:00"), _at(SATURDAY, "12:00")) is None
    assert open_intervals(expr, SATURDAY) is None


def test_earliest_open():
    def earliest(day: date, hhmm: str) -> datetime | None:
        return earliest_open_at_or_after(WEEKDAYS, _at(day, hhmm), _at(day, "20:00"))

    assert earliest(WEDNESDAY, "09:00") == _at(WEDNESDAY, "10:00")  # wait for opening
    assert earliest(WEDNESDAY, "12:15") == _at(WEDNESDAY, "12:15")  # already open
    assert earliest(WEDNESDAY, "18:30") is None  # closed for the rest of the day
    assert earliest(SATURDAY, "09:00") is None


# --- schedule ----------------------------------------------------------------


def test_schedule_waits_for_opening():
    museum = _cand("museum", hours="Mo-Su 10:00-18:00", dlat=0.009)  # 1301 m street = 17.3 min
    statue = _cand("statue", kind="artwork", minutes=20, dlat=0.0045)  # 650 m back = 8.7 min
    result = schedule([museum, statue], START, _brief(start="09:00"))

    first, second = result.stops
    assert first.walk_minutes_from_prev == 18
    assert first.arrive == _at(WEDNESDAY, "10:00")  # came at 09:18, waited for opening
    assert first.depart == _at(WEDNESDAY, "10:30")
    assert second.arrive == _at(WEDNESDAY, "10:39")
    assert second.depart == _at(WEDNESDAY, "10:59")
    assert result.total_walk_m == 1951
    assert result.total_minutes == 119


def test_schedule_uses_routed_leg_lengths():
    order = [_cand("a", dlat=0.009), _cand("b", dlat=0.018)]
    result = schedule(order, START, _brief(), leg_meters=[450.0, 900.0])  # 6 and 12 min at 4.5 km/h

    assert [s.walk_meters_from_prev for s in result.stops] == [450, 900]
    assert [s.arrive for s in result.stops] == [_at(WEDNESDAY, "10:06"), _at(WEDNESDAY, "10:48")]
    assert result.total_walk_m == 1350


# --- optimizer ---------------------------------------------------------------


def test_optimizer_respects_end_time():
    candidates = [_cand(f"p{i}", minutes=45, dlat=0.001 * i) for i in range(8)]
    brief = _brief(end="12:00")
    result = optimize(candidates, START, brief)

    stops = schedule(result.order, START, brief).stops
    assert len(stops) == 2
    assert stops[-1].depart <= _at(WEDNESDAY, "12:00")
    assert sorted(result.dropped + _ids(result.order)) == sorted(_ids(candidates))


def test_optimizer_skips_place_closed_all_day():
    closed = _cand("closed", score=1.0, hours=WEEKDAYS)
    others = [
        _cand("a", kind="artwork", score=0.5, dlat=0.002),
        _cand("b", kind="park", score=0.5, dlat=0.004),
    ]
    result = optimize([closed, *others], START, _brief(day=SATURDAY))

    assert result.dropped == ["closed"]
    assert sorted(_ids(result.order)) == ["a", "b"]


def test_required_place_is_kept_and_reported():
    required = _cand("wanted", score=0.3, hours=WEEKDAYS, required=True)
    others = [_cand("a", kind="artwork", dlat=0.002), _cand("b", kind="park", dlat=0.004)]
    brief = _brief(day=SATURDAY)
    result = optimize([required, *others], START, brief)

    assert sorted(_ids(result.order)) == ["a", "b", "wanted"]  # the rest is still planned around it
    s = schedule(result.order, START, brief)
    plan = Plan(
        brief=brief,
        start=START,
        stops=s.stops,
        total_walk_m=s.total_walk_m,
        total_minutes=s.total_minutes,
    )
    assert [(v.code, v.place_id) for v in validate(plan)] == [("closed_at_visit", "wanted")]


def test_closed_required_place_is_not_traded_for_another_closed_one():
    # "wanted" at 10:30-11:30 runs into its lunch break. Visiting "x" first would push
    # "wanted" to 12:00, but "x" itself is closed after 10:00: one closed stop for another.
    wanted = _cand("wanted", minutes=60, required=True, hours="Mo-Su 10:30-11:00,12:00-18:00")
    x = _cand("x", score=1.0, minutes=90, hours="Mo-Su 09:00-10:00")
    result = optimize([wanted, x], START, _brief(end="14:00"))

    assert _ids(result.order) == ["wanted"]
    assert result.dropped == ["x"]


def test_lunch_does_not_push_a_stop_into_its_break():
    # w, b, s fill 11:00-14:00; the cafe (opens 13:00) only fits if w makes room.
    # Without w, s would start at 12:00 and run into its 12:30-13:00 break: no lunch then.
    route = [
        _cand("w", kind="artwork", score=0.2, minutes=60),
        _cand("b", kind="museum", score=0.8, minutes=60),
        _cand("s", kind="gallery", score=0.9, minutes=60, hours="Mo-Su 12:00-12:30,13:00-18:00"),
    ]
    cafe = _cand("cafe", kind="cafe", score=0.3, minutes=90, hours="Mo-Su 13:00-22:00")
    brief = _brief(start="11:00", end="15:00")
    result = optimize([*route, cafe], START, brief)

    assert _ids(result.order) == ["w", "b", "s"]
    assert all(
        is_open(s.place.opening_hours, s.arrive, s.depart) is not False
        for s in schedule(result.order, START, brief).stops
    )


def _lunch_case(start: str) -> tuple[Candidate, list[Candidate], TripBrief]:
    kinds = ["museum", "park", "church", "artwork", "gallery"]
    sights = [
        _cand(f"s{i}", kind=kinds[i % 5], score=0.9, minutes=75, dlat=0.001 * i)
        for i in range(1, 9)
    ]
    cafe = _cand("cafe", kind="cafe", score=0.1, hours="Mo-Su 08:00-22:00", dlon=0.001)
    return cafe, sights, _brief(start=start, end="18:00")


def test_long_walk_over_lunch_gets_a_food_stop():
    # Six 75-minute sights fill 10:00-18:00 so tightly that the cafe does not fit by itself.
    cafe, sights, brief = _lunch_case(start="10:00")
    result = optimize([cafe, *sights], START, brief)

    assert "cafe" in _ids(result.order)
    lunch = next(s for s in schedule(result.order, START, brief).stops if s.place.id == "cafe")
    assert time(12) <= lunch.arrive.time() <= time(15)


def test_short_afternoon_walk_has_no_forced_lunch():
    cafe, sights, brief = _lunch_case(start="14:00")
    assert "cafe" in optimize([cafe, *sights], START, brief).dropped


def test_rain_prefers_indoor_place():
    # Same score, same distance, only one fits: the tie goes to the first id, unless it rains.
    park = _cand("a-park", kind="park", indoor=False, minutes=45, dlat=0.003)
    museum = _cand("b-museum", kind="museum", indoor=True, minutes=45, dlat=-0.003)
    brief = _brief(end="11:00")
    rain = [HourWeather(time=_at(WEDNESDAY, "10:00"), precipitation_probability=80)]

    assert _ids(optimize([park, museum], START, brief).order) == ["a-park"]
    assert _ids(optimize([park, museum], START, brief, weather=rain).order) == ["b-museum"]


def test_third_stop_of_the_same_kind_is_penalised():
    churches = [_cand(f"church{i}", kind="church", score=0.9, dlat=0.001 * i) for i in (1, 2, 3)]
    statue = _cand("statue", kind="artwork", score=0.8, dlon=0.002)
    result = optimize([*churches, statue], START, _brief(end="11:45"))  # room for three stops

    kinds = [c.place.kind for c in result.order]
    assert sorted(kinds) == ["artwork", "church", "church"]


def test_interests_have_diminishing_returns():
    # A mural takes half the time of the building, so by value per minute murals
    # would fill the hour; the decay makes a fourth mural worth less than the building.
    murals = [
        _cand(f"mural{i}", kind="artwork", minutes=10, dlat=0.0005 * i, interests=["art"])
        for i in range(1, 6)
    ]
    building = _cand("house", kind="architecture", minutes=20, dlon=0.003, interests=["arch"])
    result = optimize([*murals, building], START, _brief(end="11:00"))  # not enough time for all

    assert "house" in _ids(result.order)


def test_optimizer_is_deterministic():
    rng = random.Random(7)
    kinds = ["museum", "park", "church", "cafe", "artwork"]
    candidates = [
        _cand(
            f"p{i:02d}",
            kind=kinds[i % 5],
            score=round(rng.uniform(0.3, 1), 2),
            minutes=rng.choice([20, 40, 60]),
            hours=rng.choice([None, "Mo-Su 11:00-17:00", "24/7"]),
            dlat=rng.uniform(-0.01, 0.01),
            dlon=rng.uniform(-0.02, 0.02),
        )
        for i in range(25)
    ]
    first = optimize(candidates, START, _brief())
    second = optimize(list(reversed(candidates)), START, _brief())

    assert _ids(first.order) == _ids(second.order)
    assert sorted(first.dropped) == sorted(second.dropped)


# --- routing -----------------------------------------------------------------

ROUTE_POINTS = [
    GeoPoint(lat=59.9390, lon=30.3158),
    GeoPoint(lat=59.9343, lon=30.3248),
    GeoPoint(lat=59.9341, lon=30.3061),
]


def _fake_osrm(monkeypatch, response):
    async def get_json(url, params=None):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(http, "get_json", get_json)


async def test_walking_route_reads_osrm(monkeypatch):
    _fake_osrm(monkeypatch, json.loads((FIXTURES / "planning_osrm_route.json").read_text()))
    route = await routing.walking_route(ROUTE_POINTS)

    assert route.geometry[0] == [59.939503, 30.31507]  # GeoJSON [lon, lat] turned into [lat, lon]
    assert route.leg_meters == [1087.7, 1299.3]


@pytest.mark.parametrize(
    "response", [SourceError("down"), {"code": "NoRoute", "routes": []}], ids=["down", "no-route"]
)
async def test_walking_route_failure_gives_none(monkeypatch, response):
    _fake_osrm(monkeypatch, response)
    assert await routing.walking_route(ROUTE_POINTS) is None


async def test_slow_routing_service_gives_none(monkeypatch):
    async def hanging(url, params=None):
        await asyncio.sleep(60)

    monkeypatch.setattr(http, "get_json", hanging)
    monkeypatch.setattr(routing, "ROUTE_TIMEOUT_S", 0.01)
    assert await routing.walking_route(ROUTE_POINTS) is None


async def test_long_polyline_is_thinned(monkeypatch):
    line = [[30.0 + i * 1e-5, 59.0] for i in range(5001)]
    _fake_osrm(
        monkeypatch,
        {
            "code": "Ok",
            "routes": [
                {
                    "geometry": {"coordinates": line},
                    "legs": [{"distance": 600.0, "duration": 480.0}],
                }
            ],
        },
    )
    route = await routing.walking_route(ROUTE_POINTS[:2])

    assert len(route.geometry) <= routing.MAX_POLYLINE_POINTS
    assert route.geometry[0] == [59.0, 30.0] and route.geometry[-1] == [59.0, line[-1][0]]


# --- validator ---------------------------------------------------------------


def _plan(
    museum_hours: str | None = "Mo-Su 10:00-18:00",
    statue_indoor: bool | None = None,
    stops: int = 2,
    **brief,
) -> Plan:
    """Clean by default: a museum 11:00-12:00, then a statue 12:10-12:30."""
    museum = Stop(
        place=_place("museum", hours=museum_hours, indoor=True, name="Музей"),
        arrive=_at(WEDNESDAY, "11:00"),
        depart=_at(WEDNESDAY, "12:00"),
        score=0.8,
        interests=["History"],
        reason="test",
    )
    statue = Stop(
        place=_place("statue", kind="artwork", indoor=statue_indoor, name="Памятник"),
        arrive=_at(WEDNESDAY, "12:10"),
        depart=_at(WEDNESDAY, "12:30"),
        score=0.8,
        interests=["street art"],
        reason="test",
    )
    return Plan(
        brief=_brief(**brief),
        start=START,
        stops=[museum, statue][:stops],
        total_walk_m=1000,
        total_minutes=150,
    )


def test_clean_plan_has_no_violations():
    plan = _plan(
        interests=[Interest(label="history", weight=1), Interest(label="coffee", weight=0.5)]
    )
    assert validate(plan) == []


RAIN_AT_NOON = [HourWeather(time=_at(WEDNESDAY, "12:00"), precipitation_mm=1.2)]


@pytest.mark.parametrize(
    "plan, weather, code, severity",
    [
        pytest.param(
            _plan(museum_hours="Mo-Su 12:00-18:00"), None, "closed_at_visit", "error", id="closed"
        ),
        pytest.param(_plan(end="12:15"), None, "over_time", "error", id="late"),
        pytest.param(_plan(museum_hours=None), None, "unknown_hours", "warning", id="no-hours"),
        pytest.param(
            _plan(statue_indoor=False), RAIN_AT_NOON, "rain_outdoor", "warning", id="rain"
        ),
        pytest.param(_plan(avoid=["без музеев"]), None, "avoid_violation", "error", id="by-kind"),
        pytest.param(_plan(avoid=["памятник"]), None, "avoid_violation", "error", id="by-name"),
        pytest.param(_plan(max_walk_km=0.5), None, "long_walk", "warning", id="long-walk"),
        pytest.param(
            _plan(interests=[Interest(label="parks", weight=0.8)]),
            None,
            "uncovered_interest",
            "warning",
            id="interest",
        ),
        pytest.param(_plan(stops=1), None, "too_few_stops", "error", id="one-stop"),
        pytest.param(
            _plan(must_include=["Эрмитаж"]), None, "missing_must_include", "warning", id="must"
        ),
    ],
)
def test_each_violation(plan, weather, code, severity):
    violations = validate(plan, weather)
    assert [(v.code, v.severity) for v in violations] == [(code, severity)]


def test_avoid_phrase_matches_whole_words_of_the_name():
    plan = _plan(avoid=["кафе"])
    plan.stops[0].place.name = "Кафедральный собор"  # "кафе" is only the start of a word here
    assert validate(plan) == []


@pytest.mark.parametrize(
    "avoid, kind, name, expected",
    [
        ("музеи", "landmark", "Русский музей", True),  # a Wikipedia place: the name decides
        ("храмы", "landmark", "Исаакиевский собор", True),
        ("кафе", "cafe", "Кофемания", True),
        ("бары", "landmark", "Дом Баранова", False),  # short stems never match names
        ("кафе", "landmark", "Кафедральный собор", False),
        # A word of the phrase is a stem plus a short ending, not any word it starts.
        ("барокко", "bar", "Бар", False),
        ("без парковок", "park", "Парк", False),
        ("без кинотеатров", "cinema", "Аврора", True),
        ("no shopping", "shop", "Shop", True),
        ("без пивнушек", "pub", "Паб", True),
        ("алкогольные напитки", "bar", "Бар", True),
        ("ресторанчики", "landmark", "Ресторанчик у моста", True),
    ],
)
def test_avoid_matches_kind_or_a_word_of_the_name(avoid, kind, name, expected):
    assert matches_avoid(avoid, kind, name) is expected


@pytest.mark.parametrize(
    "text, kinds",
    [
        ("без музеев", {"museum", "gallery"}),
        ("Бары и тусовки", {"bar", "pub"}),
        ("барокко", set()),
        ("без парковок", set()),
    ],
)
def test_named_kinds_is_what_the_brief_merge_and_the_validator_share(text, kinds):
    assert named_kinds(text) == kinds


@pytest.mark.parametrize(
    "wanted, name, expected",
    [
        ("Казанский собор", "Казанский кафедральный собор", True),
        ("Эрмитаж", "Государственный Эрмитаж", True),
        ("Спас на Крови", "Храм Спаса на Крови", True),
        ("Эрмитаж", "Эрмитажный театр", False),
    ],
)
def test_names_place_matches_words_not_substrings(wanted, name, expected):
    assert names_place(wanted, name) is expected


def test_messages_name_the_place_and_time():
    [ru] = validate(_plan(museum_hours="Mo-Su 12:00-18:00"))
    [en] = validate(_plan(museum_hours="Mo-Su 12:00-18:00", language="en"))
    assert ru.message == "«Музей»: в 11:00–12:00 закрыто; в этот день открыто 12:00–18:00"
    assert en.message == "Музей is closed at 11:00–12:00; open 12:00–18:00 that day"
    assert ru.place_id == "museum"
