"""The whole pipeline offline: scripted LLM, fake sources, real optimizer and validator."""

import copy
from datetime import date, datetime, time

import pytest

from app import pipeline
from app.llm import ToolCall
from app.models import GeoPoint, HourWeather, Place, TripBrief
from app.planning.routing import RouteResult
from app.sources import geocode, osm, weather
from app.sources.geocode import CityInfo
from app.sources.http import SourceError
from tests.test_agent import FakeLLM, Recorder, call, turn

CENTER = GeoPoint(lat=55.7963, lon=49.1088, label="Казанский кремль")
DAY = date(2026, 10, 24)


def _place(n: int, kind: str, dlat: float) -> Place:
    return Place(
        id=f"osm:node/{n}",
        name=f"Место {n}",
        lat=CENTER.lat + dlat,
        lon=CENTER.lon,
        kind=kind,
        source="osm",
        indoor=kind == "museum",
    )


PLACES = [_place(1, "museum", 0.002), _place(2, "church", 0.004), _place(3, "artwork", -0.002)]
PLACES.append(_place(4, "cafe", -0.004))
SPARE = _place(5, "museum", 0.003)  # found by search, shortlisted only as a replacement


@pytest.fixture
def sources(monkeypatch):
    async def locate_city(city, country=None):
        return CityInfo(name="Казань", center=CENTER, bbox=(55.7, 49.0, 55.9, 49.2))

    async def no_route(points):
        return None  # the pipeline falls back to straight lines

    async def hourly(point, day, today):
        return []

    async def search_places(center, radius_m, categories, limit=40):
        return [*PLACES, SPARE]

    monkeypatch.setattr(geocode, "locate_city", locate_city)
    monkeypatch.setattr(weather, "hourly", hourly)
    monkeypatch.setattr(osm, "search_places", search_places)
    monkeypatch.setattr(pipeline, "walking_route", no_route)


def _brief_reply() -> dict:
    return {
        "city": "Казань",
        "date": DAY.isoformat(),
        "start_time": "10:00",
        "end_time": "13:00",
        "interests": [{"label": "история", "weight": 1.0}],
    }


def _research_turns() -> list:
    shortlist = [
        call(
            "shortlist_place",
            place_id=p.id,
            score=0.8,
            interests=["история"],
            reason=f"{p.name} — по теме",
            visit_minutes=30,
        )
        for p in PLACES
    ]
    return [
        turn(call("search_osm", categories=["museum", "church"], radius_m=1000)),
        turn(*shortlist, call("finish_research", summary="готово")),
    ]


def _texts(stop_ids: list[str]) -> dict:
    return {
        "title": "Казань за утро",
        "summary": "Коротко.",
        "stops": [{"place_id": i, "blurb": "Текст.", "tip": None} for i in stop_ids],
    }


def _shortlist_spare() -> ToolCall:
    return call(
        "shortlist_place",
        place_id=SPARE.id,
        score=0.7,
        interests=["история"],
        reason="замена",
        visit_minutes=30,
    )


async def _planned(events: Recorder, **request: str) -> tuple[pipeline.Session, FakeLLM]:
    llm = FakeLLM(
        turns=_research_turns(), structured=[_brief_reply(), _texts([p.id for p in PLACES])]
    )
    req = pipeline.PlanRequest(request="Казань, 24 октября, с 10 до 13, история", **request)
    return await pipeline.plan_trip(req, llm, events, DAY), llm


async def test_plan_then_remove_a_stop(sources):
    events = Recorder()
    session, llm = await _planned(events)

    done = [e.data["name"] for e in events if e.type == "stage" and e.data["status"] == "done"]
    assert done == ["brief", "weather", "research", "optimize", "route", "validate", "write"]
    plan = session.plan
    assert {s.place.id for s in plan.stops} == {p.id for p in PLACES}
    assert plan.title == "Казань за утро"
    assert plan.geometry is None  # no walking route: the UI draws dotted straight lines
    assert not [v for v in plan.violations if v.severity == "error"]
    assert pipeline.get_session(session.id) is session

    # The "Убрать" button: no LLM interpretation, a short research run that must find a
    # replacement (finish is refused below the minimum), then the same optimizer and
    # validator.
    removed = plan.stops[0].place
    llm.turns = [
        turn(call("finish_research", summary="замена не нужна")),
        turn(_shortlist_spare(), call("finish_research", summary="нашёл замену")),
    ]
    llm.structured_replies = [_texts([SPARE.id, *(p.id for p in PLACES if p.id != removed.id)])]
    session, reply = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(exclude_place_ids=[removed.id]), llm, events
    )

    stops = {s.place.id for s in session.plan.stops}
    assert removed.id not in stops and SPARE.id in stops
    assert removed.id in session.state.excluded
    assert f"The traveller removed: {removed.name}." in llm.seen[-3][1]["content"]
    assert reply


async def test_new_avoid_phrase_drops_matching_places_without_new_research(sources):
    events = Recorder()
    session, llm = await _planned(events)
    museum = PLACES[0].id
    llm.structured_replies = [
        {"add_avoid": ["музеи"], "reply": "Уберу музеи."},
        _texts([p.id for p in PLACES]),
    ]

    session, _ = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(feedback="без музеев"), llm, events
    )

    assert museum not in {s.place.id for s in session.plan.stops}
    assert museum in session.state.excluded
    assert not [v for v in session.plan.violations if v.severity == "error"]


async def test_feedback_without_new_places_still_gets_the_repair_round(sources, monkeypatch):
    events = Recorder()
    session, llm = await _planned(events)
    kept = session.plan.stops[0].place.id
    others = [s.place.id for s in session.plan.stops[1:]]
    monkeypatch.setattr(pipeline, "REPAIR_STEPS", 1)
    # The replanner asks for no research, yet one stop is too few: the validator's error
    # starts the repair round, whose agent finds a second place.
    llm.turns = [turn(_shortlist_spare())]
    llm.structured_replies = [
        {"exclude_place_ids": others, "reply": "Оставляю одно место."},
        _texts([kept, SPARE.id]),
    ]

    session, _ = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(feedback="оставь только первое"), llm, events
    )

    assert {s.place.id for s in session.plan.stops} == {kept, SPARE.id}
    assert "The route check found these problems" in llm.seen[-2][1]["content"]


async def test_failed_feedback_keeps_the_previous_plan_and_state(sources, monkeypatch):
    events = Recorder()
    session, llm = await _planned(events)
    plan, state = session.plan, copy.deepcopy(session.state)
    monkeypatch.setattr(pipeline, "REPAIR_STEPS", 1)
    llm.turns = [turn(content="Ничего не нашёл.")]
    every = [s.place.id for s in plan.stops]
    llm.structured_replies = [
        {"exclude_place_ids": every, "end_time": "12:00", "reply": "Убираю всё."}
    ]

    # Every stop removed and the repair finds nothing: an empty route is an error,
    # not a result, and the session is left as the traveller last saw it.
    with pytest.raises(pipeline.PipelineError, match="ни одно найденное место"):
        await pipeline.apply_feedback(
            session, pipeline.FeedbackRequest(feedback="убери всё"), llm, events
        )

    assert session.plan is plan
    assert session.state == state


@pytest.mark.parametrize(
    "answer, message",
    [(None, "Не нашёл город «Атлантида»"), (SourceError("HTTP 429"), "Геокодер")],
    ids=["unknown", "down"],
)
async def test_city_lookup_failure_is_a_user_facing_error(sources, monkeypatch, answer, message):
    async def locate_city(city, country=None):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(geocode, "locate_city", locate_city)
    llm = FakeLLM(structured=[{**_brief_reply(), "city": "Атлантида"}])

    with pytest.raises(pipeline.PipelineError, match=message):
        await pipeline.plan_trip(pipeline.PlanRequest(request="Атлантида"), llm, Recorder(), DAY)


async def test_start_point_lookup_failure_starts_from_the_centre(sources, monkeypatch):
    async def geocoder_down(query, city=None, country=None):
        raise SourceError("nominatim.openstreetmap.org: HTTP 503")

    monkeypatch.setattr(geocode, "geocode", geocoder_down)
    events = Recorder()

    session, _ = await _planned(events, start_point="Кремль")

    assert session.plan.start.label == "центр, Казань"
    assert any(e.type == "warning" and "Кремль" in e.data["message"] for e in events)


async def test_real_streets_longer_than_the_limit_drop_the_weakest_stop(sources, monkeypatch):
    # The optimizer's estimate fits the traveller's 3.5 km; OSRM's streets are 1 km a leg,
    # so four stops (4 km) break the limit while they still fit the time window.
    async def long_streets(points):
        return RouteResult(
            geometry=[[p.lat, p.lon] for p in points], leg_meters=[1000.0] * (len(points) - 1)
        )

    monkeypatch.setattr(pipeline, "walking_route", long_streets)
    llm = FakeLLM(
        turns=_research_turns(),
        structured=[{**_brief_reply(), "max_walk_km": 3.5}, _texts([p.id for p in PLACES])],
    )
    req = pipeline.PlanRequest(request="Казань, с 10 до 13, история, не больше 3,5 км")
    session = await pipeline.plan_trip(req, llm, Recorder(), DAY)

    assert session.plan.total_walk_m <= 3_500
    assert len(session.plan.stops) == 3 and len(session.plan.dropped) == 1


def test_rain_from_feedback_marks_the_remaining_hours_rainy():
    brief = TripBrief(city="Казань", date=DAY, start_time=time(10), end_time=time(18), interests=[])
    forecast = [HourWeather(time=datetime(2026, 10, 24, 15), precipitation_probability=10)]

    hours = pipeline._with_rain(forecast, brief, time(15, 30))

    assert [h.time.hour for h in hours] == [15, 16, 17]
    assert all(h.precipitation_probability == 100 for h in hours)
