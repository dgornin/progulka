"""The whole pipeline offline: scripted LLM, fake sources, real optimizer and validator."""

from datetime import date, datetime, time

import pytest

from app import pipeline
from app.models import AgentEvent, GeoPoint, HourWeather, Place, TripBrief
from app.sources import geocode, osm, weather
from app.sources.geocode import CityInfo
from tests.test_agent import FakeLLM, call, turn

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
        return CityInfo(
            name="Казань", center=CENTER, bbox=(55.7, 49.0, 55.9, 49.2), country_code="ru"
        )

    async def no_route(points):
        return None  # the pipeline falls back to straight lines

    async def hourly(point, day):
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


async def test_plan_then_feedback(sources):
    ids = [p.id for p in PLACES]
    llm = FakeLLM(turns=_research_turns(), structured=[_brief_reply(), _texts(ids)])
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    session = await pipeline.plan_trip(
        pipeline.PlanRequest(request="Казань, 24 октября, с 10 до 13, история"), llm, emit, DAY
    )

    done = [e.data["name"] for e in events if e.type == "stage" and e.data["status"] == "done"]
    assert done == ["brief", "weather", "research", "optimize", "route", "validate", "write"]
    plan = session.plan
    assert {s.place.id for s in plan.stops} == set(ids)
    assert plan.title == "Казань за утро"
    assert not [v for v in plan.violations if v.severity == "error"]
    assert pipeline.get_session(session.id) is session

    # Removing a stop from the UI: no LLM interpretation, a short research run that must
    # find a replacement (finish is refused below the minimum), then the same optimizer
    # and validator.
    removed = plan.stops[0].place.id
    replacement = call(
        "shortlist_place",
        place_id=SPARE.id,
        score=0.7,
        interests=["история"],
        reason="замена",
        visit_minutes=30,
    )
    llm.turns = [
        turn(call("finish_research", summary="замена не нужна")),
        turn(replacement, call("finish_research", summary="нашёл замену")),
    ]
    llm.structured_replies = [_texts([SPARE.id, *(i for i in ids if i != removed)])]
    session, reply = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(exclude_place_ids=[removed]), llm, emit
    )

    stops = {s.place.id for s in session.plan.stops}
    assert removed not in stops and SPARE.id in stops
    assert removed in session.state.excluded
    assert reply


async def test_unknown_city_is_a_user_facing_error(sources, monkeypatch):
    async def nowhere(city, country=None):
        return None

    monkeypatch.setattr(geocode, "locate_city", nowhere)
    llm = FakeLLM(structured=[{**_brief_reply(), "city": "Атлантида"}])

    async def emit(event: AgentEvent) -> None:
        pass

    with pytest.raises(pipeline.PipelineError, match="Атлантида"):
        await pipeline.plan_trip(pipeline.PlanRequest(request="Атлантида"), llm, emit, DAY)


def test_rain_from_feedback_marks_the_remaining_hours_rainy():
    brief = TripBrief(city="Казань", date=DAY, start_time=time(10), end_time=time(18), interests=[])
    forecast = [HourWeather(time=datetime(2026, 10, 24, 15), precipitation_probability=10)]

    hours = pipeline._with_rain(forecast, brief, time(15, 30))

    assert [h.time.hour for h in hours] == [15, 16, 17]
    assert all(h.precipitation_probability == 100 for h in hours)
