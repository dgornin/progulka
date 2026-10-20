"""The whole pipeline offline: scripted LLM, fake sources, real optimizer and validator."""

import copy
import json
from datetime import date, datetime, time

import pytest

from app import pipeline
from app.agent.prompts import MEMORY, RESEARCHER
from app.llm import ToolCall
from app.models import GeoPoint, HourWeather, Place, TripBrief
from app.planning.routing import RouteResult
from app.profile import Profile, WalkPreferences
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
    assert session.learned == []  # no memory passed: the caller keeps none, nothing is learned

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


async def test_feedback_asking_for_an_avoided_topic_lifts_the_old_avoid(sources):
    events = Recorder()
    session, llm = await _planned(events)
    church = PLACES[1].id
    # As if the profile said «не предлагать храмы» when the trip was planned.
    session.state.brief = session.state.brief.model_copy(update={"avoid": ["Храмы"]})
    llm.structured_replies = [
        {"add_interests": [{"label": "Храмы", "weight": 1.0, "source": "profile"}], "reply": "Ок."},
        _texts([p.id for p in PLACES]),
    ]

    session, _ = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(feedback="всё-таки хочу в храм"), llm, events
    )

    # The old avoid would have made the church an error and started a repair round.
    assert church in {s.place.id for s in session.plan.stops}
    assert not any(e.type == "stage" and e.data["name"] == "repair" for e in events)
    brief = session.state.brief
    assert brief.avoid == []
    assert [(i.label, i.source) for i in brief.interests] == [
        ("Храмы", "request"),
        ("история", "request"),
    ]


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


async def test_profile_and_memory_reach_the_agent_and_new_facts_are_learned(sources):
    profile = Profile(home_city="Казань", diet=["вегетарианец"], interests={"nature": 0.6})
    memory = ["не любит толпы"]
    learned = {"facts": ["любит тихие дворы", "Не любит толпы."]}
    brief = {**_brief_reply(), "interests": [{"label": "История", "weight": 1.0}]}
    llm = FakeLLM(
        turns=_research_turns(), structured=[brief, _texts([p.id for p in PLACES]), learned]
    )
    events = Recorder()
    req = pipeline.PlanRequest(request="Казань, 24 октября, с 10 до 13, история. Обожаю дворики")

    session = await pipeline.plan_trip(req, llm, events, DAY, profile=profile, memory=memory)

    stages = [e.data for e in events if e.type == "stage" and e.data["status"] == "done"]
    brief_detail = next(d["detail"] for d in stages if d["name"] == "brief")
    # The request has one interest: the profile fills in, discounted, after it.
    assert [(i["label"], i["weight"], i["source"]) for i in brief_detail["interests"]] == [
        ("История", 1.0, "request"),
        ("Природа и парки", 0.42, "profile"),
    ]
    task = llm.seen[1][1]["content"]
    assert llm.seen[1][0]["content"].startswith(RESEARCHER)
    assert "История: search_osm historic, monument, memorial" in task
    assert "Природа и парки: search_osm park, garden" in task
    assert '["не любит толпы", "Питание: вегетарианец"]' in task
    writer_input = json.loads(llm.seen[3][1]["content"])
    assert writer_input["traveller"]["known"] == ["не любит толпы", "Питание: вегетарианец"]
    # Memory reads the request, knows the profile, and keeps only what is new.
    assert llm.seen[4][0]["content"] == MEMORY
    memory_input = json.loads(llm.seen[4][1]["content"])
    assert memory_input["messages"] == [req.request]
    assert "Питание: вегетарианец" in memory_input["known_facts"]
    assert session.learned == ["любит тихие дворы"]
    assert [e.data for e in events if e.type == "memory"] == [{"facts": ["любит тихие дворы"]}]


async def test_empty_request_with_a_city_skips_the_brief_llm_call(sources):
    llm = FakeLLM(turns=_research_turns(), structured=[_texts([p.id for p in PLACES])])
    req = pipeline.PlanRequest(
        city="Казань",
        date=DAY,
        start_time=time(10),
        end_time=time(13),
        preferences=WalkPreferences(more=["history"], pace="relaxed"),
    )

    session = await pipeline.plan_trip(req, llm, Recorder(), DAY, memory=[])

    brief = session.state.brief
    assert llm.seen[0][0]["content"].startswith(RESEARCHER)  # the first LLM call is the agent's
    assert (brief.city, brief.language, brief.pace) == ("Казань", "ru", "relaxed")
    assert [(i.label, i.source) for i in brief.interests] == [("История", "walk")]
    assert len(session.plan.stops) == 4
    assert session.learned == []  # no text, nothing to learn from


async def test_empty_request_needs_a_city_from_the_form_or_the_profile(sources):
    llm = FakeLLM()
    with pytest.raises(pipeline.PipelineError, match="Опишите прогулку или укажите город"):
        await pipeline.plan_trip(pipeline.PlanRequest(request="   "), llm, Recorder(), DAY)
    assert llm.seen == []

    llm = FakeLLM(turns=_research_turns(), structured=[_texts([p.id for p in PLACES])])
    profile = Profile(home_city="Казань", interests={"history": 1.0}, max_walk_km=6)
    req = pipeline.PlanRequest(start_time=time(10), end_time=time(13))
    session = await pipeline.plan_trip(req, llm, Recorder(), DAY, profile)
    brief = session.state.brief
    assert (brief.city, brief.max_walk_km) == ("Казань", 6)
    assert [(i.label, i.weight, i.source) for i in brief.interests] == [("История", 0.7, "profile")]


async def test_feedback_text_is_learned_from_and_the_remove_button_is_not(sources):
    events = Recorder()
    session, llm = await _planned(events)
    llm.structured_replies = [
        {"reply": "Учту."},
        _texts([p.id for p in PLACES]),
        {"facts": ["вегетарианец"]},
    ]

    session, _ = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(feedback="кстати, я вегетарианец"), llm, events, memory=[]
    )

    assert session.learned == ["вегетарианец"]
    assert events[-1] == pipeline.AgentEvent(type="memory", data={"facts": ["вегетарианец"]})

    removed = session.plan.stops[0].place.id
    llm.turns = [turn(_shortlist_spare(), call("finish_research", summary="нашёл замену"))]
    llm.structured_replies = [_texts([SPARE.id])]
    session, _ = await pipeline.apply_feedback(
        session, pipeline.FeedbackRequest(exclude_place_ids=[removed]), llm, events, memory=[]
    )

    assert session.learned == []  # the last run's facts are not reported twice
    assert llm.structured_replies == []  # and the button made no memory call


def test_rain_from_feedback_marks_the_remaining_hours_rainy():
    brief = TripBrief(city="Казань", date=DAY, start_time=time(10), end_time=time(18), interests=[])
    forecast = [HourWeather(time=datetime(2026, 10, 24, 15), precipitation_probability=10)]

    hours = pipeline._with_rain(forecast, brief, time(15, 30))

    assert [h.time.hour for h in hours] == [15, 16, 17]
    assert all(h.precipitation_probability == 100 for h in hours)
