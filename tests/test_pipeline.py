"""The whole pipeline offline: scripted LLM, fake sources, real optimizer and validator."""

import asyncio
import json
from datetime import date, datetime, time

import pytest

from app import pipeline
from app.agent.prompts import MEMORY, RESEARCHER
from app.agent.tools import ResearchState
from app.llm import ToolCall
from app.models import Candidate, GeoPoint, HourWeather, Place, TripBrief
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


async def _planned(events: Recorder, **request: str) -> tuple[pipeline.TripRun, FakeLLM]:
    llm = FakeLLM(
        turns=_research_turns(), structured=[_brief_reply(), _texts([p.id for p in PLACES])]
    )
    req = pipeline.PlanRequest(request="Казань, 24 октября, с 10 до 13, история", **request)
    return await pipeline.plan_trip(req, llm, events, DAY), llm


async def test_plan_then_remove_a_stop(sources):
    events = Recorder()
    trip_run, llm = await _planned(events)

    done = [e.data["name"] for e in events if e.type == "stage" and e.data["status"] == "done"]
    assert done == ["brief", "weather", "research", "optimize", "route", "validate", "write"]
    plan = trip_run.plan
    assert {s.place.id for s in plan.stops} == {p.id for p in PLACES}
    assert plan.title == "Казань за утро"
    assert plan.geometry is None  # no walking route: the UI draws dotted straight lines
    assert not [v for v in plan.violations if v.severity == "error"]
    assert trip_run.learned == []  # no memory passed: the caller keeps none, nothing is learned

    # The "Убрать" button: no LLM interpretation, a short research run that must find a
    # replacement (finish is refused below the minimum), then the same optimizer and
    # validator.
    removed = plan.stops[0].place
    llm.turns = [
        turn(call("finish_research", summary="замена не нужна")),
        turn(_shortlist_spare(), call("finish_research", summary="нашёл замену")),
    ]
    llm.structured_replies = [_texts([SPARE.id, *(p.id for p in PLACES if p.id != removed.id)])]
    trip_run, reply = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(exclude_place_ids=[removed.id]), llm, events
    )

    stops = {s.place.id for s in trip_run.plan.stops}
    assert removed.id not in stops and SPARE.id in stops
    assert removed.id in trip_run.state.excluded
    assert f"The traveller removed: {removed.name}." in llm.seen[-3][1]["content"]
    assert reply


async def test_removing_a_place_the_traveller_asked_for_stops_asking_for_it(sources):
    asked = PLACES[0]
    llm = FakeLLM(
        turns=_research_turns(),
        structured=[{**_brief_reply(), "must_include": [asked.name]}, _texts([asked.id])],
    )
    req = pipeline.PlanRequest(request=f"Казань, с 10 до 13, история, обязательно {asked.name}")
    trip_run = await pipeline.plan_trip(req, llm, Recorder(), DAY)
    assert trip_run.state.shortlist[asked.id].required

    llm.turns = [turn(_shortlist_spare(), call("finish_research", summary="нашёл замену"))]
    llm.structured_replies = [_texts([SPARE.id])]
    events = Recorder()
    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(exclude_place_ids=[asked.id]), llm, events
    )

    # Otherwise finish is refused until the agent finds the removed place again.
    assert trip_run.state.brief.must_include == []
    finish = next(
        e for e in events if e.type == "tool_result" and e.data["name"] == "finish_research"
    )
    assert finish.data["error"] is None
    assert "missing_must_include" not in {v.code for v in trip_run.plan.violations}


async def test_new_avoid_phrase_drops_matching_places_until_it_is_lifted(sources):
    events = Recorder()
    trip_run, llm = await _planned(events)
    museum = PLACES[0].id
    llm.structured_replies = [
        {"add_avoid": ["музеи"], "reply": "Уберу музеи."},
        _texts([p.id for p in PLACES]),
    ]

    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(feedback="без музеев"), llm, events
    )

    # No new research: the avoid phrase alone drops the museum.
    assert museum not in {s.place.id for s in trip_run.plan.stops}
    assert museum not in trip_run.state.shortlist
    assert not [v for v in trip_run.plan.violations if v.severity == "error"]
    # Avoided is not rejected: asking for museums again lifts the avoid, and the agent
    # may shortlist the same museum again.
    assert museum not in trip_run.state.excluded
    llm.turns = [
        turn(
            call(
                "shortlist_place",
                place_id=museum,
                score=0.9,
                interests=["Музеи"],
                reason="снова хочет в музей",
                visit_minutes=30,
            ),
            call("finish_research", summary="музей вернулся"),
        )
    ]
    llm.structured_replies = [
        {
            "add_interests": [{"label": "Музеи", "weight": 1.0}],
            "research_goal": "Найди музей",
            "reply": "Верну музей.",
        },
        _texts([p.id for p in PLACES]),
    ]

    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(feedback="всё-таки хочу в музей"), llm, events
    )

    assert trip_run.state.brief.avoid == []
    assert museum in {s.place.id for s in trip_run.plan.stops}
    assert "rejected" not in llm.seen[-2][1]["content"]


async def test_the_route_leaves_out_what_the_current_avoid_list_rules_out(sources):
    trip_run, _ = await _planned(Recorder())
    museum = PLACES[0].id
    trip_run.state.brief = trip_run.state.brief.model_copy(update={"avoid": ["без музеев"]})

    plan = await pipeline._build(trip_run, Recorder())

    assert museum not in {s.place.id for s in plan.stops}
    assert "avoid_violation" not in {v.code for v in plan.violations}
    assert museum in trip_run.state.shortlist  # left out, not removed


async def test_feedback_asking_for_an_avoided_topic_lifts_the_old_avoid(sources):
    events = Recorder()
    trip_run, llm = await _planned(events)
    church = PLACES[1].id
    # As if the profile said «не предлагать храмы» when the trip was planned.
    trip_run.state.brief = trip_run.state.brief.model_copy(update={"avoid": ["Храмы"]})
    llm.structured_replies = [
        {"add_interests": [{"label": "Храмы", "weight": 1.0, "source": "profile"}], "reply": "Ок."},
        _texts([p.id for p in PLACES]),
    ]

    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(feedback="всё-таки хочу в храм"), llm, events
    )

    # The old avoid would have made the church an error and started a repair round.
    assert church in {s.place.id for s in trip_run.plan.stops}
    assert not any(e.type == "stage" and e.data["name"] == "repair" for e in events)
    brief = trip_run.state.brief
    assert brief.avoid == []
    assert [(i.label, i.source) for i in brief.interests] == [
        ("Храмы", "request"),
        ("история", "request"),
    ]


async def test_feedback_without_new_places_still_gets_the_repair_round(sources, monkeypatch):
    events = Recorder()
    trip_run, llm = await _planned(events)
    kept = trip_run.plan.stops[0].place.id
    others = [s.place.id for s in trip_run.plan.stops[1:]]
    monkeypatch.setattr(pipeline, "REPAIR_STEPS", 1)
    # The replanner asks for no research, yet one stop is too few: the validator's error
    # starts the repair round, whose agent finds a second place.
    llm.turns = [turn(_shortlist_spare())]
    llm.structured_replies = [
        {"exclude_place_ids": others, "reply": "Оставляю одно место."},
        _texts([kept, SPARE.id]),
    ]

    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(feedback="оставь только первое"), llm, events
    )

    assert {s.place.id for s in trip_run.plan.stops} == {kept, SPARE.id}
    assert "The route check found these problems" in llm.seen[-2][1]["content"]


async def test_feedback_that_leaves_no_stops_is_an_error_not_a_route(sources, monkeypatch):
    events = Recorder()
    trip_run, llm = await _planned(events)
    monkeypatch.setattr(pipeline, "REPAIR_STEPS", 1)
    llm.turns = [turn(content="Ничего не нашёл.")]
    every = [s.place.id for s in trip_run.plan.stops]
    llm.structured_replies = [
        {"exclude_place_ids": every, "end_time": "12:00", "reply": "Убираю всё."}
    ]

    # Every stop removed and the repair finds nothing: an empty route is an error,
    # not a result (and the caller saves nothing).
    with pytest.raises(pipeline.PipelineError, match="ни одно найденное место"):
        await pipeline.apply_feedback(
            trip_run, pipeline.FeedbackRequest(feedback="убери всё"), llm, events
        )


async def test_a_provider_that_hangs_on_the_brief_or_the_feedback_is_a_user_facing_error(
    sources, monkeypatch
):
    trip_run, llm = await _planned(Recorder())

    async def hang(*_, **__):
        await asyncio.Event().wait()

    llm.structured = hang
    monkeypatch.setattr(pipeline, "LLM_STEP_TIMEOUT_S", 0.01)

    with pytest.raises(pipeline.PipelineError, match="не ответил вовремя"):
        await pipeline.plan_trip(pipeline.PlanRequest(request="Казань"), llm, Recorder(), DAY)
    with pytest.raises(pipeline.PipelineError, match="не ответил вовремя"):
        await pipeline.apply_feedback(
            trip_run, pipeline.FeedbackRequest(feedback="покороче"), llm, Recorder()
        )


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


@pytest.mark.parametrize(
    "answer, message",
    [
        (None, "Не нашёл «Кремль», начинаю из центра города"),
        (SourceError("HTTP 503"), "Геокодер OpenStreetMap не ответил на поиск «Кремль»"),
    ],
    ids=["unknown", "down"],
)
async def test_start_point_lookup_failure_starts_from_the_centre(
    sources, monkeypatch, answer, message
):
    async def geocoder(query, city=None, country=None):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(geocode, "geocode", geocoder)
    events = Recorder()

    trip_run, _ = await _planned(events, start_point="Кремль")

    assert trip_run.plan.start.label == "центр, Казань"
    [warning] = [e.data["message"] for e in events if e.type == "warning"]
    assert warning.startswith(message)


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
    trip_run = await pipeline.plan_trip(req, llm, Recorder(), DAY)

    assert trip_run.plan.total_walk_m <= 3_500
    assert len(trip_run.plan.stops) == 3 and len(trip_run.plan.dropped) == 1


async def test_a_required_stop_that_ends_late_does_not_strip_the_route(sources):
    # The museum the traveller asked for opens at 15:00, so the walk ends at 16:30, after
    # the window. The optimizer fits seven monuments before it; routing must not drop
    # them for a lateness they did not cause.
    museum = Place(
        id="osm:node/100",
        name="Музей",
        lat=CENTER.lat + 0.004,
        lon=CENTER.lon,
        kind="museum",
        source="osm",
        opening_hours="Mo-Su 15:00-20:00",
    )
    monuments = [_place(10 + n, "monument", 0.001 * n) for n in range(7)]
    shortlist = {
        p.id: Candidate(place=p, score=0.6, reason="рядом", visit_minutes=30) for p in monuments
    }
    shortlist[museum.id] = Candidate(
        place=museum, score=0.9, reason="просили", visit_minutes=90, required=True
    )
    brief = TripBrief(
        city="Казань", date=DAY, start_time=time(10), end_time=time(15, 30), must_include=["Музей"]
    )
    state = ResearchState(brief=brief, start=CENTER, weather_summary="", shortlist=shortlist)
    events = Recorder()

    plan = await pipeline._build(pipeline.TripRun(id="t", state=state, weather=[]), events)

    assert len(plan.stops) == 8 and plan.dropped == []
    route = next(e.data for e in events if e.type == "stage" and e.data["name"] == "route")
    assert "убрано" not in route["label"]


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

    trip_run = await pipeline.plan_trip(req, llm, events, DAY, profile=profile, memory=memory)

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
    assert trip_run.learned == ["любит тихие дворы"]
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

    trip_run = await pipeline.plan_trip(req, llm, Recorder(), DAY, memory=[])

    brief = trip_run.state.brief
    assert llm.seen[0][0]["content"].startswith(RESEARCHER)  # the first LLM call is the agent's
    assert (brief.city, brief.language, brief.pace) == ("Казань", "ru", "relaxed")
    assert [(i.label, i.source) for i in brief.interests] == [("История", "walk")]
    assert len(trip_run.plan.stops) == 4
    assert trip_run.learned == []  # no text, nothing to learn from


async def test_empty_request_needs_a_city_from_the_form_or_the_profile(sources):
    llm = FakeLLM()
    with pytest.raises(pipeline.PipelineError, match="Опишите прогулку или укажите город"):
        await pipeline.plan_trip(pipeline.PlanRequest(request="   "), llm, Recorder(), DAY)
    assert llm.seen == []

    llm = FakeLLM(turns=_research_turns(), structured=[_texts([p.id for p in PLACES])])
    profile = Profile(home_city="Казань", interests={"history": 1.0}, max_walk_km=6)
    req = pipeline.PlanRequest(start_time=time(10), end_time=time(13))
    trip_run = await pipeline.plan_trip(req, llm, Recorder(), DAY, profile)
    brief = trip_run.state.brief
    assert (brief.city, brief.max_walk_km) == ("Казань", 6)
    assert [(i.label, i.weight, i.source) for i in brief.interests] == [("История", 0.7, "profile")]


async def test_a_window_too_short_to_walk_is_refused_before_the_research(sources):
    llm = FakeLLM()
    req = pipeline.PlanRequest(city="Казань", start_time=time(23, 59))

    with pytest.raises(pipeline.PipelineError, match="23:59–23:59 слишком короткое"):
        await pipeline.plan_trip(req, llm, Recorder(), DAY)
    assert llm.seen == []


def test_a_form_whose_end_is_not_after_its_start_is_invalid():
    with pytest.raises(ValueError, match="Конец прогулки должен быть позже начала"):
        pipeline.PlanRequest(city="Казань", start_time=time(22), end_time=time(1))


async def test_feedback_text_is_learned_from_and_the_remove_button_is_not(sources):
    events = Recorder()
    trip_run, llm = await _planned(events)
    llm.structured_replies = [
        {"reply": "Учту."},
        _texts([p.id for p in PLACES]),
        {"facts": ["вегетарианец"]},
    ]

    trip_run, _ = await pipeline.apply_feedback(
        trip_run,
        pipeline.FeedbackRequest(feedback="кстати, я вегетарианец"),
        llm,
        events,
        memory=[],
    )

    assert trip_run.learned == ["вегетарианец"]
    assert events[-2] == pipeline.AgentEvent(type="memory", data={"facts": ["вегетарианец"]})
    assert events[-1].data == {"name": "memory", "status": "done", "label": "Новых фактов: 1"}

    removed = trip_run.plan.stops[0].place.id
    llm.turns = [turn(_shortlist_spare(), call("finish_research", summary="нашёл замену"))]
    llm.structured_replies = [_texts([SPARE.id])]
    trip_run, _ = await pipeline.apply_feedback(
        trip_run, pipeline.FeedbackRequest(exclude_place_ids=[removed]), llm, events, memory=[]
    )

    assert trip_run.learned == []  # the last run's facts are not reported twice
    assert llm.structured_replies == []  # and the button made no memory call


def test_rain_from_feedback_marks_the_remaining_hours_rainy():
    brief = TripBrief(city="Казань", date=DAY, start_time=time(10), end_time=time(18), interests=[])
    forecast = [HourWeather(time=datetime(2026, 10, 24, 15), precipitation_probability=10)]

    hours = pipeline._with_rain(forecast, brief, time(15, 30))

    assert [h.time.hour for h in hours] == [15, 16, 17]
    assert all(h.precipitation_probability == 100 for h in hours)
