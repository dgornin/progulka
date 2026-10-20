"""Coordinated edits use real route checks, with scripted LLM replies and no network."""

import asyncio
import json
from copy import deepcopy
from datetime import time

import pytest

from app import pipeline
from app.agent.researcher import ResearchResult
from app.store import journey_view
from tests.test_agent import Recorder
from tests.test_journey import TripLLM


@pytest.fixture(autouse=True)
def no_routing(monkeypatch):
    async def route(points):
        return None

    monkeypatch.setattr(pipeline, "walking_route", route)


def _llm(**edit):
    return TripLLM({}, {"JourneyEdit": [{"reply": "Переношу места.", **edit}]})


async def _apply(journey, llm, events=None):
    return await pipeline.apply_journey_feedback(
        journey,
        pipeline.JourneyFeedbackRequest(feedback="Поменяй местами первые остановки дней 1 и 2"),
        llm,
        events if events is not None else Recorder(),
    )


SWAP = [
    {"source_day": 0, "target_day": 1, "place_id": "osm:node/1"},
    {"source_day": 1, "target_day": 0, "place_id": "osm:node/11"},
]


async def test_swap_reserves_places_keeps_dates_and_leaves_other_days_untouched(make_journey):
    original = make_journey(days=3)
    original.days[0].state.brief.must_include = ["Место 1"]
    before = deepcopy(original)
    events = Recorder()
    llm = _llm(moves=SWAP)

    updated, reply, changed = await _apply(original, llm, events)

    assert changed == [0, 1] and reply == "Переношу места."
    assert original == before
    assert updated.days[2] == before.days[2]
    assert updated.outline[2] == before.outline[2]
    assert {s.place.id for s in updated.days[0].plan.stops} == {"osm:node/2", "osm:node/11"}
    assert {s.place.id for s in updated.days[1].plan.stops} == {"osm:node/1", "osm:node/12"}
    for i in changed:
        assert updated.days[i].state.brief.date == before.days[i].state.brief.date
        assert updated.days[i].weather == before.days[i].weather
        assert not [v for v in updated.days[i].plan.violations if v.severity == "error"]
    assert updated.days[0].state.brief.must_include == ["Место 11"]
    assert updated.days[1].state.brief.must_include == ["Место 1"]
    assert updated.outline[0].must_include == ["Место 11"]
    assert updated.days[1].state.shortlist["osm:node/1"].required
    assert updated.days[0].state.registry is updated.days[1].state.registry
    assert not llm.calls("chat")  # transferring known stops needs no new search
    assert {e.data.get("day") for e in events} == {None, 0, 1}
    assert any(e.data.get("detail") == {"changed_days": [0, 1]} for e in events)
    context = json.loads(llm.calls("JourneyEdit")[0][1]["content"])
    assert [d["weekday"] for d in context["days"]] == ["Saturday", "Sunday", "Monday"]
    assert "osm:node/21" in context["days"][2]["route"]


async def test_full_programme_swap_moves_all_stops_and_updates_themes(make_journey):
    journey = make_journey(days=3)
    moves = [
        {"source_day": i, "target_day": 1 - i, "place_id": stop.place.id}
        for i in (0, 1)
        for stop in journey.days[i].plan.stops
    ]
    llm = _llm(
        moves=moves,
        days=[
            {"day_index": 0, "theme": "Новая тема", "reply": "Меняю."},
            {"day_index": 1, "rain_from": "15:00", "reply": "Учитываю дождь."},
        ],
    )
    updated, _, _ = await _apply(journey, llm)
    assert updated.outline[0].theme == "Новая тема"
    assert {s.place.id for s in updated.days[0].plan.stops} == {"osm:node/11", "osm:node/12"}
    assert {s.place.id for s in updated.days[1].plan.stops} == {"osm:node/1", "osm:node/2"}
    assert any(h.precipitation_probability == 100 for h in updated.days[1].weather)
    assert updated.days[0].weather == journey.days[0].weather


async def test_research_runs_only_for_requested_days_in_parallel(make_journey, monkeypatch):
    journey = make_journey(days=3)
    active = set()
    both_started = asyncio.Event()

    async def research(day, llm, emit, **kwargs):
        active.add(day.id)
        if len(active) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 1)
        return ResearchResult(list(day.state.shortlist.values()), "finished", 1)

    monkeypatch.setattr(pipeline, "_research", research)
    llm = _llm(
        days=[
            {"day_index": i, "research_goal": "Найти ещё архитектуру", "reply": "Ищу."}
            for i in (0, 1)
        ]
    )
    updated, _, changed = await _apply(journey, llm)
    assert changed == [0, 1]
    assert active == {journey.days[0].id, journey.days[1].id}
    assert updated.days[2] == journey.days[2]


async def test_earlier_research_sees_removals_from_later_days(make_journey, monkeypatch):
    journey = make_journey(days=2)
    state = journey.days[1].state
    template = state.shortlist["osm:node/12"]
    spare = template.place.model_copy(update={"id": "osm:node/14", "name": "Запасное"})
    state.registry[spare.id] = spare
    state.shortlist[spare.id] = template.model_copy(update={"place": spare})
    observed = []

    async def research(day, llm, emit, **kwargs):
        observed.append(set(day.state.rejected_elsewhere))
        return ResearchResult(list(day.state.shortlist.values()), "finished", 1)

    monkeypatch.setattr(pipeline, "_research", research)
    await _apply(
        journey,
        _llm(
            days=[
                {"day_index": 0, "research_goal": "Найти архитектуру", "reply": "Ищу."},
                {"day_index": 1, "exclude_place_ids": ["osm:node/11"], "reply": "Убираю."},
            ]
        ),
    )
    assert observed and "osm:node/11" in observed[0]


async def test_coordinator_timeout_is_actionable_and_keeps_the_original(make_journey, monkeypatch):
    journey = make_journey(days=2)
    before = deepcopy(journey)

    async def never_finishes(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "interpret_journey_feedback", never_finishes)
    monkeypatch.setattr(pipeline, "LLM_STEP_TIMEOUT_S", 0.001)
    with pytest.raises(pipeline.PipelineError, match="не ответил вовремя"):
        await _apply(journey, _llm())
    assert journey == before


@pytest.mark.parametrize(
    "failure", [pipeline.PipelineError("Не удалось"), asyncio.CancelledError()]
)
async def test_late_failure_or_cancellation_does_not_mutate_input(
    make_journey, monkeypatch, failure
):
    journey = make_journey(days=3)
    before = deepcopy(journey)
    build = pipeline._build_checked
    built = []

    async def fail_second(day, *args, **kwargs):
        built.append(day.id)
        if len(built) == 2:
            raise failure
        return await build(day, *args, **kwargs)

    monkeypatch.setattr(pipeline, "_build_checked", fail_second)
    llm = _llm(moves=SWAP)
    with pytest.raises(type(failure)):
        await _apply(journey, llm)
    assert len(built) == 2 and journey == before
    assert not llm.calls("PlanTexts")


async def test_closed_transferred_stop_rejects_the_whole_edit(make_journey, monkeypatch):
    journey = make_journey(days=2)
    place = journey.days[0].state.shortlist["osm:node/1"].place
    place.kind = "cafe"
    place.opening_hours = "Sa 10:00-20:00; Su off"

    async def unchanged_repair(llm, state, emit, **kwargs):
        return ResearchResult(list(state.shortlist.values()), "finished", 1)

    monkeypatch.setattr(pipeline, "research", unchanged_repair)
    before = deepcopy(journey)
    with pytest.raises(pipeline.PipelineError, match="День 2:.*закрыто"):
        await _apply(journey, _llm(moves=SWAP))
    assert journey == before


async def test_source_alias_cannot_reintroduce_a_transferred_place(make_journey):
    journey = make_journey(days=2)
    candidate = journey.days[0].state.shortlist["osm:node/1"]
    alias = candidate.place.model_copy(update={"id": "wikipedia:ru:1", "source": "wikipedia"})
    journey.days[0].state.registry[alias.id] = alias
    journey.days[0].state.shortlist[alias.id] = candidate.model_copy(update={"place": alias})

    updated, _, _ = await _apply(journey, _llm(moves=SWAP))
    assert alias.id not in {s.place.id for s in updated.days[0].plan.stops}
    assert "osm:node/1" in {s.place.id for s in updated.days[1].plan.stops}


async def test_a_finished_source_future_does_not_prevent_copying(make_journey):
    journey = make_journey(days=2)
    future = asyncio.get_running_loop().create_future()
    future.set_result([])
    searches = {"cached": future}
    for day in journey.days:
        day.state.searches = searches
    updated, _, _ = await _apply(journey, _llm(moves=SWAP))
    assert updated.days[0].state.searches is updated.days[1].state.searches
    assert not updated.days[0].state.searches
    assert journey.days[0].state.searches["cached"] is future


@pytest.mark.parametrize(
    "edit",
    [
        {"days": [{"day_index": 5, "reply": "Меняю."}]},
        {"days": [{"day_index": 0, "reply": "Меняю."}] * 2},
        {"days": [{"day_index": 0, "start_time": "20:00", "reply": "Меняю."}]},
        {"days": [{"day_index": 0, "exclude_place_ids": ["unknown"], "reply": "Меняю."}]},
        {"moves": [{"source_day": 0, "target_day": 5, "place_id": "osm:node/1"}]},
        {"moves": [{"source_day": 0, "target_day": 0, "place_id": "osm:node/1"}]},
        {"moves": [{"source_day": 0, "target_day": 1, "place_id": "invented"}]},
        {"moves": [SWAP[0], SWAP[0]]},
        {
            "moves": SWAP,
            "days": [
                {"day_index": 0, "exclude_place_ids": ["osm:node/1"], "reply": "Удаляю."},
            ],
        },
        {},
    ],
)
async def test_invalid_coordinator_output_leaves_every_day_unchanged(make_journey, edit):
    journey = make_journey(days=2)
    before = deepcopy(journey)
    llm = _llm(**edit)
    with pytest.raises(pipeline.PipelineError):
        await _apply(journey, llm)
    assert journey == before
    assert len(llm.seen) == 1


async def test_one_day_trip_can_use_the_same_endpoint(make_journey):
    journey = make_journey()
    updated, _, changed = await _apply(
        journey, _llm(days=[{"day_index": 0, "end_time": "16:00", "reply": "Сокращаю."}])
    )
    assert changed == [0]
    assert updated.days[0].plan.brief.end_time == time(16)
    assert journey.days[0].plan.brief.end_time == time(17)
    assert journey_view(updated).title == updated.days[0].plan.title
