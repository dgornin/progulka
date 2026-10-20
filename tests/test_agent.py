"""The code around the model: tool validation, grounding, loop termination and the
checks applied to structured output.

The LLM is a scripted fake and the sources are monkeypatched, so the tests run
offline and without an API key.
"""

import asyncio
import itertools
import json
from datetime import date, datetime, time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from app.agent import tools, writer
from app.agent.brief import parse_brief
from app.agent.prompts import BUDGET_WARNING, NUDGE
from app.agent.replanner import interpret_feedback
from app.agent.researcher import research
from app.agent.tools import ResearchState, ToolOutcome, execute, tool_schemas
from app.agent.writer import write_texts
from app.llm import LLM, ChatTurn, LLMError, ToolCall
from app.models import AgentEvent, GeoPoint, Interest, Place, Plan, Stop, TripBrief
from app.sources import osm, wikipedia, wikivoyage
from app.sources.http import SourceError
from tests.test_speech import provider

START = GeoPoint(lat=59.9343, lon=30.3351, label="Невский проспект")


def category(kind: str, label: str, indoor: bool, minutes: int) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind, label_ru=label, filters=[], indoor=indoor, visit_minutes=minutes
    )


CATALOG = {
    "museum": category("museum", "Музей", True, 60),
    "artwork": category("artwork", "Стрит-арт", False, 10),
    "cafe": category("cafe", "Кафе", True, 40),
}
MUSEUM = Place(
    id="osm:node/1",
    name="Музей Фаберже",
    lat=59.9349,
    lon=30.3437,
    kind="museum",
    source="osm",
    opening_hours="Mo-Su 10:00-20:45",
    indoor=True,
)
MURAL = Place(
    id="osm:node/2",
    name="Стена с граффити",
    lat=59.9302,
    lon=30.3480,
    kind="artwork",
    source="osm",
)
CAFE = Place(id="osm:node/3", name="Пышечная", lat=59.9375, lon=30.3224, kind="cafe", source="osm")
SINGER = Place(
    id="wikipedia:ru:4321",
    name="Дом Зингера",
    lat=59.9357,
    lon=30.3259,
    kind="architecture",
    source="wikipedia",
    description="Здание компании «Зингер» на Невском, памятник модерна.",
)
OSM_PLACES = [MUSEUM, MURAL, CAFE]
EXTRACT = "Дом Зингера — здание в стиле модерн, построенное в 1902–1904 годах."


@pytest.fixture(autouse=True)
def sources(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Fake sources; returns the log of their calls."""
    calls: list[tuple[Any, ...]] = []

    async def search_places(center, radius_m, categories, limit=40):
        calls.append(("osm", tuple(categories), radius_m))
        return [p for p in OSM_PLACES if p.kind in categories]

    async def wiki_search(query, center, radius_m, lang="ru", limit=20):
        calls.append(("wikipedia", query, radius_m))
        return [SINGER]

    async def page_extract(place_id, max_chars=1500):
        return EXTRACT

    async def listings(city, lang="ru", sections=None):
        return []

    monkeypatch.setattr(osm, "CATEGORIES", CATALOG)
    monkeypatch.setattr(osm, "search_places", search_places)
    monkeypatch.setattr(wikipedia, "search", wiki_search)
    monkeypatch.setattr(wikipedia, "page_extract", page_extract)
    monkeypatch.setattr(wikivoyage, "listings", listings)
    return calls


class FakeLLM:
    """Replays scripted replies (an exception is raised instead of returned) and keeps
    a snapshot of every conversation it was sent. Structured replies come in call order,
    or by schema name for calls that run at the same time."""

    def __init__(self, turns: list[Any] = (), structured: list[Any] | dict[str, list[Any]] = ()):
        self.turns = list(turns)
        self.structured_replies = structured if isinstance(structured, dict) else list(structured)
        self.seen: list[list[dict[str, Any]]] = []

    async def chat(self, messages, tools=None, **_):
        self.seen.append(json.loads(json.dumps(messages)))  # the loop keeps appending to it
        return self._next(self.turns)

    async def structured(self, messages, schema, **_):
        self.seen.append(messages)
        replies = self.structured_replies
        if isinstance(replies, dict):
            replies = replies[schema.__name__]
        return schema.model_validate(self._next(replies))

    @staticmethod
    def _next(replies: list[Any]) -> Any:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


_call_ids = itertools.count(1)


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_call_ids)}", name=name, arguments=json.dumps(args))


def turn(*calls: ToolCall, content: str | None = None) -> ChatTurn:
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in calls
        ]
    return ChatTurn(content=content, tool_calls=list(calls), message=message)


def shortlist_call(place: Place, interests: list[str] = ()) -> ToolCall:
    return call(
        "shortlist_place", place_id=place.id, score=0.8, interests=list(interests), reason="ок"
    )


def make_brief(**changes: Any) -> TripBrief:
    fields: dict[str, Any] = {
        "city": "Санкт-Петербург",
        "date": date(2026, 10, 24),
        "start_time": time(10),
        "end_time": time(13),  # 3 hours -> at least 4 candidates
        "interests": [Interest(label="модерн", weight=1.0)],
    }
    return TripBrief(**{**fields, **changes})


def make_state(**brief_changes: Any) -> ResearchState:
    return ResearchState(
        brief=make_brief(**brief_changes), start=START, weather_summary="сухо, +12"
    )


def make_plan() -> Plan:
    stops = [
        Stop(
            place=place,
            arrive=datetime(2026, 10, 24, 10 + n),
            depart=datetime(2026, 10, 24, 10 + n, 40),
            score=0.8,
            reason="подходит",
        )
        for n, place in enumerate([MUSEUM, SINGER])
    ]
    return Plan(brief=make_brief(), start=START, stops=stops, total_walk_m=1500, total_minutes=160)


SHORTLIST_ARGS = {"score": 0.9, "interests": [], "reason": "подходит"}


class Recorder(list[AgentEvent]):
    async def __call__(self, event: AgentEvent) -> None:
        self.append(event)


async def run_tool(state: ResearchState, name: str, **args: Any) -> ToolOutcome:
    return await execute(state, name, json.dumps(args))


async def search_all(state: ResearchState) -> None:
    await run_tool(state, "search_osm", categories=list(CATALOG))
    await run_tool(state, "search_wikipedia", query="модерн")


# --- tools ---------------------------------------------------------------------------


def test_category_enum_follows_the_catalog() -> None:
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in tool_schemas()}
    assert schemas["search_osm"]["properties"]["categories"]["items"]["enum"] == list(CATALOG)


async def test_search_shows_25_nearest_compact_results_and_registers_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    many = [
        Place(
            id=f"osm:node/{n}",
            name=f"Музей {n}",
            lat=START.lat + n * 0.001,
            lon=START.lon,
            kind="museum",
            source="osm",
            description="очень длинное описание " * 20,
        )
        for n in range(30, 0, -1)
    ]

    async def search_places(center, radius_m, categories, limit=40):
        return many

    monkeypatch.setattr(osm, "search_places", search_places)
    state = make_state()
    state.excluded.add("osm:node/1")

    outcome = await run_tool(state, "search_osm", categories=["museum"])

    shown = json.loads(outcome.content)["places"]
    assert outcome.count == 29 and len(shown) == 25
    assert [p["dist_m"] for p in shown] == sorted(p["dist_m"] for p in shown)
    assert all(len(p["snippet"]) <= 160 for p in shown)
    assert len(state.registry) == 29 and "osm:node/1" not in state.registry


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("search_osm", "{not json", "invalid arguments"),
        ("search_osm", '{"categories": ["museum"], "radius_m": "far"}', "radius_m"),
        ("search_osm", '{"categories": ["museum"], "radius_m": 99999}', "radius_m"),
        ("search_osm", '{"categories": ["casino"]}', "unknown categories"),
        ("get_wikivoyage_listings", '{"section": "sleep"}', "section"),
        ("book_hotel", "{}", "unknown tool"),
    ],
)
async def test_bad_arguments_come_back_as_text(name: str, arguments: str, expected: str) -> None:
    outcome = await execute(make_state(), name, arguments)
    assert outcome.error and expected in outcome.error
    assert expected in json.loads(outcome.content)["error"]
    assert not outcome.finished


async def broken_source(*args: Any, **kwargs: Any) -> list[Place]:
    raise SourceError("overpass-api.de: HTTP 504")


async def slow_source(*args: Any, **kwargs: Any) -> list[Place]:
    await asyncio.sleep(5)
    return []


async def buggy_source(*args: Any, **kwargs: Any) -> list[Place]:
    raise KeyError("elements")  # a parser meeting a response it did not expect


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (broken_source, "HTTP 504"),
        (slow_source, "timed out"),
        (buggy_source, "failed unexpectedly"),
    ],
)
async def test_source_failure_comes_back_as_text(
    monkeypatch: pytest.MonkeyPatch, source: Any, expected: str
) -> None:
    monkeypatch.setattr(osm, "search_places", source)
    monkeypatch.setattr(tools, "TOOL_TIMEOUT_S", 0.05)
    outcome = await run_tool(make_state(), "search_osm", categories=["museum"])
    assert expected in outcome.error


async def test_shortlist_accepts_only_found_and_not_rejected_places() -> None:
    state = make_state(must_include=["фаберже"])
    await search_all(state)
    state.excluded.add(MURAL.id)
    common = {"score": 0.9, "reason": "подходит"}

    made_up = await run_tool(state, "shortlist_place", place_id="osm:x", interests=[], **common)
    rejected = await run_tool(state, "shortlist_place", place_id=MURAL.id, interests=[], **common)
    wrong_label = await run_tool(
        state, "shortlist_place", place_id=MUSEUM.id, interests=["яйца"], **common
    )
    accepted = await run_tool(
        state, "shortlist_place", place_id=MUSEUM.id, interests=["Модерн"], **common
    )

    assert "unknown place_id" in made_up.error
    assert "rejected" in rejected.error
    assert "unknown interest labels" in wrong_label.error
    assert accepted.error is None and accepted.candidate_event["action"] == "add"
    assert list(state.shortlist) == [MUSEUM.id]
    candidate = state.shortlist[MUSEUM.id]
    assert candidate.required  # "фаберже" from must_include is in the name
    assert candidate.visit_minutes == 60  # the catalog default for museums
    assert candidate.interests == ["модерн"]  # the brief's label, not the model's spelling


@pytest.mark.parametrize(
    "building",
    [
        # the same Wikidata link, whatever the names
        Place(
            id="osm:way/9",
            name="Здание компании Зингер",
            lat=59.9358,
            lon=30.3259,
            kind="architecture",
            source="osm",
            tags={"wikidata": "Q4"},
        ),
        # no link, but one name contains the other and they stand side by side
        Place(
            id="osm:way/9",
            name="Дом Зингера",
            lat=59.9358,
            lon=30.3259,
            kind="architecture",
            source="osm",
        ),
    ],
)
async def test_second_record_of_the_same_place_is_rejected(building: Place) -> None:
    state = make_state()
    article = SINGER.model_copy(
        update={"name": "Дом Зингера (Санкт-Петербург)", "tags": {"wikidata": "Q4"}}
    )
    state.registry.update({article.id: article, building.id: building})
    common = {"score": 0.8, "interests": [], "reason": "модерн"}

    await run_tool(state, "shortlist_place", place_id=article.id, **common)
    duplicate = await run_tool(state, "shortlist_place", place_id=building.id, **common)

    assert "same place" in duplicate.error and list(state.shortlist) == [article.id]


async def test_details_keep_the_wikipedia_extract_for_the_writer() -> None:
    state = make_state()
    await search_all(state)
    await execute(state, "shortlist_place", shortlist_call(SINGER).arguments)
    outcome = await run_tool(state, "get_place_details", place_id=SINGER.id)
    assert json.loads(outcome.content)["description"] == EXTRACT
    assert state.registry[SINGER.id].description == EXTRACT
    assert state.shortlist[SINGER.id].place.description == EXTRACT  # what the plan carries


async def test_finish_is_rejected_while_shortlist_is_short_then_accepted_on_third_try() -> None:
    state = make_state()
    await search_all(state)
    await run_tool(
        state, "shortlist_place", place_id=MUSEUM.id, score=0.7, interests=[], reason="музей"
    )

    first = await run_tool(state, "finish_research", summary="хватит")
    second = await run_tool(state, "finish_research", summary="хватит")
    third = await run_tool(state, "finish_research", summary="хватит")

    assert not first.finished and "need at least 4" in first.error
    assert "no candidate for the interest 'модерн'" in first.error
    assert not second.finished
    assert third.finished and "не хватает" in third.summary


async def test_finish_is_accepted_when_shortlist_is_complete() -> None:
    state = make_state()
    await search_all(state)
    for place in [*OSM_PLACES, SINGER]:
        interests = ["модерн"] if place is SINGER else []
        await run_tool(
            state,
            "shortlist_place",
            place_id=place.id,
            score=0.8,
            interests=interests,
            reason="подходит",
        )
    outcome = await run_tool(state, "finish_research", summary="готово")
    assert outcome.finished and outcome.error is None


async def test_finish_is_rejected_while_a_must_include_place_is_missing() -> None:
    state = make_state(must_include=["Эрмитаж"])
    await search_all(state)
    for place in [*OSM_PLACES, SINGER]:
        await execute(state, "shortlist_place", shortlist_call(place, ["модерн"]).arguments)

    outcome = await run_tool(state, "finish_research", summary="готово")

    assert not outcome.finished and "must_include 'Эрмитаж'" in outcome.error


async def test_places_another_day_planned_are_hidden_and_cannot_be_shortlisted() -> None:
    state = make_state()
    await search_all(state)
    await execute(state, "shortlist_place", shortlist_call(MURAL, ["модерн"]).arguments)
    state.taken = {MUSEUM.id, MURAL.id}

    found = json.loads((await run_tool(state, "search_osm", categories=list(CATALOG))).content)
    taken = await run_tool(state, "shortlist_place", place_id=MUSEUM.id, **SHORTLIST_ARGS)
    finish = await run_tool(state, "finish_research", summary="готово")

    assert [p["id"] for p in found["places"]] == [CAFE.id]
    assert "another day" in taken.error
    # The shortlisted mural is planned on another day: it does not count for this one.
    assert "the shortlist has 0 places" in finish.error
    assert list(state.available()) == [] and list(state.shortlist) == [MURAL.id]


async def test_days_sharing_their_searches_send_one_query_and_retry_a_failed_one(
    monkeypatch: pytest.MonkeyPatch, sources: list[tuple[Any, ...]]
) -> None:
    first, second = make_state(), make_state()
    second.registry, second.searches = first.registry, first.searches

    await asyncio.gather(
        run_tool(first, "search_osm", categories=["museum"]),
        run_tool(second, "search_osm", categories=["museum"]),
    )
    assert sources == [("osm", ("museum",), 1500)]
    assert MUSEUM.id in second.registry

    working = wikipedia.search
    monkeypatch.setattr(wikipedia, "search", broken_source)
    failed = await run_tool(first, "search_wikipedia", query="модерн")
    monkeypatch.setattr(wikipedia, "search", working)
    retried = await run_tool(second, "search_wikipedia", query="модерн")

    assert "HTTP 504" in failed.error and retried.count == 1


async def test_a_day_sees_the_sources_record_whatever_another_day_looked_up() -> None:
    first, second = make_state(), make_state()
    second.registry = first.registry
    await run_tool(first, "search_wikipedia", query="модерн")
    await run_tool(first, "get_place_details", place_id=SINGER.id)

    outcome = await run_tool(second, "search_wikipedia", query="модерн")

    [shown] = json.loads(outcome.content)["places"]
    assert shown["snippet"] == SINGER.description  # not the extract the first day fetched
    assert first.registry[SINGER.id].description == EXTRACT  # saved with the first day


async def test_each_day_looks_a_place_up_in_its_own_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One building in two categories: its record depends on the category searched.
    as_cafe = MUSEUM.model_copy(update={"kind": "cafe", "indoor": None})

    async def search_places(center, radius_m, categories, limit=40):
        return [MUSEUM if "museum" in categories else as_cafe]

    monkeypatch.setattr(osm, "search_places", search_places)
    first, second = make_state(), make_state()
    second.registry = first.registry
    await run_tool(first, "search_osm", categories=["museum"])
    await run_tool(second, "search_osm", categories=["cafe"])

    details = [
        json.loads((await run_tool(day, "get_place_details", place_id=MUSEUM.id)).content)
        for day in (first, second)
    ]
    await run_tool(second, "shortlist_place", place_id=MUSEUM.id, **SHORTLIST_ARGS)

    assert [d["kind"] for d in details] == ["museum", "cafe"]
    assert second.shortlist[MUSEUM.id].place.kind == "cafe"


async def test_a_search_its_last_day_gave_up_on_is_not_awaited_by_the_next() -> None:
    queries: list[int] = []

    async def fetch() -> list[Place]:
        queries.append(len(queries))
        await asyncio.sleep(1 if len(queries) == 1 else 0)
        return [MUSEUM]

    state = make_state()
    first = asyncio.create_task(tools._shared(state, "osm museum", fetch))
    await asyncio.sleep(0)
    first.cancel()  # its tool call timed out
    # Another day sends the same search before the abandoned query has been cancelled.
    second = asyncio.create_task(tools._shared(state, "osm museum", fetch))

    results = await asyncio.gather(first, second, return_exceptions=True)

    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] == [MUSEUM] and queries == [0, 1]
    assert state.searches["osm museum"].task.result() == [MUSEUM]


# --- research loop -------------------------------------------------------------------

# The loop's budgets are the caller's to set; these are generous enough for every script.
BUDGET = {"max_steps": 14, "time_budget_s": 60.0}


async def test_loop_runs_tools_answers_every_call_id_and_finishes() -> None:
    searches = [
        call("search_osm", categories=list(CATALOG)),
        call("search_wikipedia", query="модерн"),
    ]
    llm = FakeLLM(
        [
            turn(*searches, content="Ищу музеи, кафе и модерн."),
            # run concurrently, yet finish_research sees the four places shortlisted before it
            turn(
                *[shortlist_call(p) for p in OSM_PLACES],
                shortlist_call(SINGER, ["модерн"]),
                call("finish_research", summary="Нашёл четыре места."),
            ),
        ]
    )
    events = Recorder()

    result = await research(llm, make_state(), events, **BUDGET)

    assert (result.stop_reason, result.steps, len(result.candidates)) == ("finished", 2, 4)
    assistant, *tool_messages = llm.seen[1][-3:]
    assert [c["id"] for c in assistant["tool_calls"]] == [c.id for c in searches]
    assert [m["role"] for m in tool_messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [c.id for c in searches]
    assert SINGER.id in tool_messages[1]["content"]
    assert [e.type for e in events[:5]] == [
        "thought",
        "tool_call",
        "tool_call",
        "tool_result",
        "tool_result",
    ]
    assert sum(e.type == "candidate" for e in events) == 4
    # The traveller's language is pinned in the system prompt, not only in the task.
    assert "must be in Russian" in llm.seen[0][0]["content"]


async def test_loop_stops_at_max_steps_and_warns_before() -> None:
    searches = [call("search_osm", categories=["museum"], radius_m=r) for r in (500, 1000, 1500)]
    llm = FakeLLM([turn(c) for c in searches])
    result = await research(llm, make_state(), Recorder(), max_steps=3, time_budget_s=60.0)
    assert (result.stop_reason, result.steps) == ("max_steps", 3)
    assert llm.seen[1][-1] == {"role": "user", "content": BUDGET_WARNING}


async def test_loop_refuses_a_repeated_call_and_stops_on_the_third(sources: list) -> None:
    llm = FakeLLM(
        [
            turn(call("search_osm", categories=["museum"], radius_m=1000)),
            turn(call("search_osm", radius_m=1000, categories=["museum"])),  # same, other key order
            turn(call("search_osm", categories=["museum"], radius_m=1000)),
        ]
    )
    result = await research(llm, make_state(), Recorder(), **BUDGET)
    assert (result.stop_reason, result.steps) == ("repeated_action", 3)
    assert "already called" in llm.seen[2][-1]["content"]
    assert sources == [("osm", ("museum",), 1000)]


async def test_loop_nudges_once_then_stops_without_tool_calls() -> None:
    llm = FakeLLM([turn(content="Думаю, что делать."), turn(content="Готово.")])
    result = await research(llm, make_state(), Recorder(), **BUDGET)
    assert (result.stop_reason, result.steps) == ("no_tool_calls", 2)
    assert llm.seen[1][-1] == {"role": "user", "content": NUDGE}


async def test_loop_stops_on_the_time_budget() -> None:
    class SlowLLM:
        async def chat(self, messages, tools=None, **_):
            await asyncio.sleep(5)

    result = await research(SlowLLM(), make_state(), Recorder(), max_steps=14, time_budget_s=0.05)
    assert (result.stop_reason, result.steps) == ("time_budget", 1)


async def test_llm_failure_keeps_a_non_empty_shortlist_and_propagates_otherwise() -> None:
    failure = LLMError("rate limited", 429)
    search = call("search_osm", categories=["museum"])
    llm = FakeLLM([turn(search), turn(shortlist_call(MUSEUM)), failure])
    result = await research(llm, make_state(), Recorder(), **BUDGET)
    assert result.stop_reason == "llm_error"
    assert [c.place.id for c in result.candidates] == [MUSEUM.id]

    with pytest.raises(LLMError):
        await research(FakeLLM([failure]), make_state(), Recorder(), **BUDGET)


# --- brief ---------------------------------------------------------------------------


async def test_brief_defaults_fill_what_the_request_does_not_say() -> None:
    llm = FakeLLM(structured=[{"city": "Санкт-Петербург", "date": "2025-10-25"}])
    brief = await parse_brief(llm, "Погулять по Питеру", today=date(2026, 10, 19))
    assert "2026-10-19 (Monday)" in llm.seen[0][1]["content"]
    assert brief.date == date(2026, 10, 19)  # a date in the past falls back to today
    assert (brief.start_time, brief.end_time) == (time(10), time(18))
    assert (brief.language, brief.pace) == ("ru", "normal")


async def test_brief_overrides_win_and_an_empty_window_is_fixed() -> None:
    draft = {
        "city": "Москва",
        "language": "en",
        "start_time": "18:00",
        "end_time": "21:00",
        "interests": [{"label": "street art", "weight": 1.0}],
    }
    overrides = {"city": "Казань", "date": "2026-10-25", "end_time": "11:00"}
    llm = FakeLLM(structured=[draft])
    brief = await parse_brief(llm, "street art walk", date(2026, 10, 19), overrides)
    assert (brief.city, brief.date, brief.language) == ("Казань", date(2026, 10, 25), "en")
    # The form's 11:00 outranks the text's 18:00: the walk moves to the eight hours before it.
    assert (brief.start_time, brief.end_time) == (time(3), time(11))
    assert brief.interests == [Interest(label="street art", weight=1.0)]


@pytest.mark.parametrize(
    ("overrides", "window"),
    [
        ({"end_time": "09:00"}, (time(1), time(9))),  # before the default 10:00 start
        ({"end_time": "03:00"}, (time(0), time(3))),  # not before midnight
        ({"start_time": "16:00", "end_time": "12:00"}, (time(16), time(23, 59))),
        ({"start_time": "23:59"}, (time(23, 59), time(23, 59))),  # plan_journey refuses it
    ],
)
async def test_brief_window_keeps_the_forms_end_time(overrides, window) -> None:
    brief = await parse_brief(FakeLLM(), "", date(2026, 10, 19), {"city": "Казань", **overrides})
    assert (brief.start_time, brief.end_time) == window


@pytest.mark.parametrize(
    ("said", "form", "days"),
    [
        ({"date": "2026-10-24", "date_to": "2026-10-28"}, {}, ("2026-10-24", "2026-10-28")),
        ({"date": "2026-10-24", "days": 5}, {}, ("2026-10-24", "2026-10-28")),
        ({"days": 1}, {}, ("2026-10-19", None)),
        ({"date": "2026-10-24", "date_to": "2026-10-23"}, {}, ("2026-10-24", None)),
        # The form's calendar moves the start; the length said in the text stays.
        ({"date": "2026-10-24", "days": 3}, {"date": "2026-11-01"}, ("2026-11-01", "2026-11-03")),
        (
            {"days": 3},
            {"date": "2026-11-01", "date_to": "2026-11-02"},
            ("2026-11-01", "2026-11-02"),
        ),
    ],
)
async def test_brief_reads_the_days_of_the_trip(
    said: dict[str, Any], form: dict[str, Any], days: tuple[str, str | None]
) -> None:
    llm = FakeLLM(structured=[{"city": "Казань", **said}])
    brief = await parse_brief(llm, "Казань на несколько дней", date(2026, 10, 19), form)
    expected = tuple(date.fromisoformat(d) if d else None for d in days)
    assert (brief.date, brief.date_to) == expected


async def test_brief_without_a_city_is_an_error() -> None:
    with pytest.raises(ValueError, match="city"):
        await parse_brief(FakeLLM(structured=[{}]), "просто погулять", date(2026, 10, 19))


# --- writer and replanner ------------------------------------------------------------


async def test_writer_fills_known_stops_and_ignores_unknown_ids() -> None:
    texts = {
        "title": "Модерн и Фаберже",
        "summary": "Прогулка по Невскому.",
        "stops": [
            {"place_id": MUSEUM.id, "blurb": "Коллекция Фаберже.", "tip": "Открыт до 20:45."},
            {"place_id": "osm:node/404", "blurb": "Выдуманное место."},
        ],
    }
    plan = await write_texts(FakeLLM(structured=[texts]), make_plan())
    assert plan.title == "Модерн и Фаберже"
    assert (plan.stops[0].blurb, plan.stops[0].tip) == ("Коллекция Фаберже.", "Открыт до 20:45.")
    assert plan.stops[1].blurb is None  # no text for it: the UI shows the reason
    assert [s.place.id for s in plan.stops] == [MUSEUM.id, SINGER.id]


async def test_writer_gets_what_we_know_about_the_traveller() -> None:
    texts = {"title": "Модерн", "summary": "Коротко.", "stops": []}
    llm = FakeLLM(structured=[texts])
    await write_texts(llm, make_plan(), ["вегетарианец"])
    traveller = json.loads(llm.seen[0][1]["content"])["traveller"]
    assert traveller == {
        "interests": ["модерн"],
        "with_kids": False,
        "notes": None,
        "known": ["вегетарианец"],
    }


async def test_writer_failure_or_timeout_keeps_the_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowLLM:
        async def structured(self, messages, schema, **_):
            await asyncio.sleep(5)

    monkeypatch.setattr(writer, "WRITER_TIMEOUT_S", 0.01)
    plan = make_plan()
    assert await write_texts(FakeLLM(structured=[LLMError("rate limited", 429)]), plan) == plan
    assert await write_texts(SlowLLM(), plan) == plan


async def test_replanner_keeps_only_ids_of_current_stops_and_a_valid_window() -> None:
    edit = {
        "exclude_place_ids": [SINGER.id, "osm:node/404", SINGER.id],
        "end_time": "09:00",
        "research_goal": "найти кафе вместо Дома Зингера",
        "reply": "Заменю Дом Зингера на кафе.",
    }
    llm = FakeLLM(structured=[edit])
    result = await interpret_feedback(llm, make_plan(), "замени второе место на кафе")
    assert "2. id=wikipedia:ru:4321 | Дом Зингера" in llm.seen[0][1]["content"]
    assert result.exclude_place_ids == [SINGER.id]
    assert result.end_time is None  # 09:00 is before the 10:00 start
    assert result.reply.endswith("Время оставляю прежним: окно 10:00–09:00 получилось пустым.")
    assert result.research_goal == "найти кафе вместо Дома Зингера"


async def test_replanner_says_when_it_found_nothing_to_remove() -> None:
    edit = {"exclude_place_ids": ["osm:node/404"], "reply": "Убираю собор."}
    result = await interpret_feedback(FakeLLM(structured=[edit]), make_plan(), "убери собор")
    assert result.exclude_place_ids == []
    assert result.reply == "Убираю собор. Не нашёл в маршруте остановок, которые нужно убрать."


# --- LLM client ----------------------------------------------------------------------


async def test_provider_error_passes_on_only_the_providers_message() -> None:
    # OpenRouter's error body also names the account; the UI must not get it.
    def provider(request: httpx.Request) -> httpx.Response:
        body = {"error": {"message": "No auth credentials found", "code": 401}, "user_id": "u_42"}
        return httpx.Response(401, json=body)

    client = AsyncOpenAI(
        base_url="https://llm.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    with pytest.raises(LLMError) as error:
        await LLM(client=client, model="test").chat([{"role": "user", "content": "hi"}])
    assert (str(error.value), error.value.status) == ("No auth credentials found", 401)


async def test_answers_are_recorded_and_replayed_only_with_replay_on(tmp_path) -> None:
    sent: list[dict[str, Any]] = []
    hi = [{"role": "user", "content": "hi"}]

    await LLM(client=provider(sent), model="test", cache_dir=tmp_path).chat(hi)
    assert not list(tmp_path.iterdir())  # replay off: no visitor's words kept on disk

    replaying = LLM(client=provider(sent), model="test", cache_dir=tmp_path, replay=True)
    first, second = await replaying.chat(hi), await replaying.chat(hi)

    assert first.content == second.content and len(sent) == 2
    assert (replaying.usage.calls, replaying.usage.cached_calls) == (1, 1)


async def test_a_short_call_has_its_own_timeout_and_the_same_replay_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    llm = LLM(client=provider(sent), model="test", cache_dir=tmp_path, replay=True)
    create = llm.client.chat.completions.create
    timeouts: list[float | None] = []

    async def spy(**request: Any) -> Any:
        timeouts.append(request.get("timeout"))
        return await create(**request)

    monkeypatch.setattr(llm.client.chat.completions, "create", spy)
    hi = [{"role": "user", "content": "hi"}]

    await llm.chat(hi, timeout=25)
    await llm.chat(hi)  # the same request with the client's own timeout: replayed

    assert timeouts == [25] and llm.usage.cached_calls == 1
