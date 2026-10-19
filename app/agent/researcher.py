"""The research agent: a hand-written ReAct loop over the tools in tools.py.

The pipeline around it is fixed (brief -> research -> optimize -> validate -> write).
Only this step is an agent loop, because only here the next action depends on what
was found: "constructivism" needs a Wikipedia text search, "street art" an OSM
artwork search, "street food" both OSM and the Wikivoyage eat section.

Every run ends with an explicit, logged stop reason:
- finished: finish_research was accepted;
- max_steps / time_budget: the step or wall-clock budget ran out;
- repeated_action: the same call with the same arguments came a third time
  (the second time it is not executed and the model gets an error instead);
- no_tool_calls: two replies in a row without tool calls (the first gets a nudge);
- llm_error: the provider failed after its retries, but the shortlist is not empty
  (with an empty shortlist the error propagates: there is nothing to plan from).
Whatever the reason, the shortlist built so far is the result.
"""

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.agent.prompts import BUDGET_WARNING, NUDGE, RESEARCHER
from app.agent.tools import (
    ResearchState,
    ToolOutcome,
    error_outcome,
    execute,
    min_candidates,
    tool_schemas,
)
from app.llm import LLM, LLMError, ToolCall
from app.models import AgentEvent, Candidate
from app.sources import osm

log = logging.getLogger(__name__)

Emit = Callable[[AgentEvent], Awaitable[None]]
StopReason = Literal[
    "finished", "max_steps", "time_budget", "repeated_action", "no_tool_calls", "llm_error"
]

REPEAT_ERROR = "you already called this with these arguments; the results are above"


@dataclass
class ResearchResult:
    candidates: list[Candidate]
    stop_reason: StopReason
    steps: int


async def research(
    llm: LLM,
    state: ResearchState,
    emit: Emit,
    max_steps: int = 14,
    goal: str | None = None,
    time_budget_s: float = 150.0,
) -> ResearchResult:
    state.finish_attempts = 0
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RESEARCHER},
        {"role": "user", "content": _task(state, goal, max_steps)},
    ]
    schemas = tool_schemas()
    calls_seen: Counter[tuple[str, str]] = Counter()
    steps = 0
    idle_replies = 0
    stop_reason: StopReason = "max_steps"
    try:
        async with asyncio.timeout(time_budget_s):  # covers LLM calls and tools alike
            while steps < max_steps:
                steps += 1
                # Room for a dozen parallel tool calls: a cut-off call arrives as broken JSON.
                turn = await llm.chat(messages, tools=schemas, max_tokens=6000)
                messages.append(turn.message)
                if turn.content and turn.content.strip():
                    await emit(AgentEvent(type="thought", data={"text": turn.content.strip()}))

                if not turn.tool_calls:
                    idle_replies += 1
                    if idle_replies == 2:
                        stop_reason = "no_tool_calls"
                        break
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                idle_replies = 0

                seen_counts = [_count_call(calls_seen, call) for call in turn.tool_calls]
                if max(seen_counts) >= 3:
                    stop_reason = "repeated_action"
                    break
                repeated = [count == 2 for count in seen_counts]
                outcomes = await _run_tools(state, turn.tool_calls, repeated, emit)
                messages.extend(
                    {"role": "tool", "tool_call_id": call.id, "content": outcome.content}
                    for call, outcome in zip(turn.tool_calls, outcomes, strict=True)
                )
                if any(outcome.finished for outcome in outcomes):
                    stop_reason = "finished"
                    break
                if steps == max_steps - 2:
                    messages.append({"role": "user", "content": BUDGET_WARNING})
    except TimeoutError:
        stop_reason = "time_budget"
    except LLMError as e:
        if not state.shortlist:
            raise
        log.warning("research: LLM failed, keeping the shortlist: %s", e)
        stop_reason = "llm_error"

    # Any stop other than an accepted finish is a budget or a guard firing: a warning.
    log.log(
        logging.INFO if stop_reason == "finished" else logging.WARNING,
        "research stopped: %s after %d steps, %d candidates",
        stop_reason,
        steps,
        len(state.shortlist),
    )
    return ResearchResult(list(state.shortlist.values()), stop_reason, steps)


def _count_call(calls_seen: Counter[tuple[str, str]], call: ToolCall) -> int:
    """How many times this exact call has been made, this one included."""
    # finish_research is exempt: its answer depends on the shortlist, not only on its
    # arguments, and tools.py already bounds it to three attempts.
    if call.name == "finish_research":
        return 1
    args = _parsed(call.arguments)
    normalized = json.dumps(args, sort_keys=True) if isinstance(args, dict) else args.strip()
    key = (call.name, normalized)
    calls_seen[key] += 1
    return calls_seen[key]


async def _run_tools(
    state: ResearchState, calls: list[ToolCall], repeated: list[bool], emit: Emit
) -> list[ToolOutcome]:
    """Run one turn's tool calls concurrently; report them to the UI in call order."""
    for call in calls:
        await emit(
            AgentEvent(
                type="tool_call",
                data={"id": call.id, "name": call.name, "args": _parsed(call.arguments)},
            )
        )

    async def run(call: ToolCall, is_repeat: bool) -> ToolOutcome:
        if is_repeat:
            return error_outcome(REPEAT_ERROR)
        return await execute(state, call.name, call.arguments)

    # The shortlist tools never await, so even under gather they change the state
    # in the order the model called them (shortlist_place before finish_research).
    outcomes = await asyncio.gather(*(run(c, r) for c, r in zip(calls, repeated, strict=True)))
    for call, outcome in zip(calls, outcomes, strict=True):
        await emit(
            AgentEvent(
                type="tool_result",
                data={
                    "id": call.id,
                    "name": call.name,
                    "args": _parsed(call.arguments),
                    "summary": outcome.summary,
                    "count": outcome.count,
                    "error": outcome.error,
                },
            )
        )
        if outcome.candidate_event:
            await emit(AgentEvent(type="candidate", data=outcome.candidate_event))
    return list(outcomes)


_LANGUAGES = {"ru": "Russian", "en": "English"}


def _task(state: ResearchState, goal: str | None, max_steps: int) -> str:
    brief = state.brief
    start = state.start
    catalog = "\n".join(f"- {key}: {spec.label_ru}" for key, spec in osm.CATEGORIES.items())
    parts = [
        f"Trip brief: {brief.model_dump_json(exclude_none=True)}",
        f"Start point: {start.label or 'start'} ({start.lat:.5f}, {start.lon:.5f})",
        f"Date: {brief.date:%Y-%m-%d, %A}; walk from {brief.start_time:%H:%M} "
        f"to {brief.end_time:%H:%M}",
        f"Weather: {state.weather_summary}",
        f"Write your notes and reasons in {_LANGUAGES[brief.language]}.",
        f"Target: at least {min_candidates(brief)} shortlisted places. Budget: {max_steps} steps.",
        f"OSM categories for search_osm:\n{catalog}",
    ]
    if state.shortlist:
        current = [
            {"id": c.place.id, "name": c.place.name, "kind": c.place.kind, "score": c.score}
            for c in state.shortlist.values()
        ]
        parts.append(
            "Current shortlist (keep what still fits, remove what does not): "
            + json.dumps(current, ensure_ascii=False)
        )
    if state.excluded:
        rejected = "; ".join(
            f"{state.registry[i].name} ({i})" if i in state.registry else i
            for i in sorted(state.excluded)
        )
        parts.append(f"The traveller rejected these places; never shortlist them: {rejected}")
    if goal:
        parts.append(f"Goal of this run: {goal}")
    return "\n\n".join(parts)


def _parsed(arguments: str) -> dict[str, Any] | str:
    """Arguments as a dict when they are a JSON object, else the raw text."""
    try:
        value = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return value if isinstance(value, dict) else arguments
