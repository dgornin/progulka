"""The macro-structure: a fixed pipeline with one agent loop inside it.

    brief (LLM, structured output) -> city and start point (Nominatim)
    -> weather (Open-Meteo) -> research (ReAct agent with tools)
    -> optimize (code) -> walking route (OSRM) -> validate (code)
    -> one bounded repair round if the validator found errors
    -> texts (LLM, structured output)

Only the research step is an agent, because only there the next step depends on
what was found. Everything that can be computed (geocoding, time arithmetic, route
order, opening hours) is code, and the validator is the external critic the repair
round listens to.

Feedback re-enters at research (when new places are needed) or at optimize, and
every route, first or rebuilt, goes through the same validation and repair round.
A failed feedback run is not saved: the caller stores the trip only after success.

Personal context: the planner panel, the request, the profile and later the feedback
are merged into the brief by code (app/personalization.py); long-term memory and the
profile's notes reach the research agent and the writer as data; after a successful
run the traveller's own words are mined for new durable facts (app/agent/memory.py),
which the pipeline returns in TripRun.learned for the caller to store. The pipeline
keeps nothing between runs: the caller persists each TripRun, so a trip survives a restart.
"""

import asyncio
import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent.brief import parse_brief
from app.agent.memory import extract_memory
from app.agent.prompts import REPAIR_GOAL, REPLACE_GOAL
from app.agent.replanner import PlanEdit, interpret_feedback
from app.agent.researcher import research
from app.agent.tools import ResearchState
from app.agent.writer import write_texts
from app.config import settings
from app.llm import LLM
from app.models import (
    AgentEvent,
    Candidate,
    Emit,
    GeoPoint,
    HourWeather,
    LocalTime,
    Plan,
    TripBrief,
)
from app.personalization import (
    brief_defaults,
    brief_overrides,
    personalize,
    profile_facts,
    traveller_facts,
    with_feedback,
)
from app.planning.geo import walk_limit_km
from app.planning.optimizer import optimize, schedule
from app.planning.routing import walking_route
from app.planning.validator import matches_avoid, names_place, validate
from app.profile import Profile, WalkPreferences
from app.sources import geocode, weather
from app.sources.http import SourceError

# Budgets of the research runs: steps and wall-clock seconds. A repair or a feedback
# run continues a finished plan, so it gets less than the first research.
RESEARCH_BUDGET_S = 150.0
FEEDBACK_STEPS, FEEDBACK_BUDGET_S = 8, 90.0
REPAIR_STEPS, REPAIR_BUDGET_S = 6, 60.0
# The city is required, the forecast is optional: neither may stall the run for the
# minutes the shared HTTP client spends on retries.
GEOCODE_TIMEOUT_S = 30
WEATHER_TIMEOUT_S = 10
# The brief and the feedback are one structured LLM call each, which the LLM client's
# retries can stretch to minutes; the research and the writer have budgets of their own.
LLM_STEP_TIMEOUT_S = 60
# In a shorter window the optimizer can place no stop: refused before the research.
MIN_WINDOW = timedelta(minutes=30)


class PipelineError(RuntimeError):
    """A failure the user can act on (unknown city, empty result); shown as is."""


# Raised in two places each; the API refuses an empty form itself, before a run starts.
EMPTY_REQUEST = "Опишите прогулку или укажите город"
LLM_TIMEOUT = "LLM-провайдер не ответил вовремя, попробуйте ещё раз"


class PlanRequest(BaseModel):
    """The planner form. The text may be empty when the city is known from the form or
    from the profile: then the brief is built from the structured inputs alone."""

    request: str = Field("", max_length=2000)
    city: str | None = Field(None, max_length=100)
    # `dt.` because a field named `date` shadows the type inside the class body.
    date: dt.date | None = None
    start_time: LocalTime | None = None
    end_time: LocalTime | None = None
    start_point: str | None = Field(None, max_length=200)
    preferences: WalkPreferences = Field(default_factory=WalkPreferences)

    @field_validator("request")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("city", "start_point")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return (value or "").strip() or None

    @model_validator(mode="after")
    def _window(self) -> "PlanRequest":
        if self.start_time and self.end_time and self.end_time <= self.start_time:
            raise ValueError("Конец прогулки должен быть позже начала")
        return self


def has_place(req: PlanRequest, profile: Profile | None) -> bool:
    """The request can be planned: its text, the form or the profile names a city."""
    return bool(req.request or req.city or (profile and profile.home_city))


class FeedbackRequest(BaseModel):
    feedback: str = Field("", max_length=1000)
    exclude_place_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class TripRun:
    """A trip's working state: what a feedback run continues from. The id is the trip's."""

    id: str
    state: ResearchState
    weather: list[HourWeather]
    plan: Plan | None = None
    # New long-term memory facts learned in the last run, for the caller to store.
    learned: list[str] = field(default_factory=list)


async def stage(emit: Emit, name: str, status: str, label: str, detail: Any = None) -> None:
    data: dict[str, Any] = {"name": name, "status": status, "label": label}
    if detail is not None:
        data["detail"] = detail
    await emit(AgentEvent(type="stage", data=data))


async def plan_trip(
    req: PlanRequest,
    llm: LLM,
    emit: Emit,
    today: date,
    profile: Profile | None = None,
    memory: list[str] | None = None,
) -> TripRun:
    """`memory` is the traveller's long-term memory; None means the caller keeps none
    (the eval, an anonymous run), and then nothing is learned either."""
    if not has_place(req, profile):
        raise PipelineError(EMPTY_REQUEST)
    await stage(emit, "brief", "start", "Разбираю запрос")
    form = req.model_dump(exclude={"request", "preferences"}, exclude_none=True)
    try:
        async with asyncio.timeout(LLM_STEP_TIMEOUT_S):
            brief = await parse_brief(
                llm,
                req.request,
                today,
                overrides=brief_overrides(form, req.preferences),
                defaults=brief_defaults(profile),
            )
    except ValueError as e:
        raise PipelineError("В запросе не указан город: напишите, где гуляем") from e
    except TimeoutError as e:
        raise PipelineError(LLM_TIMEOUT) from e
    brief = personalize(brief, req.preferences, profile)
    await stage(emit, "brief", "done", "Запрос разобран", brief.model_dump(mode="json"))
    window = datetime.combine(brief.date, brief.end_time) - datetime.combine(
        brief.date, brief.start_time
    )
    if window < MIN_WINDOW:
        raise PipelineError(
            f"Окно прогулки {brief.start_time:%H:%M}–{brief.end_time:%H:%M} слишком короткое: "
            f"нужно хотя бы {MIN_WINDOW.seconds // 60} минут"
        )

    await stage(emit, "weather", "start", "Ищу город и смотрю прогноз погоды")
    try:
        async with asyncio.timeout(GEOCODE_TIMEOUT_S):
            city = await geocode.locate_city(brief.city, brief.country)
    except (SourceError, TimeoutError) as e:
        raise PipelineError("Геокодер OpenStreetMap не отвечает, попробуйте ещё раз") from e
    if city is None:
        raise PipelineError(f"Не нашёл город «{brief.city}» в OpenStreetMap")
    start = GeoPoint(lat=city.center.lat, lon=city.center.lon, label=f"центр, {city.name}")
    if brief.start_point:
        # Not found or not answered, the city centre is a fine start; the warning says which.
        try:
            async with asyncio.timeout(GEOCODE_TIMEOUT_S):
                found = await geocode.geocode(
                    brief.start_point, city=brief.city, country=brief.country
                )
        except (SourceError, TimeoutError):
            await emit(
                _warning(
                    f"Геокодер OpenStreetMap не ответил на поиск «{brief.start_point}», "
                    "начинаю из центра города"
                )
            )
        else:
            if found:
                start = found
            else:
                await emit(_warning(f"Не нашёл «{brief.start_point}», начинаю из центра города"))

    try:
        async with asyncio.timeout(WEATHER_TIMEOUT_S):
            hours = await weather.hourly(start, brief.date, today)
    except (SourceError, TimeoutError):
        hours = []
    summary = weather.summarize(hours, brief.start_time, brief.end_time, brief.language)
    await stage(
        emit,
        "weather",
        "done",
        f"Погода: {summary}",
        {"summary": summary, "start": start.model_dump()},
    )

    state = ResearchState(
        brief=brief,
        start=start,
        weather_summary=summary,
        registry={},
        shortlist={},
        excluded=set(),
    )
    trip_run = TripRun(id=uuid.uuid4().hex[:12], state=state, weather=hours)
    await _research_and_build(
        trip_run,
        llm,
        emit,
        goal=None,
        max_steps=settings.agent_max_steps,
        time_budget_s=RESEARCH_BUDGET_S,
        facts=traveller_facts(profile, memory or []),
    )
    trip_run.learned = await _learn(llm, emit, req.request, profile, memory)
    return trip_run


async def apply_feedback(
    trip_run: TripRun,
    req: FeedbackRequest,
    llm: LLM,
    emit: Emit,
    profile: Profile | None = None,
    memory: list[str] | None = None,
) -> tuple[TripRun, str | None]:
    """`profile` and `memory` as in plan_trip; the brief was personalised when the
    trip was planned, here they only inform the agent, the writer and the learning."""
    if trip_run.plan is None:
        raise PipelineError("У этой поездки ещё нет маршрута")
    trip_run.learned = []
    facts = traveller_facts(profile, memory or [])
    await stage(emit, "feedback", "start", "Разбираю правку")
    stops = {stop.place.id: stop.place.name for stop in trip_run.plan.stops}
    if req.feedback.strip():
        try:
            async with asyncio.timeout(LLM_STEP_TIMEOUT_S):
                edit = await interpret_feedback(llm, trip_run.plan, req.feedback)
        except TimeoutError as e:
            raise PipelineError(LLM_TIMEOUT) from e
    else:
        # The "Убрать" button: nothing to interpret, so no LLM call. The traveller did
        # not ask for a shorter walk, so the agent looks for a replacement. The names
        # keep the request's order: the goal is part of the LLM replay cache key.
        removed = [stops[i] for i in dict.fromkeys(req.exclude_place_ids) if i in stops]
        edit = PlanEdit(
            reply="Убираю выбранные места и ищу им замену.",
            research_goal=REPLACE_GOAL.format(names=", ".join(removed)) if removed else None,
        )
    # Only current stops: ids from the client end up in the agent's prompt.
    excluded = {
        i: stops[i] for i in [*edit.exclude_place_ids, *req.exclude_place_ids] if i in stops
    }

    _apply_edit(trip_run, edit, excluded)
    await stage(emit, "feedback", "done", edit.reply, edit.model_dump(mode="json"))
    if edit.research_goal:
        await _research_and_build(
            trip_run,
            llm,
            emit,
            goal=edit.research_goal,
            max_steps=FEEDBACK_STEPS,
            time_budget_s=FEEDBACK_BUDGET_S,
            facts=facts,
        )
    else:
        plan = await _build_checked(trip_run, llm, emit, facts)
        await _write(trip_run, llm, emit, plan, facts)
    trip_run.learned = await _learn(llm, emit, req.feedback, profile, memory)
    return trip_run, edit.reply


async def _learn(
    llm: LLM, emit: Emit, text: str, profile: Profile | None, memory: list[str] | None
) -> list[str]:
    """New long-term facts from the traveller's own words of this run.

    Runs only after the run succeeded, and not for the "Убрать" button: an empty text
    has nothing to learn from. A stage of its own: the plan is sent only after it.
    """
    if memory is None or not text.strip():
        return []
    await stage(emit, "memory", "start", "Запоминаю, что узнал о вас")
    learned = await extract_memory(llm, [text], known=[*memory, *profile_facts(profile)])
    if learned:
        await emit(AgentEvent(type="memory", data={"facts": learned}))
    await stage(emit, "memory", "done", f"Новых фактов: {len(learned)}")
    return learned


async def _research_and_build(
    trip_run: TripRun,
    llm: LLM,
    emit: Emit,
    goal: str | None,
    max_steps: int,
    time_budget_s: float,
    facts: Sequence[str],
) -> None:
    await stage(emit, "research", "start", "Агент ищет места")
    result = await research(
        llm,
        trip_run.state,
        emit,
        max_steps=max_steps,
        goal=goal,
        time_budget_s=time_budget_s,
        traveller_facts=facts,
    )
    await stage(
        emit,
        "research",
        "done",
        f"Поиск завершён ({_STOP_REASONS.get(result.stop_reason, result.stop_reason)})",
        {
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "candidates": len(result.candidates),
        },
    )
    if not trip_run.state.shortlist:
        if result.stop_reason == "finished":
            raise PipelineError("Агент не нашёл подходящих мест. Попробуйте расширить запрос.")
        # A budget or a guard fired: the request itself may be fine.
        reason = _STOP_REASONS.get(result.stop_reason, result.stop_reason)
        raise PipelineError(f"Агент остановился, не найдя мест: {reason}. Попробуйте ещё раз.")
    await _write(trip_run, llm, emit, await _build_checked(trip_run, llm, emit, facts), facts)


async def _build_checked(trip_run: TripRun, llm: LLM, emit: Emit, facts: Sequence[str]) -> Plan:
    """Build and validate the route; on errors, one bounded repair round.

    The validator's findings become the research agent's goal. A route with no stops
    is never returned as a result.
    """
    plan = await _build(trip_run, emit)
    errors = [v for v in plan.violations if v.severity == "error"]
    if errors:
        await stage(emit, "repair", "start", "Валидатор нашёл проблемы — агент исправляет")
        # The messages stay in the traveller's language: they are data for the agent.
        goal = REPAIR_GOAL.format(problems="\n".join(f"- {v.message}" for v in errors))
        result = await research(
            llm,
            trip_run.state,
            emit,
            max_steps=REPAIR_STEPS,
            goal=goal,
            time_budget_s=REPAIR_BUDGET_S,
            traveller_facts=facts,
        )
        plan = await _build(trip_run, emit)
        await stage(emit, "repair", "done", "Исправление завершено", {"steps": result.steps})
    if not plan.stops:
        raise PipelineError(
            "Не удалось собрать маршрут: ни одно найденное место не подошло. "
            "Попробуйте расширить запрос."
        )
    return plan


async def _write(trip_run: TripRun, llm: LLM, emit: Emit, plan: Plan, facts: Sequence[str]) -> None:
    await stage(emit, "write", "start", "Пишу описания остановок")
    trip_run.plan = await write_texts(llm, plan, facts)
    await stage(emit, "write", "done", "Маршрут готов")


async def _build(trip_run: TripRun, emit: Emit) -> Plan:
    state = trip_run.state
    # What the traveller rejected, and what the current avoid list rules out.
    candidates = [
        c
        for pid, c in state.shortlist.items()
        if pid not in state.excluded
        and not any(matches_avoid(a, c.place.kind, c.place.name) for a in state.brief.avoid)
    ]

    await stage(emit, "optimize", "start", "Строю порядок обхода")
    result = optimize(candidates, state.start, state.brief, trip_run.weather)
    order: list[Candidate] = list(result.order)
    dropped = list(result.dropped)
    await stage(
        emit,
        "optimize",
        "done",
        f"Вошло мест: {len(order)}, не поместилось: {len(dropped)}",
        {"chosen": len(order), "dropped": len(dropped)},
    )

    await stage(emit, "route", "start", "Прокладываю пешеходный маршрут")
    # The bounds are what the optimizer accepted, not the absolute limits: a required
    # place may already end the walk late or make it long, and the optional stops
    # around it are not to blame for that.
    estimate = schedule(order, state.start, state.brief)
    window_end = datetime.combine(state.brief.date, state.brief.end_time)
    end_limit = max(window_end, estimate.stops[-1].depart) if estimate.stops else window_end
    walk_limit_m = max(walk_limit_km(state.brief) * 1000, estimate.total_walk_m)
    while True:
        points = [state.start] + [GeoPoint(lat=c.place.lat, lon=c.place.lon) for c in order]
        route = await walking_route(points) if order else None
        timed = schedule(order, state.start, state.brief, route.leg_meters if route else None)
        # Real streets can be longer than the optimizer's estimate; if the walk no
        # longer fits those bounds, drop the weakest optional stop and route again.
        late = timed.stops and timed.stops[-1].depart > end_limit
        too_long = timed.total_walk_m > walk_limit_m
        optional = [c for c in order if not c.required]
        if not (late or too_long) or len(order) <= 2 or not optional:
            break
        weakest = min(optional, key=lambda c: (c.score, c.place.id))
        order.remove(weakest)
        dropped.append(weakest.place.id)
    routed_out = len(dropped) - len(result.dropped)
    # No route: no geometry, and the UI joins the stops with dotted straight lines.
    geometry = route.geometry if route else None
    await stage(
        emit,
        "route",
        "done",
        f"{timed.total_walk_m / 1000:.1f} км пешком"
        + (f", убрано после маршрутизации: {routed_out}" if routed_out else "")
        + ("" if route else " (по прямой: сервис маршрутов недоступен)"),
    )

    plan = Plan(
        brief=state.brief,
        start=state.start,
        stops=timed.stops,
        total_walk_m=timed.total_walk_m,
        total_minutes=timed.total_minutes,
        geometry=geometry,
        dropped=dropped,
    )
    await stage(emit, "validate", "start", "Проверяю маршрут")
    plan.violations = validate(plan, trip_run.weather)
    n_err = sum(v.severity == "error" for v in plan.violations)
    n_warn = len(plan.violations) - n_err
    await stage(
        emit,
        "validate",
        "done",
        "Ошибок нет" if not n_err else f"Ошибок: {n_err}",
        {"errors": n_err, "warnings": n_warn},
    )
    return plan


def _apply_edit(trip_run: TripRun, edit: PlanEdit, excluded: dict[str, str]) -> None:
    """`excluded`: the stops the traveller removed, id -> name."""
    state = trip_run.state
    # A new avoid phrase applies to the whole shortlist at once, so the rebuilt route
    # cannot pick an avoided place that did not fit before.
    avoided = {
        pid: c.place.name
        for pid, c in state.shortlist.items()
        if any(matches_avoid(avoid, c.place.kind, c.place.name) for avoid in edit.add_avoid)
    }
    for pid in excluded:
        state.excluded.add(pid)
        state.shortlist.pop(pid, None)
    # Avoided places are not rejected for good: _build enforces the avoid list as it is
    # at the time, so feedback that lifts the avoid makes them eligible again.
    for pid in avoided:
        state.shortlist.pop(pid, None)
    # A place the traveller asked for and now removes is no longer asked for: research
    # must not search it again, nor the validator report it missing.
    names = [*excluded.values(), *avoided.values()]
    wanted = [w for w in state.brief.must_include if not any(names_place(w, n) for n in names)]
    # The feedback is the newest layer of the brief's precedence: «добавь музей» must
    # lift the profile's «не предлагать музеи», or the route leaves the museum out.
    brief = with_feedback(
        state.brief.model_copy(update={"must_include": wanted}),
        edit.add_interests,
        edit.remove_interests,
        edit.add_avoid,
    )
    update: dict[str, Any] = {}
    if edit.start_time:
        update["start_time"] = edit.start_time
    if edit.end_time:
        update["end_time"] = edit.end_time
    state.brief = brief.model_copy(update=update)
    if edit.rain_from:
        # The traveller's own forecast wins: mark those hours rainy, and the optimizer
        # and validator treat outdoor stops in them exactly as they treat forecast rain.
        trip_run.weather = _with_rain(trip_run.weather, state.brief, edit.rain_from)
        state.weather_summary += f"; the traveller expects rain from {edit.rain_from:%H:%M}"


def _with_rain(hours: list[HourWeather], brief: TripBrief, rain_from: time) -> list[HourWeather]:
    by_time = {h.time: h for h in hours}
    t = datetime.combine(brief.date, rain_from.replace(minute=0))
    end = datetime.combine(brief.date, brief.end_time)
    while t < end:
        hour = by_time.get(t) or HourWeather(time=t)
        by_time[t] = hour.model_copy(update={"precipitation_probability": 100})
        t += timedelta(hours=1)
    return sorted(by_time.values(), key=lambda h: h.time)


def _warning(message: str) -> AgentEvent:
    return AgentEvent(type="warning", data={"message": message})


_STOP_REASONS = {
    "finished": "агент решил, что мест достаточно",
    "max_steps": "исчерпан лимит шагов",
    "time_budget": "исчерпан лимит времени",
    "repeated_action": "агент начал повторяться",
    "no_tool_calls": "агент перестал вызывать инструменты",
    "llm_error": "LLM-провайдер не ответил, беру найденное",
}
