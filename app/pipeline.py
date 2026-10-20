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
A feedback run that fails or is cancelled leaves the session as it was.

Personal context: the planner panel, the request, the profile and later the feedback
are merged into the brief by code (app/personalization.py); long-term memory and the
profile's notes reach the research agent and the writer as data; after a successful
run the traveller's own words are mined for new durable facts (app/agent/memory.py),
which the pipeline returns in Session.learned for the caller to store. The pipeline
keeps nothing between runs: the caller persists sessions, so a trip survives a restart.
"""

import asyncio
import copy
import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator

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
from app.planning.validator import matches_avoid, validate
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


class PipelineError(RuntimeError):
    """A failure the user can act on (unknown city, empty result); shown as is."""


class PlanRequest(BaseModel):
    """The planner form. The text may be empty when the city is known from the form or
    from the profile: then the brief is built from the structured inputs alone."""

    request: str = Field("", max_length=2000)
    city: str | None = Field(None, max_length=100)
    # `dt.` because a field named `date` shadows the type inside the class body.
    date: dt.date | None = None
    start_time: dt.time | None = None
    end_time: dt.time | None = None
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


class FeedbackRequest(BaseModel):
    feedback: str = Field("", max_length=1000)
    exclude_place_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class Session:
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
) -> Session:
    """`memory` is the traveller's long-term memory; None means the caller keeps none
    (the eval, an anonymous run), and then nothing is learned either."""
    if not req.request and not req.city and not (profile and profile.home_city):
        raise PipelineError("Опишите прогулку или укажите город")
    await stage(emit, "brief", "start", "Разбираю запрос")
    form = req.model_dump(exclude={"request", "preferences"}, exclude_none=True)
    try:
        brief = await parse_brief(
            llm,
            req.request,
            today,
            overrides=brief_overrides(form, req.preferences),
            defaults=brief_defaults(profile),
        )
    except ValueError as e:
        raise PipelineError("В запросе не указан город: напишите, где гуляем") from e
    brief = personalize(brief, req.preferences, profile)
    await stage(emit, "brief", "done", "Запрос разобран", brief.model_dump(mode="json"))

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
        try:
            async with asyncio.timeout(GEOCODE_TIMEOUT_S):
                found = await geocode.geocode(
                    brief.start_point, city=brief.city, country=brief.country
                )
        except (SourceError, TimeoutError):
            found = None  # the city centre is a fine start: same fallback as "not found"
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
    session = Session(id=uuid.uuid4().hex[:12], state=state, weather=hours)
    await _research_and_build(
        session,
        llm,
        emit,
        goal=None,
        max_steps=settings.agent_max_steps,
        time_budget_s=RESEARCH_BUDGET_S,
        facts=traveller_facts(profile, memory or []),
    )
    session.learned = await _learn(llm, emit, req.request, profile, memory)
    return session


async def apply_feedback(
    session: Session,
    req: FeedbackRequest,
    llm: LLM,
    emit: Emit,
    profile: Profile | None = None,
    memory: list[str] | None = None,
) -> tuple[Session, str | None]:
    """`profile` and `memory` as in plan_trip; the brief was personalised when the
    trip was planned, here they only inform the agent, the writer and the learning."""
    if session.plan is None:
        raise PipelineError("У этой сессии ещё нет маршрута")
    session.learned = []
    facts = traveller_facts(profile, memory or [])
    await stage(emit, "feedback", "start", "Разбираю правку")
    stops = {stop.place.id: stop.place.name for stop in session.plan.stops}
    if req.feedback.strip():
        edit = await interpret_feedback(llm, session.plan, req.feedback)
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
    excluded = {i for i in [*edit.exclude_place_ids, *req.exclude_place_ids] if i in stops}

    saved = copy.deepcopy(session.state), list(session.weather)
    try:
        _apply_edit(session, edit, excluded)
        await stage(emit, "feedback", "done", edit.reply, edit.model_dump(mode="json"))
        if edit.research_goal:
            await _research_and_build(
                session,
                llm,
                emit,
                goal=edit.research_goal,
                max_steps=FEEDBACK_STEPS,
                time_budget_s=FEEDBACK_BUDGET_S,
                facts=facts,
            )
        else:
            plan = await _build_checked(session, llm, emit, facts)
            await _write(session, llm, emit, plan, facts)
    except BaseException:
        # Also on cancellation (the client went away): the next edit must start from
        # the plan the traveller sees, not from half of this one.
        session.state, session.weather = saved
        raise
    session.learned = await _learn(llm, emit, req.feedback, profile, memory)
    return session, edit.reply


async def _learn(
    llm: LLM, emit: Emit, text: str, profile: Profile | None, memory: list[str] | None
) -> list[str]:
    """New long-term facts from the traveller's own words of this run.

    Runs only after the run succeeded, and not for the "Убрать" button: an empty text
    has nothing to learn from.
    """
    if memory is None or not text.strip():
        return []
    learned = await extract_memory(llm, [text], known=[*memory, *profile_facts(profile)])
    if learned:
        await emit(AgentEvent(type="memory", data={"facts": learned}))
    return learned


async def _research_and_build(
    session: Session,
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
        session.state,
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
    if not session.state.shortlist:
        if result.stop_reason == "finished":
            raise PipelineError("Агент не нашёл подходящих мест. Попробуйте расширить запрос.")
        # A budget or a guard fired: the request itself may be fine.
        reason = _STOP_REASONS.get(result.stop_reason, result.stop_reason)
        raise PipelineError(f"Агент остановился, не найдя мест: {reason}. Попробуйте ещё раз.")
    await _write(session, llm, emit, await _build_checked(session, llm, emit, facts), facts)


async def _build_checked(session: Session, llm: LLM, emit: Emit, facts: Sequence[str]) -> Plan:
    """Build and validate the route; on errors, one bounded repair round.

    The validator's findings become the research agent's goal, and the stops that
    break the avoid list are rejected outright. A route with no stops is never
    returned as a result.
    """
    plan = await _build(session, emit)
    errors = [v for v in plan.violations if v.severity == "error"]
    if errors:
        await stage(emit, "repair", "start", "Валидатор нашёл проблемы — агент исправляет")
        for v in errors:
            if v.code == "avoid_violation" and v.place_id:
                session.state.excluded.add(v.place_id)
                session.state.shortlist.pop(v.place_id, None)
        # The messages stay in the traveller's language: they are data for the agent.
        goal = REPAIR_GOAL.format(problems="\n".join(f"- {v.message}" for v in errors))
        result = await research(
            llm,
            session.state,
            emit,
            max_steps=REPAIR_STEPS,
            goal=goal,
            time_budget_s=REPAIR_BUDGET_S,
            traveller_facts=facts,
        )
        plan = await _build(session, emit)
        await stage(emit, "repair", "done", "Исправление завершено", {"steps": result.steps})
    if not plan.stops:
        raise PipelineError(
            "Не удалось собрать маршрут: ни одно найденное место не подошло. "
            "Попробуйте расширить запрос."
        )
    return plan


async def _write(session: Session, llm: LLM, emit: Emit, plan: Plan, facts: Sequence[str]) -> None:
    await stage(emit, "write", "start", "Пишу описания остановок")
    session.plan = await write_texts(llm, plan, facts)
    await stage(emit, "write", "done", "Маршрут готов")


async def _build(session: Session, emit: Emit) -> Plan:
    state = session.state
    candidates = [c for pid, c in state.shortlist.items() if pid not in state.excluded]

    await stage(emit, "optimize", "start", "Строю порядок обхода")
    result = optimize(candidates, state.start, state.brief, session.weather)
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
    window_end = datetime.combine(state.brief.date, state.brief.end_time)
    while True:
        points = [state.start] + [GeoPoint(lat=c.place.lat, lon=c.place.lon) for c in order]
        route = await walking_route(points) if order else None
        timed = schedule(order, state.start, state.brief, route.leg_meters if route else None)
        # Real streets can be longer than the optimizer's estimate; if the walk no
        # longer fits the window or the distance limit, drop the weakest optional stop
        # and route again.
        late = timed.stops and timed.stops[-1].depart > window_end
        too_long = timed.total_walk_m > walk_limit_km(state.brief) * 1000
        optional = [c for c in order if not c.required]
        if not (late or too_long) or len(order) <= 2 or not optional:
            break
        weakest = min(optional, key=lambda c: (c.score, c.place.id))
        order.remove(weakest)
        dropped.append(weakest.place.id)
    # No route: no geometry, and the UI joins the stops with dotted straight lines.
    geometry = route.geometry if route else None
    await stage(
        emit,
        "route",
        "done",
        f"{timed.total_walk_m / 1000:.1f} км пешком"
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
    plan.violations = validate(plan, session.weather)
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


def _apply_edit(session: Session, edit: PlanEdit, excluded: set[str]) -> None:
    state = session.state
    # A new avoid phrase applies to the whole shortlist at once, so the rebuilt route
    # cannot pick an avoided place that did not fit before.
    avoided = [
        pid
        for pid, c in state.shortlist.items()
        if any(matches_avoid(avoid, c.place.kind, c.place.name) for avoid in edit.add_avoid)
    ]
    for pid in [*excluded, *avoided]:
        state.excluded.add(pid)
        state.shortlist.pop(pid, None)
    # The feedback is the newest layer of the brief's precedence: «добавь музей» must
    # lift the profile's «не предлагать музеи», or the validator rejects the museum.
    brief = with_feedback(state.brief, edit.add_interests, edit.remove_interests, edit.add_avoid)
    update: dict[str, Any] = {}
    if edit.start_time:
        update["start_time"] = edit.start_time
    if edit.end_time:
        update["end_time"] = edit.end_time
    state.brief = brief.model_copy(update=update)
    if edit.rain_from:
        # The traveller's own forecast wins: mark those hours rainy, and the optimizer
        # and validator treat outdoor stops in them exactly as they treat forecast rain.
        session.weather = _with_rain(session.weather, state.brief, edit.rain_from)
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
