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
"""

import datetime as dt
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from pydantic import BaseModel, Field

from app.agent.brief import parse_brief
from app.agent.replanner import PlanEdit, interpret_feedback
from app.agent.researcher import research
from app.agent.tools import ResearchState
from app.agent.writer import write_texts
from app.config import settings
from app.llm import LLM
from app.models import (
    AgentEvent,
    Candidate,
    GeoPoint,
    HourWeather,
    Interest,
    Plan,
    TripBrief,
)
from app.planning.optimizer import optimize, schedule
from app.planning.routing import walking_route
from app.planning.validator import has_errors, validate
from app.sources import geocode, weather
from app.sources.http import SourceError

Emit = Callable[[AgentEvent], Awaitable[None]]

MAX_SESSIONS = 100
REPAIR_STEPS = 6
FEEDBACK_STEPS = 8


class PipelineError(RuntimeError):
    """A failure the user can act on (unknown city, empty result); shown as is."""


class PlanRequest(BaseModel):
    request: str = Field(..., min_length=3, max_length=2000)
    city: str | None = Field(None, max_length=100)
    # `dt.` because a field named `date` shadows the type inside the class body.
    date: dt.date | None = None
    start_time: dt.time | None = None
    end_time: dt.time | None = None
    start_point: str | None = Field(None, max_length=200)


class FeedbackRequest(BaseModel):
    feedback: str = Field("", max_length=1000)
    exclude_place_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class Session:
    id: str
    state: ResearchState
    weather: list[HourWeather]
    plan: Plan | None = None


# In-memory and bounded: a session only has to live while the user refines one walk.
SESSIONS: OrderedDict[str, Session] = OrderedDict()


async def stage(emit: Emit, name: str, status: str, label: str, detail: Any = None) -> None:
    data: dict[str, Any] = {"name": name, "status": status, "label": label}
    if detail is not None:
        data["detail"] = detail
    await emit(AgentEvent(type="stage", data=data))


async def plan_trip(req: PlanRequest, llm: LLM, emit: Emit, today: date) -> Session:
    await stage(emit, "brief", "start", "Разбираю запрос")
    overrides = req.model_dump(exclude={"request"}, exclude_none=True)
    try:
        brief = await parse_brief(llm, req.request, today, overrides)
    except ValueError as e:
        raise PipelineError("В запросе не указан город: напишите, где гуляем") from e
    await stage(emit, "brief", "done", "Запрос разобран", brief.model_dump(mode="json"))

    city = await geocode.locate_city(brief.city, brief.country)
    if city is None:
        raise PipelineError(f"Не нашёл город «{brief.city}» в OpenStreetMap")
    start = GeoPoint(lat=city.center.lat, lon=city.center.lon, label=f"центр, {city.name}")
    if brief.start_point:
        found = await geocode.geocode(brief.start_point, city=brief.city, country=brief.country)
        if found:
            start = found
        else:
            await emit(_warning(f"Не нашёл «{brief.start_point}», начинаю из центра города"))

    await stage(emit, "weather", "start", "Смотрю прогноз погоды")
    try:
        hours = await weather.hourly(start, brief.date)
    except SourceError:
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
    _remember(session)
    await _research_and_build(session, llm, emit, goal=None, max_steps=settings.agent_max_steps)
    return session


async def apply_feedback(
    session: Session, req: FeedbackRequest, llm: LLM, emit: Emit
) -> tuple[Session, str | None]:
    if session.plan is None:
        raise PipelineError("У этой сессии ещё нет маршрута")
    await stage(emit, "feedback", "start", "Разбираю правку")
    if req.feedback.strip():
        edit = await interpret_feedback(llm, session.plan, req.feedback)
    else:
        edit = PlanEdit(reply="Убираю выбранные места и перестраиваю маршрут.")
    excluded = set(edit.exclude_place_ids) | set(req.exclude_place_ids)
    _apply_edit(session, edit, excluded)
    await stage(emit, "feedback", "done", edit.reply, edit.model_dump(mode="json"))

    goal = edit.research_goal
    if goal is None and excluded:
        names = [session.state.registry[i].name for i in excluded if i in session.state.registry]
        goal = "Пользователь убрал: " + ", ".join(names) + ". Найди им замену по его интересам."
    if goal:
        await _research_and_build(session, llm, emit, goal=goal, max_steps=FEEDBACK_STEPS)
    else:
        await _build_and_write(session, llm, emit)
    return session, edit.reply


def get_session(session_id: str) -> Session | None:
    return SESSIONS.get(session_id)


async def _research_and_build(
    session: Session, llm: LLM, emit: Emit, goal: str | None, max_steps: int
) -> None:
    await stage(emit, "research", "start", "Агент ищет места")
    result = await research(llm, session.state, emit, max_steps=max_steps, goal=goal)
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
        raise PipelineError("Агент не нашёл подходящих мест. Попробуйте расширить запрос.")

    plan = await _build(session, emit)
    errors = [v for v in plan.violations if v.severity == "error"]
    if errors:
        # One bounded repair round: the validator's findings become the agent's goal.
        await stage(emit, "repair", "start", "Валидатор нашёл проблемы — агент исправляет")
        for v in plan.violations:
            if v.code == "avoid_violation" and v.place_id:
                session.state.excluded.add(v.place_id)
                session.state.shortlist.pop(v.place_id, None)
        goal = "Проверка маршрута нашла проблемы:\n" + "\n".join(f"- {v.message}" for v in errors)
        result = await research(llm, session.state, emit, max_steps=REPAIR_STEPS, goal=goal)
        plan = await _build(session, emit)
        await stage(emit, "repair", "done", "Исправление завершено", {"steps": result.steps})
    await _write(session, llm, emit, plan)


async def _build_and_write(session: Session, llm: LLM, emit: Emit) -> None:
    plan = await _build(session, emit)
    await _write(session, llm, emit, plan)


async def _write(session: Session, llm: LLM, emit: Emit, plan: Plan) -> None:
    await stage(emit, "write", "start", "Пишу описания остановок")
    session.plan = await write_texts(llm, plan)
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
        # longer fits the window, drop the weakest optional stop and route again.
        late = timed.stops and timed.stops[-1].depart > window_end
        optional = [c for c in order if not c.required]
        if not late or len(order) <= 2 or not optional:
            break
        weakest = min(optional, key=lambda c: (c.score, c.place.id))
        order.remove(weakest)
        dropped.append(weakest.place.id)
    geometry = route.geometry if route else [[p.lat, p.lon] for p in points]
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
        "Ошибок нет" if not has_errors(plan.violations) else f"Ошибок: {n_err}",
        {"errors": n_err, "warnings": n_warn},
    )
    return plan


def _apply_edit(session: Session, edit: PlanEdit, excluded: set[str]) -> None:
    state = session.state
    state.excluded |= excluded
    for pid in excluded:
        state.shortlist.pop(pid, None)
    brief = state.brief
    removed = {label.lower() for label in edit.remove_interests}
    interests = [i for i in brief.interests if i.label.lower() not in removed]
    known = {i.label.lower() for i in interests}
    interests += [
        Interest(**i.model_dump()) for i in edit.add_interests if i.label.lower() not in known
    ]
    update: dict[str, Any] = {
        "interests": interests,
        "avoid": list(dict.fromkeys([*brief.avoid, *edit.add_avoid])),
    }
    if edit.start_time:
        update["start_time"] = edit.start_time
    if edit.end_time:
        update["end_time"] = edit.end_time
    state.brief = brief.model_copy(update=update)
    if edit.rain_from:
        # The traveller's own forecast wins: mark those hours rainy, and the optimizer
        # and validator treat outdoor stops in them exactly as they treat forecast rain.
        session.weather = _with_rain(session.weather, state.brief, edit.rain_from)
        state.weather_summary += f"; путешественник ждёт дождь с {edit.rain_from:%H:%M}"


def _with_rain(hours: list[HourWeather], brief: TripBrief, rain_from: time) -> list[HourWeather]:
    by_time = {h.time: h for h in hours}
    t = datetime.combine(brief.date, rain_from.replace(minute=0))
    end = datetime.combine(brief.date, brief.end_time)
    while t < end:
        hour = by_time.get(t) or HourWeather(time=t)
        by_time[t] = hour.model_copy(update={"precipitation_probability": 100})
        t += timedelta(hours=1)
    return sorted(by_time.values(), key=lambda h: h.time)


def _remember(session: Session) -> None:
    SESSIONS[session.id] = session
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)


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
