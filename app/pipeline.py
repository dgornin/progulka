"""The macro-structure: Plan-and-Execute over the days of a trip, with a fixed
pipeline and one agent loop inside every day.

    brief (LLM, structured output) -> city and start point (Nominatim)
    -> weather for every day (Open-Meteo)
    -> outline: the day plan (LLM, structured output; only for two days or more)
    -> research: one ReAct agent per day, the days in parallel
    -> per day, in order: optimize (code) -> walking route (OSRM) -> validate (code)
       -> one bounded repair round if the validator found errors
    -> texts (LLM, structured output), the days in parallel, and beside them
       long-term memory (LLM, structured output): it reads only the request

A trip of several days is Plan-and-Execute (lecture 04): the outline is the plan (a
theme, the focus interests, an area and the named places of every day,
app/agent/outline.py), the research agents are the executors, one per day, and the
replanner (app/agent/replanner.py) edits one day when the traveller gives feedback.
A one-day trip, a single walk, makes no outline call and runs exactly as a walk
always did. The forecast comes before the outline, so the planner can put the
museums on the rainy day.

Why the days are researched in parallel but optimized one by one. Research is the
slow part (LLM turns, Overpass), and there the days do not depend on each other:
they share one registry and one set of source queries (app/agent/tools.py), at most
MAX_PARALLEL_DAYS at a time to stay polite to Overpass and the LLM provider. Choosing
stops is where they do depend on each other: a place belongs to one day. So the
optimizer runs day by day over the pool of all the shortlists minus what earlier days
took, compared by real place, not by id (tools.same_place: the Wikipedia article of
a cathedral an earlier day visits is out too). A day's own candidates keep their
score and another day's get x OTHER_DAY_FACTOR: each day prefers what its own
research found, yet can use a good place another day did not need. Such a place
moves to this day's shortlist, so a later edit of the day keeps it. A place the
traveller named waits for the day the outline gave it, whichever day's research
found it.

Only the research step is an agent, because only there the next step depends on
what was found. Everything that can be computed (geocoding, time arithmetic, route
order, opening hours) is code, and the validator is the external critic the repair
round listens to.

Feedback on a day re-enters at research (when new places are needed) or at
optimize, with the stops of the other days taken and the places removed from them
kept out, and every route, first or rebuilt, goes through the same validation and
repair round. A stop of another day is out for good; a place removed from another
day is out until the traveller asks for it on this one: «Убрать» on one day and
«добавь X» on another is how a place moves between days. A failed feedback run is
not saved: the caller stores the trip only after success.

Personal context: the planner panel, the request, the profile and later the feedback
are merged into the brief by code (app/personalization.py); long-term memory and the
profile's notes reach the research agents and the writer as data; after a successful
run the traveller's own words are mined for new durable facts (app/agent/memory.py),
which the pipeline returns in Journey.learned for the caller to store. The pipeline
keeps nothing between runs: the caller persists the journey, so it survives a restart.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent.brief import parse_brief
from app.agent.memory import extract_memory
from app.agent.outline import draft_days, locate_areas, one_day, trip_dates
from app.agent.prompts import DAY_GOAL, REPAIR_GOAL, REPLACE_GOAL
from app.agent.replanner import PlanEdit, interpret_feedback
from app.agent.researcher import ResearchResult, research
from app.agent.tools import ResearchState, same_place
from app.agent.writer import write_texts
from app.config import settings
from app.llm import LLM, LLMError
from app.models import (
    MAX_DAYS,
    AgentEvent,
    Candidate,
    DayOutline,
    Emit,
    GeoPoint,
    HourWeather,
    LocalTime,
    Place,
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
# The brief, the outline and the feedback are one structured LLM call each, which the
# LLM client's retries can stretch to minutes; research and writer have budgets of their own.
LLM_STEP_TIMEOUT_S = 60
# In a shorter window the optimizer can place no stop: refused before the research.
MIN_WINDOW = timedelta(minutes=30)
# Days researched and written at once (see the module docstring).
MAX_PARALLEL_DAYS = 4
# On its day a focus interest weighs at least FOCUS_WEIGHT (a must-have), the others
# are discounted below MUST_HAVE_WEIGHT: that day's agent need not cover them.
FOCUS_WEIGHT = 0.9
OFF_FOCUS_FACTOR = 0.6
OTHER_DAY_FACTOR = 0.6


class PipelineError(RuntimeError):
    """A failure the user can act on (unknown city, empty result); shown as is."""


# Raised in two places each; the API refuses an empty form itself, before a run starts.
EMPTY_REQUEST = "Опишите путешествие или укажите город"
LLM_TIMEOUT = "LLM-провайдер не ответил вовремя, попробуйте ещё раз"
TOO_MANY_DAYS = f"Не больше {MAX_DAYS} дней в одной поездке: сократите даты"


class PlanRequest(BaseModel):
    """The planner form. The text may be empty when the city is known from the form or
    from the profile: then the brief is built from the structured inputs alone."""

    request: str = Field("", max_length=2000)
    city: str | None = Field(None, max_length=100)
    date_from: date | None = None
    date_to: date | None = Field(None, description="Last day, inclusive; None = one day")
    # The daily window: the same on every day of the trip.
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
        if self.date_to:
            if not self.date_from:
                raise ValueError("Укажите первый день поездки")
            if self.date_to < self.date_from:
                raise ValueError("Последний день поездки раньше первого")
            if (self.date_to - self.date_from).days >= MAX_DAYS:
                raise ValueError(TOO_MANY_DAYS)
        return self


def has_place(req: PlanRequest, profile: Profile | None) -> bool:
    """The request can be planned: its text, the form or the profile names a city."""
    return bool(req.request or req.city or (profile and profile.home_city))


class FeedbackRequest(BaseModel):
    feedback: str = Field("", max_length=1000)
    exclude_place_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass
class DayRun:
    """One day's working state: what a feedback run continues from. The id is the day's."""

    id: str
    state: ResearchState
    weather: list[HourWeather]
    plan: Plan | None = None


@dataclass
class Journey:
    """A journey of one or more days in one city. The id is the journey's."""

    id: str
    brief: TripBrief  # the whole journey: date..date_to; each day's brief is in its state
    start: GeoPoint  # where the traveller stays
    title: str
    outline: list[DayOutline]
    days: list[DayRun]  # one per outline day, in order
    # New long-term memory facts learned in the last run, for the caller to store.
    learned: list[str] = field(default_factory=list)


async def stage(emit: Emit, name: str, status: str, label: str, detail: Any = None) -> None:
    data: dict[str, Any] = {"name": name, "status": status, "label": label}
    if detail is not None:
        data["detail"] = detail
    await emit(AgentEvent(type="stage", data=data))


async def plan_journey(
    req: PlanRequest,
    llm: LLM,
    emit: Emit,
    today: date,
    profile: Profile | None = None,
    memory: list[str] | None = None,
) -> Journey:
    """`memory` is the traveller's long-term memory; None means the caller keeps none
    (the eval, an anonymous run), and then nothing is learned either."""
    if not has_place(req, profile):
        raise PipelineError(EMPTY_REQUEST)
    brief = await _brief(req, llm, emit, today, profile)
    dates = trip_dates(brief)

    await stage(emit, "weather", "start", "Ищу город и смотрю прогноз погоды")
    start = await _locate(brief, emit)
    forecasts = await asyncio.gather(*(_forecast(start, day, today) for day in dates))
    summaries = [
        weather.summarize(hours, brief.start_time, brief.end_time, brief.language)
        for hours in forecasts
    ]
    await stage(
        emit,
        "weather",
        "done",
        "Погода: " + "; ".join(_by_day(dates, summaries)),
        {"start": start.model_dump()},  # the map flies there; the forecasts are in the label
    )

    title, outline = await _outline(llm, emit, brief, start, summaries)
    # One registry and one set of source queries for all the days (app/agent/tools.py).
    registry: dict[str, Place] = {}
    searches: dict[str, Any] = {}
    days = [
        DayRun(
            id=_new_id(),
            state=ResearchState(
                brief=_day_brief(brief, day, len(outline)),
                start=day.start or start,
                weather_summary=summary,
                registry=registry,
                searches=searches,
            ),
            weather=hours,
        )
        for day, hours, summary in zip(outline, forecasts, summaries, strict=True)
    ]
    facts = traveller_facts(profile, memory or [])
    failures: list[LLMError] = []

    async def research_day(i: int) -> ResearchResult:
        """On a trip of several days, a day whose LLM calls failed before it shortlisted
        anything is built from the other days' places, like a day whose agent found
        nothing: one failed day must not cancel the healthy ones. A single walk has no
        other day to borrow from, and its error is reported as is."""
        day_emit = _for_day(emit, i)
        try:
            return await _research(
                days[i],
                llm,
                day_emit,
                goal=_day_goal(outline, i),
                max_steps=settings.agent_max_steps,
                time_budget_s=RESEARCH_BUDGET_S,
                facts=facts,
            )
        except LLMError as e:
            if len(days) == 1:
                raise
            failures.append(e)
            await stage(day_emit, "research", "done", "Поиск прерван: LLM-провайдер не ответил")
            await day_emit(
                _warning(f"День {i + 1}: LLM-провайдер не ответил, соберу день из мест других дней")
            )
            return ResearchResult([], "llm_error", 0)

    results = await _each_day(len(days), research_day)
    if not any(day.state.shortlist for day in days):
        # Every day failed or found nothing: the provider's error says more, if any.
        raise failures[0] if failures else _nothing_found(results[0])

    plans = []
    scheduled: set[str] = set()
    for i, day in enumerate(days):
        day.state.taken = scheduled | _named_elsewhere(days, i)
        borrowed = _borrowed(days, i, day.state.taken)
        try:
            plan = await _build_checked(day, llm, _for_day(emit, i), facts, borrowed)
        except PipelineError as e:
            if len(days) == 1:
                raise
            raise PipelineError(f"День {i + 1}: {e}") from e
        for stop in plan.stops:
            if stop.place.id in borrowed and stop.place.id not in day.state.shortlist:
                day.state.shortlist[stop.place.id] = borrowed[stop.place.id]
        scheduled |= {stop.place.id for stop in plan.stops}
        plans.append(plan)
    # The memory call needs only the request, not the texts: it runs beside the writers.
    _, learned = await asyncio.gather(
        _each_day(len(days), lambda i: _write(days[i], llm, _for_day(emit, i), plans[i], facts)),
        _learn(llm, emit, req.request, profile, memory),
    )
    return Journey(
        id=_new_id(),
        brief=brief,
        start=start,
        title=title or _single_title(days[0]),
        outline=outline,
        days=days,
        learned=learned,
    )


async def apply_day_feedback(
    journey: Journey,
    day_index: int,
    req: FeedbackRequest,
    llm: LLM,
    emit: Emit,
    profile: Profile | None = None,
    memory: list[str] | None = None,
) -> tuple[Journey, str | None]:
    """Feedback on one day; the other days stay as they are and keep their stops.

    The caller checks `day_index` (main.feedback). `profile` and `memory` as in
    plan_journey; the brief was personalised when the trip was planned, here they only
    inform the agent, the writer and the learning."""
    day = journey.days[day_index]
    assert day.plan is not None  # store and plan_journey never keep a day without a route
    # The other days' stops are taken. What the traveller removed from them may still be
    # on this day's shortlist and stays out too, unless the traveller asks for it on this
    # day (_apply_edit): then the place moves here. Recomputed every run, so the move
    # lasts while this day's brief asks for the place.
    others = [other for other in journey.days if other is not day]
    day.state.taken = {stop.place.id for other in others for stop in other.plan.stops}
    day.state.rejected_elsewhere = {pid for other in others for pid in other.state.excluded}
    facts = traveller_facts(profile, memory or [])
    reply = await _feedback(day, day.plan, req, llm, _for_day(emit, day_index), facts)
    if len(journey.days) == 1:
        journey.title = _single_title(day)  # a single walk is named after its route
    journey.learned = await _learn(llm, emit, req.feedback, profile, memory)
    return journey, reply


async def _brief(
    req: PlanRequest, llm: LLM, emit: Emit, today: date, profile: Profile | None
) -> TripBrief:
    await stage(emit, "brief", "start", "Разбираю запрос")
    form = req.model_dump(
        include={"city", "date_to", "start_time", "end_time", "start_point"}, exclude_none=True
    )
    if req.date_from:
        form["date"] = req.date_from  # the brief's first day
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
        raise PipelineError("В запросе не указан город: напишите, куда едете") from e
    except TimeoutError as e:
        raise PipelineError(LLM_TIMEOUT) from e
    brief = personalize(brief, req.preferences, profile)
    await stage(emit, "brief", "done", "Запрос разобран", brief.model_dump(mode="json"))
    if len(trip_dates(brief)) > MAX_DAYS:
        raise PipelineError(TOO_MANY_DAYS)
    window = datetime.combine(brief.date, brief.end_time) - datetime.combine(
        brief.date, brief.start_time
    )
    if window < MIN_WINDOW:
        raise PipelineError(
            f"Окно прогулки {brief.start_time:%H:%M}–{brief.end_time:%H:%M} слишком короткое: "
            f"нужно хотя бы {MIN_WINDOW.seconds // 60} минут"
        )
    return brief


async def _locate(brief: TripBrief, emit: Emit) -> GeoPoint:
    """Where the trip starts: the traveller's start point, else the city centre."""
    try:
        async with asyncio.timeout(GEOCODE_TIMEOUT_S):
            city = await geocode.locate_city(brief.city, brief.country)
    except (SourceError, TimeoutError) as e:
        raise PipelineError("Геокодер OpenStreetMap не отвечает, попробуйте ещё раз") from e
    if city is None:
        raise PipelineError(f"Не нашёл город «{brief.city}» в OpenStreetMap")
    centre = GeoPoint(lat=city.center.lat, lon=city.center.lon, label=f"центр, {city.name}")
    if not brief.start_point:
        return centre
    # Not found or not answered, the city centre is a fine start; the warning says which.
    try:
        async with asyncio.timeout(GEOCODE_TIMEOUT_S):
            found = await geocode.geocode(brief.start_point, city=brief.city, country=brief.country)
    except (SourceError, TimeoutError):
        await emit(
            _warning(
                f"Геокодер OpenStreetMap не ответил на поиск «{brief.start_point}», "
                "начинаю из центра города"
            )
        )
        return centre
    if found is None:
        await emit(_warning(f"Не нашёл «{brief.start_point}», начинаю из центра города"))
    return found or centre


async def _forecast(start: GeoPoint, day: date, today: date) -> list[HourWeather]:
    try:
        async with asyncio.timeout(WEATHER_TIMEOUT_S):
            return await weather.hourly(start, day, today)
    except (SourceError, TimeoutError):
        return []


async def _outline(
    llm: LLM, emit: Emit, brief: TripBrief, start: GeoPoint, forecasts: list[str]
) -> tuple[str, list[DayOutline]]:
    """The trip's title and days; a one-day trip gets no title here (its route names it)."""
    if brief.date_to is None:
        return "", [one_day(brief, start)]
    await stage(emit, "outline", "start", "Распределяю дни поездки")
    try:
        async with asyncio.timeout(LLM_STEP_TIMEOUT_S):
            title, days = await draft_days(llm, brief, start, forecasts)
    except TimeoutError as e:
        raise PipelineError(LLM_TIMEOUT) from e
    days = await locate_areas(days, brief, start)
    await stage(
        emit,
        "outline",
        "done",
        f"План по дням: {title}",
        {"title": title, "days": [day.model_dump(mode="json") for day in days]},
    )
    return title, days


def _day_brief(brief: TripBrief, day: DayOutline, days: int) -> TripBrief:
    """What one day's agent, optimizer and validator work from: its date, and on a trip
    of several days its focus, its named places and its area."""
    single = brief.model_copy(update={"date": day.date, "date_to": None})
    if days == 1:
        return single
    focus = {label.casefold() for label in day.focus}
    interests = [
        i.model_copy(
            update={
                "weight": max(i.weight, FOCUS_WEIGHT)
                if i.label.casefold() in focus
                else round(i.weight * OFF_FOCUS_FACTOR, 2)
            }
        )
        for i in brief.interests
    ]
    return single.model_copy(
        update={
            "interests": interests,
            "must_include": day.must_include,
            "start_point": day.area or brief.start_point,
        }
    )


def _day_goal(outline: list[DayOutline], index: int) -> str | None:
    """The research goal of a day of a multi-day trip; a single walk has none."""
    if len(outline) == 1:
        return None
    day = outline[index]
    others = "; ".join(
        f"day {other.index + 1}: {_theme(other)}" for other in outline if other is not day
    )
    return DAY_GOAL.format(
        day=index + 1,
        days=len(outline),
        date=f"{day.date:%Y-%m-%d}",
        theme=_theme(day),
        focus=", ".join(day.focus) or "the brief's interests",
        others=others,
    )


def _theme(day: DayOutline) -> str:
    theme = day.theme or "no set theme"
    return f"{theme}, around {day.area}" if day.area else theme


def _named_elsewhere(days: list[DayRun], index: int) -> set[str]:
    """Shortlisted places the traveller named for another day: they wait for that day,
    whichever day's research found them."""
    mine = days[index].state.brief.must_include
    theirs = [w for j, day in enumerate(days) if j != index for w in day.state.brief.must_include]
    return {
        pid
        for day in days
        for pid, c in day.state.shortlist.items()
        if any(names_place(w, c.place.name) for w in theirs)
        and not any(names_place(w, c.place.name) for w in mine)
    }


def _borrowed(days: list[DayRun], index: int, taken: set[str]) -> dict[str, Candidate]:
    """Other days' candidates the day may use (see the module docstring): discounted,
    except a place the traveller named for this day, which the day must keep."""
    own = days[index].state
    pool: dict[str, Candidate] = {}
    for other in days:
        for pid, candidate in other.state.shortlist.items():
            if pid in own.shortlist or pid in taken or pid in pool:
                continue
            if any(names_place(w, candidate.place.name) for w in own.brief.must_include):
                update: dict[str, Any] = {"required": True}
            else:
                update = {"score": round(candidate.score * OTHER_DAY_FACTOR, 3)}
            pool[pid] = candidate.model_copy(update=update)
    return pool


async def _each_day[T](count: int, work: Callable[[int], Awaitable[T]]) -> list[T]:
    """work(i) for every day at once, at most MAX_PARALLEL_DAYS at a time. The first
    failure cancels the other days and is raised as is, not as an ExceptionGroup: the
    caller reports a PipelineError or an LLMError to the traveller."""
    gate = asyncio.Semaphore(MAX_PARALLEL_DAYS)

    async def limited(i: int) -> T:
        async with gate:
            return await work(i)

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(limited(i)) for i in range(count)]
    except ExceptionGroup as failed:
        raise failed.exceptions[0] from None
    return [task.result() for task in tasks]


def _for_day(emit: Emit, index: int) -> Emit:
    """Tag every event of one day's work with the day: the days run interleaved."""

    async def tagged(event: AgentEvent) -> None:
        await emit(event.model_copy(update={"data": {**event.data, "day": index}}))

    return tagged


async def _feedback(
    day: DayRun,
    plan: Plan,
    req: FeedbackRequest,
    llm: LLM,
    emit: Emit,
    facts: Sequence[str],
) -> str:
    """Apply the feedback on `plan`, the day's current route, and rebuild the route;
    return the reply to the traveller."""
    await stage(emit, "feedback", "start", "Разбираю правку")
    stops = {stop.place.id: stop.place.name for stop in plan.stops}
    if req.feedback.strip():
        try:
            async with asyncio.timeout(LLM_STEP_TIMEOUT_S):
                edit = await interpret_feedback(llm, plan, req.feedback)
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

    _apply_edit(day, edit, excluded)
    await stage(emit, "feedback", "done", edit.reply, edit.model_dump(mode="json"))
    if edit.research_goal:
        result = await _research(
            day,
            llm,
            emit,
            goal=edit.research_goal,
            max_steps=FEEDBACK_STEPS,
            time_budget_s=FEEDBACK_BUDGET_S,
            facts=facts,
        )
        if not day.state.available():
            raise _nothing_found(result)
    rebuilt = await _build_checked(day, llm, emit, facts)
    await _write(day, llm, emit, rebuilt, facts)
    return edit.reply


async def _learn(
    llm: LLM, emit: Emit, text: str, profile: Profile | None, memory: list[str] | None
) -> list[str]:
    """New long-term facts from the traveller's own words of this run.

    Runs only once the routes are built (beside the writers of a new journey, after the
    whole feedback run), and not for the "Убрать" button: an empty text has nothing to
    learn from. A stage of its own: the journey is sent only after it.
    """
    if memory is None or not text.strip():
        return []
    await stage(emit, "memory", "start", "Запоминаю, что узнал о вас")
    learned = await extract_memory(llm, [text], known=[*memory, *profile_facts(profile)])
    if learned:
        await emit(AgentEvent(type="memory", data={"facts": learned}))
    await stage(emit, "memory", "done", f"Новых фактов: {len(learned)}")
    return learned


async def _research(
    day: DayRun,
    llm: LLM,
    emit: Emit,
    goal: str | None,
    max_steps: int,
    time_budget_s: float,
    facts: Sequence[str],
) -> ResearchResult:
    await stage(emit, "research", "start", "Агент ищет места")
    result = await research(
        llm,
        day.state,
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
    return result


def _nothing_found(result: ResearchResult) -> PipelineError:
    if result.stop_reason == "finished":
        return PipelineError("Агент не нашёл подходящих мест. Попробуйте расширить запрос.")
    # A budget or a guard fired: the request itself may be fine.
    reason = _STOP_REASONS.get(result.stop_reason, result.stop_reason)
    return PipelineError(f"Агент остановился, не найдя мест: {reason}. Попробуйте ещё раз.")


async def _build_checked(
    day: DayRun,
    llm: LLM,
    emit: Emit,
    facts: Sequence[str],
    borrowed: Mapping[str, Candidate] | None = None,
) -> Plan:
    """Build and validate the route; on errors, one bounded repair round.

    The validator's findings become the research agent's goal. A route with no stops
    is never returned as a result.
    """
    plan = await _build(day, emit, borrowed)
    errors = [v for v in plan.violations if v.severity == "error"]
    if errors:
        await stage(emit, "repair", "start", "Валидатор нашёл проблемы — агент исправляет")
        # The messages stay in the traveller's language: they are data for the agent.
        goal = REPAIR_GOAL.format(problems="\n".join(f"- {v.message}" for v in errors))
        try:
            result = await research(
                llm,
                day.state,
                emit,
                max_steps=REPAIR_STEPS,
                goal=goal,
                time_budget_s=REPAIR_BUDGET_S,
                traveller_facts=facts,
            )
        except LLMError:
            # Raised only for a day with no shortlist of its own, built from the other
            # days' places: it keeps the route as built, the validator's errors shown.
            await stage(emit, "repair", "done", "Исправить не удалось: LLM-провайдер не ответил")
        else:
            plan = await _build(day, emit, borrowed)
            await stage(emit, "repair", "done", "Исправление завершено", {"steps": result.steps})
    if not plan.stops:
        raise PipelineError(
            "Не удалось собрать маршрут: ни одно найденное место не подошло. "
            "Попробуйте расширить запрос."
        )
    return plan


async def _write(day: DayRun, llm: LLM, emit: Emit, plan: Plan, facts: Sequence[str]) -> None:
    await stage(emit, "write", "start", "Пишу описания остановок")
    day.plan = await write_texts(llm, plan, facts)
    await stage(emit, "write", "done", "Маршрут готов")


async def _build(day: DayRun, emit: Emit, borrowed: Mapping[str, Candidate] | None = None) -> Plan:
    """`borrowed`: other days' candidates this day may use (see _borrowed)."""
    state = day.state
    pool = {**(borrowed or {}), **state.shortlist}
    # Out: what the traveller rejected here or on another day, what other days took,
    # what the current avoid list rules out, and any other source's record of those
    # places or of a candidate already kept. Ids alone miss the Wikipedia article of an
    # OSM stop; the registry is shared, so it holds the other days' places too.
    out = state.taken | state.excluded | state.rejected_elsewhere
    blocked = [state.registry[p] for p in out if p in state.registry]
    candidates: list[Candidate] = []
    # The day's own shortlist first, so of two records of a place it keeps its own (the
    # optimizer sorts by id itself: the order matters only there).
    for pid, c in sorted(pool.items(), key=lambda item: item[0] not in state.shortlist):
        if (
            pid in out
            or any(matches_avoid(a, c.place.kind, c.place.name) for a in state.brief.avoid)
            or any(same_place(c.place, p) for p in [*blocked, *(k.place for k in candidates)])
        ):
            continue
        candidates.append(c)

    await stage(emit, "optimize", "start", "Строю порядок обхода")
    result = optimize(candidates, state.start, state.brief, day.weather)
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
    plan.violations = validate(plan, day.weather)
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


def _apply_edit(day: DayRun, edit: PlanEdit, excluded: dict[str, str]) -> None:
    """`excluded`: the stops the traveller removed, id -> name."""
    state = day.state
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
    # must not search it again, nor the validator report it missing. A place they ask
    # to add is: the validator reports it if the route still lacks it.
    names = [*excluded.values(), *avoided.values()]
    asked = [*state.brief.must_include, *edit.add_places]
    wanted = [w for w in dict.fromkeys(asked) if not any(names_place(w, n) for n in names)]
    # The feedback is the newest layer of the brief's precedence: «добавь музей» must
    # lift the profile's «не предлагать музеи», or the route leaves the museum out.
    # And «добавь X» lifts an older «Убрать» of X, on this day or another.
    unrejected = {
        pid
        for pid in state.excluded | state.rejected_elsewhere
        if pid in state.registry and any(names_place(w, state.registry[pid].name) for w in wanted)
    }
    state.excluded -= unrejected
    state.rejected_elsewhere -= unrejected
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
        day.weather = _with_rain(day.weather, state.brief, edit.rain_from)
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


def _single_title(day: DayRun) -> str:
    brief = day.state.brief
    return (day.plan and day.plan.title) or f"{brief.city}, {brief.date:%d.%m.%Y}"


def _by_day(dates: list[date], texts: list[str]) -> list[str]:
    if len(dates) == 1:
        return texts
    return [f"{day:%d.%m} {text}" for day, text in zip(dates, texts, strict=True)]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


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
