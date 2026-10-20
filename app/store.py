"""Persistence for everything the HTTP layer needs: accounts, profiles, memory, journeys.

The ORM stays inside this module: callers get pydantic models (the API's shapes) and
pipeline objects, so a route handler never touches a database session. Every method
opens its own short transaction, which also makes the store usable from the
background task that keeps running an agent after the request handler has returned.

A journey is a pipeline Journey at rest: a row for the journey and a row per day.
Each day's ResearchState (a dataclass of pydantic models) and hourly weather go
through pydantic TypeAdapters to JSON and back, so feedback on a saved day continues
exactly where the last run stopped, restart or not. `taken` and `rejected_elsewhere`
(what the other days planned and lost) are not saved: the pipeline sets them before
every run.

In memory the days of a journey share one registry of found places, so one search
serves every day. A day is saved with only the places its state refers to (shortlist,
rejected places, its stops) rather than the whole registry once per day, and on load
the days share the union again. The price: an edit that needs a place found earlier
but never picked searches for it again, and the sources' HTTP cache answers that search.
"""

from dataclasses import replace
from datetime import UTC, date, datetime, time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.memory import fact_key
from app.agent.tools import ResearchState
from app.db import FACT_COLUMN_CHARS, AuthSession, MemoryFact, User, utcnow
from app.db import Day as DayRow
from app.db import Journey as JourneyRow
from app.models import DayOutline, GeoPoint, HourWeather, Plan, TripBrief
from app.pipeline import DayRun, Journey
from app.profile import Profile, WalkPreferences

# Old facts give way to new ones: the profile is the stable part, memory is recent.
MAX_MEMORY_FACTS = 50


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    is_admin: bool
    onboarded: bool


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    created_at: datetime


class HistoryEntry(BaseModel):
    """One run on a journey: the plan of all its days, or feedback on one day."""

    at: datetime = Field(default_factory=utcnow)
    kind: Literal["plan", "feedback"]
    day: int | None = None  # the day feedback changed; None for the plan
    text: str  # the request or the feedback, as the traveller typed it
    reply: str | None = None  # the agent's answer to feedback
    cost_usd: float = 0.0


class DayView(BaseModel):
    day_id: str
    index: int
    date: date
    theme: str
    area: str | None
    plan: Plan


class JourneyView(BaseModel):
    """A journey as the planner shows it: the route of every day, for one map."""

    id: str
    title: str
    city: str
    date_from: date
    date_to: date  # inclusive; equals date_from for one day
    start: GeoPoint
    outline: list[DayOutline]
    days: list[DayView]


class JourneySummary(BaseModel):
    id: str
    title: str
    city: str
    date_from: date
    date_to: date
    days: int
    stops: int  # of all days
    walk_km: float  # of all days
    created_at: datetime
    updated_at: datetime


class JourneyDetail(BaseModel):
    journey: JourneyView
    history: list[HistoryEntry]
    request: str
    preferences: WalkPreferences


class AdminUser(BaseModel):
    id: int
    email: str
    name: str
    is_admin: bool
    created_at: datetime
    journeys: int
    days: int  # the days of those journeys: one route each
    cost_usd: float


class AdminJourney(BaseModel):
    id: str
    title: str
    city: str
    days: int
    owner_email: str
    updated_at: datetime
    cost_usd: float


_STATE = TypeAdapter(ResearchState)
_WEATHER = TypeAdapter(list[HourWeather])
_HISTORY = TypeAdapter(list[HistoryEntry])
_OUTLINE = TypeAdapter(list[DayOutline])
# Lists read the days' copies of their plan totals: the research state, plan and
# route geometry of a day are tens to hundreds of kilobytes of JSON, and a list
# needs none of it.
_DAY_TOTALS = (
    select(
        DayRow.journey_id,
        func.count().label("days"),
        func.sum(DayRow.stops).label("stops"),
        func.round(func.sum(DayRow.walk_km), 1).label("walk_km"),
    )
    .group_by(DayRow.journey_id)
    .subquery()
)


def _email(email: str) -> str:
    return email.strip().lower()


def today_start() -> datetime:
    return datetime.combine(utcnow().date(), time(), tzinfo=UTC)


class Store:
    def __init__(self, engine: AsyncEngine):
        # Rows are read after their transaction commits: nothing may lazy-load later.
        self._db = async_sessionmaker(engine, expire_on_commit=False)

    # --- accounts and login sessions ---

    async def create_user(
        self,
        email: str,
        password_hash: str,
        name: str,
        *,
        is_admin: bool = False,
        onboarded: bool = False,
        profile: Profile | None = None,
    ) -> UserOut | None:
        """The new user, or None when the email is taken."""
        row = User(
            email=_email(email),
            name=name,
            password_hash=password_hash,
            is_admin=is_admin,
            onboarded=onboarded,
            profile=(profile or Profile()).model_dump(mode="json"),
        )
        try:
            async with self._db.begin() as s:
                s.add(row)
        except IntegrityError:
            return None
        return UserOut.model_validate(row)

    async def credentials(self, email: str) -> tuple[UserOut, str] | None:
        """The user and their password hash, for login and for seeding accounts."""
        async with self._db() as s:
            row = await s.scalar(select(User).where(User.email == _email(email)))
        return (UserOut.model_validate(row), row.password_hash) if row else None

    async def update_user(
        self, user_id: int, *, is_admin: bool, password_hash: str | None = None
    ) -> None:
        """A new password also ends the account's sessions: whoever knew the old one,
        or still holds a cookie from it, is signed out."""
        async with self._db.begin() as s:
            row = await s.get_one(User, user_id)
            row.is_admin = is_admin
            if password_hash:
                row.password_hash = password_hash
                await s.execute(delete(AuthSession).where(AuthSession.user_id == user_id))

    async def revoke_admins(self, keep: str | None) -> None:
        """Admin rights belong to the configured admin account alone: when ADMIN_EMAIL
        changes or is removed, the old account loses them at the next start."""
        query = update(User).where(User.is_admin).values(is_admin=False)
        if keep:
            query = query.where(User.email != _email(keep))
        async with self._db.begin() as s:
            await s.execute(query)

    async def create_session(self, user_id: int, token_hash: str, expires_at: datetime) -> None:
        async with self._db.begin() as s:
            # Sweeping expired sessions at each login keeps the table small without a cron.
            await s.execute(delete(AuthSession).where(AuthSession.expires_at <= utcnow()))
            s.add(AuthSession(token_hash=token_hash, user_id=user_id, expires_at=expires_at))

    async def session_user(self, token_hash: str) -> UserOut | None:
        async with self._db() as s:
            row = await s.scalar(
                select(User)
                .join(AuthSession, AuthSession.user_id == User.id)
                .where(AuthSession.token_hash == token_hash, AuthSession.expires_at > utcnow())
            )
        return UserOut.model_validate(row) if row else None

    async def delete_session(self, token_hash: str) -> None:
        async with self._db.begin() as s:
            await s.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash))

    # --- profile and long-term memory ---

    async def get_profile(self, user_id: int) -> Profile:
        async with self._db() as s:
            row = await s.get_one(User, user_id)
        return Profile.model_validate(row.profile)

    async def put_profile(self, user_id: int, profile: Profile) -> Profile:
        """Save the questionnaire; saving it once is what completes onboarding."""
        async with self._db.begin() as s:
            row = await s.get_one(User, user_id)
            row.profile = profile.model_dump(mode="json")
            row.onboarded = True
        return profile

    async def list_memory(self, user_id: int) -> list[MemoryOut]:
        async with self._db() as s:
            rows = await s.scalars(
                select(MemoryFact)
                .where(MemoryFact.user_id == user_id)
                .order_by(MemoryFact.id.desc())
            )
            return [MemoryOut.model_validate(row) for row in rows]

    async def memory_texts(self, user_id: int, limit: int) -> list[str]:
        """The newest facts first: what the agent gets as context for a run."""
        async with self._db() as s:
            rows = await s.scalars(
                select(MemoryFact.text)
                .where(MemoryFact.user_id == user_id)
                .order_by(MemoryFact.id.desc())
                .limit(limit)
            )
            return list(rows)

    async def add_memory(
        self, user_id: int, facts: list[str], source_trip_id: str | None = None
    ) -> list[str]:
        """Store the facts not known yet (by their words, as memory extraction compares
        them: case and punctuation do not make a new fact); return them.

        Beyond MAX_MEMORY_FACTS the oldest facts are dropped.
        """
        async with self._db.begin() as s:
            known = await s.scalars(select(MemoryFact.text).where(MemoryFact.user_id == user_id))
            seen = {fact_key(text) for text in known}
            added: list[str] = []
            for fact in facts:
                text = " ".join(fact.split())[:FACT_COLUMN_CHARS]
                if text and fact_key(text) not in seen:
                    seen.add(fact_key(text))
                    added.append(text)
            s.add_all(
                MemoryFact(user_id=user_id, text=text, source_trip_id=source_trip_id)
                for text in added
            )
            await s.flush()
            stale = (
                select(MemoryFact.id)
                .where(MemoryFact.user_id == user_id)
                .order_by(MemoryFact.id.desc())
                .offset(MAX_MEMORY_FACTS)
            )
            await s.execute(delete(MemoryFact).where(MemoryFact.id.in_(stale.scalar_subquery())))
        return added

    async def forget_all(self, user_id: int) -> None:
        async with self._db.begin() as s:
            await s.execute(delete(MemoryFact).where(MemoryFact.user_id == user_id))

    async def delete_memory(self, user_id: int, fact_id: int) -> bool:
        async with self._db.begin() as s:
            result = await s.execute(
                delete(MemoryFact).where(MemoryFact.id == fact_id, MemoryFact.user_id == user_id)
            )
        return result.rowcount > 0

    # --- journeys ---

    async def create_journey(
        self,
        user_id: int,
        journey: Journey,
        request: str,
        preferences: WalkPreferences,
        entry: HistoryEntry,
    ) -> None:
        row = JourneyRow(
            id=journey.id,
            user_id=user_id,
            request=request,
            preferences=preferences.model_dump(mode="json"),
            history=[],
            cost_usd=0.0,
            created_at=entry.at,
        )
        _record_journey(row, journey, entry)
        days = []
        for index, run in enumerate(journey.days):
            day = DayRow(id=run.id, user_id=user_id, journey_id=journey.id, day_index=index)
            _record_day(day, run)
            days.append(day)
        async with self._db.begin() as s:
            s.add(row)
            # Inserted first: the days refer to it, and with no relationship() between the
            # two the unit of work does not order the INSERTs by that foreign key.
            await s.flush()
            s.add_all(days)

    async def update_journey(self, user_id: int, journey: Journey, entry: HistoryEntry) -> None:
        """Save feedback on the day `entry` names: that day's row and the journey's row
        (the other days did not change)."""
        if entry.day is None:
            raise ValueError("feedback names the day it changed")
        async with self._db.begin() as s:
            row = await self._owned_journey(s, user_id, journey.id)
            day = await s.scalar(
                select(DayRow).where(DayRow.journey_id == journey.id, DayRow.day_index == entry.day)
            )
            if row is None or day is None:
                raise LookupError(f"day {entry.day} of journey {journey.id} of user {user_id}")
            _record_journey(row, journey, entry)
            _record_day(day, journey.days[entry.day])

    async def list_journeys(self, user_id: int) -> list[JourneySummary]:
        async with self._db() as s:
            rows = await s.execute(
                select(
                    JourneyRow.id,
                    JourneyRow.title,
                    JourneyRow.city,
                    JourneyRow.date_from,
                    JourneyRow.date_to,
                    _DAY_TOTALS.c.days,
                    _DAY_TOTALS.c.stops,
                    _DAY_TOTALS.c.walk_km,
                    JourneyRow.created_at,
                    JourneyRow.updated_at,
                )
                .join(_DAY_TOTALS, _DAY_TOTALS.c.journey_id == JourneyRow.id)
                .where(JourneyRow.user_id == user_id)
                .order_by(JourneyRow.updated_at.desc())
            )
            return [JourneySummary(**row._mapping) for row in rows]

    async def get_journey(self, user_id: int, journey_id: str) -> JourneyDetail | None:
        found = await self._load(user_id, journey_id)
        if found is None:
            return None
        row, days = found
        return JourneyDetail(
            journey=journey_view(_journey(row, days)),
            history=_HISTORY.validate_python(row.history),
            request=row.request,
            preferences=WalkPreferences.model_validate(row.preferences),
        )

    async def load_journey(self, user_id: int, journey_id: str) -> Journey | None:
        """The journey a feedback run continues from: every day with its research state."""
        found = await self._load(user_id, journey_id)
        return _journey(*found) if found else None

    async def delete_journey(self, user_id: int, journey_id: str) -> bool:
        """The days go with it (ON DELETE CASCADE); memory learned on them stays."""
        async with self._db.begin() as s:
            result = await s.execute(
                delete(JourneyRow).where(JourneyRow.id == journey_id, JourneyRow.user_id == user_id)
            )
        return result.rowcount > 0

    async def delete_journeys(self, user_id: int) -> None:
        """Every journey of the user, with their days."""
        async with self._db.begin() as s:
            await s.execute(delete(JourneyRow).where(JourneyRow.user_id == user_id))

    async def _load(self, user_id: int, journey_id: str) -> tuple[JourneyRow, list[DayRow]] | None:
        async with self._db() as s:
            row = await self._owned_journey(s, user_id, journey_id)
            if row is None:
                return None
            days = await s.scalars(
                select(DayRow).where(DayRow.journey_id == row.id).order_by(DayRow.day_index)
            )
            return row, list(days)

    @staticmethod
    async def _owned_journey(s: AsyncSession, user_id: int, journey_id: str) -> JourneyRow | None:
        # Someone else's journey and a missing one look the same: the API answers 404 to both.
        return await s.scalar(
            select(JourneyRow).where(JourneyRow.id == journey_id, JourneyRow.user_id == user_id)
        )

    # --- admin ---

    async def llm_cost_since(self, since: datetime) -> float:
        """LLM spend of the saved runs since `since` (failed runs are not saved)."""
        async with self._db() as s:
            histories = await s.scalars(
                select(JourneyRow.history).where(JourneyRow.updated_at >= since)
            )
            return sum(
                (
                    entry.cost_usd
                    for history in histories
                    for entry in _HISTORY.validate_python(history)
                    if entry.at >= since
                ),
                0.0,
            )

    async def totals(self) -> dict[str, int | float]:
        """Journeys, and the day routes in them, in all and of journeys started today."""
        since = today_start()
        new = func.count().filter(JourneyRow.created_at >= since)
        async with self._db() as s:
            users = await s.scalar(select(func.count()).select_from(User))
            journeys, journeys_today, cost_total = (
                await s.execute(
                    select(
                        func.count(), new, func.coalesce(func.sum(JourneyRow.cost_usd), 0.0)
                    ).select_from(JourneyRow)
                )
            ).one()
            days, days_today = (
                await s.execute(
                    select(func.count(), new)
                    .select_from(DayRow)
                    .join(JourneyRow, JourneyRow.id == DayRow.journey_id)
                )
            ).one()
        return {
            "users": users or 0,
            "journeys": journeys,
            "journeys_today": journeys_today,
            "days": days,
            "days_today": days_today,
            "llm_cost_today": round(await self.llm_cost_since(since), 5),
            "llm_cost_total": round(cost_total, 5),
        }

    async def admin_users(self) -> list[AdminUser]:
        journeys = (
            select(
                JourneyRow.user_id,
                func.count().label("journeys"),
                func.sum(JourneyRow.cost_usd).label("cost_usd"),
            )
            .group_by(JourneyRow.user_id)
            .subquery()
        )
        days = (
            select(DayRow.user_id, func.count().label("days")).group_by(DayRow.user_id).subquery()
        )
        async with self._db() as s:
            rows = await s.execute(
                select(
                    User,
                    func.coalesce(journeys.c.journeys, 0),
                    func.coalesce(days.c.days, 0),
                    func.coalesce(journeys.c.cost_usd, 0.0),
                )
                .outerjoin(journeys, journeys.c.user_id == User.id)
                .outerjoin(days, days.c.user_id == User.id)
                .order_by(User.id)
            )
            return [
                AdminUser(
                    id=user.id,
                    email=user.email,
                    name=user.name,
                    is_admin=user.is_admin,
                    created_at=user.created_at,
                    journeys=n_journeys,
                    days=n_days,
                    cost_usd=round(cost, 5),
                )
                for user, n_journeys, n_days, cost in rows
            ]

    async def admin_journeys(self, limit: int = 50) -> list[AdminJourney]:
        async with self._db() as s:
            rows = await s.execute(
                select(
                    JourneyRow.id,
                    JourneyRow.title,
                    JourneyRow.city,
                    _DAY_TOTALS.c.days,
                    User.email.label("owner_email"),
                    JourneyRow.updated_at,
                    JourneyRow.cost_usd,
                )
                .join(User, User.id == JourneyRow.user_id)
                .join(_DAY_TOTALS, _DAY_TOTALS.c.journey_id == JourneyRow.id)
                .order_by(JourneyRow.updated_at.desc())
                .limit(limit)
            )
            return [
                AdminJourney(**{**row._mapping, "cost_usd": round(row.cost_usd, 5)}) for row in rows
            ]


def journey_view(journey: Journey) -> JourneyView:
    date_from, date_to = _dates(journey.brief)
    return JourneyView(
        id=journey.id,
        title=journey.title,
        city=journey.brief.city,
        date_from=date_from,
        date_to=date_to,
        start=journey.start,
        outline=journey.outline,
        days=[
            DayView(
                day_id=run.id,
                index=day.index,
                date=day.date,
                theme=day.theme,
                area=day.area,
                plan=run.plan,
            )
            for day, run in zip(journey.outline, journey.days, strict=True)
        ],
    )


def _dates(brief: TripBrief) -> tuple[date, date]:
    return brief.date, brief.date_to or brief.date


def _record_journey(row: JourneyRow, journey: Journey, entry: HistoryEntry) -> None:
    """Copy a finished run into the journey's row: what the journey is now, one history line."""
    row.title = journey.title
    row.city = journey.brief.city
    row.date_from, row.date_to = _dates(journey.brief)
    row.brief = journey.brief.model_dump(mode="json")
    row.start = journey.start.model_dump(mode="json")
    row.outline = _OUTLINE.dump_python(journey.outline, mode="json")
    # A new list, not append(): plain JSON columns do not track in-place changes.
    row.history = [*row.history, entry.model_dump(mode="json")]
    row.cost_usd += entry.cost_usd
    row.updated_at = entry.at


def _record_day(row: DayRow, run: DayRun) -> None:
    plan = run.plan
    if plan is None:
        raise ValueError("only a day that has a route is saved")
    row.date = plan.brief.date
    row.stops = len(plan.stops)
    row.walk_km = round(plan.total_walk_m / 1000, 1)
    row.state = _saved_state(run.state, plan)
    row.weather = _WEATHER.dump_python(run.weather, mode="json")
    row.plan = plan.model_dump(mode="json")


def _saved_state(state: ResearchState, plan: Plan) -> dict[str, Any]:
    """The day's state with the part of the shared registry it refers to."""
    kept = {*state.shortlist, *state.excluded, *(stop.place.id for stop in plan.stops)}
    registry = {pid: place for pid, place in state.registry.items() if pid in kept}
    return _STATE.dump_python(replace(state, registry=registry), mode="json")


def _journey(row: JourneyRow, days: list[DayRow]) -> Journey:
    runs = [
        DayRun(
            id=day.id,
            state=_STATE.validate_python(day.state),
            weather=_WEATHER.validate_python(day.weather),
            plan=Plan.model_validate(day.plan),
        )
        for day in days
    ]
    registry = {pid: place for run in runs for pid, place in run.state.registry.items()}
    for run in runs:
        run.state.registry = registry
    return Journey(
        id=row.id,
        brief=TripBrief.model_validate(row.brief),
        start=GeoPoint.model_validate(row.start),
        title=row.title,
        outline=_OUTLINE.validate_python(row.outline),
        days=runs,
        learned=[],
    )
