"""Persistence for everything the HTTP layer needs: accounts, profiles, memory, trips.

The ORM stays inside this module: callers get pydantic models (the API's shapes) and
pipeline objects, so a route handler never touches a database session. Every method
opens its own short transaction, which also makes the store usable from the
background task that keeps running an agent after the request handler has returned.

A trip is a pipeline TripRun at rest: ResearchState (a dataclass of pydantic models)
and the hourly weather go through pydantic TypeAdapters to JSON and back, so feedback
on a saved trip continues exactly where the last run stopped, restart or not.
"""

from datetime import UTC, date, datetime, time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import load_only

from app.agent.memory import fact_key
from app.agent.tools import ResearchState
from app.db import FACT_COLUMN_CHARS, AuthSession, MemoryFact, Trip, User, utcnow
from app.models import HourWeather, Plan
from app.pipeline import TripRun
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
    """One run on a trip: the first plan or a feedback round."""

    at: datetime = Field(default_factory=utcnow)
    kind: Literal["plan", "feedback"]
    text: str  # the request or the feedback, as the traveller typed it
    reply: str | None = None  # the agent's answer to feedback
    cost_usd: float = 0.0


class TripSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    city: str
    date: date
    stops: int
    walk_km: float
    created_at: datetime
    updated_at: datetime


class TripDetail(BaseModel):
    trip: TripSummary
    plan: dict[str, Any]  # Plan JSON as saved: shown as is, not re-validated
    history: list[HistoryEntry]
    request: str
    preferences: WalkPreferences


class AdminUser(BaseModel):
    id: int
    email: str
    name: str
    is_admin: bool
    created_at: datetime
    trips: int
    cost_usd: float


class AdminTrip(BaseModel):
    id: str
    title: str
    city: str
    owner_email: str
    updated_at: datetime
    cost_usd: float


_STATE = TypeAdapter(ResearchState)
_WEATHER = TypeAdapter(list[HourWeather])
_HISTORY = TypeAdapter(list[HistoryEntry])
# Lists read only these columns: the research state, plan and route geometry of a trip
# are tens to hundreds of kilobytes of JSON, and a list of trips needs none of it.
_SUMMARY_ONLY = load_only(
    Trip.title,
    Trip.city,
    Trip.date,
    Trip.stops,
    Trip.walk_km,
    Trip.cost_usd,
    Trip.created_at,
    Trip.updated_at,
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

    async def delete_memory(self, user_id: int, fact_id: int) -> bool:
        async with self._db.begin() as s:
            result = await s.execute(
                delete(MemoryFact).where(MemoryFact.id == fact_id, MemoryFact.user_id == user_id)
            )
        return result.rowcount > 0

    # --- trips ---

    async def create_trip(
        self,
        user_id: int,
        trip_run: TripRun,
        request: str,
        preferences: WalkPreferences,
        entry: HistoryEntry,
    ) -> None:
        row = Trip(
            id=trip_run.id,
            user_id=user_id,
            request=request,
            preferences=preferences.model_dump(mode="json"),
            history=[],
            cost_usd=0.0,
            created_at=entry.at,
        )
        _record_run(row, trip_run, entry)
        async with self._db.begin() as s:
            s.add(row)

    async def update_trip(self, user_id: int, trip_run: TripRun, entry: HistoryEntry) -> None:
        async with self._db.begin() as s:
            row = await self._owned_trip(s, user_id, trip_run.id)
            if row is None:
                raise LookupError(f"trip {trip_run.id} of user {user_id} is gone")
            _record_run(row, trip_run, entry)

    async def list_trips(self, user_id: int) -> list[TripSummary]:
        async with self._db() as s:
            rows = await s.scalars(
                select(Trip)
                .options(_SUMMARY_ONLY)
                .where(Trip.user_id == user_id)
                .order_by(Trip.updated_at.desc())
            )
            return [TripSummary.model_validate(row) for row in rows]

    async def get_trip(self, user_id: int, trip_id: str) -> TripDetail | None:
        async with self._db() as s:
            row = await self._owned_trip(s, user_id, trip_id)
        if row is None:
            return None
        return TripDetail(
            trip=TripSummary.model_validate(row),
            plan=row.plan,
            history=_HISTORY.validate_python(row.history),
            request=row.request,
            preferences=WalkPreferences.model_validate(row.preferences),
        )

    async def load_trip_run(self, user_id: int, trip_id: str) -> TripRun | None:
        """The state a feedback run continues from."""
        async with self._db() as s:
            row = await self._owned_trip(s, user_id, trip_id)
        if row is None:
            return None
        return TripRun(
            id=row.id,
            state=_STATE.validate_python(row.state),
            weather=_WEATHER.validate_python(row.weather),
            plan=Plan.model_validate(row.plan),
        )

    async def delete_trip(self, user_id: int, trip_id: str) -> bool:
        async with self._db.begin() as s:
            result = await s.execute(
                delete(Trip).where(Trip.id == trip_id, Trip.user_id == user_id)
            )
        return result.rowcount > 0

    @staticmethod
    async def _owned_trip(s: AsyncSession, user_id: int, trip_id: str) -> Trip | None:
        # Someone else's trip and a missing one look the same: the API answers 404 to both.
        return await s.scalar(select(Trip).where(Trip.id == trip_id, Trip.user_id == user_id))

    # --- admin ---

    async def llm_cost_since(self, since: datetime) -> float:
        """LLM spend of the saved runs since `since` (failed runs are not saved)."""
        async with self._db() as s:
            histories = await s.scalars(select(Trip.history).where(Trip.updated_at >= since))
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
        since = today_start()
        async with self._db() as s:
            users = await s.scalar(select(func.count()).select_from(User))
            trips = await s.scalar(select(func.count()).select_from(Trip))
            trips_today = await s.scalar(
                select(func.count()).select_from(Trip).where(Trip.created_at >= since)
            )
            cost_total = await s.scalar(select(func.coalesce(func.sum(Trip.cost_usd), 0.0)))
        return {
            "users": users or 0,
            "trips": trips or 0,
            "trips_today": trips_today or 0,
            "llm_cost_today": round(await self.llm_cost_since(since), 5),
            "llm_cost_total": round(cost_total or 0.0, 5),
        }

    async def admin_users(self) -> list[AdminUser]:
        per_user = (
            select(
                Trip.user_id,
                func.count().label("trips"),
                func.sum(Trip.cost_usd).label("cost_usd"),
            )
            .group_by(Trip.user_id)
            .subquery()
        )
        async with self._db() as s:
            rows = await s.execute(
                select(
                    User,
                    func.coalesce(per_user.c.trips, 0),
                    func.coalesce(per_user.c.cost_usd, 0.0),
                )
                .outerjoin(per_user, per_user.c.user_id == User.id)
                .order_by(User.id)
            )
            return [
                AdminUser(
                    id=user.id,
                    email=user.email,
                    name=user.name,
                    is_admin=user.is_admin,
                    created_at=user.created_at,
                    trips=trips,
                    cost_usd=round(cost, 5),
                )
                for user, trips, cost in rows
            ]

    async def admin_trips(self, limit: int = 50) -> list[AdminTrip]:
        async with self._db() as s:
            rows = await s.execute(
                select(Trip, User.email)
                .options(_SUMMARY_ONLY)
                .join(User, User.id == Trip.user_id)
                .order_by(Trip.updated_at.desc())
                .limit(limit)
            )
            return [
                AdminTrip(
                    id=trip.id,
                    title=trip.title,
                    city=trip.city,
                    owner_email=email,
                    updated_at=trip.updated_at,
                    cost_usd=round(trip.cost_usd, 5),
                )
                for trip, email in rows
            ]


def _record_run(row: Trip, trip_run: TripRun, entry: HistoryEntry) -> None:
    """Copy a finished run into the trip row: the new plan, its state, one history line."""
    plan = trip_run.plan
    if plan is None:
        raise ValueError("only a run that produced a plan is saved")
    brief = trip_run.state.brief
    row.title = plan.title or f"{brief.city}, {brief.date:%d.%m.%Y}"
    row.city = brief.city
    row.date = brief.date
    row.stops = len(plan.stops)
    row.walk_km = round(plan.total_walk_m / 1000, 1)
    row.state = _STATE.dump_python(trip_run.state, mode="json")
    row.weather = _WEATHER.dump_python(trip_run.weather, mode="json")
    row.plan = plan.model_dump(mode="json")
    # A new list, not append(): plain JSON columns do not track in-place changes.
    row.history = [*row.history, entry.model_dump(mode="json")]
    row.cost_usd += entry.cost_usd
    row.updated_at = entry.at
