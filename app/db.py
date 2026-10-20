"""Database schema: accounts, login sessions, long-term memory, journeys and their days.

SQLite through SQLAlchemy's async engine (aiosqlite): one file on a Docker volume is
all a single-server course project needs, there is no database server to run, and
the async driver keeps the event loop free while the agent streams. The ORM is typed
(Mapped[...]) and the URL comes from settings: another database needs only its async
driver and DATABASE_URL.

A journey (1..7 days in one city) is a row in `journeys`; each of its days is a row in
`trips` that keeps the day's whole research state as JSON (places found, the shortlist,
rejected places, the day's brief) next to its route: that is what lets one day be
continued with feedback after a server restart, without searching again.

The schema is created with create_all. The one change since the first deploy (a trip,
once a whole walk, became a day of a journey) is migrated at startup by
`_journeys_from_trips` rather than with Alembic: one SQLite file and one change need a
few ALTERs and a copy, not a migration framework with its own history table, scripts
and upgrade step. The next schema change is the moment to add Alembic, with this
schema as its baseline.
"""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import (
    JSON,
    Connection,
    Date,
    DateTime,
    ForeignKey,
    RowMapping,
    String,
    Text,
    TypeDecorator,
    event,
    insert,
    inspect,
    make_url,
    text,
    update,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.agent.outline import one_day
from app.models import DayOutline, GeoPoint, TripBrief

# Width of a stored memory fact; facts the pipeline learns are far shorter (memory.py).
FACT_COLUMN_CHARS = 300


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Aware UTC datetimes in and out: SQLite stores naive ones and drops the zone."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.astimezone(UTC).replace(tzinfo=None) if value else None

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return value.replace(tzinfo=UTC) if value else None


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime, dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)  # stored lowercased
    name: Mapped[str] = mapped_column(String(60))
    password_hash: Mapped[str] = mapped_column(String(200))
    is_admin: Mapped[bool] = mapped_column(default=False)
    onboarded: Mapped[bool] = mapped_column(default=False)  # the questionnaire was saved
    profile: Mapped[dict[str, Any]] = mapped_column(default=dict)  # app.profile.Profile
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    # sha256 of the cookie token: a copy of the database does not let anyone sign in.
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime]


class MemoryFact(Base):
    __tablename__ = "memory_facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(String(FACT_COLUMN_CHARS))
    # The day it was learned on, which leads to its journey.
    source_trip_id: Mapped[str | None] = mapped_column(ForeignKey("trips.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Journey(Base):
    __tablename__ = "journeys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # pipeline Journey.id
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    # Copies of the brief, so the journey list never parses it.
    city: Mapped[str] = mapped_column(String(100))
    date_from: Mapped[date]
    date_to: Mapped[date]  # inclusive; equals date_from for one day
    brief: Mapped[dict[str, Any]]  # TripBrief of the whole journey; each day has its own
    start: Mapped[dict[str, Any]]  # GeoPoint: the hotel or the start point
    outline: Mapped[list[Any]]  # [DayOutline]: the theme and area of every day
    request: Mapped[str] = mapped_column(Text)  # the first request, as typed
    preferences: Mapped[dict[str, Any]]  # app.profile.WalkPreferences of the first request
    history: Mapped[list[Any]]  # [store.HistoryEntry]: every run with its day, reply and cost
    cost_usd: Mapped[float] = mapped_column(default=0.0)  # LLM spend of all runs
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


class Day(Base):
    """One day of a journey: its route and the research state feedback continues from."""

    __tablename__ = "trips"  # historical: a trip became a day of a journey

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # pipeline DayRun.id
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    journey_id: Mapped[str] = mapped_column(
        ForeignKey("journeys.id", ondelete="CASCADE"), index=True
    )
    day_index: Mapped[int]
    date: Mapped[date]
    # Copies of plan fields, so the journey list never parses the plans.
    stops: Mapped[int]
    walk_km: Mapped[float]
    state: Mapped[dict[str, Any]]  # ResearchState: what feedback continues from
    weather: Mapped[list[Any]]  # [HourWeather], with the traveller's own rain marks
    plan: Mapped[dict[str, Any]]


def connect(url: str) -> AsyncEngine:
    """An engine for `url`; for a SQLite file, its directory is created and FKs enforced."""
    parsed = make_url(url)
    engine = create_async_engine(url)
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

        # SQLite ignores ON DELETE unless every connection turns foreign keys on.
        @event.listens_for(engine.sync_engine, "connect")
        def _foreign_keys(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_journeys_from_trips)


# What a trip row held while a trip was a whole walk; all of it now belongs to its journey.
_MOVED_TO_JOURNEY = (
    "title",
    "city",
    "request",
    "preferences",
    "history",
    "cost_usd",
    "created_at",
    "updated_at",
)
_OUTLINE = TypeAdapter(list[DayOutline])


def _journeys_from_trips(conn: Connection) -> None:
    """Make every trip saved before journeys existed a one-day journey of its own.

    The journey takes the trip's id, so old links (/app/trips/<id>) open it. Each step
    checks the schema before it acts: the migration runs at every start, does nothing
    once done, and a start interrupted halfway finishes it at the next one.
    """
    columns = {column["name"] for column in inspect(conn).get_columns("trips")}
    if "journey_id" not in columns:
        # Nullable: SQLite adds a REFERENCES column to a table with rows only as NULL.
        conn.exec_driver_sql(
            "ALTER TABLE trips ADD COLUMN journey_id VARCHAR(32) "
            "REFERENCES journeys (id) ON DELETE CASCADE"
        )
    if "day_index" not in columns:
        conn.exec_driver_sql("ALTER TABLE trips ADD COLUMN day_index INTEGER NOT NULL DEFAULT 0")
    # All of them, not just one: a start interrupted between the DROP COLUMNs below
    # left some (each DDL commits on its own when there was no row to copy).
    if columns.issuperset(_MOVED_TO_JOURNEY):
        old = conn.execute(
            text(
                "SELECT id, user_id, title, city, date, request, preferences, state, history,"
                " cost_usd, created_at, updated_at FROM trips WHERE journey_id IS NULL"
            ).columns(
                date=Date,
                preferences=JSON,
                state=JSON,
                history=JSON,
                created_at=UTCDateTime,
                updated_at=UTCDateTime,
            )
        ).all()
        if old:
            conn.execute(insert(Journey), [_journey_of_trip(row._mapping) for row in old])
            conn.execute(update(Day).where(Day.journey_id.is_(None)).values(journey_id=Day.id))
    moved = [column for column in _MOVED_TO_JOURNEY if column in columns]
    if moved:
        # SQLite does not drop an indexed column.
        conn.exec_driver_sql("DROP INDEX IF EXISTS ix_trips_updated_at")
        for column in moved:
            conn.exec_driver_sql(f"ALTER TABLE trips DROP COLUMN {column}")
    for index in Day.__table__.indexes:
        index.create(conn, checkfirst=True)


def _journey_of_trip(trip: RowMapping) -> dict[str, Any]:
    """The journeys row of an old trip row: a single walk, outlined as the pipeline
    outlines one today."""
    brief = TripBrief.model_validate(trip["state"]["brief"])
    outline = [one_day(brief, GeoPoint.model_validate(trip["state"]["start"]))]
    return {
        "id": trip["id"],
        "user_id": trip["user_id"],
        "title": trip["title"],
        "city": trip["city"],
        "date_from": trip["date"],
        "date_to": trip["date"],
        "brief": trip["state"]["brief"],
        "start": trip["state"]["start"],
        "outline": _OUTLINE.dump_python(outline, mode="json"),
        "request": trip["request"],
        "preferences": trip["preferences"],
        # The first run planned the whole trip; every later one changed its only day.
        "history": [
            {**entry, "day": None if entry["kind"] == "plan" else 0} for entry in trip["history"]
        ],
        "cost_usd": trip["cost_usd"],
        "created_at": trip["created_at"],
        "updated_at": trip["updated_at"],
    }
