"""Database schema: accounts, login sessions, long-term memory and saved trips.

SQLite through SQLAlchemy's async engine (aiosqlite): one file on a Docker volume is
all a single-server course project needs, there is no database server to run, and
the async driver keeps the event loop free while the agent streams. The ORM is typed
(Mapped[...]) and the URL comes from settings: another database needs only its async
driver and DATABASE_URL.

A trip keeps the whole research state of the pipeline as JSON (places found, the
shortlist, rejected places, the parsed brief) next to the plan: that is what lets a
trip be continued with feedback after a server restart, without searching again.

The schema is created with create_all at startup and there are no migrations: the
tables are new and have no production data to carry over yet. When a column changes
after real users appear, that is the moment to add Alembic.
"""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, TypeDecorator, event, make_url
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

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
    source_trip_id: Mapped[str | None] = mapped_column(ForeignKey("trips.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # pipeline TripRun.id
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    city: Mapped[str] = mapped_column(String(100))
    date: Mapped[date]
    # Copies of plan fields, so the trip list never parses the plans.
    stops: Mapped[int]
    walk_km: Mapped[float]
    request: Mapped[str] = mapped_column(Text)  # the first request, as typed
    preferences: Mapped[dict[str, Any]]  # app.profile.WalkPreferences of the first request
    state: Mapped[dict[str, Any]]  # ResearchState: what feedback continues from
    weather: Mapped[list[Any]]  # [HourWeather], with the traveller's own rain marks
    plan: Mapped[dict[str, Any]]
    history: Mapped[list[Any]]  # [store.HistoryEntry]: every run with its reply and cost
    cost_usd: Mapped[float] = mapped_column(default=0.0)  # LLM spend of all runs
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)


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
