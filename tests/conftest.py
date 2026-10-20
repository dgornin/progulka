"""Shared fixtures: no popularity lookups unless a test asks for them, and for the
persistence tests a fresh SQLite store, finished journeys, and a database file as the app
left it before journeys existed."""

import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from app.agent.tools import ResearchState
from app.db import connect, create_schema
from app.models import (
    Candidate,
    DayOutline,
    GeoPoint,
    HourWeather,
    Interest,
    Place,
    Plan,
    Stop,
    TripBrief,
)
from app.pipeline import DayRun, Journey
from app.sources import wikidata
from app.store import Store

DAY = date(2026, 10, 24)
START = GeoPoint(lat=59.93, lon=30.36, label="центр, Санкт-Петербург")
# In the registry of every journey below, but no day refers to it: saving drops it.
UNPICKED = Place(id="osm:node/99", name="Не выбрано", lat=59.9, lon=30.3, kind="cafe", source="osm")


@pytest.fixture(autouse=True)
def no_popularity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every search tool scores its places through Wikimedia (app/sources/wikidata.py).
    Tests of other sources fake only their own source's answers: here no place gets a
    score, so they stay focused and never reach the network. tests/test_wikidata.py puts
    the real lookup back on top of its own fake of the HTTP client."""

    async def unknown(places: object, timeout_s: float | None = None) -> dict[str, float]:
        return {}

    monkeypatch.setattr(wikidata, "popularity", unknown)
    monkeypatch.setattr(wikidata, "_down_until", 0.0)  # no outage left over from a test


@pytest.fixture
async def store(tmp_path) -> AsyncIterator[Store]:
    engine = connect(f"sqlite+aiosqlite:///{tmp_path / 'data' / 'test.db'}")
    await create_schema(engine)
    yield Store(engine)
    await engine.dispose()


def _brief(day: date = DAY, date_to: date | None = None) -> TripBrief:
    return TripBrief(
        city="Санкт-Петербург",
        date=day,
        date_to=date_to,
        start_time=time(11),
        end_time=time(17),
        interests=[Interest(label="архитектура", weight=1.0)],
        avoid=["музеи"],
    )


def _day_run(
    day_id: str = "trip1",
    title: str | None = "Конструктивизм и кофе",
    day: date = DAY,
    first: int = 1,
) -> DayRun:
    """One finished day: places first..first+2, the first two on the route, the third
    rejected by the traveller."""
    brief = _brief(day)
    places = [
        Place(
            id=f"osm:node/{n}",
            name=f"Место {n}",
            lat=59.93 + n / 1000,
            lon=30.36,
            kind="artwork",
            source="osm",
            opening_hours="Mo-Su 10:00-20:00",
        )
        for n in range(first, first + 3)
    ]
    shortlist = {
        p.id: Candidate(
            place=p, score=0.8, interests=["архитектура"], reason="фасад", visit_minutes=20
        )
        for p in places[:2]
    }
    state = ResearchState(
        brief=brief,
        start=START,
        weather_summary="облачно, без осадков",
        registry={p.id: p for p in places},
        shortlist=shortlist,
        excluded={places[2].id},
    )
    weather = [
        HourWeather(
            time=datetime.combine(day, time(h)), temperature_c=8.5, precipitation_probability=10
        )
        for h in range(11, 17)
    ]
    stops = [
        Stop(
            place=p,
            arrive=datetime.combine(day, time(11 + i)),
            depart=datetime.combine(day, time(11 + i, 20)),
            walk_minutes_from_prev=12,
            walk_meters_from_prev=900,
            score=0.8,
            reason="фасад",
        )
        for i, p in enumerate(places[:2])
    ]
    plan = Plan(
        brief=brief,
        start=START,
        stops=stops,
        total_walk_m=3456,
        total_minutes=200,
        geometry=[[59.93, 30.36], [59.931, 30.36]],
        title=title,
    )
    return DayRun(id=day_id, state=state, weather=weather, plan=plan)


def _journey(journey_id: str = "j1", days: int = 1, title: str = "Петербург") -> Journey:
    """What plan_journey returns: days with their own places, one shared registry."""
    runs = [
        _day_run(f"{journey_id}-{i}", day=DAY + timedelta(days=i), first=1 + 10 * i)
        for i in range(days)
    ]
    registry = {pid: p for run in runs for pid, p in run.state.registry.items()}
    registry[UNPICKED.id] = UNPICKED
    for run in runs:
        run.state.registry = registry
    outline = [
        DayOutline(
            index=i,
            date=DAY + timedelta(days=i),
            theme=f"Тема дня {i + 1}" if days > 1 else "",
            focus=["архитектура"],
        )
        for i in range(days)
    ]
    last = DAY + timedelta(days=days - 1)
    return Journey(
        id=journey_id,
        brief=_brief(DAY, last if days > 1 else None),
        start=START,
        title=title,
        outline=outline,
        days=runs,
        learned=[],
    )


@pytest.fixture
def make_journey() -> Callable[..., Journey]:
    return _journey


# The schema as create_all made it before journeys existed (only the tables it changed
# or refers to; create_all adds the rest).
LEGACY_SCHEMA = """
CREATE TABLE users (
    id INTEGER NOT NULL, email VARCHAR(254) NOT NULL, name VARCHAR(60) NOT NULL,
    password_hash VARCHAR(200) NOT NULL, is_admin BOOLEAN NOT NULL, onboarded BOOLEAN NOT NULL,
    profile JSON NOT NULL, created_at DATETIME NOT NULL, PRIMARY KEY (id), UNIQUE (email)
);
CREATE TABLE trips (
    id VARCHAR(32) NOT NULL, user_id INTEGER NOT NULL, title VARCHAR(200) NOT NULL,
    city VARCHAR(100) NOT NULL, date DATE NOT NULL, stops INTEGER NOT NULL,
    walk_km DOUBLE NOT NULL, request TEXT NOT NULL, preferences JSON NOT NULL,
    state JSON NOT NULL, weather JSON NOT NULL, "plan" JSON NOT NULL, history JSON NOT NULL,
    cost_usd DOUBLE NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL,
    PRIMARY KEY (id), FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE INDEX ix_trips_user_id ON trips (user_id);
CREATE INDEX ix_trips_updated_at ON trips (updated_at);
CREATE TABLE memory_facts (
    id INTEGER NOT NULL, user_id INTEGER NOT NULL, text VARCHAR(300) NOT NULL,
    source_trip_id VARCHAR(32), created_at DATETIME NOT NULL, PRIMARY KEY (id),
    FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY(source_trip_id) REFERENCES trips (id) ON DELETE SET NULL
);
CREATE INDEX ix_memory_facts_user_id ON memory_facts (user_id);
"""
LEGACY_HISTORY = [
    {
        "at": "2026-10-19T10:00:00Z",
        "kind": "plan",
        "text": "Суббота в Петербурге",
        "reply": None,
        "cost_usd": 0.01,
    },
    {
        "at": "2026-10-19T10:05:00Z",
        "kind": "feedback",
        "text": "без соборов",
        "reply": "Убрал",
        "cost_usd": 0.002,
    },
]


def _legacy_database(path: Path, password_hash: str = "hash") -> None:
    """A database file with one user, one trip (id "trip1") saved as the old app saved
    it, and a memory fact learned on that trip."""
    day = _day_run()
    with sqlite3.connect(path) as db:
        db.executescript(LEGACY_SCHEMA)
        db.execute(
            "INSERT INTO users VALUES (1, 'anna@example.com', 'Анна', ?, 0, 1, '{}',"
            " '2026-10-01 09:00:00.000000')",
            (password_hash,),
        )
        db.execute(
            "INSERT INTO trips VALUES ('trip1', 1, 'Конструктивизм и кофе', 'Санкт-Петербург',"
            " '2026-10-24', 2, 3.5, 'Суббота в Петербурге', ?, ?, ?, ?, ?, 0.012,"
            " '2026-10-19 10:00:00.000000', '2026-10-19 10:05:00.000000')",
            (
                json.dumps({"more": ["street_art"]}),
                TypeAdapter(ResearchState).dump_json(day.state).decode(),
                TypeAdapter(list[HourWeather]).dump_json(day.weather).decode(),
                day.plan.model_dump_json(),
                json.dumps(LEGACY_HISTORY),
            ),
        )
        db.execute(
            "INSERT INTO memory_facts VALUES (1, 1, 'Любит кофе', 'trip1',"
            " '2026-10-19 10:00:00.000000')"
        )
    db.close()


@pytest.fixture
def make_legacy_database() -> Callable[..., None]:
    return _legacy_database
