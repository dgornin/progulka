"""Shared fixtures of the persistence tests: a fresh SQLite store and a finished trip run."""

from collections.abc import AsyncIterator, Callable
from datetime import date, datetime, time

import pytest

from app.agent.tools import ResearchState
from app.db import connect, create_schema
from app.models import Candidate, GeoPoint, HourWeather, Interest, Place, Plan, Stop, TripBrief
from app.pipeline import TripRun
from app.store import Store

DAY = date(2026, 10, 24)


@pytest.fixture
async def store(tmp_path) -> AsyncIterator[Store]:
    engine = connect(f"sqlite+aiosqlite:///{tmp_path / 'data' / 'test.db'}")
    await create_schema(engine)
    yield Store(engine)
    await engine.dispose()


def _trip_run(trip_id: str = "trip1", title: str | None = "Конструктивизм и кофе") -> TripRun:
    """What the pipeline returns after a run: state with all three collections, a plan."""
    brief = TripBrief(
        city="Санкт-Петербург",
        date=DAY,
        start_time=time(11),
        end_time=time(17),
        interests=[Interest(label="архитектура", weight=1.0)],
        avoid=["музеи"],
    )
    start = GeoPoint(lat=59.93, lon=30.36, label="центр, Санкт-Петербург")
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
        for n in (1, 2, 3)
    ]
    shortlist = {
        p.id: Candidate(
            place=p, score=0.8, interests=["архитектура"], reason="фасад", visit_minutes=20
        )
        for p in places[:2]
    }
    state = ResearchState(
        brief=brief,
        start=start,
        weather_summary="облачно, без осадков",
        registry={p.id: p for p in places},
        shortlist=shortlist,
        excluded={places[2].id},
    )
    weather = [
        HourWeather(time=datetime(2026, 10, 24, h), temperature_c=8.5, precipitation_probability=10)
        for h in range(11, 17)
    ]
    stops = [
        Stop(
            place=p,
            arrive=datetime(2026, 10, 24, 11 + i),
            depart=datetime(2026, 10, 24, 11 + i, 20),
            walk_minutes_from_prev=12,
            walk_meters_from_prev=900,
            score=0.8,
            reason="фасад",
        )
        for i, p in enumerate(places[:2])
    ]
    plan = Plan(
        brief=brief,
        start=start,
        stops=stops,
        total_walk_m=3456,
        total_minutes=200,
        geometry=[[59.93, 30.36], [59.931, 30.36]],
        title=title,
    )
    return TripRun(id=trip_id, state=state, weather=weather, plan=plan)


@pytest.fixture
def make_trip_run() -> Callable[..., TripRun]:
    return _trip_run
