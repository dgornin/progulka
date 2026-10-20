import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest

from app import store as store_module
from app.db import connect, create_schema
from app.profile import Profile, WalkPreferences
from app.store import HistoryEntry, Store

# The trips table once a migration has finished: a trip row is a day of a journey.
DAY_COLUMNS = {
    "id",
    "user_id",
    "journey_id",
    "day_index",
    "date",
    "stops",
    "walk_km",
    "state",
    "weather",
    "plan",
}


async def _user(store, email="anna@example.com"):
    return await store.create_user(email, "hash", "Анна")


async def _create(store, user, journey, text="Три дня в Петербурге", cost=0.01, at=None):
    entry = HistoryEntry(kind="plan", text=text, cost_usd=cost, **({"at": at} if at else {}))
    await store.create_journey(user.id, journey, text, WalkPreferences(), entry)
    return entry


async def test_email_is_unique_case_insensitively(store):
    assert await _user(store, "Anna@Example.com") is not None
    assert await _user(store, "anna@example.com ") is None


async def test_profile_defaults_and_saving_it_completes_onboarding(store):
    user = await _user(store)
    assert await store.get_profile(user.id) == Profile()

    profile = Profile(home_city="Казань", interests={"nature": 1.0}, avoid=["nightlife"])
    await store.put_profile(user.id, profile)

    assert await store.get_profile(user.id) == profile
    found, _ = await store.credentials(user.email)
    assert found.onboarded


async def test_journey_round_trips_through_the_database(store, make_journey):
    user = await _user(store)
    journey = make_journey(days=2, title="Петербург за два дня")
    journey.days[0].state.taken = {"osm:node/11"}
    await _create(store, user, journey)

    loaded = await store.load_journey(user.id, "j1")

    assert (loaded.id, loaded.title, loaded.brief, loaded.start, loaded.outline) == (
        journey.id,
        journey.title,
        journey.brief,
        journey.start,
        journey.outline,
    )
    # Everything feedback continues from, day by day: the shortlist, the rejected ids,
    # the brief, the weather with its hours and the route.
    for saved, original in zip(loaded.days, journey.days, strict=True):
        assert saved.id == original.id
        assert saved.state.shortlist == original.state.shortlist
        assert saved.state.excluded == original.state.excluded
        assert saved.state.brief == original.state.brief
        assert (saved.weather, saved.plan) == (original.weather, original.plan)
        assert saved.state.taken == set()  # the pipeline sets it before every run
    # Each day was saved with the places it refers to; loaded, the days share them again.
    registry = loaded.days[0].state.registry
    assert loaded.days[1].state.registry is registry
    assert set(registry) == set(journey.days[0].state.registry) - {"osm:node/99"}


async def test_journey_summary_detail_and_history(store, make_journey):
    user = await _user(store)
    first = await _create(store, user, make_journey(days=2, title="Петербург за два дня"))

    [summary] = await store.list_journeys(user.id)
    assert (summary.title, summary.city, summary.days, summary.stops, summary.walk_km) == (
        "Петербург за два дня",
        "Санкт-Петербург",
        2,
        4,
        7.0,
    )
    assert (summary.date_from, summary.date_to) == (date(2026, 10, 24), date(2026, 10, 25))

    edited = await store.load_journey(user.id, "j1")
    edited.days[1].plan.stops.pop()
    edited.days[0].plan.stops.clear()  # not the day of the feedback: must not be saved
    reply = HistoryEntry(kind="feedback", day=1, text="убери второе", reply="Убрал", cost_usd=0.005)
    await store.update_journey(user.id, edited, reply)

    detail = await store.get_journey(user.id, "j1")
    view = detail.journey
    assert [len(day.plan.stops) for day in view.days] == [2, 1]
    assert [(day.index, day.theme, day.day_id) for day in view.days] == [
        (0, "Тема дня 1", "j1-0"),
        (1, "Тема дня 2", "j1-1"),
    ]
    assert [(h.kind, h.day, h.reply) for h in detail.history] == [
        ("plan", None, None),
        ("feedback", 1, "Убрал"),
    ]
    assert detail.request == "Три дня в Петербурге"
    [summary] = await store.list_journeys(user.id)
    assert summary.stops == 3 and summary.updated_at == reply.at
    assert summary.created_at == first.at and summary.updated_at.tzinfo is not None
    [admin_row] = await store.admin_journeys()
    assert (admin_row.days, admin_row.owner_email) == (2, "anna@example.com")
    assert admin_row.cost_usd == pytest.approx(0.015)


async def test_journeys_are_scoped_to_their_owner(store, make_journey):
    anna, boris = await _user(store), await _user(store, "boris@example.com")
    await _create(store, anna, make_journey())

    assert await store.list_journeys(boris.id) == []
    assert await store.get_journey(boris.id, "j1") is None
    assert await store.load_journey(boris.id, "j1") is None
    assert not await store.delete_journey(boris.id, "j1")
    with pytest.raises(LookupError):
        await store.update_journey(
            boris.id, make_journey(), HistoryEntry(kind="feedback", day=0, text="x")
        )
    assert await store.delete_journey(anna.id, "j1")
    assert await store.list_journeys(anna.id) == []


async def test_deleting_a_journey_deletes_its_days_and_keeps_what_was_learned(
    store, make_journey, tmp_path
):
    user = await _user(store)
    await _create(store, user, make_journey(days=2))
    await store.add_memory(user.id, ["Любит кофе"], source_trip_id="j1-0")

    await store.delete_journey(user.id, "j1")

    assert [f.text for f in await store.list_memory(user.id)] == ["Любит кофе"]
    with sqlite3.connect(tmp_path / "data" / "test.db") as db:
        assert db.execute("SELECT count(*) FROM trips").fetchone() == (0,)


async def test_memory_dedupes_by_words_ignoring_case_and_punctuation(store):
    user = await _user(store)

    added = await store.add_memory(user.id, ["Любит  кофе", "любит кофе", " ", "Не ест мясо"])
    assert added == ["Любит кофе", "Не ест мясо"]
    assert await store.add_memory(user.id, ["ЛЮБИТ КОФЕ!", "Гуляет с собакой"]) == [
        "Гуляет с собакой"
    ]
    assert await store.memory_texts(user.id, limit=2) == ["Гуляет с собакой", "Не ест мясо"]


async def test_memory_keeps_only_the_newest_facts(store, monkeypatch):
    monkeypatch.setattr(store_module, "MAX_MEMORY_FACTS", 3)
    user = await _user(store)
    for n in range(5):
        await store.add_memory(user.id, [f"факт {n}"])

    assert [f.text for f in await store.list_memory(user.id)] == ["факт 4", "факт 3", "факт 2"]


async def test_memory_is_deleted_only_by_its_owner(store):
    anna, boris = await _user(store), await _user(store, "boris@example.com")
    await store.add_memory(anna.id, ["Любит кофе"])
    [fact] = await store.list_memory(anna.id)

    assert not await store.delete_memory(boris.id, fact.id)
    assert await store.delete_memory(anna.id, fact.id)
    assert await store.list_memory(anna.id) == []


async def test_admin_totals_and_cost_of_today(store, make_journey):
    anna = await _user(store)
    await _user(store, "boris@example.com")
    yesterday = datetime.now(UTC) - timedelta(days=1)
    await _create(store, anna, make_journey("old", days=2), cost=0.5, at=yesterday)
    feedback = HistoryEntry(kind="feedback", day=1, text="y", cost_usd=0.02)
    await store.update_journey(anna.id, make_journey("old", days=2), feedback)
    await _create(store, anna, make_journey("new", days=3), cost=0.01)

    totals = await store.totals()

    assert (totals["users"], totals["journeys"], totals["journeys_today"]) == (2, 2, 1)
    assert (totals["days"], totals["days_today"]) == (5, 3)
    assert totals["llm_cost_today"] == pytest.approx(0.03)
    assert totals["llm_cost_total"] == pytest.approx(0.53)
    users = {u.email: u for u in await store.admin_users()}
    assert (users["anna@example.com"].journeys, users["anna@example.com"].days) == (2, 5)
    assert users["anna@example.com"].cost_usd == pytest.approx(0.53)
    assert users["boris@example.com"].journeys == 0 and users["boris@example.com"].cost_usd == 0


async def test_a_trip_saved_before_journeys_becomes_a_one_day_journey(
    tmp_path, make_legacy_database
):
    path = tmp_path / "old.db"
    make_legacy_database(path)
    engine = connect(f"sqlite+aiosqlite:///{path}")
    await create_schema(engine)
    await create_schema(engine)  # every start runs it: the second does nothing
    store = Store(engine)

    [summary] = await store.list_journeys(1)
    detail = await store.get_journey(1, "trip1")
    journey = await store.load_journey(1, "trip1")
    await engine.dispose()

    assert (summary.id, summary.title, summary.days, summary.stops) == (
        "trip1",
        "Конструктивизм и кофе",
        1,
        2,
    )
    assert summary.created_at == datetime(2026, 10, 19, 10, tzinfo=UTC)
    assert [(h.kind, h.day, h.text) for h in detail.history] == [
        ("plan", None, "Суббота в Петербурге"),
        ("feedback", 0, "без соборов"),
    ]
    assert detail.preferences.more == ["street_art"]
    # The day keeps the trip's id, so the memory fact learned on it still points at it.
    [day] = journey.days
    assert day.id == "trip1" and day.state.excluded == {"osm:node/3"}
    [outline] = journey.outline
    assert (outline.date, outline.focus, outline.area) == (
        date(2026, 10, 24),
        ["архитектура"],
        None,
    )
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(trips)")}
        source = db.execute("SELECT source_trip_id FROM memory_facts").fetchone()
    assert columns == DAY_COLUMNS
    assert source == ("trip1",)


async def test_a_migrated_journey_is_deleted_with_its_day(tmp_path, make_legacy_database):
    path = tmp_path / "old.db"
    make_legacy_database(path)
    engine = connect(f"sqlite+aiosqlite:///{path}")
    await create_schema(engine)

    assert await Store(engine).delete_journey(1, "trip1")
    await engine.dispose()

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM trips").fetchone() == (0,)
        assert db.execute("SELECT source_trip_id FROM memory_facts").fetchone() == (None,)


async def test_a_migration_cut_short_finishes_at_the_next_start(tmp_path, make_legacy_database):
    path = tmp_path / "old.db"
    make_legacy_database(path)
    # SQLite commits an ALTER at once: a start stopped right after the first one.
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE trips ADD COLUMN journey_id VARCHAR(32)")
    engine = connect(f"sqlite+aiosqlite:///{path}")
    await create_schema(engine)

    detail = await Store(engine).get_journey(1, "trip1")
    await engine.dispose()

    assert [day.day_id for day in detail.journey.days] == ["trip1"]
    assert detail.request == "Суббота в Петербурге"


async def test_a_migration_cut_short_between_its_drops_finishes_at_the_next_start(
    tmp_path, make_legacy_database
):
    path = tmp_path / "old.db"
    make_legacy_database(path)
    # With no trip to copy every ALTER commits at once: a start stopped after two drops.
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM memory_facts")
        db.execute("DELETE FROM trips")
        db.execute("ALTER TABLE trips ADD COLUMN journey_id VARCHAR(32)")
        db.execute("ALTER TABLE trips ADD COLUMN day_index INTEGER NOT NULL DEFAULT 0")
        db.execute("DROP INDEX ix_trips_updated_at")
        db.execute("ALTER TABLE trips DROP COLUMN title")
        db.execute("ALTER TABLE trips DROP COLUMN city")
    engine = connect(f"sqlite+aiosqlite:///{path}")
    await create_schema(engine)
    await engine.dispose()

    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(trips)")}
    assert columns == DAY_COLUMNS
