from datetime import UTC, datetime, timedelta

import pytest

from app import store as store_module
from app.profile import Profile, WalkPreferences
from app.store import HistoryEntry


async def _user(store, email="anna@example.com"):
    return await store.create_user(email, "hash", "Анна")


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


async def test_trip_run_round_trips_through_the_database(store, make_trip_run):
    user = await _user(store)
    trip_run = make_trip_run()
    prefs = WalkPreferences(more=["nature"], pace="relaxed")
    entry = HistoryEntry(kind="plan", text="Суббота в Петербурге", cost_usd=0.012)

    await store.create_trip(user.id, trip_run, "Суббота в Петербурге", prefs, entry)
    loaded = await store.load_trip_run(user.id, trip_run.id)

    # Everything feedback continues from: the registry, the shortlist, the rejected ids,
    # the brief and the weather with its hours.
    assert loaded.state == trip_run.state
    assert loaded.state.excluded == {"osm:node/3"}
    assert loaded.weather == trip_run.weather
    assert loaded.plan == trip_run.plan


async def test_trip_summary_detail_and_history(store, make_trip_run):
    user = await _user(store)
    first = HistoryEntry(kind="plan", text="Прогулка", cost_usd=0.01)
    await store.create_trip(
        user.id, make_trip_run(title=None), "Прогулка", WalkPreferences(), first
    )

    [summary] = await store.list_trips(user.id)
    assert summary.title == "Санкт-Петербург, 24.10.2026"  # the writer gave no title
    assert (summary.city, summary.stops, summary.walk_km) == ("Санкт-Петербург", 2, 3.5)

    rebuilt = make_trip_run(title="Без музеев")
    rebuilt.plan.stops.pop()
    reply = HistoryEntry(kind="feedback", text="убери второе", reply="Убрал", cost_usd=0.005)
    await store.update_trip(user.id, rebuilt, reply)

    detail = await store.get_trip(user.id, "trip1")
    assert detail.trip.title == "Без музеев" and detail.trip.stops == 1
    assert [h.kind for h in detail.history] == ["plan", "feedback"]
    assert detail.history[1].reply == "Убрал"
    assert detail.trip.updated_at == reply.at and detail.trip.created_at == first.at
    assert detail.trip.updated_at.tzinfo is not None
    assert detail.request == "Прогулка"
    [admin_row] = await store.admin_trips()
    assert admin_row.cost_usd == pytest.approx(0.015)


async def test_trips_are_scoped_to_their_owner(store, make_trip_run):
    anna, boris = await _user(store), await _user(store, "boris@example.com")
    entry = HistoryEntry(kind="plan", text="x")
    await store.create_trip(anna.id, make_trip_run(), "x", WalkPreferences(), entry)

    assert await store.list_trips(boris.id) == []
    assert await store.get_trip(boris.id, "trip1") is None
    assert await store.load_trip_run(boris.id, "trip1") is None
    assert not await store.delete_trip(boris.id, "trip1")
    with pytest.raises(LookupError):
        await store.update_trip(boris.id, make_trip_run(), entry)
    assert await store.delete_trip(anna.id, "trip1")
    assert await store.list_trips(anna.id) == []


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


async def test_deleting_a_trip_keeps_what_was_learned_on_it(store, make_trip_run):
    user = await _user(store)
    entry = HistoryEntry(kind="plan", text="x")
    await store.create_trip(user.id, make_trip_run(), "x", WalkPreferences(), entry)
    await store.add_memory(user.id, ["Любит кофе"], source_trip_id="trip1")

    await store.delete_trip(user.id, "trip1")

    assert [f.text for f in await store.list_memory(user.id)] == ["Любит кофе"]


async def test_admin_totals_and_cost_of_today(store, make_trip_run):
    anna = await _user(store)
    await _user(store, "boris@example.com")
    yesterday = datetime.now(UTC) - timedelta(days=1)
    old = HistoryEntry(at=yesterday, kind="plan", text="x", cost_usd=0.5)
    await store.create_trip(anna.id, make_trip_run("old"), "x", WalkPreferences(), old)
    await store.update_trip(
        anna.id, make_trip_run("old"), HistoryEntry(kind="feedback", text="y", cost_usd=0.02)
    )
    new = HistoryEntry(kind="plan", text="z", cost_usd=0.01)
    await store.create_trip(anna.id, make_trip_run("new"), "z", WalkPreferences(), new)

    totals = await store.totals()

    assert totals["users"] == 2 and totals["trips"] == 2 and totals["trips_today"] == 1
    assert totals["llm_cost_today"] == pytest.approx(0.03)
    assert totals["llm_cost_total"] == pytest.approx(0.53)
    users = {u.email: u for u in await store.admin_users()}
    assert users["anna@example.com"].trips == 2
    assert users["boris@example.com"].trips == 0 and users["boris@example.com"].cost_usd == 0
