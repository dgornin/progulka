"""The HTTP API end to end: the real store on a temporary SQLite file, a fake pipeline."""

import asyncio
import base64
import io
import json
import wave
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.auth import hash_password
from app.db import connect
from app.limits import BUDGET_SPENT, HOUR_S, Limits, SlidingWindow
from app.models import AgentEvent
from app.pipeline import TOO_MANY_DAYS, PipelineError
from app.store import Store

DEMO = ("demo@progulka.app", "progulka-demo")
ADMIN = ("admin@progulka.app", "admin-secret-1")
PASSWORD = "secret-123"
THREE_DAYS = {
    "request": "Три дня в Петербурге, хочу стрит-арт",
    "date_from": "2026-10-24",
    "date_to": "2026-10-26",
    "preferences": {"more": ["street_art"]},
}


class FakePipeline:
    """Stands in for plan_journey and apply_day_feedback: records its inputs, spends a cent."""

    def __init__(self, make_journey):
        self.make_journey = make_journey
        self.calls: list[dict] = []
        self.learned = ["Любит конструктивизм"]
        self.error: Exception | None = None

    async def plan_journey(self, req, llm, emit, today, profile=None, memory=None):
        self.calls.append({"req": req, "profile": profile, "memory": memory})
        await emit(AgentEvent(type="stage", data={"name": "brief", "status": "start"}))
        llm.usage.cost_usd += 0.01
        if self.error:
            raise self.error
        days = (req.date_to - req.date_from).days + 1 if req.date_to else 1
        journey = self.make_journey(f"j{len(self.calls)}", days=days)
        journey.learned = list(self.learned)
        return journey

    async def apply_day_feedback(
        self, journey, day_index, req, llm, emit, profile=None, memory=None
    ):
        self.calls.append({"journey": journey, "day": day_index, "req": req, "memory": memory})
        stage = {"name": "feedback", "status": "start", "day": day_index}
        await emit(AgentEvent(type="stage", data=stage))
        llm.usage.cost_usd += 0.002
        plan = journey.days[day_index].plan
        plan.stops = [s for s in plan.stops if s.place.id not in req.exclude_place_ids]
        plan.title = "Без лишнего"
        if self.error:
            raise self.error  # after changing the loaded state, as a real failure can
        journey.learned = ["Не любит толпы"]
        return journey, "Убрал"


def _settings(tmp_path, **changes):
    values = {
        "api_key": "test-key",
        "model_name": "test/model",
        "llm_replay": False,
        "today": None,
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        "cookie_secure": False,
        "demo_email": DEMO[0],
        "demo_password": DEMO[1],
        "admin_email": ADMIN[0],
        "admin_password": ADMIN[1],
    }
    return replace(main.settings, **{**values, **changes})


@pytest.fixture
def pipeline(make_journey, monkeypatch):
    fake = FakePipeline(make_journey)
    monkeypatch.setattr(main, "plan_journey", fake.plan_journey)
    monkeypatch.setattr(main, "apply_day_feedback", fake.apply_day_feedback)
    return fake


@pytest.fixture
def speech(monkeypatch):
    """Stands in for the speech model: records what it was sent, spends a tenth of a cent."""
    received: list[str] = []

    async def transcribe(llm, audio):
        received.append(audio)
        return "Три дня в Казани", 0.001

    monkeypatch.setattr(main, "transcribe", transcribe)
    return received


def _fresh_throttles(monkeypatch, registrations=None, logins=None):
    window = main.AUTH_WINDOW_S
    monkeypatch.setattr(main, "login_failures", SlidingWindow(main.LOGIN_FAILURES, window))
    monkeypatch.setattr(main, "logins", SlidingWindow(logins or main.LOGINS, window))
    monkeypatch.setattr(
        main, "registrations", SlidingWindow(registrations or main.REGISTRATIONS, window)
    )
    monkeypatch.setattr(main, "transcriptions", SlidingWindow(main.TRANSCRIPTIONS_PER_HOUR, HOUR_S))


@pytest.fixture
def client(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "limits", Limits(0, 0, 0))
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "busy_journeys", set())
    with TestClient(main.app) as test_client:
        yield test_client


def _register(client, email="anna@example.com", name="Анна"):
    client.cookies.clear()
    response = client.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD, "name": name}
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]


def _login(client, email, password):
    client.cookies.clear()
    return client.post("/api/auth/login", json={"email": email, "password": password})


def _events(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def _journey_data(response):
    [event] = [e for e in _events(response) if e["type"] == "journey"]
    return event["data"]


def _feedback(client, journey_id="j1", day=0, **body):
    return client.post(f"/api/journeys/{journey_id}/days/{day}/feedback", json=body)


async def _remember(database_url, email, facts):
    """Store facts for an account straight in its database, between two app starts."""
    engine = connect(database_url)
    try:
        store = Store(engine)
        user, _ = await store.credentials(email)
        await store.add_memory(user.id, facts)
    finally:
        await engine.dispose()


def _wav(seconds=1.0, rate=16_000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(bytes(2 * channels * int(seconds * rate)))
    return base64.b64encode(buffer.getvalue()).decode()


def _overrun_wav():
    """A WAV whose format chunk claims a megabyte the file does not have."""
    data = bytearray(base64.b64decode(_wav()))
    data[16:20] = (1 << 20).to_bytes(4, "little")
    return base64.b64encode(data).decode()


def test_traveller_plans_continues_and_deletes_a_journey(client, pipeline):
    user = _register(client)
    assert user["name"] == "Анна" and not user["onboarded"] and not user["is_admin"]
    assert client.get("/api/me").json()["user"]["email"] == "anna@example.com"

    profile = {"home_city": "Казань", "interests": {"nature": 1.0, "coffee": 0.6}}
    assert client.put("/api/profile", json=profile).status_code == 200
    assert client.get("/api/me").json()["user"]["onboarded"]
    assert client.get("/api/profile").json()["interests"] == {"nature": 1.0, "coffee": 0.6}

    events = _events(client.post("/api/plan", json=THREE_DAYS))
    assert [e["type"] for e in events] == ["stage", "journey", "done"]
    data = events[1]["data"]
    assert (data["journey_id"], data["reply"], data["learned"]) == (
        "j1",
        None,
        ["Любит конструктивизм"],
    )
    assert data["usage"]["cost_usd"] == 0.01
    view = data["journey"]
    assert (view["id"], view["date_from"], view["date_to"]) == ("j1", "2026-10-24", "2026-10-26")
    assert [(d["index"], d["day_id"], d["date"], d["theme"]) for d in view["days"]] == [
        (0, "j1-0", "2026-10-24", "Тема дня 1"),
        (1, "j1-1", "2026-10-25", "Тема дня 2"),
        (2, "j1-2", "2026-10-26", "Тема дня 3"),
    ]
    assert [len(d["plan"]["stops"]) for d in view["days"]] == [2, 2, 2]
    assert len(view["outline"]) == 3 and view["start"]["lat"] == 59.93
    assert pipeline.calls[0]["profile"].home_city == "Казань"
    assert pipeline.calls[0]["memory"] == []

    [summary] = client.get("/api/journeys").json()
    assert (summary["id"], summary["title"], summary["days"], summary["stops"]) == (
        "j1",
        "Петербург",
        3,
        6,
    )
    assert [m["text"] for m in client.get("/api/memory").json()] == ["Любит конструктивизм"]

    events = _events(_feedback(client, day=1, exclude_place_ids=["osm:node/11"]))
    # The day's events say which day they are about; the final event, the journey.
    assert events[0]["data"]["day"] == 1
    data = _journey_data(_feedback(client, day=2, feedback="без соборов"))
    assert (data["journey_id"], data["reply"], data["learned"]) == ("j1", "Убрал", [])
    assert [len(d["plan"]["stops"]) for d in data["journey"]["days"]] == [2, 1, 2]
    # Feedback continued the saved journey: every day with its research state, the
    # days sharing one registry again, the memory as context.
    first_feedback = pipeline.calls[1]
    assert first_feedback["day"] == 1 and len(first_feedback["journey"].days) == 3
    assert first_feedback["journey"].days[1].state.excluded == {"osm:node/13"}
    assert "osm:node/1" in first_feedback["journey"].days[1].state.registry
    assert first_feedback["memory"] == ["Любит конструктивизм"]

    detail = client.get("/api/journeys/j1").json()
    assert detail["journey"]["days"][1]["plan"]["title"] == "Без лишнего"
    assert [(h["kind"], h["day"], h["text"], h["reply"]) for h in detail["history"]] == [
        ("plan", None, "Три дня в Петербурге, хочу стрит-арт", None),
        ("feedback", 1, "Убрать: Место 11", "Убрал"),
        ("feedback", 2, "без соборов", "Убрал"),
    ]
    assert [h["cost_usd"] for h in detail["history"]] == [0.01, 0.002, 0.002]
    assert detail["preferences"]["more"] == ["street_art"]
    assert detail["request"] == THREE_DAYS["request"]

    assert client.delete("/api/journeys/j1").status_code == 204
    assert client.get("/api/journeys").json() == []
    assert client.get("/api/journeys/j1").status_code == 404
    assert _feedback(client, feedback="ещё").status_code == 404


def test_known_facts_are_not_reported_as_learned_again(client):
    _register(client)
    _journey_data(client.post("/api/plan", json={"request": "Прогулка по Петербургу"}))

    again = _journey_data(client.post("/api/plan", json={"request": "Ещё одна"}))

    assert again["learned"] == []
    assert len(client.get("/api/memory").json()) == 1


def test_a_failed_run_saves_nothing(client, pipeline):
    _register(client)
    pipeline.error = PipelineError("Не нашёл город «Атлантида» в OpenStreetMap")

    events = _events(client.post("/api/plan", json={"request": "Атлантида"}))

    assert [e["type"] for e in events] == ["stage", "error", "done"]
    assert events[1]["data"]["message"].startswith("Не нашёл город")
    assert client.get("/api/journeys").json() == []


def test_a_failed_feedback_leaves_the_saved_journey_as_it_was(client, pipeline):
    _register(client)
    _journey_data(client.post("/api/plan", json=THREE_DAYS))
    saved = client.get("/api/journeys/j1").json()
    pipeline.error = PipelineError("Не удалось собрать маршрут")

    events = _events(_feedback(client, day=1, exclude_place_ids=["osm:node/11"]))

    assert [e["type"] for e in events] == ["stage", "error", "done"]
    assert client.get("/api/journeys/j1").json() == saved


def test_someone_elses_journey_and_memory_look_missing(client):
    _register(client)
    _journey_data(client.post("/api/plan", json={"request": "Прогулка"}))
    [fact] = client.get("/api/memory").json()

    _register(client, "boris@example.com", "Борис")

    assert client.get("/api/journeys").json() == []
    assert client.get("/api/journeys/j1").status_code == 404
    assert client.delete("/api/journeys/j1").status_code == 404
    assert _feedback(client, feedback="x").status_code == 404
    assert client.delete(f"/api/memory/{fact['id']}").status_code == 404
    assert _login(client, "anna@example.com", PASSWORD).status_code == 200
    assert len(client.get("/api/journeys").json()) == 1


@pytest.mark.parametrize("day", [3, -1])
def test_feedback_on_a_day_the_journey_does_not_have(client, pipeline, day):
    _register(client)
    _journey_data(client.post("/api/plan", json=THREE_DAYS))

    response = _feedback(client, day=day, feedback="без соборов")

    assert response.status_code == 404
    assert response.json() == {"detail": main.DAY_NOT_FOUND}
    assert len(pipeline.calls) == 1 and main.busy_journeys == set()


def test_a_journey_being_rebuilt_refuses_feedback_and_delete(client):
    _register(client)
    _journey_data(client.post("/api/plan", json=THREE_DAYS))
    main.busy_journeys.add("j1")

    # Any day: the run on day 0 must not lose the places it takes to day 2.
    feedback = _feedback(client, day=2, feedback="без соборов")

    assert feedback.status_code == 409
    assert feedback.json() == {"detail": "Путешествие уже перестраивается"}
    assert client.delete("/api/journeys/j1").status_code == 409


def test_removing_a_stop_that_is_already_gone_uses_no_run(client, pipeline, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=2, daily_budget_usd=0, max_concurrent=0))
    _register(client)
    _journey_data(client.post("/api/plan", json=THREE_DAYS))

    # As from a second tab, where the stop was removed by an earlier run; or a stop of
    # another day.
    for stale in ("osm:node/9", "osm:node/1"):
        response = _feedback(client, day=1, exclude_place_ids=[stale])
        assert response.status_code == 409
        assert response.json() == {"detail": "Этого места уже нет в маршруте"}

    assert len(pipeline.calls) == 1 and main.busy_journeys == set()
    _journey_data(_feedback(client, day=1, feedback="без соборов"))


def test_runs_are_rate_limited_per_address(client, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=1, daily_budget_usd=0, max_concurrent=0))
    _register(client)
    _journey_data(client.post("/api/plan", json=THREE_DAYS))  # three days, one run

    refused = client.post("/api/plan", json={"request": "Ещё"})
    feedback = _feedback(client, feedback="без соборов")

    assert refused.status_code == 429 and "в час" in refused.json()["detail"]
    assert feedback.status_code == 429
    assert main.busy_journeys == set()  # a refused feedback does not leave the journey locked


def test_the_admin_is_never_rate_limited(client, speech, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=1, daily_budget_usd=0, max_concurrent=1))
    monkeypatch.setattr(main, "transcriptions", SlidingWindow(1, HOUR_S))
    _register(client)
    _journey_data(client.post("/api/plan", json={"request": "Прогулка"}))
    assert client.post("/api/transcribe", json={"audio": _wav()}).status_code == 200

    _login(client, *ADMIN)  # from the same address, after it used up its runs

    for _ in range(2):
        _journey_data(client.post("/api/plan", json={"request": "Прогулка"}))
        assert client.post("/api/transcribe", json={"audio": _wav()}).status_code == 200
    assert main.limits.running == 0


def test_an_empty_form_is_refused_before_it_uses_a_run(client, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=1, daily_budget_usd=0, max_concurrent=0))
    _register(client)  # no home city in a fresh profile

    empty = client.post("/api/plan", json={"request": "  "})
    _journey_data(client.post("/api/plan", json={"request": "", "city": "Казань"}))

    assert empty.status_code == 422 and "укажите город" in empty.json()["detail"]


@pytest.mark.parametrize(
    ("form", "detail"),
    [
        ({"start_time": "12:00", "end_time": "11:00"}, "Конец прогулки должен быть позже начала"),
        ({"date_from": "2026-10-24", "date_to": "2026-11-01"}, TOO_MANY_DAYS),
    ],
)
def test_a_wrong_form_is_one_russian_sentence(client, form, detail):
    _register(client)

    response = client.post("/api/plan", json={"city": "Казань", **form})

    assert response.status_code == 422
    assert response.json()["detail"] == detail


def test_an_old_trip_opens_as_a_journey_after_the_upgrade(
    tmp_path, monkeypatch, pipeline, make_legacy_database
):
    make_legacy_database(tmp_path / "api.db", asyncio.run(hash_password(PASSWORD)))
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "busy_journeys", set())
    with TestClient(main.app) as client:
        assert _login(client, "anna@example.com", PASSWORD).status_code == 200

        assert client.get("/app/trips/trip1").status_code == 200  # the old link
        detail = client.get("/api/journeys/trip1").json()
        assert [d["day_id"] for d in detail["journey"]["days"]] == ["trip1"]
        assert [h["day"] for h in detail["history"]] == [None, 0]
        data = _journey_data(_feedback(client, "trip1", exclude_place_ids=["osm:node/1"]))
        assert len(data["journey"]["days"][0]["plan"]["stops"]) == 1
        assert [m["text"] for m in client.get("/api/memory").json()] == [
            "Не любит толпы",
            "Любит кофе",
        ]


def test_voice_is_transcribed_and_its_cost_counts_toward_the_budget(client, speech, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(0, daily_budget_usd=0.001, max_concurrent=0))
    _register(client)
    audio = _wav(seconds=60)  # the longest the recorder sends

    response = client.post("/api/transcribe", json={"audio": audio})

    assert response.status_code == 200
    assert response.json() == {"text": "Три дня в Казани"}
    assert speech == [audio]
    for refused in (
        client.post("/api/transcribe", json={"audio": audio}),
        client.post("/api/plan", json={"request": "Три дня в Казани"}),
    ):
        assert refused.status_code == 429
        assert refused.json() == {"detail": BUDGET_SPENT}


def test_transcriptions_have_a_window_per_address(client, speech, monkeypatch):
    monkeypatch.setattr(main, "transcriptions", SlidingWindow(1, HOUR_S))
    _register(client)
    assert client.post("/api/transcribe", json={"audio": _wav()}).status_code == 200

    refused = client.post("/api/transcribe", json={"audio": _wav()})

    assert refused.status_code == 429 and "голосовых вводов в час" in refused.json()["detail"]
    assert client.post("/api/plan", json={"request": "Прогулка"}).status_code == 200


@pytest.mark.parametrize(
    ("audio", "status", "detail"),
    [
        (_wav(seconds=62), 413, "Запись длиннее минуты"),
        ("A" * 3_400_000, 413, "Запись длиннее минуты"),
        ("это не base64", 422, "Запись не похожа на WAV"),
        (base64.b64encode(b"RIFF....OggS").decode(), 422, "Запись не похожа на WAV"),
        (_overrun_wav(), 422, "Запись не похожа на WAV"),
        (_wav(rate=44_100), 422, "Нужна запись WAV 16 кГц, моно, 16 бит"),
        (_wav(channels=2), 422, "Нужна запись WAV 16 кГц, моно, 16 бит"),
        (_wav(seconds=0), 422, "Запись пустая"),
    ],
)
def test_a_recording_the_recorder_never_sends_costs_nothing(client, speech, audio, status, detail):
    _register(client)

    response = client.post("/api/transcribe", json={"audio": audio})

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert speech == []


def test_a_failed_transcription_says_what_happened(client, monkeypatch):
    async def transcribe(llm, audio):
        raise PipelineError("Распознавание речи не ответило вовремя, попробуйте ещё раз")

    monkeypatch.setattr(main, "transcribe", transcribe)
    _register(client)

    response = client.post("/api/transcribe", json={"audio": _wav()})

    assert response.status_code == 502
    assert response.json()["detail"].startswith("Распознавание речи не ответило")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/me"),
        ("get", "/api/profile"),
        ("get", "/api/memory"),
        ("get", "/api/journeys"),
        ("get", "/api/journeys/j1"),
        ("delete", "/api/journeys/j1"),
        ("post", "/api/plan"),
        ("post", "/api/journeys/j1/days/0/feedback"),
        ("post", "/api/transcribe"),
    ],
)
def test_signed_out_requests_are_refused(client, method, path):
    response = client.request(method, path, json={"request": "Прогулка", "audio": _wav()})
    assert response.status_code == 401
    assert response.json() == {"detail": "Нужно войти"}


def test_logout_ends_the_session(client):
    _register(client)
    token = client.cookies.get("progulka_session")

    assert client.post("/api/auth/logout").status_code == 204

    assert client.get("/api/me").status_code == 401
    client.cookies.set("progulka_session", token)  # a copied cookie is revoked too
    assert client.get("/api/me").status_code == 401


def test_wrong_password_and_login_throttle(client):
    _register(client)
    for _ in range(main.LOGIN_FAILURES):  # the right password is not a failure
        assert _login(client, "anna@example.com", PASSWORD).status_code == 200

    wrong = _login(client, "anna@example.com", "not-my-password")
    assert wrong.status_code == 401
    assert wrong.json() == {"detail": "Неверная почта или пароль"}
    for _ in range(main.LOGIN_FAILURES - 1):
        assert _login(client, "anna@example.com", "not-my-password").status_code == 401

    assert _login(client, "anna@example.com", PASSWORD).status_code == 429


def test_successful_logins_have_a_cap_of_their_own(client, monkeypatch):
    _fresh_throttles(monkeypatch, logins=2)
    for _ in range(2):
        assert _login(client, *DEMO).status_code == 200

    assert _login(client, *DEMO).status_code == 429


def test_a_wave_of_sign_ups_does_not_lock_out_logins(client, monkeypatch):
    _fresh_throttles(monkeypatch, registrations=2)
    _register(client, "a@example.com")
    _register(client, "b@example.com")
    client.cookies.clear()

    third = client.post(
        "/api/auth/register", json={"email": "c@example.com", "password": PASSWORD, "name": "В"}
    )

    assert third.status_code == 429
    assert _login(client, *DEMO).status_code == 200  # the same address, the demo account


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"email": "anna", "password": PASSWORD, "name": "Анна"}, "Укажите настоящую почту"),
        ({"email": "a@b.ru", "password": "short", "name": "Анна"}, "не короче 8 символов"),
        ({"email": "a@b.ru", "password": PASSWORD, "name": "  "}, "Укажите имя"),
        ({"email": "a@b.ru", "password": PASSWORD}, "Проверьте поле «name»"),
    ],
)
def test_registration_errors_are_one_russian_sentence(client, body, detail):
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_email_can_be_registered_once(client):
    _register(client, "Anna@Example.com")
    client.cookies.clear()

    again = client.post(
        "/api/auth/register",
        json={"email": "anna@example.com", "password": PASSWORD, "name": "Анна"},
    )

    assert again.status_code == 409


def test_profile_rejects_unknown_interests(client):
    _register(client)
    response = client.put("/api/profile", json={"interests": {"skydiving": 1.0}})
    assert response.status_code == 422
    assert "skydiving" in response.json()["detail"]


def test_config_shows_the_demo_account_and_never_the_admin(client):
    response = client.get("/api/config")

    assert response.json() == {"demo": {"email": DEMO[0], "password": DEMO[1]}}
    assert ADMIN[0] not in response.text and ADMIN[1] not in response.text


def test_demo_account_is_ready_to_use(client):
    user = _login(client, *DEMO).json()["user"]

    assert user["onboarded"] and not user["is_admin"]
    profile = client.get("/api/profile").json()
    assert profile["home_city"] == "Санкт-Петербург"
    assert profile["interests"]["architecture"] == 1.0


def test_admin_pages_are_for_the_admin_only(client):
    _register(client)
    two_days = {"request": "Выходные", "date_from": "2026-10-24", "date_to": "2026-10-25"}
    _journey_data(client.post("/api/plan", json=two_days))
    for path in ("/api/admin/stats", "/api/admin/users", "/api/admin/trips"):
        assert client.get(path).status_code == 403

    assert _login(client, *ADMIN).json()["user"]["is_admin"]

    stats = client.get("/api/admin/stats").json()
    assert (stats["users"], stats["journeys"], stats["journeys_today"]) == (3, 1, 1)
    assert (stats["days"], stats["days_today"]) == (2, 2)  # one route each
    assert stats["llm_cost_today"] == stats["llm_cost_total"] == 0.01
    assert stats["limits"] == {
        "per_hour": 0,
        "daily_budget_usd": 0,
        "max_concurrent": 0,
        "running": 0,
    }
    users = {u["email"]: u for u in client.get("/api/admin/users").json()}
    assert (users["anna@example.com"]["journeys"], users["anna@example.com"]["days"]) == (1, 2)
    assert users[ADMIN[0]]["is_admin"]
    [journey] = client.get("/api/admin/trips").json()
    assert (journey["id"], journey["owner_email"], journey["days"], journey["cost_usd"]) == (
        "j1",
        "anna@example.com",
        2,
        0.01,
    )


def test_seeded_passwords_follow_the_env_across_restarts(tmp_path, monkeypatch, pipeline):
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    with TestClient(main.app) as client:
        assert _login(client, *DEMO).status_code == 200
        old_cookie = client.cookies.get("progulka_session")

    monkeypatch.setattr(main, "settings", _settings(tmp_path, demo_password="rotated-pass"))
    with TestClient(main.app) as client:
        client.cookies.set("progulka_session", old_cookie)
        assert client.get("/api/me").status_code == 401  # a reset signs the old password out
        assert _login(client, *DEMO).status_code == 401
        assert _login(client, DEMO[0], "rotated-pass").status_code == 200


def test_the_shared_demo_account_cannot_be_wiped_and_a_restart_resets_it(
    tmp_path, monkeypatch, pipeline
):
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "limits", Limits(0, 0, 0))
    monkeypatch.setattr(main, "busy_journeys", set())
    with TestClient(main.app) as client:
        _login(client, *DEMO)
        # Planning and feedback work in it; deleting and the questionnaire do not.
        journey_id = _journey_data(client.post("/api/plan", json=THREE_DAYS))["journey_id"]
        assert _feedback(client, journey_id, feedback="без музеев").status_code == 200
        for refused in (
            client.put("/api/profile", json={"home_city": "Казань"}),
            client.delete("/api/memory/1"),
            client.delete(f"/api/journeys/{journey_id}"),
        ):
            assert refused.status_code == 403 and refused.json()["detail"] == main.DEMO_IS_SHARED
    # Facts an older version let the demo's visitors teach it.
    asyncio.run(_remember(main.settings.database_url, DEMO[0], ["Любит конструктивизм"]))

    with TestClient(main.app) as client:
        _login(client, *DEMO)
        assert client.get("/api/profile").json()["home_city"] == "Санкт-Петербург"
        assert client.get("/api/memory").json() == []  # what visitors taught it is gone
        assert client.get("/api/journeys").json() == []  # and what they typed


def test_the_shared_demo_account_learns_nothing_from_its_visitors(client, pipeline):
    """Its password is public: a fact learned from one visitor's words would reach the
    next stranger's agent. Its runs get no memory, so the real pipeline makes no
    extraction call, and main stores nothing even when a pipeline reports facts."""
    _login(client, *DEMO)
    assert client.get("/api/me").json()["user"]["is_demo"]

    plan = _journey_data(client.post("/api/plan", json=THREE_DAYS))
    feedback = _journey_data(_feedback(client, plan["journey_id"], feedback="я вегетарианец"))

    assert plan["learned"] == feedback["learned"] == []
    assert [call["memory"] for call in pipeline.calls] == [None, None]
    assert client.get("/api/memory").json() == []

    _register(client)  # anyone else keeps a memory of their own, and is told so
    assert not client.get("/api/me").json()["user"]["is_demo"]
    assert _journey_data(client.post("/api/plan", json=THREE_DAYS))["learned"]
    assert pipeline.calls[-1]["memory"] == []


def test_admin_rights_follow_the_env_across_restarts(tmp_path, monkeypatch, pipeline):
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    with TestClient(main.app) as client:
        assert _login(client, *ADMIN).json()["user"]["is_admin"]

    new_admin = ("root@progulka.app", "root-secret-1")
    monkeypatch.setattr(
        main, "settings", _settings(tmp_path, admin_email=new_admin[0], admin_password=new_admin[1])
    )
    with TestClient(main.app) as client:
        assert not _login(client, *ADMIN).json()["user"]["is_admin"]  # replaced: no rights
        assert client.get("/api/admin/stats").status_code == 403
        assert _login(client, *new_admin).json()["user"]["is_admin"]

    monkeypatch.setattr(main, "settings", _settings(tmp_path, admin_email="", admin_password=""))
    with TestClient(main.app) as client:
        assert not _login(client, *new_admin).json()["user"]["is_admin"]


def test_feedback_continues_a_journey_saved_before_a_restart(tmp_path, monkeypatch, pipeline):
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "busy_journeys", set())
    with TestClient(main.app) as client:
        _register(client)
        _journey_data(client.post("/api/plan", json=THREE_DAYS))
        cookie = client.cookies.get("progulka_session")

    with TestClient(main.app) as client:
        client.cookies.set("progulka_session", cookie)
        feedback = _feedback(client, day=1, feedback="без соборов")

        assert _journey_data(feedback)["reply"] == "Убрал"
        restored = pipeline.calls[1]["journey"].days[1]
        assert restored.state.excluded == {"osm:node/13"}
        # The places of all three days, without the one no day refers to.
        assert len(restored.state.registry) == 9
        history = client.get("/api/journeys/j1").json()["history"]
        assert [h["text"] for h in history] == [THREE_DAYS["request"], "без соборов"]


def test_restart_keeps_todays_spend_in_the_budget(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "limits", Limits(0, daily_budget_usd=0.01, max_concurrent=0))
    with TestClient(main.app) as client:
        _register(client)
        _journey_data(client.post("/api/plan", json={"request": "Прогулка"}))

    monkeypatch.setattr(main, "limits", Limits(0, daily_budget_usd=0.01, max_concurrent=0))
    with TestClient(main.app) as client:
        assert _login(client, "anna@example.com", PASSWORD).status_code == 200
        assert client.post("/api/plan", json={"request": "Ещё"}).status_code == 429


def test_catalog_is_public(client):
    interests = client.get("/api/catalog").json()["interests"]
    assert {"key": "nature", "label": "Природа и парки"} in interests


@pytest.mark.parametrize(
    ("path", "page"),
    [
        ("/", "landing.html"),
        ("/login", "auth.html"),
        ("/register", "auth.html"),
        ("/app/trips", "trips.html"),
        ("/app/new", "planner.html"),
        ("/app/trips/j1", "planner.html"),
        ("/app/profile", "profile.html"),
        ("/app/onboarding", "profile.html"),
        ("/app/admin", "admin.html"),
    ],
)
def test_pages_are_served(client, path, page):
    response = client.get(path)
    assert response.status_code == 200
    assert response.text == (main.STATIC / page).read_text()
    assert response.headers["cache-control"] == "no-cache"


def test_static_files_are_revalidated_after_a_deploy(client):
    response = client.get("/static/js/api.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    again = client.get("/static/js/api.js", headers={"if-none-match": response.headers["etag"]})
    assert again.status_code == 304


def test_cabinet_root_redirects_to_the_trips(client):
    response = client.get("/app", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/app/trips"
