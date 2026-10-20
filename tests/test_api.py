"""The HTTP API end to end: the real store on a temporary SQLite file, a fake pipeline."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.limits import Limits, SlidingWindow
from app.models import AgentEvent
from app.pipeline import PipelineError

DEMO = ("demo@progulka.app", "progulka-demo")
ADMIN = ("admin@progulka.app", "admin-secret-1")
PASSWORD = "secret-123"


class FakePipeline:
    """Stands in for plan_trip and apply_feedback: records its inputs, spends a cent."""

    def __init__(self, make_session):
        self.make_session = make_session
        self.calls: list[dict] = []
        self.learned = ["Любит конструктивизм"]
        self.error: Exception | None = None

    async def plan_trip(self, req, llm, emit, today, profile=None, memory=None):
        self.calls.append({"req": req, "profile": profile, "memory": memory})
        await emit(AgentEvent(type="stage", data={"name": "brief", "status": "start"}))
        llm.usage.cost_usd += 0.01
        if self.error:
            raise self.error
        session = self.make_session(trip_id=f"trip{len(self.calls)}")
        session.learned = list(self.learned)
        return session

    async def apply_feedback(self, session, req, llm, emit, profile=None, memory=None):
        self.calls.append({"session": session, "req": req, "memory": memory})
        llm.usage.cost_usd += 0.002
        plan = session.plan
        plan.stops = [s for s in plan.stops if s.place.id not in req.exclude_place_ids]
        plan.title = "Без лишнего"
        session.learned = ["Не любит толпы"]
        return session, "Убрал"


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
def pipeline(make_session, monkeypatch):
    fake = FakePipeline(make_session)
    monkeypatch.setattr(main, "plan_trip", fake.plan_trip)
    monkeypatch.setattr(main, "apply_feedback", fake.apply_feedback)
    return fake


def _fresh_throttles(monkeypatch, registrations=None):
    window = main.AUTH_WINDOW_S
    monkeypatch.setattr(main, "login_failures", SlidingWindow(main.LOGIN_FAILURES, window))
    monkeypatch.setattr(
        main, "registrations", SlidingWindow(registrations or main.REGISTRATIONS, window)
    )


@pytest.fixture
def client(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "limits", Limits(0, 0, 0))
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "busy_trips", set())
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


def _plan_data(response):
    [event] = [e for e in _events(response) if e["type"] == "plan"]
    return event["data"]


def test_traveller_plans_continues_and_deletes_a_trip(client, pipeline):
    user = _register(client)
    assert user["name"] == "Анна" and not user["onboarded"] and not user["is_admin"]
    assert client.get("/api/me").json()["user"]["email"] == "anna@example.com"

    profile = {"home_city": "Казань", "interests": {"nature": 1.0, "coffee": 0.6}}
    assert client.put("/api/profile", json=profile).status_code == 200
    assert client.get("/api/me").json()["user"]["onboarded"]
    assert client.get("/api/profile").json()["interests"] == {"nature": 1.0, "coffee": 0.6}

    request = {"request": "Суббота, хочу стрит-арт", "preferences": {"more": ["street_art"]}}
    events = _events(client.post("/api/plan", json=request))
    assert [e["type"] for e in events] == ["stage", "plan", "done"]
    plan = events[1]["data"]
    assert plan["trip_id"] == "trip1" and plan["reply"] is None
    assert plan["learned"] == ["Любит конструктивизм"]
    assert plan["usage"]["cost_usd"] == 0.01
    assert pipeline.calls[0]["profile"].home_city == "Казань"
    assert pipeline.calls[0]["memory"] == []

    [summary] = client.get("/api/trips").json()
    assert (summary["id"], summary["title"], summary["stops"]) == (
        "trip1",
        "Конструктивизм и кофе",
        2,
    )
    assert [m["text"] for m in client.get("/api/memory").json()] == ["Любит конструктивизм"]

    feedback = client.post("/api/trips/trip1/feedback", json={"exclude_place_ids": ["osm:node/1"]})
    plan = _plan_data(feedback)
    assert (plan["trip_id"], plan["reply"], plan["learned"]) == (
        "trip1",
        "Убрал",
        ["Не любит толпы"],
    )
    # Feedback continued the saved research state, with the memory as context.
    assert pipeline.calls[1]["session"].state.excluded == {"osm:node/3"}
    assert pipeline.calls[1]["memory"] == ["Любит конструктивизм"]

    detail = client.get("/api/trips/trip1").json()
    assert (detail["trip"]["title"], detail["trip"]["stops"]) == ("Без лишнего", 1)
    assert [(h["kind"], h["text"], h["reply"]) for h in detail["history"]] == [
        ("plan", "Суббота, хочу стрит-арт", None),
        ("feedback", "Убрать: Место 1", "Убрал"),
    ]
    assert [h["cost_usd"] for h in detail["history"]] == [0.01, 0.002]
    assert len(detail["plan"]["stops"]) == 1
    assert detail["preferences"]["more"] == ["street_art"]

    assert client.delete("/api/trips/trip1").status_code == 204
    assert client.get("/api/trips").json() == []
    assert client.get("/api/trips/trip1").status_code == 404
    assert client.post("/api/trips/trip1/feedback", json={"feedback": "ещё"}).status_code == 404


def test_known_facts_are_not_reported_as_learned_again(client):
    _register(client)
    _plan_data(client.post("/api/plan", json={"request": "Прогулка по Петербургу"}))

    again = _plan_data(client.post("/api/plan", json={"request": "Ещё одна"}))

    assert again["learned"] == []
    assert len(client.get("/api/memory").json()) == 1


def test_a_failed_run_saves_nothing(client, pipeline):
    _register(client)
    pipeline.error = PipelineError("Не нашёл город «Атлантида» в OpenStreetMap")

    events = _events(client.post("/api/plan", json={"request": "Атлантида"}))

    assert [e["type"] for e in events] == ["stage", "error", "done"]
    assert events[1]["data"]["message"].startswith("Не нашёл город")
    assert client.get("/api/trips").json() == []


def test_someone_elses_trip_and_memory_look_missing(client):
    _register(client)
    _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))
    [fact] = client.get("/api/memory").json()

    _register(client, "boris@example.com", "Борис")

    assert client.get("/api/trips").json() == []
    assert client.get("/api/trips/trip1").status_code == 404
    assert client.delete("/api/trips/trip1").status_code == 404
    assert client.post("/api/trips/trip1/feedback", json={"feedback": "x"}).status_code == 404
    assert client.delete(f"/api/memory/{fact['id']}").status_code == 404
    assert _login(client, "anna@example.com", PASSWORD).status_code == 200
    assert len(client.get("/api/trips").json()) == 1


def test_a_trip_being_rebuilt_refuses_feedback_and_delete(client):
    _register(client)
    _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))
    main.busy_trips.add("trip1")

    feedback = client.post("/api/trips/trip1/feedback", json={"feedback": "без соборов"})

    assert feedback.status_code == 409
    assert feedback.json() == {"detail": "Маршрут уже перестраивается"}
    assert client.delete("/api/trips/trip1").status_code == 409


def test_runs_are_rate_limited_per_address(client, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=1, daily_budget_usd=0, max_concurrent=0))
    _register(client)
    _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))

    refused = client.post("/api/plan", json={"request": "Ещё"})
    feedback = client.post("/api/trips/trip1/feedback", json={"feedback": "без соборов"})

    assert refused.status_code == 429 and "в час" in refused.json()["detail"]
    assert feedback.status_code == 429
    assert main.busy_trips == set()  # a refused feedback does not leave the trip locked


def test_an_empty_form_is_refused_before_it_uses_a_run(client, monkeypatch):
    monkeypatch.setattr(main, "limits", Limits(per_hour=1, daily_budget_usd=0, max_concurrent=0))
    _register(client)  # no home city in a fresh profile

    empty = client.post("/api/plan", json={"request": "  "})
    _plan_data(client.post("/api/plan", json={"request": "", "city": "Казань"}))

    assert empty.status_code == 422 and "укажите город" in empty.json()["detail"]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/me"),
        ("get", "/api/profile"),
        ("get", "/api/memory"),
        ("get", "/api/trips"),
        ("post", "/api/plan"),
        ("post", "/api/trips/trip1/feedback"),
    ],
)
def test_signed_out_requests_are_refused(client, method, path):
    response = client.request(method, path, json={"request": "Прогулка"})
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
    _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))
    for path in ("/api/admin/stats", "/api/admin/users", "/api/admin/trips"):
        assert client.get(path).status_code == 403

    assert _login(client, *ADMIN).json()["user"]["is_admin"]

    stats = client.get("/api/admin/stats").json()
    assert (stats["users"], stats["trips"], stats["trips_today"]) == (3, 1, 1)
    assert stats["llm_cost_today"] == stats["llm_cost_total"] == 0.01
    assert stats["limits"] == {
        "per_hour": 0,
        "daily_budget_usd": 0,
        "max_concurrent": 0,
        "running": 0,
    }
    users = {u["email"]: u for u in client.get("/api/admin/users").json()}
    assert users["anna@example.com"]["trips"] == 1 and users[ADMIN[0]]["is_admin"]
    [trip] = client.get("/api/admin/trips").json()
    assert (trip["id"], trip["owner_email"], trip["cost_usd"]) == (
        "trip1",
        "anna@example.com",
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


def test_feedback_continues_a_trip_saved_before_a_restart(tmp_path, monkeypatch, pipeline):
    _fresh_throttles(monkeypatch)
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    with TestClient(main.app) as client:
        _register(client)
        _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))
        cookie = client.cookies.get("progulka_session")

    with TestClient(main.app) as client:
        client.cookies.set("progulka_session", cookie)
        feedback = client.post("/api/trips/trip1/feedback", json={"feedback": "без соборов"})

        assert _plan_data(feedback)["reply"] == "Убрал"
        restored = pipeline.calls[1]["session"]
        assert restored.state.excluded == {"osm:node/3"} and len(restored.state.registry) == 3
        history = client.get("/api/trips/trip1").json()["history"]
        assert [h["text"] for h in history] == ["Прогулка", "без соборов"]


def test_restart_keeps_todays_spend_in_the_budget(tmp_path, monkeypatch, pipeline):
    monkeypatch.setattr(main, "settings", _settings(tmp_path))
    monkeypatch.setattr(main, "limits", Limits(0, daily_budget_usd=0.01, max_concurrent=0))
    with TestClient(main.app) as client:
        _register(client)
        _plan_data(client.post("/api/plan", json={"request": "Прогулка"}))

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
        ("/app/trips/trip1", "planner.html"),
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
