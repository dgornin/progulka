"""HTTP layer: pages, accounts, saved trips, and the agent's work streamed as NDJSON.

Each agent run (a new walk, or feedback on a saved one) runs the pipeline in a
background task that pushes AgentEvents into a queue; the response streams them one
JSON object per line, so the UI shows every tool call while the agent is still
working. The run is saved before the final `plan` event, so the trip id in that event
always points at a stored trip; `done` closes the stream with the run's LLM usage.

The traveller's profile and recent memory facts go into every run; facts the run
learned are stored and reported in the `plan` event.

Pages are static HTML: each asks /api/me who is signed in and sends a stranger to
/login. The API enforces access on its own, and every trip query is scoped to its owner.

Run locally:   uvicorn app.main:app --reload
Run in Docker: docker compose up --build
"""

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.types import Scope

from app.agent.tools import tool_schemas
from app.auth import (
    AdminDep,
    StoreDep,
    UserDep,
    authenticate,
    end_session,
    hash_password,
    seed_account,
    start_session,
)
from app.config import settings
from app.db import connect, create_schema
from app.limits import Limits, SlidingWindow
from app.llm import LLM, LLMError
from app.models import AgentEvent, Emit
from app.pipeline import (
    EMPTY_REQUEST,
    FeedbackRequest,
    PipelineError,
    PlanRequest,
    TripRun,
    apply_feedback,
    has_place,
    plan_trip,
)
from app.profile import Profile, catalog
from app.store import (
    AdminTrip,
    AdminUser,
    HistoryEntry,
    MemoryOut,
    Store,
    TripDetail,
    TripSummary,
    UserOut,
    today_start,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("progulka")
STATIC = Path(__file__).parent / "static"

MEMORY_IN_CONTEXT = 20  # newest facts the agent sees; older ones stay in the profile page
# "Today" for the brief is Moscow's date, not the server's (UTC in Docker): between
# 00:00 and 03:00 MSK the server still says yesterday. No DST, so a fixed offset.
MOSCOW = timezone(timedelta(hours=3))
LOGIN_FAILURES, LOGINS, REGISTRATIONS, AUTH_WINDOW_S = 10, 100, 20, 300.0  # per address
NO_REQUEST_TEXT = "По анкете и настройкам прогулки"
TRIP_NOT_FOUND = "Маршрут не найден"
TRIP_BUSY = "Маршрут уже перестраивается"
TOO_MANY_ATTEMPTS = "Слишком много попыток, подождите несколько минут"
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
DEMO_PROFILE = Profile(
    home_city="Санкт-Петербург",
    interests={"architecture": 1.0, "history": 0.6, "coffee": 0.6, "street_art": 0.6},
    pace="normal",
)

limits = Limits(
    settings.rate_limit_per_hour, settings.daily_budget_usd, settings.max_concurrent_runs
)
# Password guessing and mass sign-ups (each costs a scrypt hash), counted apart: a
# classroom behind one NAT signing up at once must not lock everyone there out of
# logging in, the demo account included. Logins of either outcome have a cap of their
# own, far above a classroom's, so a script cannot spend the CPU on correct passwords.
login_failures = SlidingWindow(LOGIN_FAILURES, AUTH_WINDOW_S)
logins = SlidingWindow(LOGINS, AUTH_WINDOW_S)
registrations = SlidingWindow(REGISTRATIONS, AUTH_WINDOW_S)
# Trips with a run in progress. A second feedback or a delete is refused, not queued:
# a plain set, not per-trip locks, because nothing ever waits for a trip.
busy_trips: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # The LLM client first: without a key the app fails at startup, not on a visitor's run.
    app.state.llm = LLM.from_settings(settings)
    engine = connect(settings.database_url)
    await create_schema(engine)
    store = Store(engine)
    if settings.demo_email and settings.demo_password:
        await seed_account(
            store,
            settings.demo_email,
            settings.demo_password,
            "Демо",
            is_admin=False,
            profile=DEMO_PROFILE,
        )
    admin = settings.admin_email if settings.admin_email and settings.admin_password else None
    if admin:
        await seed_account(store, admin, settings.admin_password, "Администратор", is_admin=True)
    await store.revoke_admins(keep=admin)
    # Every deploy restarts the app; today's budget must not start from zero again.
    limits.spend(await store.llm_cost_since(today_start()))
    app.state.store = store
    yield
    await engine.dispose()


app = FastAPI(
    title="Прогулка",
    description="LLM agent that plans a walking route through real places in a city.",
    version="0.2.0",
    lifespan=lifespan,
)


class RevalidatedFiles(StaticFiles):
    """Static files the browser revalidates on every use (a 304 when unchanged): after a
    deploy every visitor gets the new scripts at once, with no build step to version them."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatedFiles(directory=STATIC), name="static")


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """One Russian sentence instead of FastAPI's error list: the UI shows `detail` as is."""
    error = exc.errors()[0]
    if error["type"] == "value_error":  # our validators speak Russian already
        message = str(error["ctx"]["error"])
    else:
        field = ".".join(str(part) for part in error["loc"][1:] if isinstance(part, str))
        message = f"Проверьте поле «{field}»" if field else "Некорректный запрос"
    return JSONResponse(status_code=422, content={"detail": message})


# --- pages ---


def _page(name: str) -> FileResponse:
    return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
async def landing() -> FileResponse:
    return _page("landing.html")


@app.get("/login", include_in_schema=False)
@app.get("/register", include_in_schema=False)
async def auth_page() -> FileResponse:
    return _page("auth.html")


@app.get("/app", include_in_schema=False)
async def cabinet() -> RedirectResponse:
    return RedirectResponse("/app/trips", status_code=307)


@app.get("/app/trips", include_in_schema=False)
async def trips_page() -> FileResponse:
    return _page("trips.html")


@app.get("/app/new", include_in_schema=False)
@app.get("/app/trips/{trip_id}", include_in_schema=False)
async def planner_page() -> FileResponse:
    return _page("planner.html")


@app.get("/app/profile", include_in_schema=False)
@app.get("/app/onboarding", include_in_schema=False)
async def profile_page() -> FileResponse:
    return _page("profile.html")


@app.get("/app/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    return _page("admin.html")


# --- public API ---


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_name}


@app.get("/api/tools")
async def tools() -> list[dict[str, Any]]:
    """The exact tool schemas the research agent receives: the model never sees our code."""
    return tool_schemas()


@app.get("/api/config")
async def config() -> dict[str, Any]:
    """The demo account, shown on the login page. Never the admin account."""
    demo = (
        {"email": settings.demo_email, "password": settings.demo_password}
        if settings.demo_email and settings.demo_password
        else None
    )
    return {"demo": demo}


@app.get("/api/catalog")
async def interest_catalog() -> dict[str, Any]:
    return catalog()


class LoginForm(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class RegisterForm(LoginForm):
    name: str

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL.fullmatch(value):
            raise ValueError("Укажите настоящую почту")
        return value

    @field_validator("password")
    @classmethod
    def _password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Пароль должен быть не короче 8 символов")
        return value

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not 1 <= len(value) <= 60:
            raise ValueError("Укажите имя, до 60 символов")
        return value


@app.post("/api/auth/register", status_code=201)
async def register(
    form: RegisterForm, request: Request, response: Response, store: StoreDep
) -> dict[str, UserOut]:
    client = _client(request)
    if registrations.full(client):
        raise HTTPException(status_code=429, detail=TOO_MANY_ATTEMPTS)
    registrations.add(client)
    user = await store.create_user(form.email, await hash_password(form.password), form.name)
    if user is None:
        raise HTTPException(status_code=409, detail="Эта почта уже зарегистрирована, войдите")
    await start_session(store, response, user.id, settings.cookie_secure)
    return {"user": user}


@app.post("/api/auth/login")
async def login(
    form: LoginForm, request: Request, response: Response, store: StoreDep
) -> dict[str, UserOut]:
    client = _client(request)
    if login_failures.full(client) or logins.full(client):
        raise HTTPException(status_code=429, detail=TOO_MANY_ATTEMPTS)
    # Counted as a failure until the password checks out: parallel guesses are all
    # still hashing when the next one arrives, and must not all pass the check above.
    login_failures.add(client)
    logins.add(client)
    user = await authenticate(store, form.email, form.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Неверная почта или пароль")
    login_failures.undo(client)
    await start_session(store, response, user.id, settings.cookie_secure)
    return {"user": user}


@app.post("/api/auth/logout", status_code=204)
async def logout(request: Request, store: StoreDep) -> Response:
    response = Response(status_code=204)
    await end_session(store, request, response, settings.cookie_secure)
    return response


# --- the signed-in traveller ---


@app.get("/api/me")
async def me(user: UserDep) -> dict[str, UserOut]:
    return {"user": user}


@app.get("/api/profile")
async def get_profile(user: UserDep, store: StoreDep) -> Profile:
    return await store.get_profile(user.id)


@app.put("/api/profile")
async def put_profile(profile: Profile, user: UserDep, store: StoreDep) -> Profile:
    return await store.put_profile(user.id, profile)


@app.get("/api/memory")
async def memory(user: UserDep, store: StoreDep) -> list[MemoryOut]:
    return await store.list_memory(user.id)


@app.delete("/api/memory/{fact_id}", status_code=204)
async def forget(fact_id: int, user: UserDep, store: StoreDep) -> Response:
    if not await store.delete_memory(user.id, fact_id):
        raise HTTPException(status_code=404, detail="Такого факта нет")
    return Response(status_code=204)


@app.get("/api/trips")
async def trips(user: UserDep, store: StoreDep) -> list[TripSummary]:
    return await store.list_trips(user.id)


@app.get("/api/trips/{trip_id}")
async def trip(trip_id: str, user: UserDep, store: StoreDep) -> TripDetail:
    detail = await store.get_trip(user.id, trip_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=TRIP_NOT_FOUND)
    return detail


@app.delete("/api/trips/{trip_id}", status_code=204)
async def delete_trip(trip_id: str, user: UserDep, store: StoreDep) -> Response:
    if trip_id in busy_trips:
        raise HTTPException(status_code=409, detail=TRIP_BUSY)
    if not await store.delete_trip(user.id, trip_id):
        raise HTTPException(status_code=404, detail=TRIP_NOT_FOUND)
    return Response(status_code=204)


# --- admin ---


@app.get("/api/admin/stats")
async def admin_stats(_admin: AdminDep, store: StoreDep) -> dict[str, Any]:
    return {
        **await store.totals(),
        "limits": {
            "per_hour": limits.per_hour,
            "daily_budget_usd": limits.daily_budget_usd,
            "max_concurrent": limits.max_concurrent,
            "running": limits.running,
        },
    }


@app.get("/api/admin/users")
async def admin_users(_admin: AdminDep, store: StoreDep) -> list[AdminUser]:
    return await store.admin_users()


@app.get("/api/admin/trips")
async def admin_trips(_admin: AdminDep, store: StoreDep) -> list[AdminTrip]:
    return await store.admin_trips()


# --- agent runs ---


@dataclass
class Outcome:
    trip_run: TripRun
    reply: str | None
    learned: list[str]  # facts that were new to the traveller's memory


Job = Callable[[LLM, Emit], Awaitable[Outcome]]


@app.post("/api/plan")
async def plan(
    req: PlanRequest, request: Request, user: UserDep, store: StoreDep
) -> StreamingResponse:
    profile, facts = await _traveller(store, user.id)
    # Refused here, not inside the stream: an empty form must not use up an hourly run.
    if not has_place(req, profile):
        raise HTTPException(status_code=422, detail=EMPTY_REQUEST)

    async def job(llm: LLM, emit: Emit) -> Outcome:
        today = settings.today or datetime.now(MOSCOW).date()
        trip_run = await plan_trip(req, llm, emit, today=today, profile=profile, memory=facts)
        entry = HistoryEntry(
            kind="plan", text=req.request or NO_REQUEST_TEXT, cost_usd=llm.usage.cost_usd
        )
        await store.create_trip(user.id, trip_run, req.request, req.preferences, entry)
        learned = await store.add_memory(user.id, trip_run.learned, source_trip_id=trip_run.id)
        return Outcome(trip_run, None, learned)

    return _stream(job, request)


@app.post("/api/trips/{trip_id}/feedback")
async def feedback(
    trip_id: str, req: FeedbackRequest, request: Request, user: UserDep, store: StoreDep
) -> StreamingResponse:
    # Taken before the first await: a second request must not load the state this run
    # is about to replace and then overwrite its result. The run frees the trip when it
    # ends; until the run starts, this handler frees it on any failure.
    if trip_id in busy_trips:
        raise HTTPException(status_code=409, detail=TRIP_BUSY)
    busy_trips.add(trip_id)
    try:
        # From the database, not process memory: a trip from last week, or from before a
        # restart, is continued exactly where its last run stopped.
        trip_run = await store.load_trip_run(user.id, trip_id)
        if trip_run is None:
            raise HTTPException(status_code=404, detail=TRIP_NOT_FOUND)
        # «Убрать» from a stale tab, for a stop an earlier run already removed: nothing
        # to do, and refused before it uses up a run.
        stops = trip_run.plan.stops if trip_run.plan else []
        if not req.feedback.strip() and not any(
            stop.place.id in req.exclude_place_ids for stop in stops
        ):
            raise HTTPException(status_code=409, detail="Этого места уже нет в маршруте")
        profile, facts = await _traveller(store, user.id)
        text = _feedback_text(req, trip_run)

        async def job(llm: LLM, emit: Emit) -> Outcome:
            updated, reply = await apply_feedback(
                trip_run, req, llm, emit, profile=profile, memory=facts
            )
            entry = HistoryEntry(
                kind="feedback", text=text, reply=reply, cost_usd=llm.usage.cost_usd
            )
            await store.update_trip(user.id, updated, entry)
            learned = await store.add_memory(user.id, updated.learned, source_trip_id=trip_id)
            return Outcome(updated, reply, learned)

        return _stream(job, request, trip_id=trip_id)
    except BaseException:
        busy_trips.discard(trip_id)
        raise


async def _traveller(store: Store, user_id: int) -> tuple[Profile, list[str]]:
    return await store.get_profile(user_id), await store.memory_texts(user_id, MEMORY_IN_CONTEXT)


def _feedback_text(req: FeedbackRequest, trip_run: TripRun) -> str:
    """The history line of a feedback run; the «Убрать» button sends no text, only ids."""
    if text := req.feedback.strip():
        return text
    stops = trip_run.plan.stops if trip_run.plan else []
    names = [stop.place.name for stop in stops if stop.place.id in req.exclude_place_ids]
    return "Убрать: " + ", ".join(names)


def _client(request: Request) -> str:
    # Behind the reverse proxy uvicorn runs with --proxy-headers, so this is the visitor.
    return request.client.host if request.client else "unknown"


def _stream(job: Job, request: Request, trip_id: str | None = None) -> StreamingResponse:
    """Start the run in the background and stream its events; it frees `trip_id` at the end."""
    refusal = limits.admit(_client(request))
    if refusal:
        raise HTTPException(status_code=429, detail=refusal)
    llm: LLM = request.app.state.llm.for_run()
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await queue.put(event)

    async def run() -> None:
        started = time.monotonic()
        try:
            outcome = await job(llm, emit)
            plan = outcome.trip_run.plan
            data: dict[str, Any] = {
                "trip_id": outcome.trip_run.id,
                "plan": plan.model_dump(mode="json") if plan else None,
                "usage": llm.usage.as_dict(),
                "reply": outcome.reply,
                "learned": outcome.learned,
            }
            await emit(AgentEvent(type="plan", data=data))
        except PipelineError as e:
            await emit(AgentEvent(type="error", data={"message": str(e)}))
        except LLMError as e:
            await emit(AgentEvent(type="error", data={"message": f"Ошибка LLM-провайдера: {e}"}))
        except Exception:
            log.exception("pipeline failed")
            await emit(AgentEvent(type="error", data={"message": "Внутренняя ошибка, см. логи"}))
        finally:
            if trip_id:
                busy_trips.discard(trip_id)
            limits.release(llm.usage.cost_usd)
            elapsed = round(time.monotonic() - started, 1)
            await emit(
                AgentEvent(type="done", data={"usage": llm.usage.as_dict(), "elapsed_s": elapsed})
            )
            await queue.put(None)

    # Started here, not in lines(): the run releases its limit slot in its finally, so
    # it must run even if the client leaves before the first byte is streamed.
    task = asyncio.create_task(run())

    async def lines() -> AsyncIterator[str]:
        try:
            while (event := await queue.get()) is not None:
                yield event.model_dump_json() + "\n"
        finally:
            # The client went away: stop spending tokens on a plan nobody will see.
            task.cancel()

    return StreamingResponse(lines(), media_type="application/x-ndjson")
