"""HTTP layer: pages, accounts, saved journeys, voice input, and the agent's work as NDJSON.

Each agent run (a new journey, or feedback on one day of a saved one) runs the
pipeline in a background task that pushes AgentEvents into a queue; the response
streams them one JSON object per line, so the UI shows every tool call while the agent
is still working. The run is saved before the final `journey` event, so the journey
id in that event always points at a stored journey; `done` closes the stream with the
run's LLM usage.

The traveller's profile and recent memory facts go into every run; facts the run
learned are stored and reported in the `journey` event.

Voice input is recorded in the browser and sent here as a short WAV: an audio-capable
LLM transcribes it (app/agent/speech.py), so it works in every browser, including
those without the Web Speech API. The recording is checked before it costs a call.

Pages are static HTML: each asks /api/me who is signed in and sends a stranger to
/login. The API enforces access on its own, and every journey query is scoped to its owner.

Run locally:   uvicorn app.main:app --reload
Run in Docker: docker compose up --build
"""

import asyncio
import base64
import io
import logging
import re
import time
import wave
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

from app.agent.speech import transcribe
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
from app.limits import BUDGET_SPENT, HOUR_S, Limits, SlidingWindow
from app.llm import LLM, LLMError
from app.models import AgentEvent, Emit, Stop
from app.pipeline import (
    EMPTY_REQUEST,
    FeedbackRequest,
    Journey,
    PipelineError,
    PlanRequest,
    apply_day_feedback,
    has_place,
    plan_journey,
)
from app.profile import Profile, catalog
from app.store import (
    AdminJourney,
    AdminUser,
    HistoryEntry,
    JourneyDetail,
    JourneySummary,
    MemoryOut,
    Store,
    UserOut,
    journey_view,
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
TRANSCRIPTIONS_PER_HOUR = 30  # per address: a minute of speech each, a few minutes of talk
# What the recorder in static/js/voice.js sends: 16 kHz mono 16-bit WAV, up to a minute.
AUDIO_RATE, MAX_AUDIO_S, MAX_AUDIO_BYTES = 16_000, 60, 2_500_000
NO_REQUEST_TEXT = "По анкете и настройкам путешествия"
JOURNEY_NOT_FOUND = "Путешествие не найдено"
DAY_NOT_FOUND = "В этом путешествии нет такого дня"
JOURNEY_BUSY = "Путешествие уже перестраивается"
TOO_MANY_ATTEMPTS = "Слишком много попыток, подождите несколько минут"
DEMO_IS_SHARED = (
    "Демо-аккаунт общий: в нём нельзя удалять и менять анкету. Заведите свой — это бесплатно"
)
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
transcriptions = SlidingWindow(TRANSCRIPTIONS_PER_HOUR, HOUR_S)
# Journeys with a run in progress. A second feedback or a delete is refused, not
# queued: a plain set, not per-journey locks, because nothing ever waits for a journey.
# Feedback on one day locks the whole journey: the other days' stops are what that
# run must not take, and a parallel run on another day could take the same place.
busy_journeys: set[str] = set()


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
    description="LLM agent that plans a traveller's walks in an unfamiliar city, one route "
    "a day, through real places.",
    version="0.3.0",
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
@app.get("/app/trips/{journey_id}", include_in_schema=False)
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
    _refuse_on_shared_demo(user)
    return await store.put_profile(user.id, profile)


@app.get("/api/memory")
async def memory(user: UserDep, store: StoreDep) -> list[MemoryOut]:
    return await store.list_memory(user.id)


@app.delete("/api/memory/{fact_id}", status_code=204)
async def forget(fact_id: int, user: UserDep, store: StoreDep) -> Response:
    _refuse_on_shared_demo(user)
    if not await store.delete_memory(user.id, fact_id):
        raise HTTPException(status_code=404, detail="Такого факта нет")
    return Response(status_code=204)


@app.get("/api/journeys")
async def journeys(user: UserDep, store: StoreDep) -> list[JourneySummary]:
    return await store.list_journeys(user.id)


@app.get("/api/journeys/{journey_id}")
async def journey(journey_id: str, user: UserDep, store: StoreDep) -> JourneyDetail:
    detail = await store.get_journey(user.id, journey_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=JOURNEY_NOT_FOUND)
    return detail


@app.delete("/api/journeys/{journey_id}", status_code=204)
async def delete_journey(journey_id: str, user: UserDep, store: StoreDep) -> Response:
    _refuse_on_shared_demo(user)
    if journey_id in busy_journeys:
        raise HTTPException(status_code=409, detail=JOURNEY_BUSY)
    if not await store.delete_journey(user.id, journey_id):
        raise HTTPException(status_code=404, detail=JOURNEY_NOT_FOUND)
    return Response(status_code=204)


class TranscribeRequest(BaseModel):
    audio: str  # base64 of a WAV file


@app.post("/api/transcribe")
async def transcribe_speech(
    body: TranscribeRequest, request: Request, user: UserDep
) -> dict[str, str]:
    """Voice input: a recording in, its transcript out, for the request or feedback box."""
    client = _client(request)
    counted = not user.is_admin  # the admin is exempt, as from the run limits (limits.py)
    if counted and transcriptions.full(client):
        raise HTTPException(
            status_code=429,
            detail=f"Не больше {TRANSCRIPTIONS_PER_HOUR} голосовых вводов в час с одного адреса",
        )
    if counted and limits.out_of_budget():
        raise HTTPException(status_code=429, detail=BUDGET_SPENT)
    _check_recording(body.audio)
    if counted:
        transcriptions.add(client)
    llm: LLM = request.app.state.llm.for_run()
    try:
        text, cost_usd = await transcribe(llm, body.audio)
    except PipelineError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    limits.spend(cost_usd)
    return {"text": text}


def _check_recording(audio: str) -> None:
    """Refuse what the recorder never sends before it costs an LLM call: too long or too
    big (413); not base64, not 16 kHz mono 16-bit WAV, or empty (422)."""
    too_long = HTTPException(status_code=413, detail="Запись длиннее минуты")
    if len(audio) > (MAX_AUDIO_BYTES + 2) // 3 * 4:  # base64 of MAX_AUDIO_BYTES
        raise too_long
    try:
        with wave.open(io.BytesIO(base64.b64decode(audio, validate=True))) as wav:
            frames = wav.getnframes()
            pcm = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
    # base64's binascii.Error is a ValueError; wave raises a bare RuntimeError for a
    # chunk whose size runs past the file.
    except (ValueError, wave.Error, EOFError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail="Запись не похожа на WAV") from e
    if pcm != (1, 2, AUDIO_RATE):
        raise HTTPException(status_code=422, detail="Нужна запись WAV 16 кГц, моно, 16 бит")
    if not frames:
        raise HTTPException(status_code=422, detail="Запись пустая")
    if frames > MAX_AUDIO_S * AUDIO_RATE:
        raise too_long


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
async def admin_journeys(_admin: AdminDep, store: StoreDep) -> list[AdminJourney]:
    """The newest journeys; the path says trips, like the cabinet's /app/trips."""
    return await store.admin_journeys()


# --- agent runs ---


@dataclass
class Outcome:
    journey: Journey
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
        journey = await plan_journey(req, llm, emit, today=today, profile=profile, memory=facts)
        entry = HistoryEntry(
            kind="plan", text=req.request or NO_REQUEST_TEXT, cost_usd=llm.usage.cost_usd
        )
        await store.create_journey(user.id, journey, req.request, req.preferences, entry)
        # A fact points at the day it was learned on; the request of a journey speaks
        # for all its days, so its facts point at the first.
        learned = await store.add_memory(
            user.id, journey.learned, source_trip_id=journey.days[0].id
        )
        return Outcome(journey, None, learned)

    return _stream(job, request, user)


@app.post("/api/journeys/{journey_id}/days/{index}/feedback")
async def feedback(
    journey_id: str,
    index: int,
    req: FeedbackRequest,
    request: Request,
    user: UserDep,
    store: StoreDep,
) -> StreamingResponse:
    # Taken before the first await: a second request must not load the state this run
    # is about to replace and then overwrite its result. The run frees the journey when
    # it ends; until the run starts, this handler frees it on any failure.
    if journey_id in busy_journeys:
        raise HTTPException(status_code=409, detail=JOURNEY_BUSY)
    busy_journeys.add(journey_id)
    try:
        # From the database, not process memory: a journey from last week, or from
        # before a restart, is continued exactly where its last run stopped.
        journey = await store.load_journey(user.id, journey_id)
        if journey is None:
            raise HTTPException(status_code=404, detail=JOURNEY_NOT_FOUND)
        if not 0 <= index < len(journey.days):
            raise HTTPException(status_code=404, detail=DAY_NOT_FOUND)
        day = journey.days[index]
        stops = day.plan.stops if day.plan else []
        # «Убрать» from a stale tab, for a stop an earlier run already removed: nothing
        # to do, and refused before it uses up a run.
        if not req.feedback.strip() and not any(
            stop.place.id in req.exclude_place_ids for stop in stops
        ):
            raise HTTPException(status_code=409, detail="Этого места уже нет в маршруте")
        profile, facts = await _traveller(store, user.id)
        text = _feedback_text(req, stops)

        async def job(llm: LLM, emit: Emit) -> Outcome:
            updated, reply = await apply_day_feedback(
                journey, index, req, llm, emit, profile=profile, memory=facts
            )
            entry = HistoryEntry(
                kind="feedback", day=index, text=text, reply=reply, cost_usd=llm.usage.cost_usd
            )
            await store.update_journey(user.id, updated, entry)
            learned = await store.add_memory(
                user.id, updated.learned, source_trip_id=updated.days[index].id
            )
            return Outcome(updated, reply, learned)

        return _stream(job, request, user, journey_id=journey_id)
    except BaseException:
        busy_journeys.discard(journey_id)
        raise


def _refuse_on_shared_demo(user: UserOut) -> None:
    """The demo account's password is public: what one visitor deletes or writes into the
    questionnaire (whose notes reach the agent's prompts), every later visitor gets.
    Planning, feedback and the memory they teach still work in it, and a restart resets
    its profile and memory (auth.seed_account)."""
    demo = settings.demo_email.strip().lower() if settings.demo_password else ""
    if demo and user.email == demo:
        raise HTTPException(status_code=403, detail=DEMO_IS_SHARED)


async def _traveller(store: Store, user_id: int) -> tuple[Profile, list[str]]:
    return await store.get_profile(user_id), await store.memory_texts(user_id, MEMORY_IN_CONTEXT)


def _feedback_text(req: FeedbackRequest, stops: list[Stop]) -> str:
    """The history line of a feedback run; the «Убрать» button sends no text, only ids."""
    if text := req.feedback.strip():
        return text
    names = [stop.place.name for stop in stops if stop.place.id in req.exclude_place_ids]
    return "Убрать: " + ", ".join(names)


def _client(request: Request) -> str:
    # Behind the reverse proxy uvicorn runs with --proxy-headers, so this is the visitor.
    return request.client.host if request.client else "unknown"


def _stream(
    job: Job, request: Request, user: UserOut, journey_id: str | None = None
) -> StreamingResponse:
    """Start the run in the background and stream its events; it frees `journey_id` at the end."""
    refusal = limits.admit(_client(request), exempt=user.is_admin)
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
            data: dict[str, Any] = {
                "journey_id": outcome.journey.id,
                "journey": journey_view(outcome.journey).model_dump(mode="json"),
                "usage": llm.usage.as_dict(),
                "reply": outcome.reply,
                "learned": outcome.learned,
            }
            await emit(AgentEvent(type="journey", data=data))
        except PipelineError as e:
            await emit(AgentEvent(type="error", data={"message": str(e)}))
        except LLMError as e:
            await emit(AgentEvent(type="error", data={"message": f"Ошибка LLM-провайдера: {e}"}))
        except Exception:
            log.exception("pipeline failed")
            await emit(AgentEvent(type="error", data={"message": "Внутренняя ошибка, см. логи"}))
        finally:
            if journey_id:
                busy_journeys.discard(journey_id)
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
