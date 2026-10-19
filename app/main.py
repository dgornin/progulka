"""HTTP layer: serves the UI and streams the agent's work as NDJSON.

Each request runs the pipeline in a background task that pushes AgentEvents into a
queue; the response streams them one JSON object per line, so the UI shows every
tool call while the agent is still working. The last events are `plan` (or
`error`) and `done` with the run's LLM usage and cost.

Run locally:   uvicorn app.main:app --reload
Run in Docker: docker compose up --build
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.tools import tool_schemas
from app.config import settings
from app.limits import Limits
from app.llm import LLM, LLMError
from app.models import AgentEvent, Emit
from app.pipeline import (
    FeedbackRequest,
    PipelineError,
    PlanRequest,
    Session,
    apply_feedback,
    get_session,
    plan_trip,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("progulka")
STATIC = Path(__file__).parent / "static"

base_llm = LLM.from_settings(settings)
limits = Limits(
    settings.rate_limit_per_hour, settings.daily_budget_usd, settings.max_concurrent_runs
)

app = FastAPI(
    title="Прогулка",
    description="LLM agent that plans a walking route through real places in a city.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

Job = Callable[[LLM, Emit], Awaitable[tuple[Session, str | None]]]


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": base_llm.model}


@app.get("/api/tools")
async def tools() -> list[dict[str, Any]]:
    """The exact tool schemas the research agent receives: the model never sees our code."""
    return tool_schemas()


@app.post("/api/plan")
async def plan(req: PlanRequest, request: Request) -> StreamingResponse:
    async def job(llm: LLM, emit: Emit) -> tuple[Session, str | None]:
        return await plan_trip(req, llm, emit, today=settings.today or date.today()), None

    return _stream(job, _client(request))


@app.post("/api/sessions/{session_id}/feedback")
async def feedback(session_id: str, req: FeedbackRequest, request: Request) -> StreamingResponse:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена: постройте маршрут заново")

    async def job(llm: LLM, emit: Emit) -> tuple[Session, str | None]:
        return await apply_feedback(session, req, llm, emit)

    return _stream(job, _client(request))


def _client(request: Request) -> str:
    # Behind the reverse proxy uvicorn runs with --proxy-headers, so this is the visitor.
    return request.client.host if request.client else "unknown"


def _stream(job: Job, client: str) -> StreamingResponse:
    refusal = limits.admit(client)
    if refusal:
        raise HTTPException(status_code=429, detail=refusal)
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await queue.put(event)

    async def run() -> None:
        llm = base_llm.for_run()
        started = time.monotonic()
        try:
            session, reply = await job(llm, emit)
            data: dict[str, Any] = {
                "session_id": session.id,
                "plan": session.plan.model_dump(mode="json") if session.plan else None,
                "usage": llm.usage.as_dict(),
            }
            if reply:
                data["reply"] = reply
            await emit(AgentEvent(type="plan", data=data))
        except PipelineError as e:
            await emit(AgentEvent(type="error", data={"message": str(e)}))
        except LLMError as e:
            await emit(AgentEvent(type="error", data={"message": f"Ошибка LLM-провайдера: {e}"}))
        except Exception:
            log.exception("pipeline failed")
            await emit(AgentEvent(type="error", data={"message": "Внутренняя ошибка, см. логи"}))
        finally:
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
