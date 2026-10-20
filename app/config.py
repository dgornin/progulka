"""Settings read from the environment (and from .env for local runs).

The code reads configuration from the environment and nothing else, so the same
image runs on a laptop and on a server; docker compose passes .env via env_file.
"""

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    # python-dotenv and docker compose strip quotes, `docker run --env-file` does not:
    # API_KEY='sk-...' would reach the provider with the quotes.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    # Secrets stay out of repr(): a traceback or a log line that prints the settings
    # must not print the key or the passwords.
    api_key: str = field(repr=False)
    base_url: str
    model_name: str
    # Transcribes voice input: a chat model that accepts audio (app/agent/speech.py).
    stt_model: str
    temperature: float
    reasoning: str  # "off" | "low" | "medium" | "high"
    agent_max_steps: int
    # The trip planner agent allocates the found places to the days of a trip of several
    # days (app/agent/planner.py); off, code assembles them: the eval's ablation, whose
    # results go to eval/results/<model>-noplanner.*.
    trip_planner_agent: bool
    user_agent: str
    cache_dir: Path
    overpass_urls: tuple[str, ...]  # tried in order; the last one that answered is tried first
    llm_replay: bool  # record LLM answers on disk and replay them (demo insurance)
    # Pins "today": the brief prompt and the dates it resolves depend on it, so without
    # the pin a run rehearsed yesterday does not replay from the cache.
    today: date | None
    # Public deployment guards (app/limits.py); 0 = off.
    rate_limit_per_hour: int
    daily_budget_usd: float
    max_concurrent_runs: int
    # SQLite file in DATA_DIR (a mounted volume in Docker) unless DATABASE_URL is set.
    database_url: str
    cookie_secure: bool  # behind HTTPS: the session cookie is never sent over plain HTTP
    # Accounts created (or whose password is reset to these values) at startup, so an
    # operator rotates them by editing .env. Empty = no such account.
    demo_email: str
    demo_password: str = field(repr=False)
    admin_email: str
    admin_password: str = field(repr=False)


def load_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(REPO / ".data")))
    return Settings(
        api_key=_env("API_KEY", ""),
        base_url=_env("BASE_URL", "https://openrouter.ai/api/v1"),
        model_name=_env("MODEL_NAME", ""),
        stt_model=_env("STT_MODEL", "qwen/qwen3.8-omni-flash"),
        temperature=float(_env("LLM_TEMPERATURE", "0.2")),
        reasoning=_env("LLM_REASONING", "off"),
        agent_max_steps=int(_env("AGENT_MAX_STEPS", "14")),
        trip_planner_agent=_env("TRIP_PLANNER_AGENT", "1") != "0",
        user_agent=_env("HTTP_USER_AGENT", "progulka/0.1 (+https://github.com/dgornin/progulka)"),
        cache_dir=Path(_env("CACHE_DIR", str(REPO / ".cache"))),
        overpass_urls=tuple(
            url.strip()
            for url in _env(
                "OVERPASS_URLS",
                "https://overpass-api.de/api/interpreter,"
                "https://overpass.openstreetmap.fr/api/interpreter,"
                "https://overpass.private.coffee/api/interpreter",
            ).split(",")
            if url.strip()
        ),
        llm_replay=_env("LLM_REPLAY", "0") == "1",
        today=date.fromisoformat(pinned) if (pinned := _env("TODAY", "")) else None,
        rate_limit_per_hour=int(_env("RATE_LIMIT_PER_HOUR", "0")),
        daily_budget_usd=float(_env("DAILY_BUDGET_USD", "0")),
        max_concurrent_runs=int(_env("MAX_CONCURRENT_RUNS", "0")),
        database_url=_env("DATABASE_URL", f"sqlite+aiosqlite:///{data_dir / 'progulka.db'}"),
        cookie_secure=_env("COOKIE_SECURE", "0") == "1",
        demo_email=_env("DEMO_EMAIL", ""),
        demo_password=_env("DEMO_PASSWORD", ""),
        admin_email=_env("ADMIN_EMAIL", ""),
        admin_password=_env("ADMIN_PASSWORD", ""),
    )


settings = load_settings()
