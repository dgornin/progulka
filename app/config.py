"""Settings read from the environment (and from .env for local runs).

The code reads configuration from the environment and nothing else, so the same
image runs on a laptop and on a server; docker compose passes .env via env_file.
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    # python-dotenv and docker compose strip quotes, `docker run --env-file` does not:
    # API_KEY='sk-...' would reach the provider with the quotes.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model_name: str
    temperature: float
    reasoning: str  # "off" | "low" | "medium" | "high"
    agent_max_steps: int
    user_agent: str
    cache_dir: Path
    overpass_urls: tuple[str, ...]  # tried in order; the last one that answered is tried first
    llm_replay: bool  # serve repeated LLM requests from the on-disk cache (demo insurance)
    # Pins "today": the brief prompt and the dates it resolves depend on it, so without
    # the pin a run rehearsed yesterday does not replay from the cache.
    today: date | None
    # Public deployment guards (app/limits.py); 0 = off.
    rate_limit_per_hour: int
    daily_budget_usd: float
    max_concurrent_runs: int


def load_settings() -> Settings:
    return Settings(
        api_key=_env("API_KEY", ""),
        base_url=_env("BASE_URL", "https://openrouter.ai/api/v1"),
        model_name=_env("MODEL_NAME", ""),
        temperature=float(_env("LLM_TEMPERATURE", "0.2")),
        reasoning=_env("LLM_REASONING", "off"),
        agent_max_steps=int(_env("AGENT_MAX_STEPS", "14")),
        user_agent=_env("HTTP_USER_AGENT", "progulka/0.1 (+https://github.com/dgornin/progulka)"),
        cache_dir=Path(_env("CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cache"))),
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
    )


settings = load_settings()
