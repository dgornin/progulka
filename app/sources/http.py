"""Shared HTTP client for the open data sources.

- one User-Agent that identifies the app (Nominatim and Wikimedia require it);
- per-host minimum interval between requests (Nominatim allows 1 request/second);
- on-disk cache keyed by method + URL + params + body, so the same city is not
  fetched twice and a rehearsed demo does not depend on the network;
- retries with backoff on 429/5xx and timeouts; callers with mirrors (Overpass)
  ask for one short attempt instead and fail over themselves.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings

# Seconds between two requests to the same host.
MIN_INTERVAL = {
    "nominatim.openstreetmap.org": 1.1,
    "overpass-api.de": 1.0,
    "overpass.openstreetmap.fr": 1.0,
    "overpass.private.coffee": 1.0,
    "routing.openstreetmap.de": 0.5,
}
CACHE_TTL_S = 7 * 24 * 3600
RETRY_STATUSES = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    """A data source failed after retries; the agent sees this as a tool error."""


class HttpClient:
    def __init__(self, cache_dir: Path | None, user_agent: str):
        self.cache_dir = cache_dir
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept-Language": "ru,en"},
            timeout=httpx.Timeout(40.0, connect=10.0),
            follow_redirects=True,
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", url, params=params)

    async def post_form_json(
        self,
        url: str,
        data: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int = 3,
        cache_as: str | None = None,
        cacheable: Callable[[Any], bool] | None = None,
    ) -> Any:
        """`cache_as` replaces the URL in the cache key, so mirrors share one cache entry;
        `cacheable` keeps error payloads that arrive with HTTP 200 out of the cache."""
        return await self._request(
            "POST",
            url,
            data=data,
            timeout_s=timeout_s,
            attempts=attempts,
            cache_as=cache_as,
            cacheable=cacheable,
        )

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        attempts: int = 3,
        cache_as: str | None = None,
        cacheable: Callable[[Any], bool] | None = None,
    ) -> Any:
        key = self._key(method, cache_as or url, params, data)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)) if timeout_s else None

        host = urlparse(url).netloc
        lock = self._locks.setdefault(host, asyncio.Lock())
        delay = 2.0
        for attempt in range(1, attempts + 1):
            async with lock:
                wait = MIN_INTERVAL.get(host, 0) - (time.monotonic() - self._last.get(host, 0))
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    resp = await self._client.request(
                        method,
                        url,
                        params=params,
                        data=data,
                        **({"timeout": timeout} if timeout else {}),
                    )
                except httpx.TransportError as e:
                    resp = None
                    error = f"{host}: {type(e).__name__}"
                finally:
                    self._last[host] = time.monotonic()
            if resp is not None and resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError as e:
                    raise SourceError(f"{host}: response is not JSON") from e
                if cacheable is None or cacheable(payload):
                    self._cache_write(key, payload)
                return payload
            if resp is not None:
                error = f"{host}: HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < attempts:
                await asyncio.sleep(delay)
                delay *= 2
        raise SourceError(error)

    @staticmethod
    def _key(method: str, url: str, params: Any, data: Any) -> str:
        blob = json.dumps([method, url, params, data], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_read(self, key: str) -> Any | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / "http" / f"{key}.json"
        if not path.exists() or time.time() - path.stat().st_mtime > CACHE_TTL_S:
            return None
        return json.loads(path.read_text())

    def _cache_write(self, key: str, payload: Any) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / "http" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False))


http = HttpClient(settings.cache_dir, settings.user_agent)
