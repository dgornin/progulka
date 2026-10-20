"""Shared HTTP client for the open data sources.

- one User-Agent that identifies the app (Nominatim and Wikimedia require it);
- for the rate-limited hosts (Nominatim allows 1 request/second, Overpass a few
  at a time per IP) one request at a time with a minimum interval; other hosts
  (Wikipedia, Wikivoyage) get the agent's parallel tool calls in parallel;
- on-disk cache keyed by method + URL + params + body, so the same city is not
  fetched twice and a rehearsed demo does not depend on the network;
- retries with backoff on 429/5xx and timeouts; callers with mirrors (Overpass)
  ask for one short attempt instead and fail over themselves. A mirror that fails
  so is skipped for MIRROR_DOWN_S (HostDown): the requests queued behind it on its
  host's lock would otherwise each wait out the timeout before trying the next
  mirror. A `probe` request is sent all the same, and an answer ends the skipping:
  the caller probes when every mirror is skipped, so one expensive query that timed
  out everywhere cannot take the source away from everyone for MIRROR_DOWN_S.
"""

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import settings

# Seconds between two requests to the same rate-limited host.
MIN_INTERVAL = {
    "nominatim.openstreetmap.org": 1.1,
    "routing.openstreetmap.de": 0.5,
    **{urlparse(url).netloc: 1.0 for url in settings.overpass_urls},
}
CACHE_TTL_S = 7 * 24 * 3600
RETRY_STATUSES = {429, 500, 502, 503, 504}
MIRROR_DOWN_S = 60.0


class SourceError(RuntimeError):
    """A data source failed after retries; the agent sees this as a tool error.

    `status`: the HTTP status of the last answer; None when none came (a transport
    error or timeout), which callers with a breaker of their own tell apart from a
    4xx about the request itself."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class HostDown(SourceError):
    """Not sent: the mirror failed less than MIRROR_DOWN_S ago."""


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
        self._down_until: dict[str, float] = {}  # mirror host -> when to try it again

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        attempts: int = 3,
        cacheable: Callable[[Any], bool] | None = None,
    ) -> Any:
        """`timeout_s` and `attempts` let a caller whose answer is optional fail fast."""
        return await self._request(
            "GET", url, params=params, timeout_s=timeout_s, attempts=attempts, cacheable=cacheable
        )

    async def post_form_json(
        self,
        url: str,
        data: dict[str, Any],
        *,
        timeout_s: float | None = None,
        attempts: int = 3,
        cache_as: str | None = None,
        cacheable: Callable[[Any], bool] | None = None,
        probe: bool = False,
    ) -> Any:
        """`cache_as` replaces the URL in the cache key, so mirrors share one cache entry;
        `cacheable` keeps error payloads that arrive with HTTP 200 out of the cache;
        `probe` sends the request even to a mirror that is skipped as down."""
        return await self._request(
            "POST",
            url,
            data=data,
            timeout_s=timeout_s,
            attempts=attempts,
            cache_as=cache_as,
            cacheable=cacheable,
            probe=probe,
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
        probe: bool = False,
    ) -> Any:
        key = self._key(method, cache_as or url, params, data)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0)) if timeout_s else None

        host = urlparse(url).netloc
        gate = (
            self._locks.setdefault(host, asyncio.Lock())
            if host in MIN_INTERVAL
            else contextlib.nullcontext()
        )
        delay = 2.0
        for attempt in range(1, attempts + 1):
            async with gate:
                # Checked once the gate is ours: the request may have queued behind the
                # one that found the mirror down.
                if not probe and self._down_until.get(host, 0) > time.monotonic():
                    raise HostDown(f"{host}: failed a moment ago, skipped")
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
                mirror = attempts == 1  # the caller fails over to another mirror
                if mirror and (resp is None or resp.status_code in RETRY_STATUSES):
                    self._down_until[host] = time.monotonic() + MIRROR_DOWN_S
            if resp is not None and resp.status_code == 200:
                self._down_until.pop(host, None)  # it answers again
                try:
                    payload = resp.json()
                except ValueError as e:
                    raise SourceError(f"{host}: response is not JSON") from e
                if cacheable is None or cacheable(payload):
                    self._cache_write(key, payload)
                return payload
            status = resp.status_code if resp is not None else None
            if status is not None:
                error = f"{host}: HTTP {status}"
                if status not in RETRY_STATUSES:
                    break
            if attempt < attempts:
                await asyncio.sleep(delay)
                delay *= 2
        raise SourceError(error, status)

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
