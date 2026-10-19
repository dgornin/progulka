"""Guards for a public deployment: anyone with the link spends the project's API key.

- per-client rate limit: at most N agent runs per hour from one address;
- daily budget: once today's LLM spend reaches the cap, new runs are refused;
- concurrency: at most K runs at once; the rest are refused at once instead of
  queueing behind slow external APIs.

Every limit is off at 0, the default, so local runs are unaffected; the server's .env
turns them on. A refused request never starts the pipeline, so it costs nothing.
"""

import time
from collections import defaultdict, deque
from datetime import UTC, date, datetime

HOUR_S = 3600.0
MAX_TRACKED_CLIENTS = 10_000


class Limits:
    def __init__(self, per_hour: int, daily_budget_usd: float, max_concurrent: int):
        self.per_hour = per_hour
        self.daily_budget_usd = daily_budget_usd
        self.max_concurrent = max_concurrent
        self.running = 0
        self._runs: dict[str, deque[float]] = defaultdict(deque)
        self._day: date = datetime.now(UTC).date()
        self._spent_usd = 0.0

    def admit(self, client: str) -> str | None:
        """Register a run and return None, or return why it is refused (in Russian)."""
        now = time.monotonic()
        self._roll_day()
        if self.daily_budget_usd and self._spent_usd >= self.daily_budget_usd:
            return "Дневной лимит демо исчерпан, попробуйте завтра"
        if self.max_concurrent and self.running >= self.max_concurrent:
            return "Сейчас строится слишком много маршрутов, попробуйте через минуту"
        if len(self._runs) > MAX_TRACKED_CLIENTS:  # forget clients idle for an hour
            self._runs = defaultdict(
                deque, {c: r for c, r in self._runs.items() if r and now - r[-1] <= HOUR_S}
            )
        runs = self._runs[client]
        while runs and now - runs[0] > HOUR_S:
            runs.popleft()
        if self.per_hour and len(runs) >= self.per_hour:
            return f"Не больше {self.per_hour} маршрутов в час с одного адреса"
        runs.append(now)
        self.running += 1
        return None

    def release(self, cost_usd: float) -> None:
        self.running = max(0, self.running - 1)
        self._roll_day()
        self._spent_usd += cost_usd

    def _roll_day(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day, self._spent_usd = today, 0.0
