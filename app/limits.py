"""Guards for a public deployment: anyone with the link spends the project's API key.

- per-client rate limit: at most N agent runs per hour from one address;
- daily budget: once today's LLM spend reaches the cap, new runs are refused;
- concurrency: at most K runs at once; the rest are refused at once instead of
  queueing behind slow external APIs.

Every limit is off at 0, the default, so local runs are unaffected; the server's .env
turns them on. A refused request never starts the pipeline, so it costs nothing.
Speech transcription spends the same key: it counts toward the daily budget and is
refused once the budget is spent, but it is a single short call, so it takes no run
slot and has a per-address window of its own (app/main.py). Its spend is not saved
with a journey, so a restart forgets it: fractions of a cent per minute of speech.

The limits are per address, not per account: registration is free, so an account is
not a scarce resource, and a limit per account would not bound the spend.
"""

import time
from collections import defaultdict, deque
from datetime import UTC, date, datetime

HOUR_S = 3600.0
MAX_TRACKED_CLIENTS = 10_000
BUDGET_SPENT = "Дневной лимит демо исчерпан, попробуйте завтра"


class SlidingWindow:
    """At most `limit` events per key over the last `window_s` seconds; 0 = no limit.

    Shared by the run limit, the login and sign-up throttles and the voice input window.
    """

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def full(self, key: str) -> bool:
        if not self.limit:
            return False
        now = time.monotonic()
        if len(self._events) > MAX_TRACKED_CLIENTS:  # forget clients idle for a window
            self._events = defaultdict(
                deque,
                {k: e for k, e in self._events.items() if e and now - e[-1] <= self.window_s},
            )
        events = self._events[key]
        while events and now - events[0] > self.window_s:
            events.popleft()
        return len(events) >= self.limit

    def add(self, key: str) -> None:
        if self.limit:
            self._events[key].append(time.monotonic())

    def undo(self, key: str) -> None:
        """Take back the newest event of `key`: it turned out not to count."""
        if events := self._events.get(key):
            events.pop()


class Limits:
    def __init__(self, per_hour: int, daily_budget_usd: float, max_concurrent: int):
        self.per_hour = per_hour
        self.daily_budget_usd = daily_budget_usd
        self.max_concurrent = max_concurrent
        self.running = 0
        self._runs = SlidingWindow(per_hour, HOUR_S)
        self._day: date = datetime.now(UTC).date()
        self._spent_usd = 0.0

    def admit(self, client: str) -> str | None:
        """Register a run and return None, or return why it is refused (in Russian)."""
        if self.out_of_budget():
            return BUDGET_SPENT
        if self.max_concurrent and self.running >= self.max_concurrent:
            return "Сейчас строится слишком много маршрутов, попробуйте через минуту"
        if self._runs.full(client):
            return f"Не больше {self.per_hour} маршрутов в час с одного адреса"
        self._runs.add(client)
        self.running += 1
        return None

    def out_of_budget(self) -> bool:
        self._roll_day()
        return bool(self.daily_budget_usd) and self._spent_usd >= self.daily_budget_usd

    def release(self, cost_usd: float) -> None:
        self.running = max(0, self.running - 1)
        self.spend(cost_usd)

    def spend(self, cost_usd: float) -> None:
        """Count spend against today's budget; at startup, what was spent before a restart."""
        self._roll_day()
        self._spent_usd += cost_usd

    def _roll_day(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day, self._spent_usd = today, 0.0
