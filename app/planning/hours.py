"""Opening hours: is a place open during a visit, and when does it open?

OSM `opening_hours` is a small language ("Mo-Fr 10:00-18:00; Sa 11:00-17:00;
PH off"), so we use a real parser (opening-hours-py) instead of regexes or the
LLM. Every answer is three-valued: open, closed, or unknown (no tag, a tag we
cannot parse, or an explicit "unknown"). Unknown is not the same as closed: the
optimizer lets such places in, and the validator only warns about them.

All datetimes are naive local time of the city, the same as the trip brief.
"""

from datetime import date, datetime, time, timedelta
from functools import lru_cache

from opening_hours import OpeningHours, ParserError, State

Span = tuple[datetime, datetime, State]


@lru_cache(maxsize=1024)
def _parse(expr: str) -> OpeningHours | None:
    try:
        return OpeningHours(expr)
    except ParserError:
        return None


@lru_cache(maxsize=8192)
def _day_states(expr: str | None, day: date) -> tuple[Span, ...] | None:
    """Consecutive (start, end, state) spans covering one calendar day."""
    expr = (expr or "").strip()  # the parser reads a blank string as "24/7"
    oh = _parse(expr) if expr else None
    if oh is None:
        return None
    start = datetime.combine(day, time.min)
    return tuple((s, e, state) for s, e, state, _ in oh.intervals(start, start + timedelta(days=1)))


def _states(expr: str | None, start: datetime, end: datetime) -> list[Span] | None:
    """Spans overlapping [start, end); the visit may cross midnight."""
    spans = []
    day = start.date()
    while day <= end.date():
        states = _day_states(expr, day)
        if states is None:
            return None
        spans += [(s, e, st) for s, e, st in states if s < end and e > start]
        day += timedelta(days=1)
    return spans


def open_intervals(expr: str | None, day: date) -> list[tuple[datetime, datetime]] | None:
    """Open intervals on that day; None when the hours are missing, unparseable or unknown."""
    states = _day_states(expr, day)
    if states is None or any(st == State.UNKNOWN for _, _, st in states):
        return None
    return [(s, e) for s, e, st in states if st == State.OPEN]


@lru_cache(maxsize=65536)
def is_open(expr: str | None, start: datetime, end: datetime) -> bool | None:
    """True if open for the whole [start, end]; False if closed at any moment; None if unknown."""
    spans = _states(expr, start, end)
    if spans is None:
        return None
    states = {st for _, _, st in spans}
    if State.CLOSED in states:
        return False
    if State.UNKNOWN in states:
        return None
    return True


@lru_cache(maxsize=65536)
def earliest_open_at_or_after(expr: str | None, t: datetime, day_end: datetime) -> datetime | None:
    """First moment >= t when the place is open, or None.

    None means "never open before day_end" or "hours unknown"; in both cases the
    caller starts the visit on arrival and leaves the verdict to is_open.
    """
    spans = _states(expr, t, day_end)
    if spans is None:
        return None
    for s, _, st in spans:
        if st == State.OPEN:
            return max(s, t)
    return None
