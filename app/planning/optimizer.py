"""Choosing and ordering the stops: deterministic code, not the LLM.

The task is orienteering with time windows: from scored candidates pick a subset
and an order that maximise the total score, while every place is open during its
visit, the walk ends inside the time window and stays under the walking limit.
It is NP-hard, and LLMs are unreliable at this arithmetic, so we use a heuristic
that can be explained in a minute:

1. Required places (named by the traveller) go in first, at their cheapest
   position, even if that breaks a rule: the validator reports it and the agent
   or the traveller decides. What was asked for is never dropped silently.
2. Greedy insertion: try every remaining candidate at every position and take the
   best "value per added minute"; repeat while something fits.
3. Shorten the walk with 2-opt and relocate moves, then fill the freed time (2).
4. Lunch: a walk of 4+ hours spanning 12:00-15:00 without a food stop gets one.

A candidate's value is its score, x0.4 for an outdoor place visited in a rainy
hour and x0.7 for a third stop of the same kind in a row (five churches in a row
make a dull walk). Interests have diminishing returns: each other stop that already
covers the candidate's interest multiplies its value by 0.7, so ten 10-minute
murals do not crowd out the traveller's other interests (a candidate counts by its
least covered interest). The value is evaluated at the position being tried,
because rain and repetition depend on when and where the visit happens.

"Fits" means "breaks no rule the current route does not already break": with no
required places that is simply "breaks nothing", and when a required place is
closed or far away, other stops can still be added around it. Closed stops are
compared as a set, not a count, so a closed required place is never "fixed" by
delaying it with another stop that is closed itself.
Candidates are sorted by place id and ties keep the first, so the same input
always gives the same route.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from itertools import pairwise
from math import ceil

from app.models import Candidate, GeoPoint, HourWeather, Pace, Stop, TripBrief
from app.planning.geo import street_meters, walk_limit_km, walk_minutes
from app.planning.hours import earliest_open_at_or_after, is_open
from app.planning.validator import rain_during

RAIN_FACTOR = 0.4
REPEAT_FACTOR = 0.7
REPEAT_RUN = 3  # the third same-kind stop in a row is penalised
INTEREST_DECAY = 0.7

# Kinds that make a lunch. Bars and ice cream are not a meal; "market" is left out
# because in OSM it is as often a flea market as a food hall.
MEAL_KINDS = frozenset({"cafe", "restaurant", "street_food", "bakery", "food", "pub"})
LUNCH_START, LUNCH_END = time(12, 0), time(15, 0)
LUNCH_MIN_WINDOW = timedelta(hours=4)


@dataclass
class Schedule:
    stops: list[Stop]
    total_walk_m: int
    total_minutes: int


@dataclass
class OptimizeResult:
    order: list[Candidate]
    dropped: list[str]  # place ids of candidates that did not make it into the route


@dataclass(frozen=True)
class _Visit:
    walk_m: float
    walk_min: int
    begin: datetime  # visit start: arrival, or opening time if we came early
    end: datetime


def _timeline(
    order: list[Candidate], legs_m: list[float], t0: datetime, day_end: datetime, pace: Pace
) -> list[_Visit]:
    """Walk the order leg by leg; wait at the door if a place is not open yet."""
    visits = []
    t = t0
    for cand, meters in zip(order, legs_m, strict=True):
        walk = ceil(walk_minutes(meters, pace))  # whole minutes, rounded up: times stay readable
        arrive = t + timedelta(minutes=walk)
        # None = never opens or hours unknown: start on arrival, is_open() gives the verdict.
        begin = earliest_open_at_or_after(cand.place.opening_hours, arrive, day_end) or arrive
        t = begin + timedelta(minutes=cand.visit_minutes)
        visits.append(_Visit(meters, walk, begin, t))
    return visits


def schedule(
    order: list[Candidate], start: GeoPoint, brief: TripBrief, leg_meters: list[float] | None = None
) -> Schedule:
    """Times for a fixed order; leg_meters from real routing replace the estimate."""
    t0 = datetime.combine(brief.date, brief.start_time)
    if leg_meters is None:
        points = [start, *(c.place for c in order)]
        leg_meters = [street_meters(a, b) for a, b in pairwise(points)]
    visits = _timeline(
        order, leg_meters, t0, datetime.combine(brief.date, brief.end_time), brief.pace
    )
    stops = [
        Stop(
            place=c.place,
            arrive=v.begin,
            depart=v.end,
            walk_minutes_from_prev=v.walk_min,
            walk_meters_from_prev=round(v.walk_m),
            score=c.score,
            interests=list(c.interests),
            reason=c.reason,
        )
        for c, v in zip(order, visits, strict=True)
    ]
    return Schedule(stops, round(sum(leg_meters)), _minutes(visits[-1].end - t0) if visits else 0)


def optimize(
    candidates: list[Candidate],
    start: GeoPoint,
    brief: TripBrief,
    weather: list[HourWeather] | None = None,
) -> OptimizeResult:
    """Pick and order the stops: steps 1-4 of the module docstring."""
    planner = _Planner(candidates, start, brief, weather)
    route = planner.fill(planner.insert_required())
    route = planner.fill(planner.shorten(route))
    route = planner.ensure_lunch(route)
    order = [planner.cands[i] for i in route.route]
    chosen = {c.place.id for c in order}
    return OptimizeResult(order, [c.place.id for c in candidates if c.place.id not in chosen])


@dataclass(frozen=True)
class _Trial:
    """A route (indices into _Planner.cands) with its timeline and the rules it breaks."""

    route: list[int]
    visits: list[_Visit]
    walk_m: float
    minutes: int
    breaks: tuple[frozenset[int], int, float]  # closed stops, minutes after window, metres over

    def fits(self, base: "_Trial") -> bool:
        # For the set of closed stops "<=" is "is a subset of".
        return all(new <= old for new, old in zip(self.breaks, base.breaks, strict=True))


class _Planner:
    def __init__(
        self,
        candidates: list[Candidate],
        start: GeoPoint,
        brief: TripBrief,
        weather: list[HourWeather] | None,
    ):
        self.cands = sorted(candidates, key=lambda c: c.place.id)
        places = [c.place for c in self.cands]
        self.from_start = [street_meters(start, p) for p in places]
        self.dist = [[street_meters(a, b) for b in places] for a in places]
        self.t0 = datetime.combine(brief.date, brief.start_time)
        self.day_end = datetime.combine(brief.date, brief.end_time)
        self.pace = brief.pace
        self.max_walk_m = walk_limit_km(brief) * 1000
        self.weather = weather

    def evaluate(self, route: list[int]) -> _Trial:
        legs = self._legs(route)
        visits = _timeline([self.cands[i] for i in route], legs, self.t0, self.day_end, self.pace)
        closed = frozenset(
            i
            for i, v in zip(route, visits, strict=True)
            if is_open(self.cands[i].place.opening_hours, v.begin, v.end) is False
        )
        last = visits[-1].end if visits else self.t0
        walk_m = sum(legs)
        breaks = (closed, max(0, _minutes(last - self.day_end)), max(0.0, walk_m - self.max_walk_m))
        return _Trial(route, visits, walk_m, _minutes(last - self.t0), breaks)

    def insert_required(self) -> _Trial:
        """Each required place at its cheapest position, preferring positions that fit."""
        base = self.evaluate([])
        for i, cand in enumerate(self.cands):
            if cand.required:
                options = [
                    self.evaluate(_insert(base.route, p, i)) for p in range(len(base.route) + 1)
                ]
                base = min(options, key=lambda t: (not t.fits(base), t.minutes))
        return base

    def fill(self, base: _Trial) -> _Trial:
        """Greedy insertion while some candidate fits."""
        while (better := self._best_insertion(base, range(len(self.cands)))) is not None:
            base = better
        return base

    def shorten(self, base: _Trial) -> _Trial:
        """2-opt and relocate moves, first improvement, while the walk gets shorter."""
        improved = True
        while improved:
            improved = False
            for route in _neighbours(base.route):
                if sum(self._legs(route)) < base.walk_m - 1:  # 1 m: ignore float noise
                    trial = self.evaluate(route)
                    if trial.fits(base):
                        base, improved = trial, True
                        break
        return base

    def ensure_lunch(self, base: _Trial) -> _Trial:
        """A long walk over lunchtime with no food stop gets one that starts at 12:00-15:00.

        If none fits, the stop with the lowest value per minute (not a required one)
        makes room; if a food stop still does not fit, the route stays as it was.
        """
        long_walk = self.day_end - self.t0 >= LUNCH_MIN_WINDOW
        spans_lunch = self.t0.time() <= LUNCH_START and self.day_end.time() >= LUNCH_END
        has_food = any(self.cands[i].place.kind in MEAL_KINDS for i in base.route)
        if has_food or not (long_walk and spans_lunch):
            return base
        food = [i for i, c in enumerate(self.cands) if c.place.kind in MEAL_KINDS]
        lunch = self._best_insertion(base, food, self._starts_at_lunch)
        removable = [p for p, i in enumerate(base.route) if not self.cands[i].required]
        if lunch is None and removable:
            weakest = min(removable, key=lambda p: self._removal_ratio(base, p))
            lighter = self.evaluate(base.route[:weakest] + base.route[weakest + 1 :])
            # The stops after the removed one come earlier and may hit their own break.
            if lighter.fits(base):
                lunch = self._best_insertion(lighter, food, self._starts_at_lunch)
        return base if lunch is None else self.fill(lunch)

    def _best_insertion(
        self, base: _Trial, pool: Iterable[int], accept: Callable[[_Trial, int], bool] | None = None
    ) -> _Trial | None:
        """The fitting insertion of a pool candidate with the best value per added minute."""
        best, best_ratio = None, 0.0
        for i in pool:
            if i in base.route:
                continue
            for p in range(len(base.route) + 1):
                trial = self.evaluate(_insert(base.route, p, i))
                if not trial.fits(base) or (accept is not None and not accept(trial, p)):
                    continue
                ratio = self._value(trial, p) / (trial.minutes - base.minutes + 1)
                if ratio > best_ratio:
                    best, best_ratio = trial, ratio
        return best

    def _value(self, trial: _Trial, p: int) -> float:
        """Score of the stop at position p, adjusted for rain and repetition."""
        cand, visit = self.cands[trial.route[p]], trial.visits[p]
        value = cand.score
        if cand.place.indoor is False and rain_during(self.weather, visit.begin, visit.end):
            value *= RAIN_FACTOR
        kinds = [self.cands[i].place.kind for i in trial.route]
        run = 1
        for step in (-1, 1):
            q = p + step
            while 0 <= q < len(kinds) and kinds[q] == kinds[p]:
                run, q = run + 1, q + step
        if run >= REPEAT_RUN:
            value *= REPEAT_FACTOR
        others = [self.cands[i].interests for q, i in enumerate(trial.route) if q != p]
        covered = min(
            (sum(label in labels for labels in others) for label in cand.interests), default=0
        )
        return value * INTEREST_DECAY**covered

    def _removal_ratio(self, trial: _Trial, p: int) -> float:
        without = self.evaluate(trial.route[:p] + trial.route[p + 1 :])
        return self._value(trial, p) / (trial.minutes - without.minutes + 1)

    @staticmethod
    def _starts_at_lunch(trial: _Trial, p: int) -> bool:
        return LUNCH_START <= trial.visits[p].begin.time() <= LUNCH_END

    def _legs(self, route: list[int]) -> list[float]:
        if not route:
            return []
        return [self.from_start[route[0]], *(self.dist[a][b] for a, b in pairwise(route))]


def _insert(route: list[int], p: int, i: int) -> list[int]:
    return route[:p] + [i] + route[p:]


def _neighbours(route: list[int]) -> Iterator[list[int]]:
    """Routes one move away: reverse a segment (2-opt), or move one stop elsewhere (relocate)."""
    n = len(route)
    for a in range(n - 1):
        for b in range(a + 1, n):
            yield route[:a] + route[a : b + 1][::-1] + route[b + 1 :]
    for a in range(n):
        rest = route[:a] + route[a + 1 :]
        for b in range(n):
            if b != a:
                yield _insert(rest, b, route[a])


def _minutes(delta: timedelta) -> int:
    return int(delta.total_seconds() // 60)
