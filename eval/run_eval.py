"""Agent vs. the same model with one prompt: are the routes real and walkable, and did
the model hear what the traveller asked for? Plus trips of several days.

Single walks. For every one-day request in queries.json we build two plans with the
SAME model:
- agent: the full pipeline (tools, optimizer, validator); a one-day trip is a single
  walk with no day plan, the same pipeline as a day of a longer trip;
- baseline: one prompt, no tools — "plan a walking route, return JSON".
Then both are checked by the same code:
- real places (located): the stop has coordinates. Agent stops are source records
  with coordinates, so this column is full by construction; a baseline stop counts
  when its name is found in OpenStreetMap (Nominatim) inside the city;
- closed: stops whose known opening hours say "closed" at the planned time;
- impossible legs: the time between two stops is shorter than the walk itself;
- over window: the plan ends after the requested end time;
- walk km: street-distance estimate along the planned order;
- avoid hits: stops that match a phrase of the query's "avoid" (validator.matches_avoid:
  by the place's kind or a word of its name). "avoid" is written by hand from the
  request ("Музеи не надо" -> "музеи"; "big crowds" names no kind of place and is left
  out), not taken from the agent's parsed brief: a check against the parsed brief
  would pass whenever the LLM dropped the wish. For the agent the column therefore
  tests the LLM's brief parsing (did "музеи не надо" become an avoid) together with
  the code filter behind it; the row's brief_avoid shows which of the two failed.
  Baseline stops have no kind of their own: Nominatim's category and type (tourism =
  museum) go through the Overpass category filters (osm.kind_of); a stop that is not
  found, or whose tags no category knows, is matched by the name the model wrote,
  remarks cut off. This is lenient to the baseline: a museum whose top hit is its
  building (historic=palace, not tourism=museum) and whose name does not say "музей"
  is not counted;
- interests covered: k/n, n the interests of the agent's parsed brief, optional ones
  included, k those that at least one stop serves, by the validator's own notion
  (validator.uncovered_interests: a stop's interest labels name the interest). The
  labels are the research agent's claims, so this says whether the plan addresses
  every wish the LLM heard, not whether the places really fit. The baseline has no
  per-stop labels and is not measured ("—"); an LLM-as-judge for both systems, checked
  against hand labels, is planned for the final.
For the agent, located, closed, impossible legs and over window mostly test the code
around the LLM (sources, optimizer, validator); avoid hits and interests covered test
the LLM itself.

Trips of several days: queries with "days". Only the agent runs: the one-prompt
baseline is a single walk by design, and a baseline for a whole trip would need a
prompt of its own, so comparing with it would measure that prompt. Per trip:
- days: planned / requested; "days" is the query's hand label, so a miss is the LLM's
  brief parsing (date..date_to) or the outline;
- stops per day;
- closed, impossible legs, walk km: summed over the days; over window: the days that
  end late;
- repeats: pairs of stops on different days that are one place (the same id, or
  tools.same_place: one Wikidata/Wikipedia link, or one name nearby across sources).
  The pipeline's choice of stops is built to keep it at 0;
- avoid hits and interests covered, as above, over the whole trip;
- LLM calls, cost and seconds of the whole trip.

Results, per model: eval/results/<model>.json (single walks; the slide deck reads its
rows' agent and baseline fields), <model>-journeys.json (trips), <model>.md (both
tables) and <model>.meta.json (the git commit and date of the run that wrote each JSON
file; <model>.json stays a bare list of rows for the deck). A run rewrites only the
files of the kinds of queries it ran, so `--only kazan-3-days` keeps the single walks;
the .md is rebuilt from both JSON files.

Usage (from the repo root, needs .env):
    python -m eval.run_eval                  # all queries, MODEL_NAME from .env
    python -m eval.run_eval --model openai/gpt-6-luna --only kazan-history,kazan-3-days
"""

import argparse
import asyncio
import json
import re
import subprocess
import time as clock
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from app.agent.tools import same_place
from app.config import settings
from app.llm import LLM
from app.models import AgentEvent, GeoPoint, Interest, Place, Plan, Stop, TripBrief
from app.pipeline import Journey, PlanRequest, plan_journey
from app.planning.geo import haversine_m, street_meters, walk_minutes
from app.planning.hours import is_open
from app.planning.validator import matches_avoid, uncovered_interests
from app.sources import geocode, osm
from app.sources.http import SourceError, http

HERE = Path(__file__).parent
TODAY = date(2026, 10, 20)  # the queries name concrete dates; this anchors "today"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
# A match this far from the walk's start is a namesake elsewhere in the region, not the
# place the model meant; counting it would inflate the baseline's walking distance.
MAX_FROM_START_M = 10_000


class BaselineStop(BaseModel):
    name: str
    address: str | None = None
    arrive: str  # "HH:MM"
    depart: str


class BaselinePlan(BaseModel):
    stops: list[BaselineStop]


BASELINE_PROMPT = (
    "You are an experienced local guide. Plan a walking route for the request below: "
    "pick real places, order them sensibly, and give arrival and departure times "
    "(HH:MM, local time) for each stop within the requested time window. "
    "Answer in the request's language."
)


@dataclass
class Point:
    """A stop as the checks see it, whichever system planned it."""

    name: str
    kind: str | None  # a kind of osm.CATEGORIES; None: unknown, the name alone is matched
    lat: float | None
    lon: float | None
    opening_hours: str | None
    arrive: datetime
    depart: datetime


@dataclass
class RouteCheck:
    closed: int
    impossible_legs: int
    over_window: bool
    walk_m: float


@dataclass
class Metrics:
    stops: int
    found: int
    closed: int
    impossible_legs: int
    over_window: bool
    walk_km: float
    avoid_hits: int
    interests_covered: int | None  # None: not measured (the baseline)
    interests_total: int | None
    llm_calls: int
    cost_usd: float
    seconds: float


@dataclass
class JourneyMetrics:
    days_requested: int
    days_planned: int
    stops_per_day: list[int]
    closed: int
    impossible_legs: int
    days_over_window: int
    walk_km: float
    repeats: int
    avoid_hits: int
    interests_covered: int
    interests_total: int
    llm_calls: int
    cost_usd: float
    seconds: float


def check_route(points: list[Point], start: GeoPoint, brief: TripBrief) -> RouteCheck:
    """What code can check in one day's route: hours, legs, the window, the distance."""
    located = [p for p in points if p.lat is not None]
    closed = sum(is_open(p.opening_hours, p.arrive, p.depart) is False for p in located)
    impossible, walk_m = 0, 0.0
    prev: GeoPoint | Point = start
    prev_depart = datetime.combine(brief.date, brief.start_time)
    for p in located:
        meters = street_meters(_geo(prev), _geo(p))
        walk_m += meters
        if (p.arrive - prev_depart).total_seconds() / 60 + 5 < walk_minutes(meters, brief.pace):
            impossible += 1
        prev, prev_depart = p, p.depart
    end = datetime.combine(brief.date, brief.end_time)
    over = bool(points) and max(p.depart for p in points) > end
    return RouteCheck(closed, impossible, over, walk_m)


def avoid_hits(points: Sequence[Point], avoid: Sequence[str]) -> int:
    """Stops that match any phrase of `avoid`, by kind (when known) or by name."""
    return sum(any(matches_avoid(a, p.kind or "", p.name) for a in avoid) for p in points)


def interest_coverage(stops: Sequence[Stop], interests: Sequence[Interest]) -> tuple[int, int]:
    """(covered, total) over all the interests, optional ones too: the validator insists
    only on the must-haves, but any wish the LLM heard is one the plan can miss."""
    return len(interests) - len(uncovered_interests(stops, interests)), len(interests)


def repeats_across_days(days: Sequence[Sequence[Place]]) -> int:
    """Pairs of stops on different days that are one place. same_place compares only
    records from different sources, so the same id is checked on its own."""
    return sum(
        a.id == b.id or same_place(a, b)
        for i, day in enumerate(days)
        for later in days[i + 1 :]
        for a in day
        for b in later
    )


def score(
    points: list[Point],
    start: GeoPoint,
    brief: TripBrief,
    avoid: Sequence[str],
    usage: dict,
    seconds: float,
    coverage: tuple[int, int] | None = None,
) -> Metrics:
    route = check_route(points, start, brief)
    covered, total = coverage or (None, None)
    return Metrics(
        stops=len(points),
        found=sum(p.lat is not None for p in points),
        closed=route.closed,
        impossible_legs=route.impossible_legs,
        over_window=route.over_window,
        walk_km=round(route.walk_m / 1000, 1),
        avoid_hits=avoid_hits(points, avoid),
        interests_covered=covered,
        interests_total=total,
        llm_calls=usage.get("llm_calls", 0),
        cost_usd=round(usage.get("cost_usd", 0.0), 4),
        seconds=round(seconds, 1),
    )


def score_journey(
    journey: Journey, days_requested: int, avoid: Sequence[str], usage: dict, seconds: float
) -> JourneyMetrics:
    """Each day is checked against its own brief (its date), the interests against the
    whole trip's: a wish is served if any day serves it."""
    plans = [day.plan for day in journey.days if day.plan]
    routes = [check_route(agent_points(p), p.start, p.brief) for p in plans]
    covered, total = interest_coverage([s for p in plans for s in p.stops], journey.brief.interests)
    return JourneyMetrics(
        days_requested=days_requested,
        days_planned=len(plans),
        stops_per_day=[len(p.stops) for p in plans],
        closed=sum(r.closed for r in routes),
        impossible_legs=sum(r.impossible_legs for r in routes),
        days_over_window=sum(r.over_window for r in routes),
        walk_km=round(sum(r.walk_m for r in routes) / 1000, 1),
        repeats=repeats_across_days([[s.place for s in p.stops] for p in plans]),
        avoid_hits=avoid_hits([pt for p in plans for pt in agent_points(p)], avoid),
        interests_covered=covered,
        interests_total=total,
        llm_calls=usage.get("llm_calls", 0),
        cost_usd=round(usage.get("cost_usd", 0.0), 4),
        seconds=round(seconds, 1),
    )


def agent_points(plan: Plan) -> list[Point]:
    return [
        Point(
            s.place.name,
            s.place.kind,
            s.place.lat,
            s.place.lon,
            s.place.opening_hours,
            s.arrive,
            s.depart,
        )
        for s in plan.stops
    ]


def _geo(p: GeoPoint | Point) -> GeoPoint:
    return p if isinstance(p, GeoPoint) else GeoPoint(lat=p.lat, lon=p.lon)


async def plan_with_agent(base: LLM, request: str) -> tuple[Journey, dict, float]:
    """The journey, the LLM usage of the run and its wall-clock seconds."""
    llm = base.for_run()

    async def ignore(event: AgentEvent) -> None:
        """The eval scores the result; the progress events are for the UI."""

    started = clock.monotonic()
    journey = await plan_journey(PlanRequest(request=request), llm, ignore, today=TODAY)
    return journey, llm.usage.as_dict(), clock.monotonic() - started


async def run_walk(base: LLM, query: dict, row: dict, with_baseline: bool) -> None:
    """Fills `row` as it goes: a failed baseline keeps the agent's result."""
    journey, usage, seconds = await plan_with_agent(base, query["request"])
    plan = journey.days[0].plan
    coverage = interest_coverage(plan.stops, plan.brief.interests)
    metrics = score(
        agent_points(plan), plan.start, plan.brief, query["avoid"], usage, seconds, coverage
    )
    row["brief_avoid"] = plan.brief.avoid
    row["agent"], row["agent_stops"] = asdict(metrics), [s.place.name for s in plan.stops]
    if with_baseline:
        metrics, names = await run_baseline(base, query, plan.brief, plan.start)
        row["baseline"], row["baseline_stops"] = asdict(metrics), names


async def run_journey(base: LLM, query: dict, row: dict) -> None:
    journey, usage, seconds = await plan_with_agent(base, query["request"])
    metrics = score_journey(journey, query["days"], query["avoid"], usage, seconds)
    row["brief_avoid"] = journey.brief.avoid
    row["agent"] = asdict(metrics)
    row["agent_stops"] = [[s.place.name for s in d.plan.stops] for d in journey.days if d.plan]


async def run_baseline(
    base: LLM, query: dict, brief: TripBrief, start: GeoPoint
) -> tuple[Metrics, list[str]]:
    llm = base.for_run()
    started = clock.monotonic()
    answer = await llm.structured(
        [
            {"role": "system", "content": BASELINE_PROMPT},
            {"role": "user", "content": query["request"]},
        ],
        BaselinePlan,
    )
    city = await geocode.locate_city(brief.city, brief.country)
    points = []
    for stop in answer.stops:
        hit = await _locate(stop, brief.city, city.bbox if city else None, start)
        points.append(baseline_point(stop, hit, brief))
    metrics = score(
        points, start, brief, query["avoid"], llm.usage.as_dict(), clock.monotonic() - started
    )
    return metrics, [s.name for s in answer.stops]


async def _locate(stop: BaselineStop, city: str, bbox, start: GeoPoint) -> dict | None:
    """Find the place in OpenStreetMap (Nominatim) inside the city's bounding box.

    Deliberately lenient, so the comparison does not favour the agent: the name as
    written, the name without quotes and parenthesised remarks, then the address the
    model gave. The first hit inside the city and within 10 km of the start counts.
    """
    queries = [stop.name, _plain_name(stop.name), stop.address]
    for query in dict.fromkeys(q for q in queries if q):
        params = {"q": f"{query}, {city}", "format": "jsonv2", "limit": 1, "extratags": 1}
        if bbox:
            south, west, north, east = bbox
            params.update(viewbox=f"{west},{north},{east},{south}", bounded=1)
        try:
            found = await http.get_json(NOMINATIM, params)
        except SourceError:
            continue
        if found:
            hit = found[0]
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if haversine_m(start.lat, start.lon, lat, lon) <= MAX_FROM_START_M:
                return hit
    return None


def baseline_point(stop: BaselineStop, hit: dict | None, brief: TripBrief) -> Point:
    """The model's stop with what Nominatim knows of it. The name is the model's without
    remarks («Улица Рубинштейна — уличная еда»), so the avoid check reads the place,
    not the model's comment on it."""
    name = _plain_name(stop.name) or stop.name
    arrive, depart = _at(brief, stop.arrive), _at(brief, stop.depart)
    if hit is None:
        return Point(name, None, None, None, None, arrive, depart)
    hours = (hit.get("extratags") or {}).get("opening_hours")
    lat, lon = float(hit["lat"]), float(hit["lon"])
    return Point(name, nominatim_kind(hit), lat, lon, hours, arrive, depart)


def nominatim_kind(hit: dict) -> str | None:
    """The project's kind of a Nominatim (jsonv2) hit: its main tag, category=type, and
    its extra tags through the Overpass category filters. The filters also require a
    name, only to skip what cannot be shown on the map; the model named this place, so
    an unnamed hit gets a placeholder."""
    tags = {**(hit.get("extratags") or {}), "name": hit.get("name") or "?"}
    tags[hit.get("category", "")] = hit.get("type", "")
    return osm.kind_of(tags)


def _plain_name(name: str) -> str:
    """'Дом-коммуна («Слеза социализма»)' -> 'Дом-коммуна'; 'Кул-Шариф — мечеть' -> 'Кул-Шариф'."""
    name = re.sub(r"\(.*?\)", "", name)
    name = re.split(r"\s[—–-]\s|:", name)[0]
    return name.replace("«", "").replace("»", "").replace('"', "").strip()


def _at(brief: TripBrief, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.strip()[:5].split(":"))
    return datetime.combine(brief.date, datetime.min.time().replace(hour=h % 24, minute=m))


def git_commit() -> str:
    """The commit the code ran at, "+dirty" when tracked files differ from it: a result
    can be reproduced only with the code that produced it. Untracked files do not count:
    a new module matters only once a tracked file imports it, and that file differs."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


WALKS_SECTION = (
    "Single walks: agent vs. one prompt",
    "Agent stops are source records with coordinates, so they are located by "
    "construction; for the baseline, located means found in OpenStreetMap. The baseline "
    "has no per-stop interest labels, so its interests are not measured (—).",
)
JOURNEYS_SECTION = (
    "Trips of several days: agent only",
    "The one-prompt baseline plans a single walk by design, so trips run the agent only. "
    "Closed, impossible legs and walk km are summed over the days; repeats are pairs of "
    "stops on different days that are one place.",
)


def walks_table(rows: list[dict]) -> str:
    lines = [
        "| query | system | stops | real places (located) | closed at visit | impossible legs "
        "| over window | avoid hits | interests covered | walk km | LLM calls | cost $ | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for system in ("agent", "baseline"):
            m = r.get(system)
            if m:
                cells = [
                    r["id"],
                    system,
                    m["stops"],
                    f"{m['found']}/{m['stops']}",
                    m["closed"],
                    m["impossible_legs"],
                    "yes" if m["over_window"] else "no",
                    m.get("avoid_hits", "—"),  # rows of runs before the column existed
                    _interests(m),
                    m["walk_km"],
                    m["llm_calls"],
                    m["cost_usd"],
                    m["seconds"],
                ]
                lines.append(_table_row(cells))
        lines += _error_rows(r)
    return "\n".join(lines) + "\n"


def journeys_table(rows: list[dict]) -> str:
    lines = [
        "| query | days (planned/requested) | stops per day | closed at visit | impossible legs "
        "| days over window | repeats across days | avoid hits | interests covered | walk km "
        "| LLM calls | cost $ | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r.get("agent")
        if m:
            cells = [
                r["id"],
                f"{m['days_planned']}/{m['days_requested']}",
                ", ".join(str(n) for n in m["stops_per_day"]),
                m["closed"],
                m["impossible_legs"],
                m["days_over_window"],
                m["repeats"],
                m["avoid_hits"],
                _interests(m),
                m["walk_km"],
                m["llm_calls"],
                m["cost_usd"],
                m["seconds"],
            ]
            lines.append(_table_row(cells))
        lines += _error_rows(r)
    return "\n".join(lines) + "\n"


def _interests(m: dict) -> str:
    covered = m.get("interests_covered")
    return "—" if covered is None else f"{covered}/{m['interests_total']}"


def _error_rows(row: dict) -> list[str]:
    # The first line only: a validation error spans many, the JSON keeps all of them.
    return (
        [_table_row([row["id"], "error", row["error"].splitlines()[0]])] if "error" in row else []
    )


def _table_row(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |"


def write_results(
    out: Path, slug: str, model: str, run: dict, walks: list[dict], journeys: list[dict]
) -> None:
    """Rewrites the JSON file of each kind of query this run ran, records the run in
    <slug>.meta.json, and rebuilds <slug>.md from both JSON files, each table under the
    commit and date of the run that wrote it."""
    out.mkdir(exist_ok=True)
    meta_path = out / f"{slug}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sections = []
    for path, rows, (heading, note), table in (
        (out / f"{slug}.json", walks, WALKS_SECTION, walks_table),
        (out / f"{slug}-journeys.json", journeys, JOURNEYS_SECTION, journeys_table),
    ):
        if rows:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            meta[path.name] = {**run, "queries": [r["id"] for r in rows]}
        elif path.exists():
            rows = json.loads(path.read_text())
        else:
            continue
        info = meta.get(path.name)
        provenance = (
            f"Run {info['run_at']}, commit `{info['commit']}`."
            if info
            else "Run before commits were recorded."
        )
        sections.append(f"## {heading}\n\n{provenance} {note}\n\n{table(rows)}")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    (out / f"{slug}.md").write_text(
        f"Model: `{model}`, queries anchored at {TODAY.isoformat()}. Avoid hits count stops "
        "matching the hand-written avoid of the query (eval/queries.json); the agent's "
        "parsed one is brief_avoid in the JSON.\n\n" + "\n".join(sections)
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated query ids")
    parser.add_argument("--model", help="override MODEL_NAME")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    queries = json.loads((HERE / "queries.json").read_text())
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        unknown = wanted - {q["id"] for q in queries}
        if unknown:
            parser.error(f"unknown query ids: {', '.join(sorted(unknown))}")
        queries = [q for q in queries if q["id"] in wanted]
    base = LLM.from_settings(settings)
    if args.model:
        base.model = args.model
    run = {
        "commit": git_commit(),
        "run_at": datetime.now().astimezone().isoformat(timespec="minutes"),
    }
    walks: list[dict] = []
    journeys: list[dict] = []
    for q in queries:
        row: dict = {"id": q["id"], "model": base.model}
        trip = q.get("days", 1) > 1
        try:
            if trip:
                await run_journey(base, q, row)
            else:
                await run_walk(base, q, row, with_baseline=not args.skip_baseline)
        except Exception as e:  # one failed query must not lose the others' results
            row["error"] = f"{type(e).__name__}: {e}"
        (journeys if trip else walks).append(row)
        print(json.dumps(row, ensure_ascii=False))

    slug = base.model.replace("/", "__").replace(":", "_")
    write_results(HERE / "results", slug, base.model, run, walks, journeys)


if __name__ == "__main__":
    asyncio.run(main())
