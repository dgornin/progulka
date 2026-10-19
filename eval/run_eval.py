"""Agent vs. the same model with one prompt: are the routes real and walkable?

For every request in queries.json we build two plans with the SAME model:
- agent: the full pipeline (tools, optimizer, validator);
- baseline: one prompt, no tools — "plan a walking route, return JSON".
Then both are checked by the same code:
- found: the place can be located in OpenStreetMap inside the city (agent places
  come from the sources, baseline places are geocoded by name);
- closed: stops whose known opening hours say "closed" at the planned time;
- impossible legs: the time between two stops is shorter than the walk itself;
- over window: the plan ends after the requested end time;
- walk km: street-distance estimate along the planned order.

Usage (from the repo root, needs .env):
    python -m eval.run_eval                  # all queries
    python -m eval.run_eval --only kazan-history --skip-baseline
Writes eval/results.json and eval/results.md.
"""

import argparse
import asyncio
import json
import time as clock
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.llm import LLM
from app.models import AgentEvent, GeoPoint, TripBrief
from app.pipeline import PlanRequest, plan_trip
from app.planning.geo import street_meters, walk_minutes
from app.planning.hours import is_open
from app.sources import geocode
from app.sources.http import SourceError, http

HERE = Path(__file__).parent
TODAY = date(2026, 10, 20)  # the queries name concrete dates; this anchors "today"
NOMINATIM = "https://nominatim.openstreetmap.org/search"


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
    name: str
    lat: float | None
    lon: float | None
    opening_hours: str | None
    arrive: datetime
    depart: datetime


@dataclass
class Metrics:
    stops: int
    found: int
    closed: int
    impossible_legs: int
    over_window: bool
    walk_km: float
    llm_calls: int
    cost_usd: float
    seconds: float


def score(
    points: list[Point], start: GeoPoint, brief: TripBrief, usage: dict, seconds: float
) -> Metrics:
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
    return Metrics(
        stops=len(points),
        found=len(located),
        closed=closed,
        impossible_legs=impossible,
        over_window=bool(points) and max(p.depart for p in points) > end,
        walk_km=round(walk_m / 1000, 1),
        llm_calls=usage.get("llm_calls", 0),
        cost_usd=round(usage.get("cost_usd", 0.0), 4),
        seconds=round(seconds, 1),
    )


def _geo(p: GeoPoint | Point) -> GeoPoint:
    return p if isinstance(p, GeoPoint) else GeoPoint(lat=p.lat, lon=p.lon)


async def run_agent(base: LLM, request: str) -> tuple[Metrics, TripBrief, GeoPoint, list[str]]:
    llm = base.for_run()
    events: list[AgentEvent] = []

    async def collect(event: AgentEvent) -> None:
        events.append(event)

    started = clock.monotonic()
    session = await plan_trip(PlanRequest(request=request), llm, collect, today=TODAY)
    plan = session.plan
    points = [
        Point(s.place.name, s.place.lat, s.place.lon, s.place.opening_hours, s.arrive, s.depart)
        for s in plan.stops
    ]
    metrics = score(
        points, plan.start, plan.brief, llm.usage.as_dict(), clock.monotonic() - started
    )
    return metrics, plan.brief, plan.start, [s.place.name for s in plan.stops]


async def run_baseline(
    base: LLM, request: str, brief: TripBrief, start: GeoPoint
) -> tuple[Metrics, list[str]]:
    llm = base.for_run()
    started = clock.monotonic()
    answer = await llm.structured(
        [{"role": "system", "content": BASELINE_PROMPT}, {"role": "user", "content": request}],
        BaselinePlan,
    )
    city = await geocode.locate_city(brief.city, brief.country)
    points = []
    for stop in answer.stops:
        lat, lon, hours = await _locate(stop, brief.city, city.bbox if city else None)
        points.append(
            Point(stop.name, lat, lon, hours, _at(brief, stop.arrive), _at(brief, stop.depart))
        )
    metrics = score(points, start, brief, llm.usage.as_dict(), clock.monotonic() - started)
    return metrics, [s.name for s in answer.stops]


async def _locate(
    stop: BaselineStop, city: str, bbox
) -> tuple[float | None, float | None, str | None]:
    """Find the place by name in OpenStreetMap (Nominatim), inside the city's bounding box."""
    params = {"q": f"{stop.name}, {city}", "format": "jsonv2", "limit": 1, "extratags": 1}
    if bbox:
        south, west, north, east = bbox
        params.update(viewbox=f"{west},{north},{east},{south}", bounded=1)
    try:
        found = await http.get_json(NOMINATIM, params)
    except SourceError:
        return None, None, None
    if not found:
        return None, None, None
    hit = found[0]
    return float(hit["lat"]), float(hit["lon"]), (hit.get("extratags") or {}).get("opening_hours")


def _at(brief: TripBrief, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.strip()[:5].split(":"))
    return datetime.combine(brief.date, datetime.min.time().replace(hour=h % 24, minute=m))


def to_markdown(rows: list[dict]) -> str:
    head = (
        "| query | system | stops | found in OSM | closed at visit | impossible legs "
        "| over window | walk km | LLM calls | cost $ | s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        for system in ("agent", "baseline"):
            m = r.get(system)
            if not m:
                continue
            cells = [
                r["id"],
                system,
                m["stops"],
                f"{m['found']}/{m['stops']}",
                m["closed"],
                m["impossible_legs"],
                "yes" if m["over_window"] else "no",
                m["walk_km"],
                m["llm_calls"],
                m["cost_usd"],
                m["seconds"],
            ]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return head + "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run one query id")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    queries = json.loads((HERE / "queries.json").read_text())
    if args.only:
        queries = [q for q in queries if q["id"] == args.only]
    base = LLM.from_settings(settings)
    rows = []
    for q in queries:
        row: dict = {"id": q["id"], "model": base.model}
        try:
            metrics, brief, start, names = await run_agent(base, q["request"])
            row["agent"], row["agent_stops"] = asdict(metrics), names
            if not args.skip_baseline:
                metrics, names = await run_baseline(base, q["request"], brief, start)
                row["baseline"], row["baseline_stops"] = asdict(metrics), names
        except Exception as e:  # one failed query must not lose the others' results
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    (HERE / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (HERE / "results.md").write_text(
        f"Model: `{base.model}`, run on {TODAY.isoformat()}.\n\n" + to_markdown(rows)
    )


if __name__ == "__main__":
    asyncio.run(main())
