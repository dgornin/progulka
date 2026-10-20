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
  against hand labels, is planned for the final;
- mean pop, top share: how well known the stops are, on the absolute 0..1 scale of
  app/sources/wikidata.py (Wikipedia page views and language versions). Mean pop is over
  the stops that have a score, top share counts the stops with pop >= wikidata.TOP among
  the stops whose popularity is known: scored, or without a Wikidata id (not a must-see).
  A stop whose id the lookup failed to score is left out, and when no stop with an id
  is scored both are "—": a failed lookup is unknown, not "no must-sees". Both systems
  are scored by the same lookup of the same Wikidata ids: an agent stop carries the id
  of its source record (OSM, Wikipedia, Wikivoyage), a baseline stop the one of its
  Nominatim hit by name (the OSM wikidata tag in extratags), so a baseline stop that is
  not found, found only by its address, or whose OSM object has no id, has no score. Neither
  is better the higher: a first visit should include a city's must-sees, a request for
  places off the beaten track should not; the numbers show which the plan did.
For the agent, located, closed, impossible legs and over window mostly test the code
around the LLM (sources, optimizer, validator); avoid hits and interests covered test
the LLM itself.

Harness (--harness MODEL, repeatable): a general chat assistant with web search, as a
traveller gets by typing the request into ChatGPT or Claude with browsing. One request to
MODEL with OpenRouter's web search server tool, not its "web" plugin: the plugin searches
once with the request as the query, the tool lets the model decide when and what to search,
as the chat apps do. The provider's native search, the one its own chat app uses, at most
HARNESS_SEARCH's max_uses searches a request, with the agent's reasoning setting unless
--harness-reasoning names another (Sonnet and Opus 5.5 cannot turn reasoning off; "default"
runs a model as its chat app does). The prompt is the baseline's, plus today's date (a chat
app shows it) and an invitation to check the places on the web. The answer is prose, as a
chat app shows it; the agent's model copies it into the baseline's JSON, and the same code
checks it. Tokens, cost (OpenRouter's, search fees included) and seconds are the chat
answer's, what the traveller pays and waits for, not the copy's; the baseline's seconds, as
in earlier runs, include the eval's OpenStreetMap lookups (about a second a stop). Tokens
are counted for every system from now on; runs before that have none. The harness is scored
against the brief and start of the agent's run. With --skip-agent the agent and the
baseline are not rerun and their rows in <model>.json are reused: the brief and start come
from the row, or, for rows written before they were recorded, from an earlier harness row or
the pipeline's brief step and start point again (the requests name the city, date and
window, so the new reading is the one the agent was scored against; every harness model is
scored against one reading). Single walks only, as the baseline.

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
- avoid hits, interests covered, mean pop and top share, as above, over the whole trip;
- LLM calls, cost and seconds of the whole trip;
- planner (JSON only): how the trip planner agent ended, its stop reason and steps, and
  fallback: whether code assembled the days after all; null when it did not run.
The ablation is a run with TRIP_PLANNER_AGENT=0: code assembles every trip's days, and the
results go to <model>-noplanner.* files beside the planner's, each run's meta saying which.

Results, per model: eval/results/<model>.json (single walks; the slide deck reads its
rows' agent and baseline fields), <model>-journeys.json (trips), <model>.md (both
tables) and <model>.meta.json (the git commit and date of the run that wrote each JSON
file; <model>.json stays a bare list of rows for the deck). A run rewrites only the
files of the kinds of queries it ran, so `--only kazan-3-days` keeps the single walks;
the .md is rebuilt from both JSON files. Harness rows of every harness model go to
<model>-harness.json, <model> being the agent's; a run replaces the rows of the models and
queries it ran and keeps the others (the meta lists each harness model's runs, each with
the queries whose rows it still holds), and the .md gets the harness table and a summary
of all the systems over the walks every system answered.

Usage (from the repo root, needs .env):
    python -m eval.run_eval                  # all queries, MODEL_NAME from .env
    python -m eval.run_eval --model openai/gpt-6-luna --only kazan-history,kazan-3-days
    python -m eval.run_eval --skip-agent --harness anthropic/claude-haiku-5.5 \\
        --harness anthropic/claude-sonnet-5.5   # the agent's walks from eval/results
"""

import argparse
import asyncio
import json
import re
import subprocess
import time as clock
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from app.agent.brief import parse_brief
from app.agent.tools import same_place
from app.config import settings
from app.llm import LLM
from app.models import AgentEvent, GeoPoint, Interest, Place, Plan, Stop, TripBrief
from app.pipeline import PLANNER_FALLBACK, Journey, PlanRequest, plan_journey
from app.planning.geo import haversine_m, street_meters, walk_minutes
from app.planning.hours import is_open
from app.planning.validator import matches_avoid, uncovered_interests
from app.sources import geocode, osm, wikidata
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
HARNESS_PROMPT = (
    " Today is {today}. You can search the web: use it to make sure the places exist and to "
    "check where they are and when they are open."
)
# The chat answer is prose, as a chat app shows it, and the eval copies it into the
# baseline's JSON. Asked for the JSON itself, the chat model answered at once without a
# single search: the format instruction outweighed the invitation to search.
COPY_PROMPT = (
    "Copy the walking route below into JSON exactly as written: the same stops in the same "
    "order, each with the address and the arrival and departure times the route gives. Do not "
    "add, drop, rename or move a stop. Where the route offers alternatives, keep the first; "
    "a stop without times gets the ones its neighbours imply."
)
# The cap bounds a run's cost: a frontier model's searches are its most expensive input.
HARNESS_SEARCH = {
    "type": "openrouter:web_search",
    "parameters": {"engine": "native", "max_uses": 5, "max_results": 5},
}
HARNESS_TIMEOUT_S = 180  # per attempt: several searches, then a long answer
HARNESS_MAX_TOKENS = 16000  # a chat answer, and the thinking of a model that must think


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
    wikidata: str | None = None  # the Wikidata id, for popularity


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
    mean_pop: float | None = None  # None: no stop has a score
    top_share: float | None = None  # None: no stops, or no stop with an id was scored
    prompt_tokens: int | None = None  # None: rows of runs before tokens were recorded
    completion_tokens: int | None = None
    web_searches: int | None = None  # the harness only: the searches the provider ran
    cited_pages: int | None = None  # the harness only: distinct URLs, when the engine cites


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
    mean_pop: float | None = None
    top_share: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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


async def stop_popularity(points: Iterable[Point]) -> dict[str, float]:
    """Wikidata id -> popularity, for the stops' ids that could be scored."""
    return await wikidata.scores(p.wikidata for p in points if p.wikidata)


def popularity_stats(
    points: Sequence[Point], popularity: Mapping[str, float]
) -> tuple[float | None, float | None]:
    """(mean pop over the stops with a score, share of pop >= TOP among the stops whose
    popularity is known). A stop without an id is known not to be a must-see; one whose id
    could not be scored is unknown and left out, so a failed lookup on one side of the
    comparison does not read as a plan without must-sees."""
    known = [popularity[p.wikidata] for p in points if p.wikidata in popularity]
    mean = round(sum(known) / len(known), 2) if known else None
    if not known and any(p.wikidata for p in points):
        return mean, None
    counted = len(known) + sum(not p.wikidata for p in points)
    top = sum(pop >= wikidata.TOP for pop in known)
    return mean, round(top / counted, 2) if counted else None


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
    popularity: Mapping[str, float] | None = None,
) -> Metrics:
    route = check_route(points, start, brief)
    covered, total = coverage or (None, None)
    mean_pop, top_share = popularity_stats(points, popularity or {})
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
        mean_pop=mean_pop,
        top_share=top_share,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def score_journey(
    journey: Journey,
    days_requested: int,
    avoid: Sequence[str],
    usage: dict,
    seconds: float,
    popularity: Mapping[str, float] | None = None,
) -> JourneyMetrics:
    """Each day is checked against its own brief (its date), the interests against the
    whole trip's: a wish is served if any day serves it."""
    plans = [day.plan for day in journey.days if day.plan]
    routes = [check_route(agent_points(p), p.start, p.brief) for p in plans]
    covered, total = interest_coverage([s for p in plans for s in p.stops], journey.brief.interests)
    mean_pop, top_share = popularity_stats(journey_points(journey), popularity or {})
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
        mean_pop=mean_pop,
        top_share=top_share,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
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
            s.place.tags.get("wikidata"),
        )
        for s in plan.stops
    ]


def journey_points(journey: Journey) -> list[Point]:
    return [pt for day in journey.days if day.plan for pt in agent_points(day.plan)]


def _geo(p: GeoPoint | Point) -> GeoPoint:
    return p if isinstance(p, GeoPoint) else GeoPoint(lat=p.lat, lon=p.lon)


async def plan_with_agent(base: LLM, request: str) -> tuple[Journey, dict, float, dict | None]:
    """The journey, the LLM usage of the run, its wall-clock seconds and how the trip
    planner ended (planner_outcome)."""
    llm = base.for_run()
    events: list[AgentEvent] = []

    async def keep(event: AgentEvent) -> None:
        """The eval scores the result; of the progress events it reads the planner's only."""
        events.append(event)

    started = clock.monotonic()
    journey = await plan_journey(PlanRequest(request=request), llm, keep, today=TODAY)
    return journey, llm.usage.as_dict(), clock.monotonic() - started, planner_outcome(events)


def planner_outcome(events: Sequence[AgentEvent]) -> dict | None:
    """The trip planner's stop reason and steps, and whether code assembled the days after
    all (fallback), from the events the UI gets; None when it did not run (a single walk,
    TRIP_PLANNER_AGENT=0, too few places to share). Without it a trip that code built
    after the planner gave up would be scored as the planner's."""
    done = [
        e.data.get("detail", {})
        for e in events
        if e.type == "stage"
        and (e.data.get("name"), e.data.get("status")) == ("synthesize", "done")
    ]
    if not done:
        return None
    return {
        "stop_reason": done[-1].get("stop_reason"),
        "steps": done[-1].get("steps"),
        "fallback": any(
            e.type == "warning" and e.data.get("message") == PLANNER_FALLBACK for e in events
        ),
    }


async def run_walk(base: LLM, query: dict, row: dict, with_baseline: bool) -> None:
    """Fills `row` as it goes: a failed baseline keeps the agent's result."""
    journey, usage, seconds, _ = await plan_with_agent(base, query["request"])
    plan = journey.days[0].plan
    coverage = interest_coverage(plan.stops, plan.brief.interests)
    points = agent_points(plan)
    popularity = await stop_popularity(points)
    metrics = score(
        points, plan.start, plan.brief, query["avoid"], usage, seconds, coverage, popularity
    )
    row["brief_avoid"] = plan.brief.avoid
    row["agent"], row["agent_stops"] = asdict(metrics), [s.place.name for s in plan.stops]
    # What every system of the row is scored against, for a later --skip-agent run.
    row["brief"], row["start"] = plan.brief.model_dump(mode="json"), plan.start.model_dump()
    if with_baseline:
        metrics, names = await run_baseline(base, query, plan.brief, plan.start)
        row["baseline"], row["baseline_stops"] = asdict(metrics), names


async def run_journey(base: LLM, query: dict, row: dict) -> None:
    journey, usage, seconds, planner = await plan_with_agent(base, query["request"])
    popularity = await stop_popularity(journey_points(journey))
    metrics = score_journey(journey, query["days"], query["avoid"], usage, seconds, popularity)
    row["brief_avoid"] = journey.brief.avoid
    row["agent"] = asdict(metrics)
    row["agent_stops"] = [[s.place.name for s in d.plan.stops] for d in journey.days if d.plan]
    row["planner"] = planner


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
    points = await locate_stops(answer.stops, brief, start)
    # The popularity lookup is the eval's own: seconds stay comparable with earlier runs.
    seconds = clock.monotonic() - started
    popularity = await stop_popularity(points)
    metrics = score(
        points, start, brief, query["avoid"], llm.usage.as_dict(), seconds, None, popularity
    )
    return metrics, [s.name for s in answer.stops]


async def run_harness(
    base: LLM, model: str, query: dict, brief: TripBrief, start: GeoPoint, reasoning: str
) -> tuple[Metrics, list[str], str]:
    """A chat answer with web search, copied into the baseline's JSON and checked by the
    baseline's code; the answer is returned too, to be read. LLM.chat returns the parsed
    turn only, and the searches and the citations are in the raw answer, so the call goes
    through the wrapper's retries and counters directly. Usage and seconds are the chat
    answer's, what a traveller pays and waits for; the copy into JSON is the eval's own
    work, done by the agent's model, and not counted."""
    llm = base.for_run()
    # LLM.chat's extensions: reasoning and cost accounting. "default" sends no setting: the
    # model as its chat app runs it. Given the search tool, Haiku then spent 7.6k reasoning
    # tokens on one walk, and Sonnet 5.5, which cannot turn reasoning off, searched only when
    # left to think as it does by default: at the "low" effort it answered from memory.
    extra_body: dict = {"usage": {"include": True}}
    if reasoning != "default":
        effort = {"enabled": False} if reasoning == "off" else {"effort": reasoning}
        extra_body["reasoning"] = effort
    messages = [
        {"role": "system", "content": BASELINE_PROMPT + HARNESS_PROMPT.format(today=TODAY)},
        {"role": "user", "content": query["request"]},
    ]
    request = {"model": model, "messages": messages, "temperature": 0}
    request |= {"max_tokens": HARNESS_MAX_TOKENS, "tools": [HARNESS_SEARCH]}
    started = clock.monotonic()
    raw = await llm._call_with_retries(request, extra_body, HARNESS_TIMEOUT_S)
    seconds = clock.monotonic() - started
    llm._count(raw)
    if raw["choices"][0].get("finish_reason") == "length":
        raise ValueError(f"the answer was cut at {HARNESS_MAX_TOKENS} tokens")
    text, searches, cited = harness_answer(raw)
    plan = await base.for_run().structured(
        [{"role": "system", "content": COPY_PROMPT}, {"role": "user", "content": text}],
        BaselinePlan,
    )
    points = await locate_stops(plan.stops, brief, start)
    popularity = await stop_popularity(points)
    metrics = score(
        points, start, brief, query["avoid"], llm.usage.as_dict(), seconds, None, popularity
    )
    metrics.web_searches, metrics.cited_pages = searches, cited
    return metrics, [s.name for s in plan.stops], text


def harness_answer(raw: Mapping) -> tuple[str, int, int]:
    """(text, searches, cited pages) of a chat completion with the web search server tool.
    OpenRouter counts the searches in usage.server_tool_use_details (server_tool_use in its
    docs); the pages come as url_citation annotations when the engine returns them, the
    same page cited twice counted once."""
    message = raw["choices"][0].get("message") or {}
    usage = raw.get("usage") or {}
    tools = usage.get("server_tool_use_details") or usage.get("server_tool_use") or {}
    pages = {
        a["url_citation"].get("url")
        for a in message.get("annotations") or []
        if a.get("type") == "url_citation"
    }
    return message.get("content") or "", tools.get("web_search_requests") or 0, len(pages)


def stored_reading(row: Mapping) -> tuple[TripBrief, GeoPoint] | None:
    """The brief and start a row's systems were scored against; None for rows of runs
    before they were recorded."""
    if "brief" not in row or "start" not in row:
        return None
    return TripBrief.model_validate(row["brief"]), GeoPoint.model_validate(row["start"])


async def read_request(base: LLM, query: dict) -> tuple[TripBrief, GeoPoint]:
    """The brief and start the pipeline would plan from: its brief step, then the start
    point in OpenStreetMap, else the city centre (pipeline._locate, without its events)."""
    brief = await parse_brief(base.for_run(), query["request"], TODAY)
    if brief.start_point:
        found = await geocode.geocode(brief.start_point, city=brief.city, country=brief.country)
        if found:
            return brief, found
    city = await geocode.locate_city(brief.city, brief.country)
    if city is None:
        raise ValueError(f"city not found in OpenStreetMap: {brief.city}")
    return brief, city.center


async def run_harnesses(
    base: LLM, models: Sequence[str], query: dict, row: Mapping, reasoning: str
) -> list[dict]:
    """A row per harness model, scored against the row's brief and start. A failed reading
    of the request fails the query for every model, a failed model its own row."""
    try:
        brief, start = stored_reading(row) or await read_request(base, query)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        return [{"id": query["id"], "model": m, "error": error} for m in models]
    reading = {"brief": brief.model_dump(mode="json"), "start": start.model_dump()}
    rows = []
    for model in models:
        harness: dict = {"id": query["id"], "model": model, "reasoning": reasoning, **reading}
        try:
            metrics, names, answer = await run_harness(base, model, query, brief, start, reasoning)
            harness["harness"], harness["harness_stops"] = asdict(metrics), names
            harness["harness_answer"] = answer
        except Exception as e:  # one failed model must not lose the others' results
            harness["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(harness, ensure_ascii=False))
        rows.append(harness)
    return rows


async def locate_stops(
    stops: Sequence[BaselineStop], brief: TripBrief, start: GeoPoint
) -> list[Point]:
    city = await geocode.locate_city(brief.city, brief.country)
    points = []
    for stop in stops:
        hit = await _locate(stop, brief.city, city.bbox if city else None, start)
        points.append(baseline_point(stop, hit, brief))
    return points


async def _locate(stop: BaselineStop, city: str, bbox, start: GeoPoint) -> dict | None:
    """Find the place in OpenStreetMap (Nominatim) inside the city's bounding box.

    Deliberately lenient, so the comparison does not favour the agent: the name as
    written, the name without quotes and parenthesised remarks, then the address the
    model gave. The first hit inside the city and within 10 km of the start counts.
    extratags bring the hit's OSM tags: opening hours, the kind, and the Wikidata id the
    popularity columns score. A hit by the address alone drops that id: the address
    without a house number finds the street («Кронверкский пр.»), and its fame is not
    the stop's.
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
                if query not in (stop.name, _plain_name(stop.name)):
                    extra = (hit.get("extratags") or {}).items()
                    hit = hit | {"extratags": {k: v for k, v in extra if k != "wikidata"}}
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
    extra = hit.get("extratags") or {}
    lat, lon = float(hit["lat"]), float(hit["lon"])
    hours, qid = extra.get("opening_hours"), extra.get("wikidata")
    return Point(name, nominatim_kind(hit), lat, lon, hours, arrive, depart, qid)


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
    a new module matters only once a tracked file imports it, and that file differs. Nor
    do the results themselves: a run of the trips after a run of the walks would
    otherwise call its own code dirty."""

    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True)
        return done.stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ":(top)",
            ":(top,exclude)eval/results",
        )
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
HARNESS_SECTION = (
    "Harness: chat with web search",
    "One request to a general chat model with OpenRouter's web search server tool (the "
    "provider's native search, at most {max_uses} searches of {max_results} results): the "
    "baseline's prompt, today's date and an invitation to check the places on the web. The "
    "agent's model copies the prose answer into the baseline's JSON, checked by the baseline's "
    "code. Tokens, cost (search fees included) and seconds are the chat answer's, not the "
    "copy's; the baseline's seconds include the eval's OpenStreetMap lookups. Cited pages are "
    "— when the engine returns no citations; the answers are harness_answer in the JSON.",
)
SUMMARY_SECTION = (
    "Summary: every system on the same walks",
    "Over the single walks every system answered. Sums of stops, closed and impossible legs; "
    "per-route means of walk km, mean pop (over the routes that have one), tokens (over the "
    "routes that recorded them: — for runs before tokens were recorded), cost and seconds. "
    "Seconds are not like for like: the agent's are its whole pipeline, the one prompt's "
    "include the eval's OpenStreetMap lookups, the chat's are its answer alone.",
)


def walks_table(rows: list[dict]) -> str:
    lines = [
        "| query | system | stops | real places (located) | closed at visit | impossible legs "
        "| over window | avoid hits | interests covered | mean pop | top share | walk km "
        "| LLM calls | cost $ | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
                    *_popularity(m),
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
        "| days over window | repeats across days | avoid hits | interests covered | mean pop "
        "| top share | walk km | LLM calls | cost $ | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
                *_popularity(m),
                m["walk_km"],
                m["llm_calls"],
                m["cost_usd"],
                m["seconds"],
            ]
            lines.append(_table_row(cells))
        lines += _error_rows(r)
    return "\n".join(lines) + "\n"


def harness_table(rows: list[dict]) -> str:
    """Grouped by query, in the order of the queries, then of the models."""
    queries = list(dict.fromkeys(r["id"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    lines = [
        "| query | model | stops | real places (located) | closed at visit | impossible legs "
        "| over window | avoid hits | mean pop | top share | walk km | web searches "
        "| cited pages | tokens | cost $ | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (queries.index(r["id"]), models.index(r["model"]))):
        m = r.get("harness")
        if not m:
            lines.append(_table_row([r["id"], r["model"], "error", r["error"].splitlines()[0]]))
            continue
        cells = [
            r["id"],
            r["model"],
            m["stops"],
            f"{m['found']}/{m['stops']}",
            m["closed"],
            m["impossible_legs"],
            "yes" if m["over_window"] else "no",
            m["avoid_hits"],
            *_popularity(m),
            m["walk_km"],
            m["web_searches"],
            "—" if not m["cited_pages"] else m["cited_pages"],
            _tokens(m),
            m["cost_usd"],
            m["seconds"],
        ]
        lines.append(_table_row(cells))
    return "\n".join(lines) + "\n"


def summary_systems(model: str, walks: list[dict], harness: list[dict]) -> list[tuple[str, list]]:
    """(label, metrics per route) of each system, over the walks every system answered: a
    query one system failed is left out of all, so the means compare the same walks (its
    error stays in the system's own table)."""
    systems = [
        (f"agent, `{model}`", {r["id"]: r["agent"] for r in walks if r.get("agent")}),
        (f"one prompt, `{model}`", {r["id"]: r["baseline"] for r in walks if r.get("baseline")}),
    ]
    for name in dict.fromkeys(r["model"] for r in harness):
        by_id = {r["id"]: r["harness"] for r in harness if r["model"] == name and r.get("harness")}
        systems.append((f"chat + web search, `{name}`", by_id))
    systems = [(label, by_id) for label, by_id in systems if by_id]
    if not systems:
        return []
    common = [i for i in systems[0][1] if all(i in by_id for _, by_id in systems)]
    return [(label, [by_id[i] for i in common]) for label, by_id in systems]


def summary_sections(model: str, walks: list[dict], harness: list[dict]) -> list[str]:
    """The summary over the walks every system answered and, when a harness model answered
    fewer (a run stopped for its cost), again without it, over the walks the rest share:
    one model's gaps must not shrink the comparison of all the others."""
    heading, note = SUMMARY_SECTION
    systems = summary_systems(model, walks, harness)
    sections = [f"## {heading}\n\n{note}\n\n{summary_table(systems)}"]
    walked = {r["id"] for r in walks if r.get("agent")}
    names = dict.fromkeys(r["model"] for r in harness)
    partial = [
        name
        for name in names
        if not walked <= {r["id"] for r in harness if r["model"] == name and r.get("harness")}
    ]
    if not partial or len(partial) == len(names):
        return sections
    rest = summary_systems(model, walks, [r for r in harness if r["model"] not in partial])
    if len(rest[0][1]) > len(systems[0][1]):
        without = ", ".join(f"`{name}`" for name in partial)
        sections.append(
            f"Without {without}, which answered fewer walks: over {len(rest[0][1])} walks."
            f"\n\n{summary_table(rest)}"
        )
    return sections


def summary_table(systems: list[tuple[str, list]]) -> str:
    lines = [
        "| system | routes | real places (located) | closed at visit | impossible legs "
        "| routes with an impossible leg | walk km per route | mean pop | tokens per route "
        "| cost $ per route | s per route |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, ms in systems:
        n = len(ms)
        pops = [m["mean_pop"] for m in ms if m.get("mean_pop") is not None]
        tokens = [_total_tokens(m) for m in ms if _total_tokens(m) is not None]
        cells = [
            label,
            n,
            f"{sum(m['found'] for m in ms)}/{sum(m['stops'] for m in ms)}",
            sum(m["closed"] for m in ms),
            sum(m["impossible_legs"] for m in ms),
            f"{sum(m['impossible_legs'] > 0 for m in ms)}/{n}",
            _mean(m["walk_km"] for m in ms),
            round(sum(pops) / len(pops), 2) if pops else "—",
            round(sum(tokens) / len(tokens)) if tokens else "—",
            _mean((m["cost_usd"] for m in ms), 4),
            _mean(m["seconds"] for m in ms),
        ]
        lines.append(_table_row(cells))
    return "\n".join(lines) + "\n"


def _total_tokens(m: dict) -> int | None:
    prompt, completion = m.get("prompt_tokens"), m.get("completion_tokens")
    return None if prompt is None or completion is None else prompt + completion


def _tokens(m: dict) -> int | str:
    total = _total_tokens(m)
    return "—" if total is None else total


def _mean(values: Iterable[float], digits: int = 1) -> float | str:
    values = list(values)
    return round(sum(values) / len(values), digits) if values else "—"


def _interests(m: dict) -> str:
    covered = m.get("interests_covered")
    return "—" if covered is None else f"{covered}/{m['interests_total']}"


def _popularity(m: dict) -> list:
    """Mean pop and top share; "—" when unknown, as in rows of runs before the columns."""
    return ["—" if m.get(key) is None else m[key] for key in ("mean_pop", "top_share")]


def _error_rows(row: dict) -> list[str]:
    # The first line only: a validation error spans many, the JSON keeps all of them.
    return (
        [_table_row([row["id"], "error", row["error"].splitlines()[0]])] if "error" in row else []
    )


def _table_row(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |"


def write_results(
    out: Path,
    slug: str,
    model: str,
    run: dict,
    walks: list[dict],
    journeys: list[dict],
    harness: Sequence[dict] = (),
) -> None:
    """Rewrites the JSON file of each kind of query this run ran, records the run in
    <slug>.meta.json, and rebuilds <slug>.md from the JSON files, each table under the
    commit and date of the run that wrote it. Harness rows are merged by model and query,
    so a run of one harness model keeps the others' rows and their runs."""
    out.mkdir(exist_ok=True)
    meta_path = out / f"{slug}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sections = []
    walk_rows: list[dict] = []
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
        if table is walks_table:
            walk_rows = rows
        info = meta.get(path.name)
        provenance = (
            f"Run {info['run_at']}, commit `{info['commit']}`."
            if info
            else "Run before commits were recorded."
        )
        sections.append(f"## {heading}\n\n{provenance} {note}\n\n{table(rows)}")
    harness_rows = _merge_harness(out / f"{slug}-harness.json", harness, run, meta)
    if harness_rows:
        provenance = "; ".join(
            f"`{name}`: " + "; ".join(_harness_run(info, len(runs) > 1) for info in runs)
            for name, runs in meta[f"{slug}-harness.json"].items()
        )
        heading, note = HARNESS_SECTION
        note = note.format(**HARNESS_SEARCH["parameters"])
        sections.append(f"## {heading}\n\n{provenance}. {note}\n\n{harness_table(harness_rows)}")
        sections += summary_sections(model, walk_rows, harness_rows)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    (out / f"{slug}.md").write_text(
        f"Model: `{model}`, queries anchored at {TODAY.isoformat()}. Avoid hits count stops "
        "matching the hand-written avoid of the query (eval/queries.json); the agent's "
        "parsed one is brief_avoid in the JSON. Mean pop is the mean popularity (0..1: "
        "Wikipedia page views and language versions) of the stops that have one; top share "
        f"is the share of stops with pop >= {wikidata.TOP} among those whose popularity is "
        "known (scored, or without a Wikidata id); — when the lookup failed.\n\n"
        + "\n".join(sections)
    )


def _merge_harness(path: Path, rows: Sequence[dict], run: dict, meta: dict) -> list[dict]:
    """The file's rows with this run's in place of the same model and query. meta lists
    each model's runs with the queries whose rows each still holds: a run of some of a
    model's walks (finishing the walks a stopped run left) must not credit the model's
    other rows to its own commit and reasoning."""
    kept = json.loads(path.read_text()) if path.exists() else []
    if not rows:
        return kept
    ran = {(r["model"], r["id"]) for r in rows}
    merged = [r for r in kept if (r["model"], r["id"]) not in ran] + list(rows)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    models = meta.setdefault(path.name, {})
    for name in dict.fromkeys(r["model"] for r in rows):
        ran_by = [r for r in rows if r["model"] == name]
        reasoning = {"reasoning": ran_by[0]["reasoning"]} if "reasoning" in ran_by[0] else {}
        queries = [r["id"] for r in ran_by]
        earlier = [
            info | {"queries": [q for q in info["queries"] if q not in queries]}
            for info in models.get(name, [])
        ]
        models[name] = [info for info in earlier if info["queries"]]
        models[name].append({**run, **reasoning, "queries": queries})
    return merged


def _harness_run(info: dict, with_queries: bool) -> str:
    """A harness run's provenance; the queries it holds when its model has several runs."""
    note = f"run {info['run_at']}, commit `{info['commit']}`"
    if "reasoning" in info:
        note += f", reasoning {info['reasoning']}"
    return note + (f" ({', '.join(info['queries'])})" if with_queries else "")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated query ids")
    parser.add_argument("--model", help="override MODEL_NAME")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--harness",
        action="append",
        default=[],
        metavar="MODEL",
        help="also plan each single walk with MODEL and web search (repeatable)",
    )
    parser.add_argument(
        "--harness-reasoning",
        choices=["off", "low", "medium", "high", "default"],
        help="reasoning of the --harness models: the agent's (LLM_REASONING) unless given; "
        "default: the provider's own, no setting sent",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="run --harness only, reusing the agent's single walks in eval/results",
    )
    args = parser.parse_args()
    if args.skip_agent and not args.harness:
        parser.error("--skip-agent reuses the agent's results for --harness: name a model")

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
    slug = base.model.replace("/", "__").replace(":", "_")
    if not settings.trip_planner_agent:
        slug += "-noplanner"  # the ablation (see the module docstring): it keeps the planner's
    stored: dict[str, dict] = {}
    if args.skip_agent:
        path = HERE / "results" / f"{slug}.json"
        if not path.exists():
            parser.error(f"no agent results to reuse: {path}")
        # Rows of runs before the reading was recorded borrow the one an earlier harness run
        # recorded: every harness model is scored against one reading of the request.
        harness_path = HERE / "results" / f"{slug}-harness.json"
        earlier = json.loads(harness_path.read_text()) if harness_path.exists() else []
        for r in [*earlier, *json.loads(path.read_text())]:
            if "brief" in r:
                stored[r["id"]] = r
        queries = [q for q in queries if q.get("days", 1) == 1]
    run = {
        "commit": git_commit(),
        "run_at": datetime.now().astimezone().isoformat(timespec="minutes"),
        "trip_planner_agent": settings.trip_planner_agent,
    }
    walks: list[dict] = []
    journeys: list[dict] = []
    harness: list[dict] = []
    for q in queries:
        trip = q.get("days", 1) > 1
        if args.skip_agent:
            row = stored.get(q["id"], {})
        else:
            row = {"id": q["id"], "model": base.model}
            try:
                if trip:
                    await run_journey(base, q, row)
                else:
                    await run_walk(base, q, row, with_baseline=not args.skip_baseline)
            except Exception as e:  # one failed query must not lose the others' results
                row["error"] = f"{type(e).__name__}: {e}"
            (journeys if trip else walks).append(row)
            print(json.dumps(row, ensure_ascii=False))
        if args.harness and not trip:
            reasoning = args.harness_reasoning or base.reasoning
            harness += await run_harnesses(base, args.harness, q, row, reasoning)

    write_results(HERE / "results", slug, base.model, run, walks, journeys, harness)


if __name__ == "__main__":
    asyncio.run(main())
