"""The research agent's tools: JSON schemas for the model and their executor.

Each tool's arguments are a pydantic model. The same model produces the JSON
Schema the LLM sees and validates what the LLM sends back, so the two cannot drift
apart. The executor never raises: bad JSON, wrong types, unknown or rejected place
ids, source failures and even bugs (logged with a traceback) all come back as an
error text the model can act on.

Grounding: every place a search returns is stored in `ResearchState.registry`, and
shortlist_place accepts only ids from it, so the model cannot add a place it made up.
Context hygiene: the model sees at most 25 compact results per search (id, name,
kind, distance, popularity, hours, a 160-character snippet); full records stay in the
registry. A slow source (Overpass can take a minute) is cut off after TOOL_TIMEOUT_S and
reported as an error, so one tool call cannot eat the whole research budget.

Ranking: each search scores its places once (`popularity`, 0..1: Wikipedia page views
and language versions, app/sources/wikidata.py). Within NEAR_M of the start (the walking
radius the prompt asks for) a city's must-sees come first, then its second line; all the
rest follows by distance. By distance alone the 25 shown could fill up with ordinary
places around the start while the landmark a kilometre away was cut off; lifted at any
distance, the landmarks of a whole city would fill a list without a radius (Wikivoyage)
and push out everything near. The score itself is shown too, so the model judges what a
first visit needs. The scores come through the HTTP cache, so a search whose lookup
succeeded shows the same list on a replayed run. A failed lookup only leaves the places
unscored, ordered by distance, but it is not cached: replayed, the search can show a
different list and miss the LLM cache, so a rehearsal counts only when its log has no
popularity failure.

The days of a multi-day trip are researched in parallel, one ResearchState each, and
share two things: the registry (a place one day found is known to all) and the
source queries (`searches`): days that start from the same point send the same
searches at the same moment, before the HTTP cache has an answer, and one query
serves them all. Each day still looks places up in its own records first (`seen`):
an OSM element's kind depends on the categories searched, and a day's tool results
must not depend on which day registered a place first, or a rehearsed run stops
replaying from the LLM cache. `taken` marks the places other days of the trip have
planned, `rejected_elsewhere` the places the traveller removed there: neither is
shown or shortlisted (the pipeline lifts a rejection when the traveller asks for the
place on this day).
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import time
from math import ceil
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.models import MUST_HAVE_WEIGHT, Candidate, GeoPoint, Place, TripBrief, clip
from app.planning.geo import haversine_m
from app.planning.validator import names_place
from app.sources import osm, wikidata, wikipedia, wikivoyage
from app.sources.http import SourceError

log = logging.getLogger(__name__)

TOOL_TIMEOUT_S = 60
NEAR_M = 2000  # the prompt's walking radius: well-known places nearer are listed first
MAX_RESULTS_SHOWN = 25
SNIPPET_CHARS = 160
REASON_CHARS = 220
MAX_DETAIL_TAGS = 40
DEFAULT_VISIT_MINUTES = 30
FINISH_ATTEMPTS = 3
SAME_PLACE_M = 150
LINK_TAGS = ("wikidata", "wikipedia")  # OSM and the Wikipedia source both fill these


@dataclass
class ResearchState:
    brief: TripBrief
    start: GeoPoint
    weather_summary: str
    registry: dict[str, Place] = field(default_factory=dict)  # every place a tool returned
    shortlist: dict[str, Candidate] = field(default_factory=dict)
    excluded: set[str] = field(default_factory=set)  # ids the traveller rejected
    finish_attempts: int = 0  # finish_research calls in the current research run
    # Transient, never saved (exclude=True): set by the pipeline before every run, from
    # what the other days have planned, and what the traveller removed there, at that moment.
    taken: Annotated[set[str], Field(exclude=True)] = field(default_factory=set)
    rejected_elsewhere: Annotated[set[str], Field(exclude=True)] = field(default_factory=set)
    # Transient: source query key -> _Search, shared by the days of one planning run.
    searches: Annotated[dict[str, Any], Field(exclude=True)] = field(default_factory=dict)
    # Transient: this day's own record of every place its tools returned (see the module
    # docstring); after a restart lookups fall back to the registry.
    seen: Annotated[dict[str, Place], Field(exclude=True)] = field(default_factory=dict)

    def available(self) -> dict[str, Candidate]:
        """The shortlist without the places other days of the trip have taken or lost."""
        out = self.taken | self.rejected_elsewhere
        return {pid: c for pid, c in self.shortlist.items() if pid not in out}


@dataclass
class ToolOutcome:
    content: str  # compact JSON for the role="tool" message
    summary: str  # one line for the UI trace, in Russian
    count: int | None = None
    error: str | None = None
    finished: bool = False
    candidate_event: dict[str, Any] | None = None


class SearchOsm(BaseModel):
    """Search OpenStreetMap around the start point for places of the given categories.
    Returns up to 25 places, well-known ones (pop >= 0.5) within about 2 km first, then
    by distance: id, name, kind, dist_m, pop, hours, snippet. Use it for concrete kinds of
    places: museums, parks, cafes, street art, viewpoints."""

    categories: list[str] = Field(
        ..., min_length=1, max_length=6, description="Category keys from the catalog in the task"
    )
    radius_m: int = Field(1500, ge=200, le=4000, description="Search radius in metres")

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, value: list[str]) -> list[str]:
        unknown = [c for c in value if c not in osm.CATEGORIES]
        if unknown:
            raise ValueError(f"unknown categories {unknown}; valid: {sorted(osm.CATEGORIES)}")
        return value


class SearchWikipedia(BaseModel):
    """Search Wikipedia articles about places with coordinates near the start point.
    Returns up to 25, well-known ones (pop >= 0.5) within about 2 km first, then by
    distance: id, name, kind, dist_m, pop, snippet. Use a keyword query for topics,
    styles, people and history ('конструктивизм', 'Пушкин'); omit the query to get the
    notable places nearby."""

    query: str | None = Field(
        None, description="Keywords in the traveller's language; omit for notable places nearby"
    )
    radius_m: int = Field(3000, ge=300, le=8000, description="Search radius in metres")


class GetWikivoyageListings(BaseModel):
    """Get the city's Wikivoyage travel-guide listings of one section: see (sights),
    do (activities), eat, drink, buy. Returns up to 25, well-known ones (pop >= 0.5)
    within about 2 km of the start point first, then by distance: id, name, kind, dist_m,
    pop, hours, snippet. Best for food, drink and local favourites."""

    section: Literal["see", "do", "eat", "drink", "buy"]


class GetPlaceDetails(BaseModel):
    """Get the full description of one place found earlier: the Wikipedia article
    extract for Wikipedia places, otherwise the source record (tags, opening hours,
    description). Use it when the snippet is not enough to judge the place."""

    place_id: str = Field(..., description="An id exactly as a search returned it")


class ShortlistPlace(BaseModel):
    """Add a place returned by a search to the shortlist (or update it if it is there).
    The optimizer builds the route only from shortlisted places."""

    place_id: str = Field(..., description="An id exactly as a search returned it")
    score: float = Field(..., ge=0, le=1, description="How well it fits the traveller, 0..1")
    interests: list[str] = Field(
        ..., description="Interest labels from the brief that it serves; may be empty"
    )
    reason: str = Field(
        ...,
        description=f"Why it fits, at most {REASON_CHARS} characters, based only on the tool "
        "output for this place",
        json_schema_extra={"maxLength": REASON_CHARS},
    )
    visit_minutes: int | None = Field(
        None, ge=10, le=180, description="Time to spend there; omit for the category default"
    )

    @field_validator("reason")
    @classmethod
    def _clip_reason(cls, value: str) -> str:
        # Models count characters badly: clip instead of spending a step on a retry.
        return clip(value, REASON_CHARS)


class RemovePlace(BaseModel):
    """Remove a place from the shortlist."""

    place_id: str = Field(..., description="An id on the shortlist")
    reason: str = Field(..., description="Why it no longer fits")


class FinishResearch(BaseModel):
    """Finish the research when the shortlist is ready. It is rejected, with the reason,
    while the shortlist is too small or an interest with weight 0.7 or more has no
    candidate."""

    summary: str = Field(..., description="One or two sentences on what was found")


class ToolRejected(Exception):
    """A call the model can fix: unknown id, rejected place, premature finish."""


Handler = Callable[[ResearchState, Any], Awaitable[ToolOutcome]]


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI "tools" list; built per call so the category enum follows osm.CATEGORIES."""
    schemas = []
    for name, (args_model, _) in _TOOLS.items():
        description, parameters = tool_parameters(args_model)
        if args_model is SearchOsm:
            parameters["properties"]["categories"]["items"]["enum"] = list(osm.CATEGORIES)
        schemas.append(
            {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }
        )
    return schemas


async def execute(state: ResearchState, name: str, arguments: str) -> ToolOutcome:
    if name not in _TOOLS:
        return error_outcome(f"unknown tool '{name}'; available: {', '.join(_TOOLS)}")
    args_model, handler = _TOOLS[name]
    try:
        args = args_model.model_validate_json(arguments or "{}")
    except ValidationError as e:
        return invalid_arguments(name, e)
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_S):
            return await handler(state, args)
    except ToolRejected as e:
        return error_outcome(str(e))
    except TimeoutError:
        return error_outcome(
            f"{name} timed out: the source is slow. Try fewer categories, a smaller radius "
            "or another tool."
        )
    except (SourceError, ValueError) as e:
        return error_outcome(f"{name} failed: {e}. Try other arguments or another tool.")
    except Exception:
        # A bug, e.g. a source parser meeting an unexpected response. It is logged with
        # the traceback, and the run goes on: under asyncio.gather an exception here
        # would abort the research while the turn's other calls kept running.
        log.exception("tool %s failed", name)
        return error_outcome(f"{name} failed unexpectedly. Use another tool.")


def error_outcome(message: str) -> ToolOutcome:
    return ToolOutcome(
        content=_json({"error": message}), summary=f"Ошибка: {message}", error=message
    )


def invalid_arguments(name: str, e: ValidationError) -> ToolOutcome:
    """What the model sent does not fit the tool's schema: every problem, by field."""
    problems = "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors()
    )
    return error_outcome(f"invalid arguments for {name}: {problems}")


def min_candidates(brief: TripBrief) -> int:
    """Shortlist size the optimizer needs to have a choice: about 1.3 places per hour."""
    start = _minutes(brief.start_time)
    end = _minutes(brief.end_time)
    return max(4, ceil(max(end - start, 0) / 60 * 1.3))


def same_place(a: Place, b: Place) -> bool:
    """Two records of one real place from different sources: the same Wikidata/Wikipedia
    link, or one name containing the other within SAME_PLACE_M. The shortlist and the
    pipeline's choice of stops both use it, so a sight is planned once per trip.

    Only records from different sources are compared: within one source, two nearby
    places with similar names (two murals called "Мурал") are different places.
    """
    if a.source == b.source:
        return False
    if any(a.tags.get(k) and a.tags[k] == b.tags.get(k) for k in LINK_TAGS):
        return True
    name, other = a.name.casefold(), b.name.casefold()
    return (name in other or other in name) and _metres(a, b) < SAME_PLACE_M


async def _search_osm(state: ResearchState, args: SearchOsm) -> ToolOutcome:
    places = await _shared(
        state,
        f"osm {_around(state)} {args.model_dump_json()}",
        lambda: _rated(osm.search_places(state.start, args.radius_m, args.categories)),
    )
    label = f"OpenStreetMap ({', '.join(args.categories)}, {args.radius_m} м)"
    return _search_outcome(state, places, label)


async def _search_wikipedia(state: ResearchState, args: SearchWikipedia) -> ToolOutcome:
    lang = state.brief.language
    key = f"wikipedia {lang} {_around(state)} {args.model_dump_json()}"
    if args.query:
        query = args.query
        places = await _shared(
            state,
            key,
            lambda: _rated(wikipedia.search(query, state.start, args.radius_m, lang=lang)),
        )
        label = f"Википедия «{args.query}» ({args.radius_m} м)"
    else:
        places = await _shared(
            state, key, lambda: _rated(wikipedia.nearby(state.start, args.radius_m, lang=lang))
        )
        label = f"Википедия рядом ({args.radius_m} м)"
    return _search_outcome(state, places, label)


async def _get_wikivoyage_listings(
    state: ResearchState, args: GetWikivoyageListings
) -> ToolOutcome:
    brief = state.brief
    # The listings are the city's, whatever the start: every day shares them.
    places = await _shared(
        state,
        f"wikivoyage {brief.language} {brief.city} {args.section}",
        lambda: _rated(
            wikivoyage.listings(brief.city, lang=brief.language, sections=[args.section])
        ),
    )
    return _search_outcome(state, places, f"Wikivoyage ({args.section})")


async def _get_place_details(state: ResearchState, args: GetPlaceDetails) -> ToolOutcome:
    place = _lookup(state, args.place_id)
    if place.source == "wikipedia":
        extract = await wikipedia.page_extract(place.id, max_chars=1500)
        if extract:
            # Keep the full text: the writer grounds its blurbs on place.description.
            place = place.model_copy(update={"description": extract})
            state.seen[place.id] = state.registry[place.id] = place
            if place.id in state.shortlist:
                state.shortlist[place.id].place = place
    details = {
        "id": place.id,
        "name": place.name,
        "kind": place.kind,
        "source": place.source,
        "dist_m": _distance_m(state.start, place),
        "pop": place.popularity,
        "hours": place.opening_hours,
        "indoor": place.indoor,
        "description": place.description,
        "tags": dict(list(place.tags.items())[:MAX_DETAIL_TAGS]) or None,
    }
    return ToolOutcome(
        content=_json({k: v for k, v in details.items() if v is not None}),
        summary=f"Подробности: {place.name}",
    )


async def _shortlist_place(state: ResearchState, args: ShortlistPlace) -> ToolOutcome:
    place = _lookup(state, args.place_id)
    labels = {i.label.casefold(): i.label for i in state.brief.interests}
    unknown = [label for label in args.interests if label.casefold() not in labels]
    if unknown:
        raise ToolRejected(
            f"unknown interest labels {unknown}; use the brief's labels: {list(labels.values())}"
        )
    duplicate = _same_place_on_shortlist(state, place)
    if duplicate:
        raise ToolRejected(
            f"'{place.name}' is the same place as shortlisted '{duplicate.name}' ({duplicate.id})"
        )
    candidate = Candidate(
        place=place,
        score=args.score,
        interests=[labels[label.casefold()] for label in args.interests],
        reason=args.reason,
        visit_minutes=args.visit_minutes or _default_visit_minutes(place.kind),
        required=any(names_place(wanted, place.name) for wanted in state.brief.must_include),
    )
    state.shortlist[place.id] = candidate
    return ToolOutcome(
        content=_json({"ok": True, "shortlist_size": len(state.shortlist)}),
        summary=f"В шорт-лист: {place.name} ({args.score:.2f})",
        candidate_event=_candidate_event("add", candidate, candidate.reason),
    )


async def _remove_place(state: ResearchState, args: RemovePlace) -> ToolOutcome:
    candidate = state.shortlist.pop(args.place_id, None)
    if candidate is None:
        raise ToolRejected(f"'{args.place_id}' is not on the shortlist")
    return ToolOutcome(
        content=_json({"ok": True, "shortlist_size": len(state.shortlist)}),
        summary=f"Убрано из шорт-листа: {candidate.place.name}",
        candidate_event=_candidate_event("remove", candidate, args.reason),
    )


async def _finish_research(state: ResearchState, args: FinishResearch) -> ToolOutcome:
    state.finish_attempts += 1
    missing = _missing_for_finish(state)
    if missing and state.finish_attempts < FINISH_ATTEMPTS:
        raise ToolRejected("not finished: " + "; ".join(missing))
    size = len(state.shortlist)
    summary = f"Исследование завершено, мест в шорт-листе: {size}"
    if missing:
        summary += f" (принято с {FINISH_ATTEMPTS}-й попытки, не хватает: {'; '.join(missing)})"
    return ToolOutcome(
        content=_json({"ok": True, "shortlist_size": size}), summary=summary, finished=True
    )


_TOOLS: dict[str, tuple[type[BaseModel], Handler]] = {
    "search_osm": (SearchOsm, _search_osm),
    "search_wikipedia": (SearchWikipedia, _search_wikipedia),
    "get_wikivoyage_listings": (GetWikivoyageListings, _get_wikivoyage_listings),
    "get_place_details": (GetPlaceDetails, _get_place_details),
    "shortlist_place": (ShortlistPlace, _shortlist_place),
    "remove_place": (RemovePlace, _remove_place),
    "finish_research": (FinishResearch, _finish_research),
}


def tool_parameters(args_model: type[BaseModel]) -> tuple[str, dict[str, Any]]:
    """Tool description and parameters schema, without pydantic's noise.

    Optional fields are advertised as plain optional types instead of
    anyOf [type, null]: several providers reject anyOf in tool schemas.
    """
    schema = args_model.model_json_schema()
    description = " ".join(schema.pop("description").split())
    schema.pop("title", None)
    for prop in schema["properties"].values():
        prop.pop("title", None)
        variants = prop.pop("anyOf", None)
        if variants:
            prop.update(next(v for v in variants if v.get("type") != "null"))
        if "default" in prop and prop["default"] is None:
            del prop["default"]
    return description, schema


@dataclass
class _Search:
    task: asyncio.Future[list[Place]]
    waiters: int = 0


async def _shared(
    state: ResearchState, key: str, fetch: Callable[[], Awaitable[list[Place]]]
) -> list[Place]:
    """Run a source query once for every day of the trip that sends it (see the module
    docstring). A failed query is forgotten, so a later call tries again."""
    search = state.searches.get(key)
    if search is None:
        search = state.searches[key] = _Search(asyncio.ensure_future(fetch()))

        def forget_failure(done: asyncio.Future[list[Place]]) -> None:
            if done.cancelled() or done.exception() is not None:
                _forget(state, key, search)

        search.task.add_done_callback(forget_failure)
    search.waiters += 1
    try:
        # Shielded: one day's tool timeout must not cancel the query another day awaits.
        return await asyncio.shield(search.task)
    finally:
        search.waiters -= 1
        if not search.waiters and not search.task.done():
            # Nobody waits for it any more. Forgotten at once, not when the cancellation
            # completes: a day that sends the same search in between must start a new
            # query, not await this one and get its CancelledError.
            search.task.cancel()
            _forget(state, key, search)


def _forget(state: ResearchState, key: str, search: _Search) -> None:
    """Drop the search, unless the key already belongs to a newer one."""
    if state.searches.get(key) is search:
        del state.searches[key]


async def _rated(search: Awaitable[list[Place]]) -> list[Place]:
    """The places of one source query with their popularity: one lookup for the whole
    batch, inside the shared query, so the days that send the same search share it too.

    wikidata.popularity turns Wikimedia failures into unscored places itself; this also
    keeps a bug in it from failing the search, which would cost far more than the hint.
    The lookup gets only what the tool timeout leaves, less a margin for the answer: past
    it, a slow but successful search (Overpass queued behind other days) would come back
    as a timeout and lose its places.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    places = await search
    took = loop.time() - started
    left = TOOL_TIMEOUT_S - took - 2
    if left < 1:
        log.warning("popularity of %d places skipped: the search took %.0f s", len(places), took)
        return places
    try:
        scores = await wikidata.popularity(places, left)
    except Exception:
        log.exception("popularity of %d places failed", len(places))
        return places
    return [
        p.model_copy(update={"popularity": scores[p.id]}) if p.id in scores else p for p in places
    ]


def _around(state: ResearchState) -> str:
    return f"{state.start.lat:.5f},{state.start.lon:.5f}"


def _search_outcome(state: ResearchState, places: list[Place], label: str) -> ToolOutcome:
    found = []
    out = state.excluded | state.taken | state.rejected_elsewhere
    for place in places:
        if place.id in out:
            continue
        # setdefault keeps a record already enriched by get_place_details. The model is
        # shown the source's own record: a day's tool results must not depend on what the
        # days researched beside it looked up first, or a rehearsed run stops replaying
        # from the LLM cache.
        state.registry.setdefault(place.id, place)
        state.seen.setdefault(place.id, place)
        found.append(place)
    found.sort(
        key=lambda p: (
            wikidata.tier(p.popularity) if (d := _distance_m(state.start, p)) <= NEAR_M else 2,
            d,
        )
    )
    shown = [_compact(state, p) for p in found[:MAX_RESULTS_SHOWN]]
    payload: dict[str, Any] = {"found": len(found), "places": shown}
    if not found:
        payload["hint"] = "nothing found: try a larger radius, other categories or another tool"
    return ToolOutcome(
        content=_json(payload), summary=f"{label}: найдено {len(found)}", count=len(found)
    )


def _compact(state: ResearchState, place: Place) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": place.id,
        "name": place.name,
        "kind": place.kind,
        "dist_m": _distance_m(state.start, place),
    }
    if place.popularity is not None:
        item["pop"] = place.popularity
    if place.opening_hours:
        item["hours"] = place.opening_hours
    if place.description:
        item["snippet"] = clip(place.description, SNIPPET_CHARS)
    return item


def _lookup(state: ResearchState, place_id: str) -> Place:
    if place_id in state.excluded:
        raise ToolRejected(f"the traveller rejected '{place_id}'; choose another place")
    if place_id in state.taken:
        raise ToolRejected(
            f"'{place_id}' is already in the route of another day of the trip; choose another place"
        )
    if place_id in state.rejected_elsewhere:
        raise ToolRejected(
            f"the traveller removed '{place_id}' from another day of the trip; choose another place"
        )
    place = state.seen.get(place_id) or state.registry.get(place_id)
    if place is None:
        raise ToolRejected(
            f"unknown place_id '{place_id}': use an id exactly as a search tool returned it"
        )
    return place


def _same_place_on_shortlist(state: ResearchState, place: Place) -> Place | None:
    """A shortlisted record of the same real place from another source."""
    return next((c.place for c in state.shortlist.values() if same_place(place, c.place)), None)


def _missing_for_finish(state: ResearchState) -> list[str]:
    missing = []
    usable = state.available().values()
    need = min_candidates(state.brief)
    if len(usable) < need:
        missing.append(f"the shortlist has {len(usable)} places, need at least {need}")
    covered = {label.casefold() for c in usable for label in c.interests}
    for interest in state.brief.interests:
        if interest.weight >= MUST_HAVE_WEIGHT and interest.label.casefold() not in covered:
            missing.append(f"no candidate for the interest '{interest.label}'")
    for wanted in state.brief.must_include:
        if not any(names_place(wanted, c.place.name) for c in usable):
            missing.append(f"must_include '{wanted}' is not on the shortlist: search it by name")
    return missing


def _default_visit_minutes(kind: str) -> int:
    spec = osm.CATEGORIES.get(kind)
    return spec.visit_minutes if spec else DEFAULT_VISIT_MINUTES


def _candidate_event(action: str, candidate: Candidate, reason: str) -> dict[str, Any]:
    return {
        "action": action,
        "place_id": candidate.place.id,
        "name": candidate.place.name,
        "kind": candidate.place.kind,
        "lat": candidate.place.lat,
        "lon": candidate.place.lon,
        "score": candidate.score,
        "interests": candidate.interests,
        "reason": reason,
    }


def _distance_m(start: GeoPoint, place: Place) -> int:
    """Distance shown to the model, rounded to 10 m."""
    return int(round(_metres(start, place), -1))


def _metres(a: GeoPoint | Place, b: Place) -> float:
    return haversine_m(a.lat, a.lon, b.lat, b.lon)


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
