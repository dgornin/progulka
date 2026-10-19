"""The research agent's tools: JSON schemas for the model and their executor.

Each tool's arguments are a pydantic model. The same model produces the JSON
Schema the LLM sees and validates what the LLM sends back, so the two cannot drift
apart. The executor never raises: bad JSON, wrong types, unknown or rejected place
ids, source failures and even bugs (logged with a traceback) all come back as an
error text the model can act on.

Grounding: every place a search returns is stored in `ResearchState.registry`, and
shortlist_place accepts only ids from it, so the model cannot add a place it made up.
Context hygiene: the model sees at most 25 compact results per search (id, name,
kind, distance, hours, a 160-character snippet); full records stay in the registry.
A slow source (Overpass can take a minute) is cut off after TOOL_TIMEOUT_S and
reported as an error, so one tool call cannot eat the whole research budget.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import time
from math import ceil
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.models import Candidate, GeoPoint, Place, TripBrief
from app.planning.geo import haversine_m
from app.sources import osm, wikipedia, wikivoyage
from app.sources.http import SourceError

log = logging.getLogger(__name__)

TOOL_TIMEOUT_S = 60
MAX_RESULTS_SHOWN = 25
SNIPPET_CHARS = 160
REASON_CHARS = 220
MAX_DETAIL_TAGS = 40
DEFAULT_VISIT_MINUTES = 30
STRONG_INTEREST = 0.7
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
    Returns up to 25 nearest places: id, name, kind, dist_m, hours, snippet. Use it for
    concrete kinds of places: museums, parks, cafes, street art, viewpoints."""

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
    Returns up to 25 nearest: id, name, kind, dist_m, snippet. Use a keyword query for
    topics, styles, people and history ('конструктивизм', 'Пушкин'); omit the query to
    get the notable places nearby."""

    query: str | None = Field(
        None, description="Keywords in the traveller's language; omit for notable places nearby"
    )
    radius_m: int = Field(3000, ge=300, le=8000, description="Search radius in metres")


class GetWikivoyageListings(BaseModel):
    """Get the city's Wikivoyage travel-guide listings of one section: see (sights),
    do (activities), eat, drink, buy. Returns up to 25 nearest to the start point: id,
    name, kind, dist_m, hours, snippet. Best for food, drink and local favourites."""

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
        return _clip(value, REASON_CHARS)


class RemovePlace(BaseModel):
    """Remove a place from the shortlist."""

    place_id: str = Field(..., description="An id on the shortlist")
    reason: str = Field(..., description="Why it no longer fits")


class FinishResearch(BaseModel):
    """Finish the research when the shortlist is ready. It is rejected, with the reason,
    while the shortlist is too small or a strong interest has no candidate."""

    summary: str = Field(..., description="One or two sentences on what was found")


class ToolRejected(Exception):
    """A call the model can fix: unknown id, rejected place, premature finish."""


Handler = Callable[[ResearchState, Any], Awaitable[ToolOutcome]]


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI "tools" list; built per call so the category enum follows osm.CATEGORIES."""
    schemas = []
    for name, (args_model, _) in _TOOLS.items():
        description, parameters = _parameters(args_model)
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
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}" for err in e.errors()
        )
        return error_outcome(f"invalid arguments for {name}: {problems}")
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


def min_candidates(brief: TripBrief) -> int:
    """Shortlist size the optimizer needs to have a choice: about 1.3 places per hour."""
    start = _minutes(brief.start_time)
    end = _minutes(brief.end_time)
    return max(4, ceil(max(end - start, 0) / 60 * 1.3))


async def _search_osm(state: ResearchState, args: SearchOsm) -> ToolOutcome:
    places = await osm.search_places(state.start, args.radius_m, args.categories)
    label = f"OpenStreetMap ({', '.join(args.categories)}, {args.radius_m} м)"
    return _search_outcome(state, places, label)


async def _search_wikipedia(state: ResearchState, args: SearchWikipedia) -> ToolOutcome:
    lang = state.brief.language
    if args.query:
        places = await wikipedia.search(args.query, state.start, args.radius_m, lang=lang)
        label = f"Википедия «{args.query}» ({args.radius_m} м)"
    else:
        places = await wikipedia.nearby(state.start, args.radius_m, lang=lang)
        label = f"Википедия рядом ({args.radius_m} м)"
    return _search_outcome(state, places, label)


async def _get_wikivoyage_listings(
    state: ResearchState, args: GetWikivoyageListings
) -> ToolOutcome:
    places = await wikivoyage.listings(
        state.brief.city, lang=state.brief.language, sections=[args.section]
    )
    return _search_outcome(state, places, f"Wikivoyage ({args.section})")


async def _get_place_details(state: ResearchState, args: GetPlaceDetails) -> ToolOutcome:
    place = _lookup(state, args.place_id)
    if place.source == "wikipedia":
        extract = await wikipedia.page_extract(place.id, max_chars=1500)
        if extract:
            # Keep the full text: the writer grounds its blurbs on place.description.
            place = place.model_copy(update={"description": extract})
            state.registry[place.id] = place
            if place.id in state.shortlist:
                state.shortlist[place.id].place = place
    details = {
        "id": place.id,
        "name": place.name,
        "kind": place.kind,
        "source": place.source,
        "dist_m": _distance_m(state.start, place),
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
    must_include = [name.casefold() for name in state.brief.must_include]
    candidate = Candidate(
        place=place,
        score=args.score,
        interests=[labels[label.casefold()] for label in args.interests],
        reason=args.reason,
        visit_minutes=args.visit_minutes or _default_visit_minutes(place.kind),
        required=any(name in place.name.casefold() for name in must_include),
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


def _parameters(args_model: type[BaseModel]) -> tuple[str, dict[str, Any]]:
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


def _search_outcome(state: ResearchState, places: list[Place], label: str) -> ToolOutcome:
    found = []
    for place in places:
        if place.id in state.excluded:
            continue
        # setdefault keeps a record already enriched by get_place_details.
        found.append(state.registry.setdefault(place.id, place))
    found.sort(key=lambda p: _distance_m(state.start, p))
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
    if place.opening_hours:
        item["hours"] = place.opening_hours
    if place.description:
        item["snippet"] = _clip(place.description, SNIPPET_CHARS)
    return item


def _lookup(state: ResearchState, place_id: str) -> Place:
    if place_id in state.excluded:
        raise ToolRejected(f"the traveller rejected '{place_id}'; choose another place")
    place = state.registry.get(place_id)
    if place is None:
        raise ToolRejected(
            f"unknown place_id '{place_id}': use an id exactly as a search tool returned it"
        )
    return place


def _same_place_on_shortlist(state: ResearchState, place: Place) -> Place | None:
    """A shortlisted record of the same real place from another source: the same
    Wikidata/Wikipedia link, or one name containing the other within SAME_PLACE_M.

    Only records from different sources are compared: within one source, two nearby
    places with similar names (two murals called "Мурал") are different places.
    """
    name = place.name.casefold()
    for other in (c.place for c in state.shortlist.values()):
        if other.source == place.source:
            continue
        linked = any(place.tags.get(k) and place.tags[k] == other.tags.get(k) for k in LINK_TAGS)
        other_name = other.name.casefold()
        namesake = name in other_name or other_name in name
        if linked or (namesake and _metres(place, other) < SAME_PLACE_M):
            return other
    return None


def _missing_for_finish(state: ResearchState) -> list[str]:
    missing = []
    need = min_candidates(state.brief)
    if len(state.shortlist) < need:
        missing.append(f"the shortlist has {len(state.shortlist)} places, need at least {need}")
    covered = {label.casefold() for c in state.shortlist.values() for label in c.interests}
    for interest in state.brief.interests:
        if interest.weight >= STRONG_INTEREST and interest.label.casefold() not in covered:
            missing.append(f"no candidate for the interest '{interest.label}'")
    return missing


def _default_visit_minutes(kind: str) -> int:
    for key, spec in osm.CATEGORIES.items():
        if kind in (key, spec.kind):
            return spec.visit_minutes
    return DEFAULT_VISIT_MINUTES


def _candidate_event(action: str, candidate: Candidate, reason: str) -> dict[str, Any]:
    return {
        "action": action,
        "place_id": candidate.place.id,
        "name": candidate.place.name,
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


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
