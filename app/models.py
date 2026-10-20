"""Domain models shared by every layer: sources -> agent -> planning -> API.

A Place always comes from a source (OpenStreetMap, Wikipedia, Wikivoyage) and keeps
its id and URL, so every stop in a plan can be traced back to where it was found.

A journey is one to MAX_DAYS days in one city; each day is one walk with its own
Plan, and DayOutline says what the day is about. A single walk is a one-day journey.
"""

from collections.abc import Awaitable, Callable
from datetime import date, datetime, time
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

# Models sometimes answer "11:00+03:00"; every time in the app is the city's local time,
# and comparing an aware time with a naive one raises, so the zone is dropped on input.
LocalTime = Annotated[time, AfterValidator(lambda t: t.replace(tzinfo=None))]

SourceName = Literal["osm", "wikipedia", "wikivoyage"]
Pace = Literal["relaxed", "normal", "active"]
# Where an interest came from: the request text, the planner's "больше" panel, or the
# profile (app/personalization.py merges them); the UI shows it next to the interest.
InterestSource = Literal["request", "walk", "profile"]


# Interests weighted at least this (a plain or a strong wish on the BRIEF prompt's
# scale) must be covered: finish_research is refused while one has no candidate, and
# the validator warns when no stop serves one. RESEARCHER states the same number.
MUST_HAVE_WEIGHT = 0.7
# The longest trip planned at once. Each day is a research run of its own, so the
# days bound the cost of one request; a week covers a city break with room to spare.
MAX_DAYS = 7


class Interest(BaseModel):
    label: str = Field(..., description="What the traveller wants, in their words: 'street art'")
    weight: float = Field(
        1.0, ge=0, le=1, description="1.0 strong wish, 0.7 plain wish, 0.4 optional"
    )
    # Set by code, never by a model: hidden from the JSON Schema that the brief and the
    # replanner send to the LLM, so the model cannot claim an interest came from the profile.
    source: SkipJsonSchema[InterestSource] = "request"


class TripBrief(BaseModel):
    """The structured version of the traveller's free-text request."""

    city: str
    country: str | None = None
    language: Literal["ru", "en"] = "ru"
    date: date
    date_to: date | None = Field(None, description="Last day of a multi-day trip, inclusive")
    start_time: LocalTime
    end_time: LocalTime
    start_point: str | None = Field(None, description="Address or landmark, free text")
    interests: list[Interest] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list, description="Things to skip: 'museums'")
    pace: Pace = "normal"
    with_kids: bool = False
    max_walk_km: float | None = None
    must_include: list[str] = Field(default_factory=list, description="Named places to keep")
    notes: str | None = None


class GeoPoint(BaseModel):
    lat: float
    lon: float
    label: str | None = None


class DayOutline(BaseModel):
    """One day of a trip as the day planner (app/agent/outline.py) laid it out."""

    index: int = Field(..., description="0-based")
    date: date
    theme: str = Field("", description="What the day is about; empty for a one-day trip")
    focus: list[str] = Field(default_factory=list, description="Brief interest labels")
    area: str | None = Field(None, description="District or landmark; None = the trip's start")
    must_include: list[str] = Field(
        default_factory=list, description="The brief's named places assigned to this day"
    )
    start: GeoPoint | None = Field(None, description="Where the day's walk begins")


class Place(BaseModel):
    id: str = Field(..., description="'osm:node/123', 'wikipedia:ru:4567', 'wikivoyage:ru:...'")
    name: str
    lat: float
    lon: float
    kind: str = Field(..., description="Normalised category: museum, artwork, cafe, park, ...")
    source: SourceName
    url: str | None = None
    description: str | None = Field(
        None, description="Text taken from the source, used for grounding"
    )
    opening_hours: str | None = Field(None, description="OSM opening_hours syntax, if known")
    indoor: bool | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A place the research agent decided is worth visiting, with its reasons."""

    place: Place
    score: float = Field(..., ge=0, le=1)
    interests: list[str] = Field(default_factory=list)
    reason: str
    visit_minutes: int = Field(..., ge=5, le=240)
    required: bool = Field(False, description="The traveller named this place: keep it in the plan")


class HourWeather(BaseModel):
    time: datetime
    temperature_c: float | None = None
    precipitation_probability: int | None = None
    precipitation_mm: float | None = None


class Stop(BaseModel):
    place: Place
    arrive: datetime
    depart: datetime
    walk_minutes_from_prev: int = 0
    walk_meters_from_prev: int = 0
    score: float
    interests: list[str] = Field(default_factory=list)
    reason: str
    blurb: str | None = None
    tip: str | None = None


class Violation(BaseModel):
    code: Literal[
        "closed_at_visit",
        "over_time",
        "unknown_hours",
        "rain_outdoor",
        "avoid_violation",
        "long_walk",
        "uncovered_interest",
        "too_few_stops",
        "missing_must_include",
    ]
    severity: Literal["error", "warning"]
    message: str
    place_id: str | None = None


class Plan(BaseModel):
    brief: TripBrief
    start: GeoPoint
    stops: list[Stop]
    total_walk_m: int
    total_minutes: int
    geometry: list[list[float]] | None = Field(None, description="Walking polyline as [lat, lon]")
    violations: list[Violation] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list, description="Candidate ids that did not fit")
    title: str | None = None
    summary: str | None = None


EventType = Literal[
    "stage",
    "thought",
    "tool_call",
    "tool_result",
    "candidate",
    "warning",
    "memory",
    "journey",
    "error",
    "done",
]


class AgentEvent(BaseModel):
    """One line of the live trace streamed to the UI."""

    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)


# How every layer reports progress: the pipeline and the agent call it, the API streams it.
Emit = Callable[[AgentEvent], Awaitable[None]]


def clip(text: str, limit: int) -> str:
    """One line of at most `limit` characters, cut with an ellipsis: what a model or a
    source wrote, fitted into a snippet, a reason or a title."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
