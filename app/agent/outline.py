"""The day planner of a multi-day trip: the "plan" of Plan-and-Execute (lecture 04).

Before any place is searched, one structured LLM call splits the trip into days: a
theme, the brief's interests the day emphasises, the part of the city it goes to and
the named places that belong to it. Each day is then executed by a research agent of
its own (app/pipeline.py), and feedback on a day is replanned on that day alone.

Why a plan first, and not one agent for the whole trip: five days are 40+ places and
dozens of tool calls in one context, far past where a model keeps track; and without
themes and areas every day would open with the same famous sights next to the hotel.

The model proposes, code checks what it can:
- focus labels must be the brief's (unknown ones are dropped), and every interest
  the traveller must get (weight >= MUST_HAVE_WEIGHT) is the focus of some day: a
  forgotten one goes to the day with the fewest;
- each named place (must_include) is on exactly one day: a forgotten one goes to
  the day with the fewest named places;
- an area is geocoded inside the city; when it is not found, or lies more than
  MAX_AREA_KM from the start (a namesake in another region), the day starts from the
  trip's start point instead.
A one-day trip is a single walk and needs no plan: code builds its outline.
"""

import asyncio
import json
import logging
from datetime import date, timedelta

from pydantic import BaseModel, Field

from app.agent.prompts import OUTLINE
from app.llm import LLM
from app.models import MUST_HAVE_WEIGHT, DayOutline, GeoPoint, TripBrief
from app.planning.geo import haversine_m
from app.sources import geocode
from app.sources.http import SourceError

log = logging.getLogger(__name__)

TITLE_CHARS = 60
THEME_CHARS = 80
MAX_AREA_KM = 15
AREA_TIMEOUT_S = 30


class DayDraft(BaseModel):
    theme: str = Field(..., description=f"At most {THEME_CHARS} characters")
    focus: list[str] = Field(default_factory=list, description="Interest labels, exactly as given")
    area: str | None = Field(None, description="District or landmark; null = the start point")
    must_include: list[str] = Field(
        default_factory=list, description="Named places from the input, exactly as given"
    )


class OutlineDraft(BaseModel):
    title: str = Field(..., description=f"At most {TITLE_CHARS} characters")
    days: list[DayDraft] = Field(..., description="One entry per day, in order")


def trip_dates(brief: TripBrief) -> list[date]:
    last = brief.date_to or brief.date
    return [brief.date + timedelta(days=n) for n in range((last - brief.date).days + 1)]


def one_day(brief: TripBrief, start: GeoPoint) -> DayOutline:
    """The outline of a single walk: the whole brief on one day, no LLM call."""
    return DayOutline(
        index=0,
        date=brief.date,
        focus=[i.label for i in brief.interests],
        must_include=list(brief.must_include),
        start=start,
    )


async def draft_days(
    llm: LLM, brief: TripBrief, start: GeoPoint, forecasts: list[str]
) -> tuple[str, list[DayOutline]]:
    """The trip's title and its days, checked against the brief; areas not located yet.
    `forecasts`: one weather summary per day."""
    dates = trip_dates(brief)
    payload = {
        "city": brief.city,
        "language": brief.language,
        "start": start.label,
        "days": [
            {"date": day.isoformat(), "weekday": day.strftime("%A"), "weather": forecast}
            for day, forecast in zip(dates, forecasts, strict=True)
        ],
        "daily_window": f"{brief.start_time:%H:%M}-{brief.end_time:%H:%M}",
        "interests": [{"label": i.label, "weight": i.weight} for i in brief.interests],
        "avoid": brief.avoid,
        "must_include": brief.must_include,
        "pace": brief.pace,
        "with_kids": brief.with_kids,
        "notes": brief.notes,
    }
    messages = [
        {"role": "system", "content": OUTLINE.format(max_km=MAX_AREA_KM)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    draft = await llm.structured(messages, OutlineDraft)
    title = _short(draft.title, TITLE_CHARS) or f"{brief.city}, {dates[0]:%d.%m}–{dates[-1]:%d.%m}"
    return title, _checked(draft, brief, dates)


async def locate_areas(
    days: list[DayOutline], brief: TripBrief, start: GeoPoint
) -> list[DayOutline]:
    """Every day with its start point: its area's, or the trip's when the area is unknown."""

    async def located(day: DayOutline) -> DayOutline:
        point = await _locate(day.area, brief, start) if day.area else None
        return day.model_copy(update={"area": day.area if point else None, "start": point or start})

    return list(await asyncio.gather(*(located(day) for day in days)))


def _checked(draft: OutlineDraft, brief: TripBrief, dates: list[date]) -> list[DayOutline]:
    labels = {i.label.casefold(): i.label for i in brief.interests}
    named = {name.casefold(): name for name in brief.must_include}
    assigned: set[str] = set()
    days = []
    for index, day in enumerate(dates):
        # A model that returns fewer days than asked leaves the rest without a theme.
        proposal = draft.days[index] if index < len(draft.days) else DayDraft(theme="")
        focus = [labels[f.casefold()] for f in proposal.focus if f.casefold() in labels]
        places = []
        for name in proposal.must_include:
            if name.casefold() in named and name.casefold() not in assigned:
                assigned.add(name.casefold())
                places.append(named[name.casefold()])
        days.append(
            DayOutline(
                index=index,
                date=day,
                theme=_short(proposal.theme, THEME_CHARS),
                focus=list(dict.fromkeys(focus)),
                area=_short(proposal.area or "", THEME_CHARS) or None,
                must_include=places,
            )
        )
    # What the model forgot still gets a day: the one with the least on it (the first
    # of equals, so the result does not depend on anything but the input).
    for interest in brief.interests:
        if interest.weight >= MUST_HAVE_WEIGHT and not any(interest.label in d.focus for d in days):
            min(days, key=lambda d: len(d.focus)).focus.append(interest.label)
    for key, name in named.items():
        if key not in assigned:
            min(days, key=lambda d: len(d.must_include)).must_include.append(name)
    return days


async def _locate(area: str, brief: TripBrief, start: GeoPoint) -> GeoPoint | None:
    try:
        async with asyncio.timeout(AREA_TIMEOUT_S):
            found = await geocode.geocode(area, city=brief.city, country=brief.country)
    except (SourceError, TimeoutError):
        found = None
    if found and haversine_m(start.lat, start.lon, found.lat, found.lon) <= MAX_AREA_KM * 1000:
        return found
    # Not an error for the traveller: the day starts from the hotel instead.
    log.info("outline: area %r in %s is %s", area, brief.city, "too far" if found else "unknown")
    return None


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
