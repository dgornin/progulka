"""Free-text request -> TripBrief.

The model only extracts what the traveller said, into a draft where every unknown
is null. Everything else is code, so it is the same on every run and does not
depend on how the model reads the request: each field is the first value set in
the form's overrides, the draft, the profile's defaults, the hard defaults (today,
10:00, five hours, normal pace), followed by the sanity checks.

Dates of a multi-day trip: the model reports the range as said ("с 24 по 28 октября"
-> both dates, "5 дней" -> a count) and code turns it into date..date_to. The length
said in the text survives a start date moved in the form's calendar: "5 дней" with
the form's 1 November is 1-5 November. The cap on the length is the pipeline's.

An empty request makes no LLM call at all: there is nothing to extract, and the
form and the profile already say everything the brief needs. That walk is cheaper,
faster and fully deterministic; its interests come from the panel and the profile
(app/personalization.py).
"""

import datetime as dt
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompts import BRIEF
from app.llm import LLM
from app.models import Interest, LocalTime, Pace, TripBrief
from app.profile import INTERESTS

DEFAULT_START = dt.time(10, 0)
DEFAULT_HOURS = 8.0  # 10:00–18:00, a travel day; the planner slider shows the same default
LATEST_END = dt.time(23, 59)


class BriefDraft(BaseModel):
    """What the model extracts; null means the request does not say it."""

    city: str | None = Field(
        None, description="Full city name in the nominative: 'по Питеру' -> 'Санкт-Петербург'"
    )
    country: str | None = None
    language: Literal["ru", "en"] | None = None
    date: dt.date | None = Field(
        None, description="YYYY-MM-DD, the first day of the trip, relative dates resolved"
    )
    date_to: dt.date | None = Field(
        None, description="YYYY-MM-DD, the last day when the request names an end date"
    )
    days: int | None = Field(
        None, ge=1, description="Only when the length is a count of days: '5 дней' -> 5"
    )
    start_time: LocalTime | None = Field(None, description="HH:MM, daily")
    end_time: LocalTime | None = Field(None, description="HH:MM, daily")
    duration_hours: float | None = Field(
        None, gt=0, le=14, description="Only when a length is given instead of an end time"
    )
    start_point: str | None = Field(None, description="Address or landmark to start from")
    interests: list[Interest] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    pace: Pace | None = None
    with_kids: bool | None = Field(
        None, description="true or false only when the request says whether children come along"
    )
    max_walk_km: float | None = Field(None, gt=0)
    must_include: list[str] = Field(default_factory=list)
    notes: str | None = None


class BriefOverrides(BaseModel):
    """Fields the traveller set explicitly in the form; they win over the model."""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    date: dt.date | None = None
    date_to: dt.date | None = None
    start_time: LocalTime | None = None
    end_time: LocalTime | None = None
    start_point: str | None = None
    pace: Pace | None = None
    with_kids: bool | None = None
    max_walk_km: float | None = None


class BriefDefaults(BaseModel):
    """Fields from the traveller's profile; used only where the request says nothing."""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    pace: Pace | None = None
    with_kids: bool | None = None
    max_walk_km: float | None = None


async def parse_brief(
    llm: LLM,
    request_text: str,
    today: dt.date,
    overrides: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> TripBrief:
    """Raise ValueError when neither the request, the overrides nor the defaults name a city."""
    fixed = BriefOverrides.model_validate(overrides or {})
    fallback = BriefDefaults.model_validate(defaults or {})
    draft = await _extract(llm, request_text, today) if request_text.strip() else BriefDraft()

    city = fixed.city or draft.city or fallback.city
    if not city:
        raise ValueError("the request does not name a city")
    # A date in the past is a misread weekday: there is nothing to plan for it.
    model_date = draft.date if draft.date and draft.date >= today else today
    first = fixed.date or model_date
    start = fixed.start_time or draft.start_time or DEFAULT_START
    end = (
        fixed.end_time or draft.end_time or _add_hours(start, draft.duration_hours or DEFAULT_HOURS)
    )
    if end <= start:
        if fixed.end_time and not fixed.start_time:
            # The end set in the form outranks a start from the text or the default.
            start = _add_hours(end, -DEFAULT_HOURS)
        else:
            end = _add_hours(start, DEFAULT_HOURS)

    return TripBrief(
        city=city,
        country=draft.country,
        language=draft.language or _detect_language(request_text),
        date=first,
        date_to=_last_day(first, fixed.date_to, draft),
        start_time=start,
        end_time=end,
        start_point=fixed.start_point or draft.start_point,
        # Rebuilt so that the source is set by code (see Interest.source).
        interests=[Interest(label=i.label, weight=i.weight) for i in draft.interests],
        avoid=draft.avoid,
        pace=_first(fixed.pace, draft.pace, fallback.pace) or "normal",
        with_kids=_first(fixed.with_kids, draft.with_kids, fallback.with_kids) or False,
        max_walk_km=_first(fixed.max_walk_km, draft.max_walk_km, fallback.max_walk_km),
        must_include=draft.must_include,
        notes=draft.notes,
    )


async def _extract(llm: LLM, request_text: str, today: dt.date) -> BriefDraft:
    system = BRIEF.format(topics=", ".join(f"«{spec.label}»" for spec in INTERESTS.values()))
    prompt = f"Today is {today.isoformat()} ({today.strftime('%A')}).\nRequest:\n{request_text}"
    return await llm.structured(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}], BriefDraft
    )


def _last_day(first: dt.date, fixed: dt.date | None, draft: BriefDraft) -> dt.date | None:
    """The trip's last day, None for a one-day trip: the form's end date, else the
    length the text gives (an end date or a count of days) counted from `first`."""
    if fixed:
        last = fixed
    elif draft.date_to:
        last = first + (draft.date_to - (draft.date or first))
    elif draft.days:
        last = first + dt.timedelta(days=draft.days - 1)
    else:
        return None
    return last if last > first else None


def _first[T](*values: T | None) -> T | None:
    """The first value that is set: False and 0 count as set, only None does not."""
    return next((v for v in values if v is not None), None)


def _add_hours(t: dt.time, hours: float) -> dt.time:
    """t + hours, kept within the same day (a walk does not cross midnight)."""
    base = dt.datetime.combine(dt.date(2000, 1, 1), t)
    moved = base + dt.timedelta(hours=hours)
    if moved.date() == base.date():
        return moved.time()
    return LATEST_END if hours > 0 else dt.time(0, 0)


def _detect_language(text: str) -> Literal["ru", "en"]:
    # Only a request in Latin letters alone is English: an empty one is planned in
    # the language of the interface.
    latin = re.search("[a-z]", text, re.IGNORECASE)
    return "en" if latin and not re.search("[а-яё]", text, re.IGNORECASE) else "ru"
