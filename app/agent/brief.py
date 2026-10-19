"""Free-text request -> TripBrief.

The model only extracts what the traveller said, into a draft where every unknown
is null. Defaults (today, 10:00, five hours, normal pace), the form's overrides and
the sanity checks are applied here in code, so they are the same on every run and
do not depend on how the model reads the request.
"""

import datetime as dt
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompts import BRIEF
from app.llm import LLM
from app.models import Interest, LocalTime, Pace, TripBrief

DEFAULT_START = dt.time(10, 0)
DEFAULT_HOURS = 5.0
LATEST_END = dt.time(23, 59)


class BriefDraft(BaseModel):
    """What the model extracts; null means the request does not say it."""

    city: str | None = Field(
        None, description="Full city name in the nominative: 'по Питеру' -> 'Санкт-Петербург'"
    )
    country: str | None = None
    language: Literal["ru", "en"] | None = None
    date: dt.date | None = Field(None, description="YYYY-MM-DD, relative dates resolved")
    start_time: LocalTime | None = Field(None, description="HH:MM")
    end_time: LocalTime | None = Field(None, description="HH:MM")
    duration_hours: float | None = Field(
        None, gt=0, le=14, description="Only when a length is given instead of an end time"
    )
    start_point: str | None = Field(None, description="Address or landmark to start from")
    interests: list[Interest] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    pace: Pace | None = None
    with_kids: bool = False
    max_walk_km: float | None = Field(None, gt=0)
    must_include: list[str] = Field(default_factory=list)
    notes: str | None = None


class BriefOverrides(BaseModel):
    """Fields the traveller set explicitly in the form; they win over the model."""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    date: dt.date | None = None
    start_time: LocalTime | None = None
    end_time: LocalTime | None = None
    start_point: str | None = None
    language: Literal["ru", "en"] | None = None


async def parse_brief(
    llm: LLM, request_text: str, today: dt.date, overrides: dict[str, Any] | None = None
) -> TripBrief:
    """Raise ValueError when neither the request nor the overrides name a city."""
    fixed = BriefOverrides.model_validate(overrides or {})
    prompt = f"Today is {today.isoformat()} ({today.strftime('%A')}).\nRequest:\n{request_text}"
    draft = await llm.structured(
        [{"role": "system", "content": BRIEF}, {"role": "user", "content": prompt}], BriefDraft
    )

    city = fixed.city or draft.city
    if not city:
        raise ValueError("the request does not name a city")
    # A date in the past is a misread weekday: there is nothing to plan for it.
    model_date = draft.date if draft.date and draft.date >= today else today
    start = fixed.start_time or draft.start_time or DEFAULT_START
    end = (
        fixed.end_time or draft.end_time or _add_hours(start, draft.duration_hours or DEFAULT_HOURS)
    )
    if end <= start:
        end = _add_hours(start, DEFAULT_HOURS)

    return TripBrief(
        city=city,
        country=draft.country,
        language=fixed.language or draft.language or _detect_language(request_text),
        date=fixed.date or model_date,
        start_time=start,
        end_time=end,
        start_point=fixed.start_point or draft.start_point,
        interests=draft.interests,
        avoid=draft.avoid,
        pace=draft.pace or "normal",
        with_kids=draft.with_kids,
        max_walk_km=draft.max_walk_km,
        must_include=draft.must_include,
        notes=draft.notes,
    )


def _add_hours(start: dt.time, hours: float) -> dt.time:
    """start + hours, capped at the end of the same day (a walk does not cross midnight)."""
    base = dt.datetime.combine(dt.date(2000, 1, 1), start)
    end = base + dt.timedelta(hours=hours)
    return end.time() if end.date() == base.date() else LATEST_END


def _detect_language(text: str) -> Literal["ru", "en"]:
    return "ru" if re.search("[а-яё]", text, re.IGNORECASE) else "en"
