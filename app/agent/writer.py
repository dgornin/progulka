"""Titles and per-stop texts for a finished plan, in the traveller's language.

The writer runs after the optimizer and the validator, so it can only decorate the
route, never change it. Its input is the source text of every stop, and it is told
to use nothing else. The output is checked against the plan (unknown place ids are
ignored), and an LLM failure or an answer slower than WRITER_TIMEOUT_S returns the
plan without texts instead of failing the request: the UI then shows each stop's
research reason.
"""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.agent.prompts import WRITER
from app.llm import LLM, LLMError
from app.models import Plan, Stop

SOURCE_TEXT_CHARS = 1200
WRITER_TIMEOUT_S = 60
# Tags the sources keep (osm.KEPT_TAGS, Wikivoyage's price) that describe what a
# visitor sees or needs. Address and contact tags are left out: they add nothing to
# a blurb and invite the model to pad it. Architect, artist and date are already in
# the OSM description, which goes in as source_text.
FACT_TAGS = ("artwork_type", "building:architecture", "cuisine", "fee", "price", "wheelchair")


class StopText(BaseModel):
    place_id: str
    blurb: str = Field(..., description="2-3 sentences")
    tip: str | None = Field(None, description="One practical sentence from the data, or null")


class PlanTexts(BaseModel):
    title: str
    summary: str
    stops: list[StopText]


async def write_texts(llm: LLM, plan: Plan, traveller_facts: Sequence[str] = ()) -> Plan:
    """`traveller_facts`: long-term memory and profile notes, so the texts can speak to
    the traveller's known tastes; the prompt forbids using them as facts about places."""
    if not plan.stops:
        return plan
    payload = {
        "language": plan.brief.language,
        "traveller": {
            "interests": [i.label for i in plan.brief.interests],
            "with_kids": plan.brief.with_kids,
            "notes": plan.brief.notes,
            "known": list(traveller_facts),
        },
        "city": plan.brief.city,
        "stops": [_stop_input(stop) for stop in plan.stops],
    }
    messages = [
        {"role": "system", "content": WRITER},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        async with asyncio.timeout(WRITER_TIMEOUT_S):
            texts = await llm.structured(messages, PlanTexts)
    except (LLMError, TimeoutError):
        return plan

    by_id = {entry.place_id: entry for entry in texts.stops if entry.blurb.strip()}
    stops = []
    for stop in plan.stops:
        text = by_id.get(stop.place.id)
        if text:
            tip = (text.tip or "").strip() or None
            stop = stop.model_copy(update={"blurb": text.blurb.strip(), "tip": tip})
        stops.append(stop)
    return plan.model_copy(update={"title": texts.title, "summary": texts.summary, "stops": stops})


def _stop_input(stop: Stop) -> dict[str, Any]:
    place = stop.place
    item: dict[str, Any] = {
        "place_id": place.id,
        "name": place.name,
        "kind": place.kind,
        "time": f"{stop.arrive:%H:%M}-{stop.depart:%H:%M}",
        "interests": stop.interests,
        "reason": stop.reason,
        "source_text": (place.description or "")[:SOURCE_TEXT_CHARS] or None,
        "opening_hours": place.opening_hours,
        "facts": {k: place.tags[k] for k in FACT_TAGS if k in place.tags} or None,
    }
    return {k: v for k, v in item.items() if v is not None}
