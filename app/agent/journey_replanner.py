"""One coordinator translates a trip-wide request into bounded tasks for day agents.

Transfers name existing stops by id, so language interpretation cannot invent a
place or silently move one from a day that was not part of the request.
"""

import json
from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from app.agent.prompts import JOURNEY_REPLANNER
from app.agent.replanner import PlanEdit, _context
from app.llm import LLM, SHORT_REQUEST_TIMEOUT_S
from app.models import MAX_DAYS, DayOutline, Plan


class DayEdit(PlanEdit):
    day_index: int = Field(ge=0, lt=MAX_DAYS, description="Zero-based index from the trip")
    theme: str | None = Field(None, max_length=160, description="New day theme, only if changed")


class PlaceMove(BaseModel):
    source_day: int = Field(ge=0, lt=MAX_DAYS)
    target_day: int = Field(ge=0, lt=MAX_DAYS)
    place_id: str = Field(..., max_length=300, description="Exact id of a source day's stop")


class JourneyEdit(BaseModel):
    days: list[DayEdit] = Field(default_factory=list, max_length=MAX_DAYS)
    moves: list[PlaceMove] = Field(default_factory=list, max_length=100)
    reply: str = Field(..., max_length=1000)


async def interpret_journey_feedback(
    llm: LLM, plans: Sequence[Plan], outline: Sequence[DayOutline], feedback: str
) -> JourneyEdit:
    context = [
        {
            "day_index": i,
            "weekday": plan.brief.date.strftime("%A"),
            "theme": outline[i].theme,
            "route": _context(plan),
        }
        for i, plan in enumerate(plans)
    ]
    return await llm.structured(
        [
            {"role": "system", "content": JOURNEY_REPLANNER},
            {
                "role": "user",
                "content": json.dumps({"days": context, "feedback": feedback}, ensure_ascii=False),
            },
        ],
        JourneyEdit,
        timeout=SHORT_REQUEST_TIMEOUT_S,
    )


def validate_edit(edit: JourneyEdit, plans: Sequence[Plan]) -> list[int]:
    """Reject the entire edit before any day changes; dropping one bad move could
    turn a requested swap into a one-way move or remove the wrong stop."""
    changed: set[int] = set()
    edits: dict[int, DayEdit] = {}
    for day in edit.days:
        i = day.day_index
        if i >= len(plans) or i in edits:
            raise ValueError("Не удалось однозначно определить дни правки. Уточните запрос.")
        edits[i] = day
        changed.add(i)
        stops = {s.place.id for s in plans[i].stops}
        if not set(day.exclude_place_ids) <= stops:
            raise ValueError("Не удалось найти выбранные места в маршруте. Уточните запрос.")
        brief = plans[i].brief
        start = datetime.combine(brief.date, day.start_time or brief.start_time)
        end = datetime.combine(brief.date, day.end_time or brief.end_time)
        if end - start < timedelta(minutes=30):
            raise ValueError(f"День {i + 1}: оставьте не меньше 30 минут на прогулку.")

    moved: set[str] = set()
    for move in edit.moves:
        source, target = move.source_day, move.target_day
        if source >= len(plans) or target >= len(plans) or source == target:
            raise ValueError("Не удалось определить дни переноса. Уточните запрос.")
        stops = {s.place.id for s in plans[source].stops}
        if move.place_id not in stops or move.place_id in moved:
            raise ValueError("Место для переноса отсутствует или указано несколько раз.")
        if any(move.place_id in day.exclude_place_ids for day in edit.days):
            raise ValueError("Нельзя одновременно удалить и перенести одно место.")
        moved.add(move.place_id)
        changed.update((source, target))
    if not changed:
        raise ValueError(edit.reply or "Уточните, какие дни или места нужно изменить.")
    return sorted(changed)
