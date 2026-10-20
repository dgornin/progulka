"""Traveller feedback on a plan -> a structured edit the pipeline can apply.

The model resolves references ("the second museum", "the cathedral") against a
numbered list of the current stops and says what changes; it does not rebuild the
route itself. Code then re-runs research (only when research_goal is set), the
optimizer and the validator, so a re-plan obeys the same checks as the first plan.
What code cannot apply (ids that are not stops, an empty time window) is dropped,
and a note in the reply says so: the model wrote the reply before these checks.
"""

from pydantic import BaseModel, Field

from app.agent.prompts import REPLANNER
from app.llm import LLM, SHORT_REQUEST_TIMEOUT_S
from app.models import Interest, LocalTime, Plan

NOTES = {
    "ru": {
        "stops": " Не нашёл в маршруте остановок, которые нужно убрать.",
        "time": " Время оставляю прежним: окно {start}–{end} получилось пустым.",
    },
    "en": {
        "stops": " I did not find the stops to remove in the route.",
        "time": " The times stay as they were: the window {start}–{end} would be empty.",
    },
}


class PlanEdit(BaseModel):
    """What the traveller's feedback changes; empty fields mean "no change"."""

    exclude_place_ids: list[str] = Field(
        default_factory=list, description="Ids of current stops to remove, exactly as listed"
    )
    add_interests: list[Interest] = Field(default_factory=list)
    remove_interests: list[str] = Field(
        default_factory=list, description="Interest labels as listed in the brief"
    )
    add_avoid: list[str] = Field(default_factory=list)
    start_time: LocalTime | None = Field(None, description="New start, HH:MM; null = unchanged")
    end_time: LocalTime | None = Field(None, description="New end, HH:MM; null = unchanged")
    rain_from: LocalTime | None = Field(
        None, description="The traveller expects rain from this time, HH:MM; null = no rain"
    )
    research_goal: str | None = Field(
        None, description="What the research agent must find; null when no new places are needed"
    )
    reply: str = Field(..., description="1-2 sentences to the traveller about what will change")


async def interpret_feedback(llm: LLM, plan: Plan, feedback: str) -> PlanEdit:
    messages = [
        {"role": "system", "content": REPLANNER},
        {"role": "user", "content": _context(plan) + f"\n\nFeedback: {feedback}"},
    ]
    edit = await llm.structured(messages, PlanEdit, timeout=SHORT_REQUEST_TIMEOUT_S)
    notes = NOTES[plan.brief.language]

    in_plan = {stop.place.id for stop in plan.stops}
    asked = list(dict.fromkeys(edit.exclude_place_ids))
    edit.exclude_place_ids = [i for i in asked if i in in_plan]
    if asked and not edit.exclude_place_ids:
        edit.reply += notes["stops"]
    start = edit.start_time or plan.brief.start_time
    end = edit.end_time or plan.brief.end_time
    if end <= start:  # an empty window cannot be planned: keep the current times
        edit.start_time = edit.end_time = None
        edit.reply += notes["time"].format(start=f"{start:%H:%M}", end=f"{end:%H:%M}")
    return edit


def _context(plan: Plan) -> str:
    brief = plan.brief
    interests = ", ".join(f"{i.label} ({i.weight:g})" for i in brief.interests) or "none"
    lines = [
        f"Brief language: {brief.language}",
        f"City: {brief.city}; date {brief.date:%Y-%m-%d}; "
        f"window {brief.start_time:%H:%M}-{brief.end_time:%H:%M}",
        f"Interests: {interests}",
        f"Avoid: {', '.join(brief.avoid) or 'nothing'}",
        "Current stops:",
    ]
    for n, stop in enumerate(plan.stops, start=1):
        place = stop.place
        where = {True: "indoor", False: "outdoor", None: "indoor/outdoor unknown"}[place.indoor]
        lines.append(
            f"{n}. id={place.id} | {place.name} | {place.kind} | "
            f"{stop.arrive:%H:%M}-{stop.depart:%H:%M} | {where}"
        )
    return "\n".join(lines)
