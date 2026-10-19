"""Deterministic checks of a finished plan: the external critic of the agent.

A model that re-reads its own plan rarely finds its mistakes (lecture 04); it needs
a signal from outside. This module is that signal. It does not know how the plan
was built: it re-checks every stop against opening hours, the time window, the
forecast, the walking limit and the brief, and returns concrete violations
("«Эрмитаж»: в 10:00–11:00 закрыто") the repair loop can act on. Errors must be
fixed; warnings are shown to the traveller.
"""

import re
from datetime import datetime, timedelta

from app.models import HourWeather, Plan, Stop, TripBrief, Violation
from app.planning.geo import walk_limit_km
from app.planning.hours import is_open, open_intervals

RAIN_PROBABILITY = 50  # %
RAIN_MM = 0.3  # per hour
INTEREST_MUST_HAVE = 0.7

# Kinds whose visit fails if the place is closed. A monument or a park is
# "open" from the street, so missing hours there are not worth a warning.
HOURS_MATTER = (
    {"museum", "gallery", "theatre", "cinema", "library", "zoo", "aquarium", "attraction"}
    | {"cafe", "restaurant", "street_food", "bakery", "food", "bar", "pub", "ice_cream"}
    | {"shop", "bookshop", "market"}
)

# Kinds excluded by common "avoid" phrases (ru/en), found by word stems:
# a word of the phrase matches when it starts with a stem ("без музеев" -> "музе").
AVOID_RULES: list[tuple[set[str], tuple[str, ...]]] = [
    ({"museum", "gallery"}, ("музе", "museum")),
    ({"gallery"}, ("галере", "galler")),
    ({"church"}, ("церк", "храм", "собор", "church", "cathedral")),
    ({"bar", "pub"}, ("бар", "паб", "алкогол", "пив", "bar", "pub", "alcohol", "beer")),
    ({"theatre"}, ("театр", "theat")),
    ({"cinema"}, ("кино", "cinema")),
    ({"park", "garden"}, ("парк", "park")),
    ({"shop"}, ("магазин", "шопинг", "shop")),
    ({"market"}, ("рын", "market")),
    ({"monument", "memorial"}, ("памятник", "мемориал", "monument", "memorial")),
]

MESSAGES = {
    "ru": {
        "closed_at_visit": "«{name}»: в {begin}–{end} закрыто; {hours}",
        "open_hours": "в этот день открыто {intervals}",
        "closed_all_day": "в этот день не работает",
        "over_time": "Маршрут заканчивается в {end}, позже {limit}",
        "unknown_hours": "Часы работы «{name}» неизвестны — проверьте перед визитом в {begin}",
        "rain_outdoor": "«{name}» под открытым небом, а в {begin}–{end} ожидается дождь",
        "avoid_violation": "«{name}» — из того, что вы просили исключить: «{avoid}»",
        "long_walk": "Пешком {km:.1f} км — больше лимита {limit:g} км",
        "uncovered_interest": "В маршруте нет ничего про «{label}»",
        "too_few_stops": "В маршруте меньше двух остановок",
    },
    "en": {
        "closed_at_visit": "{name} is closed at {begin}–{end}; {hours}",
        "open_hours": "open {intervals} that day",
        "closed_all_day": "closed all day",
        "over_time": "The route ends at {end}, after {limit}",
        "unknown_hours": "Opening hours of {name} are unknown — check before visiting at {begin}",
        "rain_outdoor": "{name} is outdoors and rain is expected at {begin}–{end}",
        "avoid_violation": "{name} matches what you asked to skip: “{avoid}”",
        "long_walk": "{km:.1f} km on foot, over the {limit:g} km limit",
        "uncovered_interest": "No stop covers “{label}”",
        "too_few_stops": "The route has fewer than two stops",
    },
}


def rain_during(weather: list[HourWeather] | None, start: datetime, end: datetime) -> bool:
    """True if any forecast hour overlapping [start, end) is rainy."""
    return any(
        h.time < end
        and h.time + timedelta(hours=1) > start
        and (
            (h.precipitation_probability or 0) >= RAIN_PROBABILITY
            or (h.precipitation_mm or 0) >= RAIN_MM
        )
        for h in weather or []
    )


def validate(plan: Plan, weather: list[HourWeather] | None = None) -> list[Violation]:
    """All violations of the plan, errors first."""
    text = MESSAGES[plan.brief.language]
    found: list[Violation] = []
    for stop in plan.stops:
        found += _stop_violations(stop, plan.brief, weather, text)
    found += _route_violations(plan, text)
    return sorted(found, key=lambda v: v.severity != "error")


def has_errors(violations: list[Violation]) -> bool:
    return any(v.severity == "error" for v in violations)


def _stop_violations(
    stop: Stop, brief: TripBrief, weather: list[HourWeather] | None, text: dict[str, str]
) -> list[Violation]:
    place = stop.place
    when = {"name": place.name, "begin": _hhmm(stop.arrive), "end": _hhmm(stop.depart)}
    found = []
    state = is_open(place.opening_hours, stop.arrive, stop.depart)
    if state is False:
        intervals = open_intervals(place.opening_hours, stop.arrive.date())
        hours = (
            text["open_hours"].format(
                intervals=", ".join(f"{_hhmm(s)}–{_hhmm(e)}" for s, e in intervals)
            )
            if intervals
            else text["closed_all_day"]
        )
        found.append(_violation("closed_at_visit", "error", text, place.id, hours=hours, **when))
    elif state is None and place.kind in HOURS_MATTER:
        found.append(_violation("unknown_hours", "warning", text, place.id, **when))
    if place.indoor is False and rain_during(weather, stop.arrive, stop.depart):
        found.append(_violation("rain_outdoor", "warning", text, place.id, **when))
    for avoid in brief.avoid:
        if _matches_avoid(avoid, place.kind, place.name):
            found.append(
                _violation("avoid_violation", "error", text, place.id, avoid=avoid, **when)
            )
            break
    return found


def _route_violations(plan: Plan, text: dict[str, str]) -> list[Violation]:
    brief = plan.brief
    found = []
    if len(plan.stops) < 2:
        found.append(_violation("too_few_stops", "error", text))
    limit = datetime.combine(brief.date, brief.end_time)
    if plan.stops and plan.stops[-1].depart > limit:
        found.append(
            _violation(
                "over_time", "error", text, end=_hhmm(plan.stops[-1].depart), limit=_hhmm(limit)
            )
        )
    walk_limit = walk_limit_km(brief)
    if plan.total_walk_m > walk_limit * 1000:
        found.append(
            _violation("long_walk", "warning", text, km=plan.total_walk_m / 1000, limit=walk_limit)
        )
    covered = {label.strip().lower() for stop in plan.stops for label in stop.interests}
    for interest in brief.interests:
        if interest.weight >= INTEREST_MUST_HAVE and interest.label.strip().lower() not in covered:
            found.append(_violation("uncovered_interest", "warning", text, label=interest.label))
    return found


def _matches_avoid(avoid: str, kind: str, name: str) -> bool:
    """The place's kind is excluded by the phrase, or the phrase is part of its name.

    The name must contain the phrase as whole words: "кафе" is not in "Кафедральный
    собор", "pub" is not in "Public Library". A false match is an error that makes
    the agent drop a good place.
    """
    phrase = avoid.strip().lower()
    if len(phrase) >= 3 and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", name.lower()):
        return True
    words = re.findall(r"\w+", phrase)
    return any(
        kind in kinds and any(w.startswith(stems) for w in words) for kinds, stems in AVOID_RULES
    )


def _violation(
    code: str, severity: str, text: dict[str, str], place_id: str | None = None, **fields: object
) -> Violation:
    return Violation(
        code=code, severity=severity, message=text[code].format(**fields), place_id=place_id
    )


def _hhmm(t: datetime) -> str:
    return t.strftime("%H:%M")
