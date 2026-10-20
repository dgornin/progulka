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

from app.models import MUST_HAVE_WEIGHT, HourWeather, Plan, Stop, TripBrief, Violation
from app.planning.geo import walk_limit_km
from app.planning.hours import is_open, open_intervals
from app.sources.weather import is_rainy

# Kinds whose visit fails if the place is closed. A monument or a park is
# "open" from the street, so missing hours there are not worth a warning.
HOURS_MATTER = (
    {"museum", "gallery", "theatre", "cinema", "library", "zoo", "aquarium", "attraction"}
    | {"cafe", "restaurant", "street_food", "bakery", "food", "bar", "pub", "ice_cream"}
    | {"shop", "bookshop", "market"}
)

# Kinds excluded by common "avoid" phrases (ru/en), found by word stems: a word of
# the phrase matches when it is a stem plus at most STEM_ENDING letters ("без музеев"
# -> "музе"), so "бары" and "парки" match, but "барокко" and "парковка" do not.
# A place of another kind still matches when a word of its name is a long stem plus
# an ending: Wikipedia ("landmark") and Wikivoyage ("sight") kinds do not say that
# "Русский музей" is a museum. Short stems ("бар", "пив") would hit "Дом Баранова".
AVOID_RULES: list[tuple[set[str], tuple[str, ...]]] = [
    ({"museum", "gallery"}, ("музе", "museum")),
    ({"gallery"}, ("галере", "galler")),
    ({"church"}, ("церк", "храм", "собор", "church", "cathedral")),
    (
        {"bar", "pub"},
        ("бар", "паб", "алкогол", "алкогольн", "пив", "пивнуш", "bar", "pub", "alcohol", "beer"),
    ),
    ({"cafe", "bakery"}, ("кафе", "кофе", "cafe", "coffee")),
    ({"restaurant", "food"}, ("ресторан", "ресторанч", "restaurant")),
    ({"street_food"}, ("фастфуд", "стритфуд", "fast")),
    ({"theatre"}, ("театр", "theat")),
    ({"cinema"}, ("кино", "кинотеатр", "cinema")),
    ({"park", "garden"}, ("парк", "park")),
    ({"shop"}, ("магазин", "магазинч", "шопинг", "shop", "shopping")),
    ({"market"}, ("рын", "market")),
    ({"monument", "memorial"}, ("памятник", "мемориал", "monument", "memorial")),
]
STEM_ENDING = 3
NAME_STEM_MIN = 4

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
        "missing_must_include": "«{name}», о котором вы просили, не попало в маршрут",
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
        "missing_must_include": "{name}, which you asked for, is not in the route",
    },
}


def rain_during(weather: list[HourWeather] | None, start: datetime, end: datetime) -> bool:
    """True if any forecast hour overlapping [start, end) is rainy."""
    return any(
        h.time < end and h.time + timedelta(hours=1) > start and is_rainy(h) for h in weather or []
    )


def validate(plan: Plan, weather: list[HourWeather] | None = None) -> list[Violation]:
    """All violations of the plan, errors first."""
    text = MESSAGES[plan.brief.language]
    found: list[Violation] = []
    for stop in plan.stops:
        found += _stop_violations(stop, plan.brief, weather, text)
    found += _route_violations(plan, text)
    return sorted(found, key=lambda v: v.severity != "error")


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
        if matches_avoid(avoid, place.kind, place.name):
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
        if interest.weight >= MUST_HAVE_WEIGHT and interest.label.strip().lower() not in covered:
            found.append(_violation("uncovered_interest", "warning", text, label=interest.label))
    for wanted in brief.must_include:
        if not any(names_place(wanted, stop.place.name) for stop in plan.stops):
            found.append(_violation("missing_must_include", "warning", text, name=wanted))
    return found


def matches_avoid(avoid: str, kind: str, name: str) -> bool:
    """The phrase excludes the place: by its kind, by a word of its name (a long stem
    plus at most STEM_ENDING letters), or the phrase is part of the name.

    Names are matched by whole words: "кафе" is not in "Кафедральный собор", "pub" is
    not in "Public Library". A false match is an error that makes the agent drop a
    good place.
    """
    phrase = avoid.strip().lower()
    if len(phrase) >= 3 and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", name.lower()):
        return True
    name_words = _words(name)
    for kinds, stems in _rules_named(phrase):
        long_stems = [s for s in stems if len(s) >= NAME_STEM_MIN]
        if kind in kinds or any(
            _inflects(w, s, STEM_ENDING) for w in name_words for s in long_stems
        ):
            return True
    return False


def named_kinds(text: str) -> set[str]:
    """The kinds of places the words of `text` name: «без музеев» -> museum, gallery.
    Shared with the brief's merge (app/personalization.py), so that it judges conflicts
    exactly as this validator does."""
    return {kind for kinds, _ in _rules_named(text) for kind in kinds}


def names_place(wanted: str, name: str) -> bool:
    """The traveller's name for a place is the source's name for it: every word of it is
    a word of the name, give or take a case ending. "Казанский собор" is "Казанский
    кафедральный собор" and "Спас на Крови" is "Храм Спаса на Крови", but "Эрмитаж"
    is not "Эрмитажный театр".
    """
    words, have = _words(wanted), _words(name)
    return bool(words) and all(any(_inflects(h, w, 2) for h in have) for w in words)


def _rules_named(text: str) -> list[tuple[set[str], tuple[str, ...]]]:
    words = _words(text)
    return [
        (kinds, stems)
        for kinds, stems in AVOID_RULES
        if any(_inflects(w, s, STEM_ENDING) for w in words for s in stems)
    ]


def _inflects(word: str, stem: str, ending: int) -> bool:
    """`word` is `stem` plus at most `ending` letters: an inflection, not another word."""
    return word.startswith(stem) and len(word) - len(stem) <= ending


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _violation(
    code: str, severity: str, text: dict[str, str], place_id: str | None = None, **fields: object
) -> Violation:
    return Violation(
        code=code, severity=severity, message=text[code].format(**fields), place_id=place_id
    )


def _hhmm(t: datetime) -> str:
    return t.strftime("%H:%M")
