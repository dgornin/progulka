"""Deterministic merge of everything we know about the traveller into one TripBrief.

Four sources can say how this walk should look, and they can disagree. The merge is
code, not a prompt: the same inputs always give the same brief, and the rule can be
stated in one line. Precedence, highest first:

    the planner form (fields and the "больше / меньше" panel)
    > the request text (what the brief LLM extracted)
    > the profile (the onboarding questionnaire)
    > hard defaults (10:00, five hours, normal pace)

Scalars (city, pace, children, distance) are resolved in parse_brief: the form's
values are overrides, the profile's are defaults for what the text leaves null.
Interests and the avoid list are merged here, layer by layer in the same order. A
layer only adds to what the layers above decided: it cannot avoid what a higher
layer asked for, nor ask for what a higher layer (or itself) avoids. So «меньше
музеев» in the panel beats «хочу в музей» in the text, and «бары» in the text
beats «не предлагать бары» in the profile. The second case matters: the validator
treats every stop matching an avoid phrase as an error.

Conflicts are judged by the kinds of places the validator would reject. A higher
layer gets all it asked for: its interest overrules a lower avoid phrase that would
reject any kind of place serving it. A lower layer gets what is left: its interest is
dropped only when a higher (or its own) avoid phrase rejects every kind serving it,
so «без музеев» rules out «Музеи и галереи», but «без галерей» leaves «музеи».

Feedback on a finished walk is the traveller's newest word, so it is one more layer
on top of the brief (with_feedback): «добавь музей» lifts the profile's «не
предлагать музеи», and «без парков» drops the profile's parks.

Profile interests only fill a thin request: with fewer than three interests, the
most important ones are added, discounted (× 0.7) so the request's own wishes rank
first, up to four in total. A rich request is not diluted by the questionnaire.
"""

import re
from typing import Any

from app.models import Interest, InterestSource, TripBrief
from app.planning.validator import AVOID_RULES
from app.profile import INTERESTS, LOVE, PACES, Profile, WalkPreferences

PROFILE_FILL_BELOW = 3
MAX_INTERESTS_WITH_PROFILE = 4
PROFILE_DISCOUNT = 0.7
# A word belongs to an avoid rule when it is the rule's stem plus a short ending:
# "бары", "музеев", "парки", but not "барокко" or "парковка".
STEM_ENDING = 3
SOURCE_ORDER: dict[InterestSource, int] = {"request": 0, "walk": 1, "profile": 2}


def brief_overrides(form: dict[str, Any], prefs: WalkPreferences) -> dict[str, Any]:
    """What the traveller set explicitly for this walk: it wins over the text."""
    scalars = prefs.model_dump(include={"pace", "with_kids", "max_walk_km"}, exclude_none=True)
    return {**form, **scalars}


def brief_defaults(profile: Profile | None) -> dict[str, Any]:
    """What the brief takes from the profile when the request text does not say it."""
    if profile is None:
        return {}
    values = {
        "city": profile.home_city,
        "pace": profile.pace,
        "with_kids": profile.with_kids,
        "max_walk_km": profile.max_walk_km,
    }
    return {key: value for key, value in values.items() if value is not None}


def personalize(brief: TripBrief, prefs: WalkPreferences, profile: Profile | None) -> TripBrief:
    """The brief with the panel's and the profile's interests and avoid list merged in."""
    interests: list[Interest] = []
    avoid: list[str] = []
    walk = [Interest(label=INTERESTS[key].label, weight=LOVE, source="walk") for key in prefs.more]
    _add_layer(interests, avoid, walk, [INTERESTS[key].label for key in prefs.less])
    _add_layer(interests, avoid, brief.interests, brief.avoid)
    if profile is not None:
        # Most important first; the stable sort keeps the catalog order among equals.
        ranked = sorted(
            (key for key in INTERESTS if key in profile.interests),
            key=lambda key: -profile.interests[key],
        )
        wanted = [
            Interest(
                label=INTERESTS[key].label,
                weight=round(profile.interests[key] * PROFILE_DISCOUNT, 2),
                source="profile",
            )
            for key in ranked
        ]
        thin = len(interests) < PROFILE_FILL_BELOW
        cap = MAX_INTERESTS_WITH_PROFILE if thin else len(interests)
        _add_layer(interests, avoid, wanted, [INTERESTS[key].label for key in profile.avoid], cap)
    interests.sort(key=lambda i: SOURCE_ORDER[i.source])  # stable: each layer keeps its order
    return brief.model_copy(update={"interests": interests, "avoid": avoid})


def with_feedback(
    brief: TripBrief, add: list[Interest], remove: list[str], avoid: list[str]
) -> TripBrief:
    """The brief with the feedback's wishes merged in as the top layer; they come first."""
    interests: list[Interest] = []
    merged: list[str] = []
    # Rebuilt so that the source is set by code (see Interest.source): the traveller's words.
    _add_layer(interests, merged, [Interest(label=i.label, weight=i.weight) for i in add], avoid)
    removed = {label.casefold() for label in remove}
    kept = [i for i in brief.interests if i.label.casefold() not in removed]
    _add_layer(interests, merged, kept, brief.avoid)
    return brief.model_copy(update={"interests": interests, "avoid": merged})


def search_hints(interests: list[Interest]) -> list[str]:
    """Where to search each catalog interest of the brief, for the research task."""
    by_label = {spec.label.casefold(): spec for spec in INTERESTS.values()}
    return [
        f"{interest.label}: {spec.hint}"
        for interest in interests
        if (spec := by_label.get(interest.label.casefold()))
    ]


def traveller_facts(profile: Profile | None, memory: list[str]) -> list[str]:
    """What the research agent and the writer should know beyond the brief: long-term
    memory and the free-text parts of the profile that the brief has no fields for."""
    return [*memory, *_profile_notes(profile)]


def profile_facts(profile: Profile | None) -> list[str]:
    """The whole profile in words, so that memory extraction does not re-learn it."""
    if profile is None:
        return []
    facts = []
    if profile.home_city:
        facts.append(f"Домашний город: {profile.home_city}")
    if profile.interests:
        labels = [
            INTERESTS[key].label + (" (очень)" if weight == LOVE else "")
            for key, weight in profile.interests.items()
        ]
        facts.append(f"Интересы: {', '.join(labels)}")
    if profile.avoid:
        facts.append(f"Не предлагать: {', '.join(INTERESTS[key].label for key in profile.avoid)}")
    facts.append(f"Темп прогулок: {PACES[profile.pace].lower()}")
    if profile.with_kids:
        facts.append("Гуляет с детьми")
    if profile.max_walk_km:
        facts.append(f"Проходит пешком не больше {profile.max_walk_km:g} км")
    return [*facts, *_profile_notes(profile)]


def _add_layer(
    interests: list[Interest],
    avoid: list[str],
    wanted: list[Interest],
    avoided: list[str],
    cap: int | None = None,
) -> None:
    """Add one precedence layer to the merged interests and avoid list, in place."""
    for phrase in avoided:
        overruled = any(_overrules(i.label, phrase) for i in interests)
        known = any(phrase.casefold() == a.casefold() for a in avoid)
        if not (overruled or known):
            avoid.append(phrase)
    for interest in wanted:
        if cap is not None and len(interests) >= cap:
            break
        known = any(interest.label.casefold() == i.label.casefold() for i in interests)
        if not known and not any(_rules_out(a, interest.label) for a in avoid):
            interests.append(interest)


def _overrules(label: str, phrase: str) -> bool:
    """A higher interest beats a lower avoid phrase that would reject any place serving
    it: «Рынки и магазины» in the panel beats «магазины» on the profile's avoid list."""
    return label.casefold() == phrase.casefold() or bool(_kinds(label) & _kinds(phrase))


def _rules_out(phrase: str, label: str) -> bool:
    """The avoid phrase leaves the interest nothing: the validator rejects every kind of
    place that serves it. «без музеев» rules out «Музеи и галереи», «без галерей» does not
    rule out «музеи». An interest with no known kind is ruled out only by its own name."""
    kinds = _kinds(label)
    return label.casefold() == phrase.casefold() or bool(kinds) and kinds <= _kinds(phrase)


def _kinds(text: str) -> set[str]:
    words = re.findall(r"\w+", text.casefold())
    return {
        kind
        for kinds, stems in AVOID_RULES
        if any(w.startswith(s) and len(w) - len(s) <= STEM_ENDING for w in words for s in stems)
        for kind in kinds
    }


def _profile_notes(profile: Profile | None) -> list[str]:
    if profile is None:
        return []
    notes = []
    if profile.diet:
        notes.append(f"Питание: {', '.join(profile.diet)}")
    if profile.about:
        notes.append(f"О себе: {profile.about}")
    return notes
