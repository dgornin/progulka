"""The deterministic personalization: the catalog, the questionnaire's validation and
the precedence form > request text > profile > hard defaults."""

from datetime import date, time
from typing import Any

import pytest
from pydantic import ValidationError

from app.agent.brief import parse_brief
from app.models import Interest, TripBrief
from app.personalization import (
    brief_defaults,
    brief_overrides,
    personalize,
    profile_facts,
    search_hints,
    traveller_facts,
    with_feedback,
)
from app.profile import INTERESTS, Profile, WalkPreferences, catalog
from app.sources import osm
from tests.test_agent import FakeLLM

TODAY = date(2026, 10, 19)


def make_brief(*interests: tuple[str, float], avoid: tuple[str, ...] = ()) -> TripBrief:
    return TripBrief(
        city="Казань",
        date=TODAY,
        start_time=time(10),
        end_time=time(15),
        interests=[Interest(label=label, weight=weight) for label, weight in interests],
        avoid=list(avoid),
    )


def merged(brief: TripBrief) -> list[tuple[str, float, str]]:
    return [(i.label, i.weight, i.source) for i in brief.interests]


# --- catalog and validation ----------------------------------------------------------


def test_catalog_lists_the_interests_and_paces_in_order() -> None:
    data = catalog()
    assert [i["key"] for i in data["interests"]] == [
        "nature",
        "water",
        "views",
        "history",
        "architecture",
        "museums",
        "street_art",
        "food",
        "coffee",
        "nightlife",
        "shopping",
        "quiet",
        "kids",
        "religion",
        "theatre",
    ]
    assert data["interests"][0] == {"key": "nature", "label": "Природа и парки"}
    assert [p["key"] for p in data["paces"]] == ["relaxed", "normal", "active"]


def test_catalog_hints_name_only_real_osm_categories() -> None:
    # The hints are tool arguments for the agent: a renamed OSM category must break here.
    assert {c for spec in INTERESTS.values() for c in spec.osm} <= set(osm.CATEGORIES)


def test_profile_snaps_weights_to_the_two_levels_and_avoid_wins() -> None:
    profile = Profile(
        home_city="  ",
        interests={"nature": 0.9, "history": 0.3, "kids": 0, "museums": 1.0},
        avoid=["museums", "museums"],
        diet=[" вегетарианец ", "вегетарианец", ""],
        about=" ",
    )
    assert profile.interests == {"nature": 1.0, "history": 0.6}
    assert profile.avoid == ["museums"]
    assert profile.diet == ["вегетарианец"]
    assert profile.home_city is None and profile.about is None


@pytest.mark.parametrize(
    "fields",
    [
        {"interests": {"casino": 1.0}},
        {"avoid": ["casino"]},
        {"max_walk_km": 30},
        {"about": "я" * 501},
        {"diet": ["x" * 41]},
    ],
)
def test_profile_rejects_what_the_questionnaire_cannot_produce(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Profile(**fields)


def test_walk_preferences_know_the_catalog_and_less_wins() -> None:
    prefs = WalkPreferences(more=["nature", "museums"], less=["museums"])
    assert prefs.more == ["nature"]
    with pytest.raises(ValidationError, match="Неизвестные интересы: casino"):
        WalkPreferences(more=["casino"])


# --- scalars: form > text > profile > defaults ----------------------------------------


async def brief_from(
    draft: dict[str, Any],
    prefs: WalkPreferences | None = None,
    profile: Profile | None = None,
    form: dict[str, Any] | None = None,
) -> TripBrief:
    overrides = brief_overrides(form or {}, prefs or WalkPreferences())
    llm = FakeLLM(structured=[{"city": "Москва", **draft}])
    return await parse_brief(llm, "запрос", TODAY, overrides, brief_defaults(profile))


@pytest.mark.parametrize(
    ("form", "text", "profile", "expected"),
    [
        ("active", "relaxed", "normal", "active"),
        (None, "relaxed", "active", "relaxed"),
        (None, None, "active", "active"),
        (None, None, None, "normal"),
    ],
    ids=["form", "text", "profile", "default"],
)
async def test_pace_precedence(
    form: str | None, text: str | None, profile: str | None, expected: str
) -> None:
    brief = await brief_from(
        {"pace": text},
        WalkPreferences(pace=form),
        Profile(pace=profile) if profile else None,
    )
    assert brief.pace == expected


async def test_explicit_no_beats_the_profile_and_the_form_beats_both() -> None:
    profile = Profile(with_kids=True, max_walk_km=5, home_city="Казань")

    said_nothing = await brief_from({}, profile=profile)
    said_no_kids = await brief_from({"with_kids": False, "max_walk_km": 8}, profile=profile)
    form = await brief_from(
        {"with_kids": False, "max_walk_km": 8},
        WalkPreferences(with_kids=True, max_walk_km=3),
        profile,
        form={"city": "Казань"},
    )

    assert (said_nothing.with_kids, said_nothing.max_walk_km) == (True, 5)
    assert said_nothing.city == "Москва"  # the text names a city: the home city is a fallback
    assert (said_no_kids.with_kids, said_no_kids.max_walk_km) == (False, 8)
    assert (form.with_kids, form.max_walk_km, form.city) == (True, 3, "Казань")


async def test_empty_request_makes_no_llm_call() -> None:
    llm = FakeLLM()
    brief = await parse_brief(llm, "  ", TODAY, defaults={"city": "Казань", "pace": "relaxed"})
    assert llm.seen == []
    assert (brief.city, brief.language, brief.pace, brief.date) == (
        "Казань",
        "ru",
        "relaxed",
        TODAY,
    )
    assert (brief.start_time, brief.end_time, brief.interests) == (time(10), time(18), [])


async def test_brief_prompt_offers_the_catalog_labels() -> None:
    llm = FakeLLM(structured=[{"city": "Казань"}])
    await parse_brief(llm, "люблю парки", TODAY)
    assert "«Природа и парки», «Набережные и вода»" in llm.seen[0][0]["content"]


# --- interests and avoid -------------------------------------------------------------


def test_walk_interests_follow_the_request_and_profile_fills_a_thin_one() -> None:
    profile = Profile(interests={"views": 0.6, "museums": 1.0, "coffee": 0.6, "nature": 1.0})
    prefs = WalkPreferences(more=["street_art"])

    brief = personalize(make_brief(("конструктивизм", 1.0)), prefs, profile)

    assert merged(brief) == [
        ("конструктивизм", 1.0, "request"),
        ("Стрит-арт", 1.0, "walk"),
        # Most important first, catalog order among equals, discounted, four in total.
        ("Природа и парки", 0.7, "profile"),
        ("Музеи и галереи", 0.7, "profile"),
    ]


def test_a_rich_request_is_not_diluted_by_the_profile() -> None:
    request = make_brief(("модерн", 1.0), ("стрит-арт", 0.7), ("кофе", 0.4))
    brief = personalize(request, WalkPreferences(), Profile(interests={"nature": 1.0}))
    assert [label for label, _, _ in merged(brief)] == ["модерн", "стрит-арт", "кофе"]


def test_the_panel_wins_over_the_same_interest_in_the_text() -> None:
    request = make_brief(("природа и парки", 0.4))
    brief = personalize(request, WalkPreferences(more=["nature"]), None)
    assert merged(brief) == [("Природа и парки", 1.0, "walk")]


def test_avoid_merges_all_layers_and_less_beats_the_text() -> None:
    request = make_brief(("музеи", 1.0), ("история", 0.7), avoid=("Храмы",))
    profile = Profile(avoid=["religion", "nightlife"])

    brief = personalize(request, WalkPreferences(less=["museums"]), profile)

    assert merged(brief) == [("история", 0.7, "request")]
    assert brief.avoid == ["Музеи и галереи", "Храмы", "Бары и тусовки"]


def test_the_text_beats_the_profile_avoid_and_the_profile_cannot_want_what_text_avoids() -> None:
    # The validator treats every stop matching an avoid phrase as an error, so a
    # profile's «не предлагать бары» must not veto the bar the traveller asked for.
    request = make_brief(("бары", 1.0), avoid=("музеи",))
    profile = Profile(interests={"museums": 1.0, "views": 0.6}, avoid=["nightlife"])

    brief = personalize(request, WalkPreferences(), profile)

    assert merged(brief) == [("бары", 1.0, "request"), ("Виды и панорамы", 0.42, "profile")]
    assert brief.avoid == ["музеи"]


def test_an_avoid_phrase_drops_an_interest_only_when_it_leaves_it_nothing() -> None:
    # «музеи» are museums and galleries for the validator: «не галереи» still leaves museums.
    keeps = personalize(make_brief(("музеи", 1.0), avoid=("галереи",)), WalkPreferences(), None)
    drops = personalize(make_brief(("галереи", 1.0), avoid=("музеи",)), WalkPreferences(), None)

    assert (merged(keeps), keeps.avoid) == ([("музеи", 1.0, "request")], ["галереи"])
    assert (merged(drops), drops.avoid) == ([], ["музеи"])


def test_a_higher_interest_lifts_a_lower_avoid_that_would_reject_part_of_it() -> None:
    # The panel's «Рынки и магазины» wants shops too, so the text's «магазины» must go.
    request = make_brief(avoid=("магазины", "музеи"))
    brief = personalize(request, WalkPreferences(more=["shopping"]), None)

    assert merged(brief) == [("Рынки и магазины", 1.0, "walk")]
    assert brief.avoid == ["музеи"]


def test_feedback_is_the_newest_layer_on_top_of_the_brief() -> None:
    brief = make_brief(("Природа и парки", 0.7), ("история", 1.0), avoid=("Музеи и галереи",))
    # The replanner's model may claim any source; the feedback is the traveller's words.
    add = [Interest(label="музеи", weight=1.0, source="profile")]

    edited = with_feedback(brief, add, remove=["ИСТОРИЯ"], avoid=["парки"])

    assert merged(edited) == [("музеи", 1.0, "request")]
    assert edited.avoid == ["парки"]  # «добавь музеи» lifted the old «Музеи и галереи»
    assert edited.city == brief.city


def test_search_hints_cover_only_catalog_interests() -> None:
    interests = [Interest(label="природа и ПАРКИ"), Interest(label="конструктивизм")]
    assert search_hints(interests) == [
        "природа и ПАРКИ: search_osm park, garden; search_wikipedia «парк», «сад»"
    ]


def test_facts_about_the_traveller() -> None:
    profile = Profile(
        home_city="Казань",
        interests={"nature": 1.0, "history": 0.6},
        avoid=["nightlife"],
        pace="relaxed",
        with_kids=True,
        max_walk_km=6,
        diet=["вегетарианец"],
        about="Фотографирую архитектуру",
    )
    notes = ["Питание: вегетарианец", "О себе: Фотографирую архитектуру"]

    assert traveller_facts(profile, ["не любит толпы"]) == ["не любит толпы", *notes]
    assert traveller_facts(None, []) == []
    assert profile_facts(profile) == [
        "Домашний город: Казань",
        "Интересы: Природа и парки (очень), История",
        "Не предлагать: Бары и тусовки",
        "Темп прогулок: спокойно",
        "Гуляет с детьми",
        "Проходит пешком не больше 6 км",
        *notes,
    ]
