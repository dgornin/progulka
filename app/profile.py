"""What the traveller tells us in a structured way: the interest catalog, the profile
from the onboarding questionnaire, and the per-trip preferences of the planner.

The catalog is the shared vocabulary of the questionnaire, the planner's
"больше / меньше" panel and the research agent. A free-text interest ("конструктивизм")
needs the agent's judgement to find; a catalog interest comes with a hint where to
look for it, so the agent does not have to rediscover on every run that "nature" means
OSM parks and gardens. The hints name only tool arguments that exist (OSM category
keys, Wikivoyage sections); a test keeps them in sync with the OSM catalog.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models import Pace

# The questionnaire has two levels per interest; absent means "not interesting".
LIKE, LOVE = 0.6, 1.0
MAX_DIET_TAGS, DIET_TAG_CHARS = 10, 40


@dataclass(frozen=True)
class InterestSpec:
    label: str
    osm: tuple[str, ...]  # search_osm categories that serve it
    elsewhere: str  # other sources worth a call: Wikipedia queries, Wikivoyage sections

    @property
    def hint(self) -> str:
        return f"search_osm {', '.join(self.osm)}; {self.elsewhere}"


INTERESTS: dict[str, InterestSpec] = {
    "nature": InterestSpec("Природа и парки", ("park", "garden"), "search_wikipedia «парк», «сад»"),
    "water": InterestSpec(
        "Набережные и вода",
        ("fountain", "viewpoint"),
        "search_wikipedia «набережная», «мост», «пруд»",
    ),
    "views": InterestSpec(
        "Виды и панорамы",
        ("viewpoint",),
        "search_wikipedia «смотровая площадка», «колокольня»",
    ),
    "history": InterestSpec(
        "История",
        ("historic", "monument", "memorial"),
        "search_wikipedia without a query (notable places nearby)",
    ),
    "architecture": InterestSpec(
        "Архитектура",
        ("architecture", "historic"),
        "search_wikipedia «архитектура», «особняк» or the style the traveller names",
    ),
    "museums": InterestSpec(
        "Музеи и галереи", ("museum", "gallery"), "get_wikivoyage_listings see"
    ),
    "street_art": InterestSpec(
        "Стрит-арт", ("artwork",), "search_wikipedia «стрит-арт», «граффити»"
    ),
    "food": InterestSpec(
        "Местная еда",
        ("restaurant", "street_food", "market"),
        "get_wikivoyage_listings eat (local cuisine)",
    ),
    "coffee": InterestSpec(
        "Кофейни и выпечка", ("cafe", "bakery"), "get_wikivoyage_listings eat, drink"
    ),
    "nightlife": InterestSpec("Бары и тусовки", ("bar", "pub"), "get_wikivoyage_listings drink"),
    "shopping": InterestSpec(
        "Рынки и магазины", ("market", "bookshop"), "get_wikivoyage_listings buy"
    ),
    "quiet": InterestSpec(
        "Тихие места",
        ("garden", "park", "library"),
        "search_wikipedia «сквер», «двор»; prefer small places away from the main sights",
    ),
    "kids": InterestSpec(
        "Для детей",
        ("playground", "zoo", "aquarium", "ice_cream"),
        "get_wikivoyage_listings do",
    ),
    "religion": InterestSpec("Храмы", ("church",), "search_wikipedia «собор», «храм»"),
    "theatre": InterestSpec(
        "Театры и концерты", ("theatre",), "search_wikipedia «театр», «концертный зал»"
    ),
}

PACES: dict[Pace, str] = {"relaxed": "Спокойно", "normal": "Обычно", "active": "Активно"}


def catalog() -> dict[str, Any]:
    """The vocabulary of the questionnaire and the planner panel, for GET /api/catalog."""
    return {
        "interests": [{"key": key, "label": spec.label} for key, spec in INTERESTS.items()],
        "paces": [{"key": key, "label": label} for key, label in PACES.items()],
    }


def _catalog_keys(keys: list[str]) -> list[str]:
    unknown = [key for key in keys if key not in INTERESTS]
    if unknown:
        raise ValueError(f"Неизвестные интересы: {', '.join(unknown)}")
    return list(dict.fromkeys(keys))


class Profile(BaseModel):
    """The questionnaire: filled after registration, edited in the profile."""

    home_city: str | None = Field(None, max_length=100)
    interests: dict[str, float] = Field(
        default_factory=dict, description=f"Catalog key -> {LIKE} «интересно» or {LOVE} «очень»"
    )
    avoid: list[str] = Field(default_factory=list, description="Catalog keys not to suggest")
    pace: Pace = "normal"
    max_walk_km: float | None = Field(None, ge=1, le=25)
    with_kids: bool = False
    diet: list[str] = Field(default_factory=list, description="'вегетарианец', 'без алкоголя'")
    about: str | None = Field(None, max_length=500)

    @field_validator("home_city", "about")
    @classmethod
    def _blank_is_none(cls, value: str | None) -> str | None:
        return (value or "").strip() or None

    @field_validator("interests")
    @classmethod
    def _known_interest_levels(cls, value: dict[str, float]) -> dict[str, float]:
        _catalog_keys(list(value))
        # A client may send any number: 0 or less is "not interesting", the rest snaps
        # to the nearer of the two levels the questionnaire offers.
        return {
            key: LOVE if weight >= (LIKE + LOVE) / 2 else LIKE
            for key, weight in value.items()
            if weight > 0
        }

    @field_validator("avoid")
    @classmethod
    def _known_avoid(cls, value: list[str]) -> list[str]:
        return _catalog_keys(value)

    @field_validator("diet")
    @classmethod
    def _diet_tags(cls, value: list[str]) -> list[str]:
        tags = list(dict.fromkeys(t.strip() for t in value if t.strip()))
        if len(tags) > MAX_DIET_TAGS or any(len(t) > DIET_TAG_CHARS for t in tags):
            raise ValueError(
                f"Не больше {MAX_DIET_TAGS} пунктов питания, до {DIET_TAG_CHARS} символов каждый"
            )
        return tags

    @model_validator(mode="after")
    def _avoid_wins(self) -> "Profile":
        # One key cannot be both wanted and avoided; "не предлагать" is the safer reading.
        for key in self.avoid:
            self.interests.pop(key, None)
        return self


class WalkPreferences(BaseModel):
    """The planner's panel for one trip, every day of it; it wins over the request text
    and the profile."""

    more: list[str] = Field(default_factory=list, description="Catalog keys: «больше этого»")
    less: list[str] = Field(default_factory=list, description="Catalog keys: «меньше / не надо»")
    pace: Pace | None = None
    with_kids: bool | None = None
    max_walk_km: float | None = Field(None, ge=1, le=25)

    @field_validator("more", "less")
    @classmethod
    def _known_keys(cls, value: list[str]) -> list[str]:
        return _catalog_keys(value)

    @model_validator(mode="after")
    def _less_wins(self) -> "WalkPreferences":
        self.more = [key for key in self.more if key not in self.less]
        return self
