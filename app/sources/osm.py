"""Places from OpenStreetMap through the Overpass API, by curated category.

The agent never writes Overpass QL. It picks category names from CATEGORIES and
this module turns them into one query. A bad tool argument therefore cannot
produce a broken or huge query: unknown names are rejected with the list of
valid ones, and the radius is capped.

Most categories require a name in the filter itself, because an unnamed bench or
cafe cannot be shown to the traveller. Artworks are the exception: street art is
often unnamed, so an artwork is kept when it is named, has an artist, or is a
mural, and gets a generated title.

Ranking: places with a Wikipedia/Wikidata link (notable) first, then by distance.
Each category is ranked separately and the lists are interleaved, so a dense
category (cafes) does not crowd out a sparse one (museums) in the same request.
"""

import math
import re
from dataclasses import dataclass
from itertools import zip_longest

from app.config import settings
from app.models import GeoPoint, Place
from app.planning.geo import haversine_m
from app.sources.http import SourceError, http

# Public Overpass instances are often overloaded (the main one answered in 40 s or
# with 502 while we built this), so each gets one short attempt and the first one
# that answers becomes the preferred instance for the next queries; a preferred one
# that fails passes that on to the next at once.
OVERPASS_TIMEOUT_S = 10.0
_preferred = 0
MAX_RADIUS_M = 5000
DUPLICATE_RADIUS_M = 150
MAX_DESCRIPTION_CHARS = 400


@dataclass(frozen=True)
class CategorySpec:
    kind: str
    label_ru: str
    filters: tuple[str, ...]  # Overpass tag filters; an element matching any of them qualifies
    indoor: bool | None
    visit_minutes: int


CATEGORIES: dict[str, CategorySpec] = {
    spec.kind: spec
    for spec in (
        CategorySpec("museum", "Музеи", ('["tourism"="museum"]["name"]',), True, 90),
        CategorySpec(
            "gallery",
            "Галереи и арт-пространства",
            ('["tourism"="gallery"]["name"]', '["amenity"="arts_centre"]["name"]'),
            True,
            45,
        ),
        CategorySpec(
            "artwork",
            "Стрит-арт и скульптуры",
            (
                '["tourism"="artwork"]["name"]',
                '["tourism"="artwork"]["artist_name"]',
                '["tourism"="artwork"]["artwork_type"="mural"]',
            ),
            False,
            10,
        ),
        CategorySpec("monument", "Памятники", ('["historic"="monument"]["name"]',), False, 10),
        CategorySpec(
            "memorial",
            "Мемориалы",
            ('["historic"="memorial"]["name"]["memorial"!="plaque"]',),
            False,
            10,
        ),
        CategorySpec(
            "historic",
            "Исторические места",
            ('["historic"~"^(castle|fort|ruins|building|manor|city_gate|tower)$"]["name"]',),
            None,
            20,
        ),
        CategorySpec(
            "architecture",
            "Архитектура",
            ('["building:architecture"]["name"]', '["heritage"]["building"]["name"]'),
            False,
            10,
        ),
        CategorySpec("church", "Храмы", ('["amenity"="place_of_worship"]["name"]',), True, 20),
        CategorySpec(
            "viewpoint", "Смотровые площадки", ('["tourism"="viewpoint"]["name"]',), False, 15
        ),
        CategorySpec("park", "Парки", ('["leisure"="park"]["name"]',), False, 30),
        CategorySpec("garden", "Сады", ('["leisure"="garden"]["name"]',), False, 30),
        CategorySpec(
            "attraction", "Достопримечательности", ('["tourism"="attraction"]["name"]',), None, 20
        ),
        CategorySpec("theatre", "Театры", ('["amenity"="theatre"]["name"]',), True, 120),
        CategorySpec("cinema", "Кинотеатры", ('["amenity"="cinema"]["name"]',), True, 120),
        CategorySpec("library", "Библиотеки", ('["amenity"="library"]["name"]',), True, 30),
        CategorySpec("bookshop", "Книжные магазины", ('["shop"="books"]["name"]',), True, 20),
        CategorySpec(
            "market",
            "Рынки и фудкорты",
            ('["amenity"="marketplace"]["name"]', '["amenity"="food_court"]["name"]'),
            None,
            40,
        ),
        CategorySpec("cafe", "Кафе и кофейни", ('["amenity"="cafe"]["name"]',), True, 45),
        CategorySpec("restaurant", "Рестораны", ('["amenity"="restaurant"]["name"]',), True, 75),
        CategorySpec("street_food", "Стритфуд", ('["amenity"="fast_food"]["name"]',), None, 20),
        CategorySpec(
            "bakery",
            "Пекарни и кондитерские",
            ('["shop"="bakery"]["name"]', '["shop"="pastry"]["name"]'),
            True,
            15,
        ),
        CategorySpec("bar", "Бары", ('["amenity"="bar"]["name"]',), True, 60),
        CategorySpec("pub", "Пабы", ('["amenity"="pub"]["name"]',), True, 60),
        CategorySpec(
            "ice_cream",
            "Мороженое",
            ('["amenity"="ice_cream"]["name"]', '["shop"="ice_cream"]["name"]'),
            None,
            10,
        ),
        CategorySpec("zoo", "Зоопарки", ('["tourism"="zoo"]["name"]',), False, 120),
        CategorySpec("aquarium", "Океанариумы", ('["tourism"="aquarium"]["name"]',), True, 90),
        CategorySpec(
            "playground", "Детские площадки", ('["leisure"="playground"]["name"]',), False, 30
        ),
        CategorySpec("fountain", "Фонтаны", ('["amenity"="fountain"]["name"]',), False, 10),
    )
}

# Tags passed on to the agent: links for notability and details, hours, and the
# few attributes that match interests (style, cuisine) or help the traveller.
KEPT_TAGS = (
    "wikidata",
    "wikipedia",
    "website",
    "opening_hours",
    "fee",
    "wheelchair",
    "addr:street",
    "addr:housenumber",
    "artwork_type",
    "building:architecture",
    "heritage",
    "cuisine",
)
# Informative tags turned into a short human description, in this order.
DESCRIPTION_TAGS = (
    ("description", "{}"),
    ("inscription", "Надпись: «{}»"),
    ("artist_name", "Автор: {}"),
    ("architect", "Архитектор: {}"),
    ("building:architecture", "Стиль: {}"),
    ("start_date", "Время создания: {}"),
    ("heritage", "Объект культурного наследия"),
    ("cuisine", "Кухня: {}"),
    ("operator", "Оператор: {}"),
)
_FILTER_PART = re.compile(r'\["([^"]+)"(?:(=|!=|~)"([^"]*)")?\]')


async def search_places(
    center: GeoPoint, radius_m: int, categories: list[str], limit: int = 40
) -> list[Place]:
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown or not categories:
        raise ValueError(f"Unknown categories {unknown}; valid: {', '.join(CATEGORIES)}")
    specs = [CATEGORIES[c] for c in dict.fromkeys(categories)]
    radius = min(radius_m, MAX_RADIUS_M)
    elements = await _overpass(_build_query(center, radius, specs))

    by_kind: dict[str, list[Place]] = {spec.kind: [] for spec in specs}
    for element in elements:
        tags = element.get("tags", {})
        spec = next((s for s in specs if _matches_any(tags, s.filters)), None)
        # Overpass gives ways and relations a "center"; a broken relation has none.
        if spec and ("lat" in element or "center" in element):
            place = _to_place(element, spec)
            # The query asked for a square; the corners are farther than the radius.
            if haversine_m(center.lat, center.lon, place.lat, place.lon) <= radius:
                by_kind[spec.kind].append(place)
    for places in by_kind.values():
        places.sort(
            key=lambda p: (not _is_notable(p), haversine_m(center.lat, center.lon, p.lat, p.lon))
        )
    interleaved = [p for group in zip_longest(*by_kind.values()) for p in group if p]
    return _dedupe(interleaved)[:limit]


def _build_query(center: GeoPoint, radius_m: int, specs: list[CategorySpec]) -> str:
    """A bounding-box query around the circle. A box uses Overpass's spatial index;
    the same filters with (around:...) timed out on two public instances where the
    box answered in about a second. search_places then cuts the box to the circle."""
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * math.cos(math.radians(center.lat)))
    box = (
        f"({center.lat - dlat:.5f},{center.lon - dlon:.5f},"
        f"{center.lat + dlat:.5f},{center.lon + dlon:.5f})"
    )
    statements = "\n".join(f"  nwr{f}{box};" for spec in specs for f in spec.filters)
    return f"[out:json][timeout:20];\n(\n{statements}\n);\nout center tags;"


async def _overpass(query: str) -> list[dict]:
    """Run the query on the preferred instance, failing over to the others in order."""
    global _preferred
    urls = settings.overpass_urls
    errors = []
    for i in [*range(_preferred, len(urls)), *range(_preferred)]:
        try:
            payload = await http.post_form_json(
                urls[i],
                {"data": query},
                timeout_s=OVERPASS_TIMEOUT_S,
                attempts=1,
                cache_as="overpass",
                cacheable=_is_complete,
            )
        except SourceError as e:
            errors.append(str(e))
            if _preferred == i:
                _preferred = (i + 1) % len(urls)  # queries from now on start at the next
            continue
        if not _is_complete(payload):
            errors.append(f"{urls[i]}: {payload.get('remark')}")
            continue
        _preferred = i
        return payload.get("elements", [])
    raise SourceError("; ".join(errors))


def _is_complete(payload: dict) -> bool:
    # A query timeout comes back as HTTP 200 with a partial result and a remark.
    return not str(payload.get("remark", "")).startswith("runtime error")


def _matches_any(tags: dict[str, str], filters: tuple[str, ...]) -> bool:
    """Evaluate Overpass tag filters locally: one union query, so we must tell the
    categories apart ourselves."""
    return any(_matches(tags, f) for f in filters)


def _matches(tags: dict[str, str], tag_filter: str) -> bool:
    for key, op, value in _FILTER_PART.findall(tag_filter):
        actual = tags.get(key)
        if op == "=" and actual != value:
            return False
        if op == "!=" and actual == value:
            return False
        if op == "~" and (actual is None or not re.search(value, actual)):
            return False
        if not op and actual is None:
            return False
    return True


def _to_place(element: dict, spec: CategorySpec) -> Place:
    tags = element.get("tags", {})
    # Only artworks can be unnamed here: every other filter requires ["name"].
    name = tags.get("name") or _artwork_title(tags)
    point = element.get("center") or element
    osm_ref = f"{element['type']}/{element['id']}"
    return Place(
        id=f"osm:{osm_ref}",
        name=name,
        lat=point["lat"],
        lon=point["lon"],
        kind=spec.kind,
        source="osm",
        url=f"https://www.openstreetmap.org/{osm_ref}",
        description=_describe(tags),
        opening_hours=tags.get("opening_hours"),
        indoor=spec.indoor,
        tags={k: tags[k] for k in KEPT_TAGS if k in tags},
    )


def _artwork_title(tags: dict[str, str]) -> str:
    title = "Мурал" if tags.get("artwork_type") == "mural" else "Арт-объект"
    artist = tags.get("artist_name")
    return f"{title} ({artist})" if artist else title


def _describe(tags: dict[str, str]) -> str | None:
    parts = [
        template.format(tags[key].replace(";", ", ").replace("_", " "))
        for key, template in DESCRIPTION_TAGS
        if tags.get(key)
    ]
    text = ". ".join(parts)
    if len(text) > MAX_DESCRIPTION_CHARS:
        text = text[: MAX_DESCRIPTION_CHARS - 1] + "…"
    return text or None


def _is_notable(place: Place) -> bool:
    return "wikidata" in place.tags or "wikipedia" in place.tags


def _dedupe(places: list[Place]) -> list[Place]:
    """Drop a place if a better-ranked one with the same name is within DUPLICATE_RADIUS_M.

    OSM often maps one museum twice: a node for the entrance and a building outline.
    Artworks are exempt: they are mapped as points, and two unnamed murals in one
    courtyard share a generated title but are different works.
    """
    kept: list[Place] = []
    by_name: dict[str, list[Place]] = {}
    for place in places:
        if place.kind == "artwork":
            kept.append(place)
            continue
        same_name = by_name.setdefault(place.name.casefold(), [])
        if any(
            haversine_m(p.lat, p.lon, place.lat, place.lon) < DUPLICATE_RADIUS_M for p in same_name
        ):
            continue
        same_name.append(place)
        kept.append(place)
    return kept
