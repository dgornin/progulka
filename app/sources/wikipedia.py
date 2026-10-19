"""Wikipedia as a source of notable places and of grounding text.

OSM knows where things are; Wikipedia knows why they matter. Every function makes
a single MediaWiki API request that returns pages together with their
coordinates and a plain-text intro, so the agent gets places it can put on the
map and text it can quote, without a request per page.

`search` puts the geographic limit into the search itself (CirrusSearch
`nearcoord:`), so a topic query like "конструктивизм" returns buildings in this
city instead of the general article about the art movement.
"""

from typing import Any
from urllib.parse import urlparse

from app.models import GeoPoint, Place
from app.sources.http import SourceError, http

MAX_GEOSEARCH_RADIUS_M = 10_000  # API limit for list=geosearch
MAX_DESCRIPTION_CHARS = 500
# Page data for the generators. prop=coordinates returns 10 pages unless asked
# for more; prop=extracts returns intros for at most 20 pages (an API limit).
EXTRACT_PARAMS = {
    "prop": "coordinates|extracts|description|pageprops",
    "colimit": "max",
    "exintro": 1,
    "explaintext": 1,
    "exsentences": 3,
    "exlimit": "max",
    "format": "json",
    "formatversion": 2,
}


async def nearby(center: GeoPoint, radius_m: int, lang: str = "ru", limit: int = 30) -> list[Place]:
    payload = await query_mediawiki(
        _api_url(lang),
        {
            "action": "query",
            "generator": "geosearch",
            "ggscoord": f"{center.lat}|{center.lon}",
            "ggsradius": min(radius_m, MAX_GEOSEARCH_RADIUS_M),
            "ggslimit": limit,
            **EXTRACT_PARAMS,
        },
    )
    return _to_places(payload, lang)


async def search(
    query: str, center: GeoPoint, radius_m: int, lang: str = "ru", limit: int = 20
) -> list[Place]:
    payload = await query_mediawiki(
        _api_url(lang),
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} nearcoord:{radius_m}m,{center.lat},{center.lon}",
            "gsrnamespace": 0,
            "gsrlimit": limit,
            **EXTRACT_PARAMS,
        },
    )
    return _to_places(payload, lang)


async def page_extract(place_id: str, max_chars: int = 1500) -> str | None:
    """Plain text of the article behind a "wikipedia:{lang}:{pageid}" id, cut to max_chars."""
    parts = place_id.split(":")
    if len(parts) != 3 or parts[0] != "wikipedia":
        raise ValueError(f"Not a Wikipedia place id: {place_id}")
    _, lang, pageid = parts
    payload = await query_mediawiki(
        _api_url(lang),
        {
            "action": "query",
            "pageids": pageid,
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "format": "json",
            "formatversion": 2,
        },
    )
    pages = payload.get("query", {}).get("pages", [])
    text = pages[0].get("extract") if pages else None
    return _shorten(text, max_chars) if text else None


async def query_mediawiki(api_url: str, params: dict[str, Any]) -> dict:
    """One MediaWiki API request (Wikipedia and Wikivoyage).

    MediaWiki reports its errors with HTTP 200 ("search is currently too busy"). Read
    as a payload, such an answer is "nothing found", and cached it would stay that for
    a week; as a SourceError it is a tool error the agent sees, and it is not cached.
    """
    payload = await http.get_json(api_url, params, cacheable=lambda p: "error" not in p)
    if "error" in payload:
        info = payload["error"].get("info", "API error")
        raise SourceError(f"{urlparse(api_url).netloc}: {info}")
    return payload


def _api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _to_places(payload: dict, lang: str) -> list[Place]:
    pages = sorted(payload.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    places = []
    for page in pages:
        if not page.get("coordinates"):
            continue
        coords = page["coordinates"][0]
        # Only the first 20 pages get an extract; the rest fall back to the
        # one-line Wikidata description.
        text = page.get("extract") or page.get("description")
        wikidata = page.get("pageprops", {}).get("wikibase_item")
        places.append(
            Place(
                id=f"wikipedia:{lang}:{page['pageid']}",
                name=page["title"],
                lat=coords["lat"],
                lon=coords["lon"],
                kind="landmark",
                source="wikipedia",
                url=f"https://{lang}.wikipedia.org/?curid={page['pageid']}",
                description=_shorten(text, MAX_DESCRIPTION_CHARS) if text else None,
                # Same tag names as OSM, so a place found in both can be matched.
                tags={"wikipedia": f"{lang}:{page['title']}"}
                | ({"wikidata": wikidata} if wikidata else {}),
            )
        )
    return places


def _shorten(text: str, max_chars: int) -> str:
    """Cut at the last sentence end that fits, or at a word with an ellipsis."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sentence_end = cut.rfind(". ")
    if sentence_end > max_chars // 2:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0] + "…"
