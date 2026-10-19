"""Nominatim geocoding: find the city, then places inside it.

Three traps shape this module:
- A plain search for "Санкт-Петербург" or "Москва" returns the federal-subject
  boundary first; its centre is a geometric centroid kilometres away from the
  historic centre. Nominatim also returns the settlement itself (place=city) with
  the same importance, so among the equally important results we take the one
  whose address type is a settlement. Its point is the city centre.
- A landmark name is ambiguous across the world ("Эрмитаж" is also a hotel in
  many towns), so once the city is known the search is bounded by its bbox and
  the Wikipedia-derived importance picks the famous object among the matches.
  Nominatim ranks exact name matches first, so the hotel "Эрмитаж" comes before
  "Государственный Эрмитаж"; we ask for enough results to reach the famous one.
- The city's boundary is narrower than what people mean by the city: Петергоф is
  outside Петербург's bbox. When the bounded search finds nothing, we search again
  without the bbox and name the city in the query instead.
"""

from pydantic import BaseModel

from app.models import GeoPoint
from app.sources.http import http

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
SETTLEMENT_TYPES = {"city", "town", "village"}
# Results this close in importance describe the same place (region vs settlement);
# a farther one is a different place that merely shares the name.
SAME_PLACE_IMPORTANCE = 0.05
LANDMARK_RESULTS = 20


class CityInfo(BaseModel):
    name: str
    center: GeoPoint
    bbox: tuple[float, float, float, float]  # (south, west, north, east)
    country_code: str | None


async def locate_city(city: str, country: str | None = None) -> CityInfo | None:
    query = f"{city}, {country}" if country else city
    results = await http.get_json(
        SEARCH_URL,
        {
            "q": query,
            "format": "jsonv2",
            "featureType": "settlement",
            "addressdetails": 1,
            "limit": 5,
        },
    )
    if not results:
        return None
    best = _pick_settlement(results)
    south, north, west, east = (float(v) for v in best["boundingbox"])
    name = best.get("name") or city
    return CityInfo(
        name=name,
        center=GeoPoint(lat=float(best["lat"]), lon=float(best["lon"]), label=name),
        bbox=(south, west, north, east),
        country_code=best.get("address", {}).get("country_code"),
    )


async def geocode(
    query: str, city: str | None = None, country: str | None = None
) -> GeoPoint | None:
    results = []
    city_info = await locate_city(city, country) if city else None
    if city_info:
        south, west, north, east = city_info.bbox
        results = await _search(query, viewbox=f"{west},{south},{east},{north}")
    if not results:
        results = await _search(f"{query}, {city}" if city else query)
    if not results:
        return None
    best = max(results, key=_importance)  # max() keeps Nominatim's order on ties
    return GeoPoint(lat=float(best["lat"]), lon=float(best["lon"]), label=_short_label(best))


async def _search(query: str, viewbox: str | None = None) -> list[dict]:
    params = {"q": query, "format": "jsonv2", "limit": LANDMARK_RESULTS}
    if viewbox:
        params |= {"viewbox": viewbox, "bounded": 1}
    return await http.get_json(SEARCH_URL, params)


def _pick_settlement(results: list[dict]) -> dict:
    top = _importance(results[0])
    same_place = [r for r in results if _importance(r) >= top - SAME_PLACE_IMPORTANCE]
    settlements = [r for r in same_place if r.get("addresstype") in SETTLEMENT_TYPES]
    return settlements[0] if settlements else results[0]


def _importance(result: dict) -> float:
    return float(result.get("importance") or 0)


def _short_label(result: dict) -> str:
    if result.get("name"):
        return result["name"]
    return ", ".join(result["display_name"].split(", ")[:2])
