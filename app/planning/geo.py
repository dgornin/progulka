"""Distances and walking times between points.

The optimizer compares thousands of candidate orders, so it cannot ask a routing
service for every pair. It uses the straight-line (haversine) distance times a
detour factor: in a city grid a walk is about 1.3 times longer than the straight
line. The real street route is requested once, for the final order (routing.py).

Unless the traveller names a limit, a day's walk is capped by pace: an optimizer
that only fills the time window happily plans 15 km, which nobody asked for.
"""

from math import asin, cos, radians, sin, sqrt

from app.models import GeoPoint, Pace, Place, TripBrief

EARTH_RADIUS_M = 6_371_000
WALK_SPEED_KMH: dict[Pace, float] = {"relaxed": 3.8, "normal": 4.5, "active": 5.2}
DETOUR_FACTOR = 1.3
DEFAULT_WALK_KM: dict[Pace, float] = {"relaxed": 7.0, "normal": 10.0, "active": 14.0}
WITH_KIDS_WALK_KM = 5.0


def haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Great-circle distance in metres."""
    d_lat = radians(b_lat - a_lat)
    d_lon = radians(b_lon - a_lon)
    h = sin(d_lat / 2) ** 2 + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(h))


def street_meters(a: GeoPoint | Place, b: GeoPoint | Place) -> float:
    """Estimated walking distance along streets."""
    return haversine_m(a.lat, a.lon, b.lat, b.lon) * DETOUR_FACTOR


def walk_minutes(meters: float, pace: Pace) -> float:
    return meters / (WALK_SPEED_KMH[pace] * 1000 / 60)


def walk_limit_km(brief: TripBrief) -> float:
    """The traveller's own limit, else a default by pace (lower with children)."""
    if brief.max_walk_km is not None:
        return brief.max_walk_km
    limit = DEFAULT_WALK_KM[brief.pace]
    return min(limit, WITH_KIDS_WALK_KM) if brief.with_kids else limit
