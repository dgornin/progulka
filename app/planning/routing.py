"""The real walking route along streets, for the final order only.

The optimizer works with estimated distances (geo.py); once the order is fixed we
ask OSRM (the public foot-profile server of routing.openstreetmap.de) for one
route through all stops. It gives the polyline for the map and the real length of
every leg, which schedule() then uses instead of the estimate. OSRM's durations are
not used: walking time comes from the traveller's pace (geo.walk_minutes).
The public server is best-effort, so on any failure we return None and the caller
keeps the estimate and draws straight lines. For the same reason the wait is capped:
the shared HTTP client retries for up to two minutes, and a nicer line on the map is
not worth that. After a failure the server is not asked for DOWN_S: the days of a
trip and their repair rounds are routed one after another, and each would wait out
the cap again. Only a failure of the server counts (no answer, 429, 5xx): OSRM
answers "no route through these points" (NoRoute, NoSegment) with HTTP 400, and that
says nothing about the next day's points.
"""

import asyncio
import time
from dataclasses import dataclass
from math import ceil

from app.models import GeoPoint
from app.sources.http import SourceError, http

OSRM_FOOT = "https://routing.openstreetmap.de/routed-foot/route/v1/foot/"
MAX_POLYLINE_POINTS = 2000  # plenty for a city map; keeps the plan JSON small
ROUTE_TIMEOUT_S = 10.0
DOWN_S = 300.0
_down_until = 0.0  # time.monotonic() until which the server counts as down


@dataclass
class RouteResult:
    geometry: list[list[float]]  # [lat, lon]
    leg_meters: list[float]


async def walking_route(points: list[GeoPoint]) -> RouteResult | None:
    """Route through the points in order; None if there is no route or the service failed."""
    global _down_until
    if len(points) < 2 or time.monotonic() < _down_until:
        return None
    coords = ";".join(f"{p.lon:.6f},{p.lat:.6f}" for p in points)
    try:
        async with asyncio.timeout(ROUTE_TIMEOUT_S):
            data = await http.get_json(
                OSRM_FOOT + coords,
                params={"overview": "full", "geometries": "geojson", "steps": "false"},
            )
    except TimeoutError:
        _down_until = time.monotonic() + DOWN_S
        return None
    except SourceError as e:
        if e.status is None or e.status == 429 or e.status >= 500:
            _down_until = time.monotonic() + DOWN_S
        return None
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    route = data["routes"][0]
    line = [[lat, lon] for lon, lat in route["geometry"]["coordinates"]]  # GeoJSON is [lon, lat]
    return RouteResult(
        geometry=_thin(line),
        leg_meters=[leg["distance"] for leg in route["legs"]],
    )


def _thin(line: list[list[float]]) -> list[list[float]]:
    """Keep every k-th point (and the last one) so at most MAX_POLYLINE_POINTS remain."""
    if len(line) <= MAX_POLYLINE_POINTS:
        return line
    step = ceil((len(line) - 1) / (MAX_POLYLINE_POINTS - 1))
    return line[:-1:step] + [line[-1]]
