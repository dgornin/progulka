"""Static checks of the landing, sign-in and cabinet pages: the frontend has no build step
and no JS test runner, so the invariants that would break silently in a browser are
pinned here: scripts find their elements, links and API calls lead to routes the server
has, and every day of the longest journey has its own colour."""

import re

import pytest
from starlette.routing import Route

from app import main
from app.models import MAX_DAYS

STATIC = main.STATIC
SCRIPTS = ("landing.js", "auth.js", "trips.js", "nav.js", "api.js")
# "/app/trips/{journey_id}" -> a pattern any one path segment fills
ROUTES = [
    re.compile(re.sub(r"\\\{\w+\\\}", "[^/]+", re.escape(route.path)))
    for route in main.app.routes
    if isinstance(route, Route)
]


def _js(name: str) -> str:
    return (STATIC / "js" / name).read_text()


def _served(path: str) -> bool:
    """A literal path, or a script's template such as "/api/journeys/${id}", has a route."""
    concrete = re.sub(r"\$\{[^}]*\}", "x", path)
    return any(route.fullmatch(concrete) for route in ROUTES)


@pytest.mark.parametrize(
    ("page", "script"),
    [("landing.html", "landing.js"), ("auth.html", "auth.js"), ("trips.html", "trips.js")],
)
def test_every_element_a_script_looks_up_is_on_its_page(page, script):
    js = _js(script)
    wanted = set(re.findall(r'getElementById\("([\w-]+)"\)', js))
    wanted |= set(re.findall(r'\$\("#([\w-]+)"\)', js))
    present = set(re.findall(r'\bid="([\w-]+)"', (STATIC / page).read_text()))
    assert wanted, f"the lookup patterns no longer match {script}"
    assert wanted <= present, f"missing in {page}: {sorted(wanted - present)}"


def test_links_lead_to_pages_the_server_serves():
    links = set()
    for page in ("landing.html", "auth.html", "trips.html"):
        links |= set(re.findall(r'href="(/[^"?#]*)', (STATIC / page).read_text()))
    for script in SCRIPTS:
        links |= set(re.findall(r'["`](/app[^"`?]*)', _js(script)))
    links = {link for link in links if not link.startswith("/static/")}
    assert {"/register", "/login", "/app/trips", "/app/new", "/app/profile"} <= links
    assert not sorted(link for link in links if not _served(link))


def test_scripts_call_only_api_routes_the_server_has():
    calls = {path for script in SCRIPTS for path in re.findall(r'["`](/api/[^"`?]*)', _js(script))}
    assert {"/api/journeys", "/api/me", "/api/config"} <= calls
    assert not sorted(call for call in calls if not _served(call))


def test_every_day_of_the_longest_journey_has_its_own_colour():
    css = (STATIC / "css" / "base.css").read_text()
    colours = dict(re.findall(r"--day-(\d+):\s*(#[0-9a-f]{6})", css))
    assert sorted(map(int, colours)) == list(range(1, MAX_DAYS + 1))
    assert len(set(colours.values())) == MAX_DAYS
    # day 1 is the brand colour: a one-day journey looks like the single route it replaced
    assert colours["1"] == re.search(r"--route:\s*(#[0-9a-f]{6})", css)[1]
