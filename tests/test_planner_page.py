"""Static checks of the planner page: the frontend has no build step and no JS test
runner, so the invariants that would break silently in a browser are pinned here: the
script and the markup agree on element ids, every agent tool has its icon, MapLibre is
pinned with its hashes, no script turns a string into markup, and the planner's
scripts agree with the server on routes, the voice recording and the longest journey."""

import re
from datetime import date, timedelta
from pathlib import Path

from starlette.routing import Route

from app import main
from app.agent.planner import planner_tool_schemas
from app.agent.tools import tool_schemas
from app.models import MAX_DAYS
from app.pipeline import PlanRequest

STATIC = Path(__file__).parent.parent / "app" / "static"
PLANNER_HTML = (STATIC / "planner.html").read_text()
PLANNER_JS = (STATIC / "js" / "planner.js").read_text()
PIPELINE = (Path(__file__).parent.parent / "app" / "pipeline.py").read_text()
PLANNER_SCRIPTS = ("planner.js", "planner-map.js", "calendar.js", "time-range.js", "voice.js")
MAPLIBRE = "https://unpkg.com/maplibre-gl@6.15.0/dist/maplibre-gl"
# "/api/journeys/{journey_id}" -> a pattern any one path segment fills
ROUTES = [
    re.compile(re.sub(r"\\\{\w+\\\}", "[^/]+", re.escape(route.path)))
    for route in main.app.routes
    if isinstance(route, Route)
]


def _js(name: str) -> str:
    return (STATIC / "js" / name).read_text()


def _constant(script: str, name: str) -> int:
    return int(re.search(rf"const {name} = ([\d_]+);", _js(script))[1].replace("_", ""))


def test_every_element_the_planner_script_looks_up_is_on_the_page():
    wanted = set(re.findall(r'\$\("#([\w-]+)"\)', PLANNER_JS))
    present = set(re.findall(r'\bid="([\w-]+)"', PLANNER_HTML))
    assert wanted, "the lookup pattern no longer matches planner.js"
    assert wanted <= present, f"missing in planner.html: {sorted(wanted - present)}"


def test_every_agent_tool_has_its_icon_in_the_trace():
    # The day agents' tools and the trip planner's: both stream their calls to the trace.
    icons = set(re.findall(r'\["(\w+)", \["', PLANNER_JS))
    schemas = [*tool_schemas(), *planner_tool_schemas()]
    assert icons == {schema["function"]["name"] for schema in schemas}


def test_every_stage_the_pipeline_reports_has_its_place_in_the_strip():
    strip = set(re.findall(r'^  \["(\w+)", "', PLANNER_JS, flags=re.MULTILINE))
    reported = set(re.findall(r'stage\(\s*\w+,\s*"(\w+)"', PIPELINE))
    assert "synthesize" in reported
    assert reported <= strip, f"missing in STAGES: {sorted(reported - strip)}"


def test_maplibre_is_pinned_by_version_and_hash():
    css = rf'href="{re.escape(MAPLIBRE)}\.css"\s+integrity="sha384-S\+OnE5M4[^"]+"'
    assert re.search(css, PLANNER_HTML)
    # MapLibre 6 ships only an ES module; the import map pins it and must come before
    # the first module script, or the browser ignores it.
    assert f'"{MAPLIBRE}.mjs": "sha384-' in PLANNER_HTML
    assert PLANNER_HTML.index('type="importmap"') < PLANNER_HTML.index('type="module"')


def test_no_script_turns_a_string_into_markup():
    # Place names, user names and memory facts come from outside; they must reach the
    # page as text nodes only.
    unsafe = re.compile(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML\(|\.setHTML\(")
    for path in (STATIC / "js").glob("*.js"):
        assert not unsafe.search(path.read_text()), path


def test_every_link_to_another_site_cuts_the_opener():
    # Sources and the map services are other sites: a tab they open must not get
    # window.opener back to the planner, where the traveller is logged in.
    for path in (STATIC / "js").glob("*.js"):
        for line in path.read_text().splitlines():
            if 'target: "_blank"' in line:
                assert 'rel: "noopener noreferrer"' in line, (path.name, line.strip())


def test_planner_scripts_call_and_link_only_routes_the_server_has():
    paths = {
        path
        for name in PLANNER_SCRIPTS
        for path in re.findall(r'["`](/(?:api|app)/[^"`?#]*)', _js(name))
    }
    # the journey's endpoints: a template such as "/api/journeys/${id}" fills each segment
    concrete = {re.sub(r"\$\{[^}]*\}", "x", path) for path in paths}
    assert {
        "/api/plan",
        "/api/transcribe",
        "/api/journeys/x/days/x/feedback",
        "/api/journeys/x/feedback",
    } <= concrete
    assert not sorted(p for p in concrete if not any(r.fullmatch(p) for r in ROUTES))


def test_the_recorder_sends_what_the_server_accepts():
    rate, seconds = _constant("voice.js", "RATE"), _constant("voice.js", "MAX_SECONDS")
    assert (rate, seconds) == (main.AUDIO_RATE, main.MAX_AUDIO_S)
    # a 16-bit mono WAV of the longest recording: a 44-byte header, two bytes a sample
    assert 44 + rate * seconds * 2 <= main.MAX_AUDIO_BYTES


def test_the_widest_journey_the_composer_offers_is_a_valid_request():
    days = _constant("planner.js", "MAX_DAYS")
    assert days == MAX_DAYS
    first = date(2026, 10, 24)
    # the slider's ends: 06:00, and 24:00 sent as the day's last minute (wireTime)
    assert '"23:59"' in _js("time-range.js")
    PlanRequest(
        city="Казань",
        date_from=first,
        date_to=first + timedelta(days=days - 1),
        start_time="06:00",
        end_time="23:59",
    )


def test_the_planner_draws_days_in_the_shared_palette():
    # base.css's --day-N: a day wears one colour on the map, on its tab and in the cabinet
    map_js = _js("planner-map.js")
    tokens = re.findall(r"--day-(\d+):", (STATIC / "css" / "base.css").read_text())
    assert _constant("planner-map.js", "PALETTE_SIZE") == len(tokens) == MAX_DAYS
    assert "--day-${" in map_js
    assert not re.search(r'\[\s*"#[0-9a-fA-F]{6}"(\s*,\s*"#[0-9a-fA-F]{6}"){4,}', map_js)
