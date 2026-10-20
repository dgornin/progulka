"""Static checks of the planner page: the frontend has no build step and no JS test
runner, so the invariants that would break silently in a browser are pinned here: the
script and the markup agree on element ids, every agent tool has its icon, MapLibre is
pinned with its hashes, and no script turns a string into markup."""

import re
from pathlib import Path

from app.agent.tools import tool_schemas

STATIC = Path(__file__).parent.parent / "app" / "static"
PLANNER_HTML = (STATIC / "planner.html").read_text()
PLANNER_JS = (STATIC / "js" / "planner.js").read_text()
MAPLIBRE = "https://unpkg.com/maplibre-gl@6.15.0/dist/maplibre-gl"


def test_every_element_the_planner_script_looks_up_is_on_the_page():
    wanted = set(re.findall(r'\$\("#([\w-]+)"\)', PLANNER_JS))
    present = set(re.findall(r'\bid="([\w-]+)"', PLANNER_HTML))
    assert wanted, "the lookup pattern no longer matches planner.js"
    assert wanted <= present, f"missing in planner.html: {sorted(wanted - present)}"


def test_every_agent_tool_has_its_icon_in_the_trace():
    icons = set(re.findall(r'\["(\w+)", \["', PLANNER_JS))
    assert icons == {schema["function"]["name"] for schema in tool_schemas()}


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
