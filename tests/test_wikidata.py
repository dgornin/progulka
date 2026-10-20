"""Popularity from Wikidata and Wikipedia page views (app/sources/wikidata.py), and how
the research tools rank and show it.

The Wikimedia answers were recorded from the real APIs on 2026-10-20 with the parameters
the module sends (tests/fixtures/sources_wikidata_*.json): the Hermitage (Q132783) and
the small Blockade barber shop museum in St Petersburg (Q21026905). The HTTP client is
faked, so the tests run offline.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from pydantic import TypeAdapter

from app.agent import tools
from app.agent.tools import ResearchState
from app.models import Place
from app.sources import osm, wikidata, wikipedia, wikivoyage
from app.sources.http import SourceError, http
from app.sources.wikidata import popularity as real_popularity
from tests.test_agent import START, make_state, run_tool

FIXTURES = Path(__file__).parent / "fixtures"
HERMITAGE = "Q132783"
BARBER = "Q21026905"
UNKNOWN = "Q100000000"  # well-formed, but no such item: Wikidata answers "missing"
RECORDED = {
    "www.wikidata.org": "entities",
    "ru.wikipedia.org": "pageviews_ru",
    "en.wikipedia.org": "pageviews_en",
}


def _recorded(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"sources_wikidata_{name}.json").read_text())


def _articles(params: dict[str, Any]) -> dict[str, Any]:
    """A made-up answer to either request: every id has a ru article «Статья <id>», and
    every title 10 views."""
    if "ids" in params:
        ids = params["ids"].split("|")
        links = {q: {"ruwiki": {"site": "ruwiki", "title": f"Статья {q}"}} for q in ids}
        return {"entities": {q: {"id": q, "sitelinks": links[q]} for q in ids}}
    days = {"2026-10-18": None, "2026-10-19": 10}
    return {
        "query": {"pages": [{"title": t, "pageviews": days} for t in params["titles"].split("|")]}
    }


def _place(pid: str, qid: str | None = None, north_m: float = 0, **fields: Any) -> Place:
    tags = {"wikidata": qid} if qid else {}
    lat = START.lat + north_m / 111_320
    values = {"name": pid, "kind": "museum", "source": "osm", **fields}
    return Place(id=pid, lat=lat, lon=START.lon, tags=tags, **values)


class Wikimedia:
    """Fake of http.get_json: per host the recorded answer or one a test sets, a log of
    the requests, and hosts that are down."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.options: list[dict[str, Any]] = []
        self.down: set[str] = set()
        self.answers: dict[str, Any] = {}

    async def get_json(self, url: str, params: dict[str, Any], **options: Any) -> Any:
        host = urlparse(url).netloc
        self.requests.append((host, params))
        self.options.append(options)
        if host in self.down:
            raise SourceError(f"{host}: HTTP 503", 503)
        return self.answers.get(host) or _recorded(RECORDED[host])

    def sent(self, host: str, param: str) -> list[str]:
        """The values of `param` in the requests to `host` that had it."""
        return [params[param] for h, params in self.requests if h == host and param in params]


@pytest.fixture
def wikimedia(no_popularity: None, monkeypatch: pytest.MonkeyPatch) -> Wikimedia:
    """The real lookup (conftest stubs it for every other test) on the recorded answers."""
    fake = Wikimedia()
    monkeypatch.setattr(wikidata, "popularity", real_popularity)
    monkeypatch.setattr(http, "get_json", fake.get_json)
    return fake


# --- the score -----------------------------------------------------------------------


async def test_a_world_famous_museum_outscores_a_small_one(wikimedia: Wikimedia) -> None:
    places = [
        _place("osm:way/1", HERMITAGE),
        _place("wikipedia:ru:51", HERMITAGE, source="wikipedia"),  # the same item, one score
        _place("wikipedia:ru:2", BARBER, source="wikipedia"),
        _place("osm:node/3"),  # no Wikidata id
        _place("osm:node/4", "Q1;Q2"),  # valid in OSM, but it would fail the whole request
        _place("osm:node/5", UNKNOWN),
    ]

    scores = await wikidata.popularity(places)

    # Hermitage: 22 802 + 23 592 views, 86 sitelinks; barber shop: 143 views, 1 sitelink.
    assert scores == {"osm:way/1": 0.91, "wikipedia:ru:51": 0.91, "wikipedia:ru:2": 0.31}
    assert wikidata.tier(0.91) == 0 and wikidata.tier(0.31) == 2
    assert wikimedia.sent("www.wikidata.org", "ids") == [f"{HERMITAGE}|{BARBER}|{UNKNOWN}"]
    assert wikimedia.sent("www.wikidata.org", "props") == ["sitelinks"]
    assert wikimedia.sent("ru.wikipedia.org", "titles") == ["Блокадная парикмахерская|Эрмитаж"]
    assert wikimedia.sent("en.wikipedia.org", "titles") == ["Hermitage Museum"]


async def test_places_without_a_valid_id_make_no_request(wikimedia: Wikimedia) -> None:
    places = [_place("osm:node/1"), _place("osm:node/2", "Q0"), _place("osm:node/3", "Q1234567890")]
    assert await wikidata.popularity(places) == {}
    assert wikimedia.requests == []


def test_the_score_is_absolute_and_capped() -> None:
    assert wikidata.score(0, 0) == 0.0
    assert wikidata.score(wikidata.V_REF, wikidata.S_REF) == 1.0
    assert wikidata.score(10 * wikidata.V_REF, 10 * wikidata.S_REF) == 1.0
    assert wikidata.score(1_000, 10) < wikidata.score(10_000, 10) < wikidata.score(10_000, 40)


def test_tiers() -> None:
    assert [wikidata.tier(p) for p in (1.0, wikidata.TOP, wikidata.NOTABLE)] == [0, 0, 1]
    assert [wikidata.tier(p) for p in (wikidata.NOTABLE - 0.01, 0.0, None)] == [2, 2, 2]


async def test_ids_go_fifty_a_request_and_titles_five(
    no_popularity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    async def get_json(url: str, params: dict[str, Any], **options: Any) -> dict[str, Any]:
        requests.append((urlparse(url).netloc, params))
        return _articles(params)

    monkeypatch.setattr(http, "get_json", get_json)
    places = [_place(f"osm:node/{n}", f"Q{n}") for n in range(1, 121)]

    scores = await real_popularity(places)

    assert len(scores) == 120 and set(scores.values()) == {wikidata.score(10, 1)}
    assert [len(p["ids"].split("|")) for h, p in requests if h == "www.wikidata.org"] == [
        50,
        50,
        20,
    ]
    batches = [len(p["titles"].split("|")) for h, p in requests if h == "ru.wikipedia.org"]
    assert len(batches) == 24 and max(batches) == wikidata.VIEW_BATCH


async def test_page_views_follow_the_continuation(
    wikimedia: Wikimedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MediaWiki answers views for a few titles a request, in title order, and asks to
    # continue for the rest; the pages it has not reached yet come without "pageviews".
    complete = _recorded("pageviews_ru")
    partial = json.loads(json.dumps(complete))
    next(p for p in partial["query"]["pages"] if p["title"] == "Эрмитаж").pop("pageviews")
    partial["continue"] = {"pvipcontinue": "Эрмитаж", "continue": "||"}

    async def get_json(url: str, params: dict[str, Any], **options: Any) -> Any:
        if urlparse(url).netloc == "ru.wikipedia.org":
            wikimedia.requests.append(("ru.wikipedia.org", params))
            return complete if "pvipcontinue" in params else partial
        return await wikimedia.get_json(url, params)

    monkeypatch.setattr(http, "get_json", get_json)
    scores = await wikidata.popularity([_place("a", HERMITAGE), _place("b", BARBER)])

    assert scores == {"a": 0.91, "b": 0.31}
    assert wikimedia.sent("ru.wikipedia.org", "pvipcontinue") == ["Эрмитаж"]


@pytest.mark.parametrize(
    ("down", "expected"),
    [
        ("www.wikidata.org", {}),
        ("ru.wikipedia.org", {}),
        # The barber shop has no en article: its score does not need the en views.
        ("en.wikipedia.org", {"b": 0.31}),
    ],
)
async def test_a_failed_request_leaves_its_places_unscored(
    wikimedia: Wikimedia, down: str, expected: dict[str, float]
) -> None:
    # Unknown rather than low: a Hermitage scored without its en views would sink.
    wikimedia.down.add(down)
    assert await wikidata.popularity([_place("a", HERMITAGE), _place("b", BARBER)]) == expected


@pytest.mark.parametrize(
    "answer",
    [
        {"error": {"code": "no-such-entity", "info": "Could not find an entity"}},  # HTTP 200
        {"entities": [HERMITAGE]},
        ["not", "an", "object"],
    ],
)
async def test_a_malformed_answer_leaves_the_places_unscored(
    wikimedia: Wikimedia, answer: Any
) -> None:
    wikimedia.answers["www.wikidata.org"] = answer
    assert await wikidata.popularity([_place("a", HERMITAGE)]) == {}
    # The host did answer: no outage, the next lookup asks again.
    assert await wikidata.popularity([_place("a", HERMITAGE)]) == {}
    assert len(wikimedia.requests) == 2


async def test_a_slow_wikimedia_is_given_up(
    no_popularity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hanging(url: str, params: dict[str, Any], **options: Any) -> Any:
        await asyncio.sleep(5)

    monkeypatch.setattr(http, "get_json", hanging)
    monkeypatch.setattr(wikidata, "TIMEOUT_S", 0.01)
    assert await real_popularity([_place("a", HERMITAGE)]) == {}


async def test_a_hanging_view_batch_leaves_the_other_places_scored(
    no_popularity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def get_json(url: str, params: dict[str, Any], **options: Any) -> dict[str, Any]:
        if "Статья Q1" in params.get("titles", "").split("|"):
            await asyncio.sleep(5)
        return _articles(params)

    monkeypatch.setattr(http, "get_json", get_json)
    monkeypatch.setattr(wikidata, "TIMEOUT_S", 0.1)
    places = [_place(f"osm:node/{n}", f"Q{n}") for n in range(1, 16)]

    scores = await real_popularity(places)

    # The titles go five a request in sorted order: «Статья Q1» waits with Q10..Q13, and
    # only those five stay unscored.
    unscored = {f"osm:node/{n}" for n in (1, 10, 11, 12, 13)}
    assert set(scores) == {p.id for p in places} - unscored


async def test_after_an_outage_the_lookups_are_skipped_for_a_while(
    wikimedia: Wikimedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    places = [_place("a", HERMITAGE)]
    wikimedia.down.add("www.wikidata.org")
    assert await wikidata.popularity(places) == {}
    # A request fails fast: a short timeout and one retry, not the client's defaults.
    timeouts = {(o["timeout_s"], o["attempts"]) for o in wikimedia.options}
    assert timeouts == {(wikidata.REQUEST_TIMEOUT_S, wikidata.REQUEST_ATTEMPTS)}

    wikimedia.down.clear()
    asked = len(wikimedia.requests)
    assert await wikidata.popularity(places) == {}
    assert len(wikimedia.requests) == asked  # not sent: Wikimedia failed a moment ago

    monkeypatch.setattr(wikidata, "_down_until", 0.0)  # DOWN_S later
    assert await wikidata.popularity(places) == {"a": 0.91}


# --- the research tools --------------------------------------------------------------

CHAPEL = _place("osm:node/1", BARBER, north_m=100, name="Часовня")
ANONYMOUS = _place("osm:node/2", north_m=50, name="Мурал", kind="artwork")
HERMITAGE_FAR = _place("osm:way/3", HERMITAGE, north_m=1500, name="Эрмитаж")


@pytest.fixture
def osm_places(monkeypatch: pytest.MonkeyPatch) -> None:
    async def search_places(center, radius_m, categories, limit=40):
        return [CHAPEL, ANONYMOUS, HERMITAGE_FAR]

    monkeypatch.setattr(osm, "search_places", search_places)


async def test_a_search_lists_a_must_see_before_nearer_ordinary_places(
    wikimedia: Wikimedia, osm_places: None
) -> None:
    state = make_state()

    outcome = await run_tool(state, "search_osm", categories=["museum", "artwork"])

    shown = json.loads(outcome.content)["places"]
    # The must-see first; the ordinary and the unscored after it, nearest first.
    assert [(p["id"], p.get("pop")) for p in shown] == [
        ("osm:way/3", 0.91),
        ("osm:node/2", None),
        ("osm:node/1", 0.31),
    ]
    assert len(wikimedia.sent("www.wikidata.org", "ids")) == 1  # one lookup for the search
    assert state.registry["osm:way/3"].popularity == 0.91
    details = json.loads((await run_tool(state, "get_place_details", place_id="osm:way/3")).content)
    assert details["pop"] == 0.91


async def test_wikipedia_results_are_scored_too(
    wikimedia: Wikimedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    article = _place("wikipedia:ru:51", HERMITAGE, source="wikipedia", name="Эрмитаж")

    async def nearby(center, radius_m, lang="ru", limit=30):
        return [article]

    monkeypatch.setattr(wikipedia, "nearby", nearby)
    outcome = await run_tool(make_state(), "search_wikipedia")
    assert json.loads(outcome.content)["places"][0]["pop"] == 0.91


async def test_a_must_see_goes_first_only_near_the_start(
    wikimedia: Wikimedia, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Wikivoyage lists the sights of the whole city: lifted at any distance, the far
    # landmarks would fill the list and push out what is near.
    far = _place("wikivoyage:1", HERMITAGE, north_m=5000, source="wikivoyage", name="Эрмитаж")
    near = _place("wikivoyage:2", north_m=400, source="wikivoyage", name="Сквер", kind="park")
    walkable = _place("wikivoyage:3", HERMITAGE, north_m=1500, source="wikivoyage", name="Музей")

    async def listings(city: str, lang: str = "ru", sections: Any = None) -> list[Place]:
        return [far, near, walkable]

    monkeypatch.setattr(wikivoyage, "listings", listings)

    outcome = await run_tool(make_state(), "get_wikivoyage_listings", section="see")

    shown = json.loads(outcome.content)["places"]
    assert [(p["id"], p.get("pop")) for p in shown] == [
        ("wikivoyage:3", 0.91),
        ("wikivoyage:2", None),
        ("wikivoyage:1", 0.91),
    ]


async def test_a_search_goes_on_by_distance_when_wikimedia_is_down(
    wikimedia: Wikimedia, osm_places: None
) -> None:
    wikimedia.down.add("www.wikidata.org")

    outcome = await run_tool(make_state(), "search_osm", categories=["museum", "artwork"])

    assert outcome.error is None
    shown = json.loads(outcome.content)["places"]
    assert [p["id"] for p in shown] == ["osm:node/2", "osm:node/1", "osm:way/3"]
    assert not any("pop" in p for p in shown)


async def test_a_bug_in_the_lookup_does_not_fail_the_search(
    osm_places: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(places: Any, timeout_s: Any = None) -> dict[str, float]:
        raise KeyError("entities")

    monkeypatch.setattr(wikidata, "popularity", broken)
    outcome = await run_tool(make_state(), "search_osm", categories=["museum", "artwork"])
    assert outcome.error is None and outcome.count == 3


async def test_the_lookup_gets_only_what_the_tool_timeout_leaves(
    osm_places: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    budgets: list[float] = []

    async def lookup(places: Any, timeout_s: float) -> dict[str, float]:
        budgets.append(timeout_s)
        return {}

    monkeypatch.setattr(wikidata, "popularity", lookup)
    # The source answers at once here; a shorter tool timeout stands for a slow source.
    monkeypatch.setattr(tools, "TOOL_TIMEOUT_S", 4)  # 4 s left: 2 for the lookup
    await run_tool(make_state(), "search_osm", categories=["museum"])
    monkeypatch.setattr(tools, "TOOL_TIMEOUT_S", 2.5)  # no time for a lookup
    outcome = await run_tool(make_state(), "search_osm", categories=["museum"])

    assert outcome.error is None and outcome.count == 3  # the places, unscored
    assert len(budgets) == 1 and 1.5 < budgets[0] <= 2


def test_a_journey_saved_before_popularity_loads_unscored() -> None:
    state = make_state()
    state.registry[CHAPEL.id] = CHAPEL
    saved = TypeAdapter(ResearchState).dump_python(state, mode="json")
    del saved["registry"][CHAPEL.id]["popularity"]
    loaded = TypeAdapter(ResearchState).validate_python(saved)
    assert loaded.registry[CHAPEL.id].popularity is None
