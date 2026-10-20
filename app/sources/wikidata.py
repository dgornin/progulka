"""How well known a place is: Wikipedia page views and Wikidata language versions.

A first-time visitor should not miss what everybody comes to see, and a name and a
160-character snippet do not tell the agent how much a place means to visitors.
Wikidata and the Wikimedia page-view statistics are CC0, need no key and answer from
Russia, and every place that matters already carries a Wikidata id: OSM's `wikidata`
tag, Wikipedia's `wikibase_item`, Wikivoyage's `wikidata` parameter, all kept in
`Place.tags["wikidata"]`. So the signal costs no new source and no new place ids: it is
a number attached to places the tools found anyway.

Per call, through the shared HTTP client (cached for a week, so a rehearsed run sees
the same numbers and replays from the LLM cache):
1. wbgetentities, ENTITY_BATCH ids a request, the requests in parallel, props=sitelinks
   only (claims of a famous item weigh hundreds of kilobytes): the number of sitelinks,
   i.e. language versions and sister projects, and the titles of the ru and en articles;
2. the page views of those articles over the last VIEW_DAYS days (PageViewInfo,
   prop=pageviews), ru and en summed, so a Paris museum and a Kazan mosque are measured
   on one scale whatever the traveller's language. MediaWiki answers views for about five
   titles it has not looked up lately and asks to continue for the rest (measured
   2026-10-20), so the titles go VIEW_BATCH a request, PARALLEL_VIEW_REQUESTS at a time.

score = 0.6 * min(1, log10(1 + views) / log10(1 + V_REF))
      + 0.4 * min(1, ln(1 + sitelinks) / ln(1 + S_REF)), rounded to 0.01.
The scale is absolute, not normalised per search: 0.8 means the same in every search,
day and city, so the agent and the eval can compare. Views say what people look up now;
sitelinks say how widely the place is known abroad, and keep a seasonal dip in views
from sinking a landmark. Logarithms, because fame spans five orders of magnitude.
Calibrated on 2026-10-20 (views: ru + en, the last 60 days):

    place                                views  sitelinks  score
    Eiffel Tower                       269 632       191    1.00
    Louvre                              92 407       170    0.98
    Hermitage Museum                    46 394        86    0.91
    Saint Isaac's Cathedral             41 912        56    0.87
    Kazan Kremlin                       12 463        48    0.80
    Kul Sharif mosque, Kazan             9 928        47    0.79
    Musée de Cluny, Paris                7 351        31    0.74
    Söyembikä Tower, Kazan               6 029        26    0.71
    Marjani mosque, Kazan                1 415        20    0.62
    Peter and Paul Cathedral, Kazan      1 408        15    0.60
    Old Tatar Quarter, Kazan               913         4    0.48
    Galeyev mosque, Kazan                  216         8    0.45
    House-commune of engineers, SPb        931         2    0.44
    Blockade barber shop, SPb              143         1    0.31
    an item with no ru or en article         0         0    0.00

so the world-famous land at 0.9-1, a city's must-sees at 0.7-0.9 (TOP), its second line
at 0.5-0.7 (NOTABLE), and an ordinary place with an article at 0.3-0.5.

Popularity is a hint and must never cost a search: a failed or malformed answer only
leaves its places without a score (unknown, not low). A request gets REQUEST_TIMEOUT_S
and one retry, not the client's 40 s and two; the whole lookup gives up after TIMEOUT_S,
or sooner when the source query has used most of the tool timeout (app/agent/tools.py),
and scores what answered by then. After an outage (no answer, 429 or 5xx after the
retry, or a lookup cut off at TIMEOUT_S) the lookups are skipped for DOWN_S, so a
Wikimedia that is down costs one search seconds, not every search. Nothing failed is
cached: a search whose lookup failed shows its places unscored, and on a replayed run,
where the lookup succeeds, a different list.
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.models import Place
from app.sources.http import SourceError
from app.sources.wikipedia import query_mediawiki

log = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
VIEW_WIKIS = ("ru", "en")
ENTITY_BATCH = 50  # the API's limit of ids per wbgetentities request
VIEW_BATCH = 5
VIEW_DAYS = 60  # the most PageViewInfo serves
MAX_CONTINUE = 3  # a batch of VIEW_BATCH should never need one; a safety net
PARALLEL_VIEW_REQUESTS = 12  # one wave for a usual search: up to 30 titles a wiki
TIMEOUT_S = 8.0
REQUEST_TIMEOUT_S = 4.0  # a request answers in under 1.6 s (measured 2026-10-20)
REQUEST_ATTEMPTS = 2  # one retry: a single 429 or 503 still gets its answer
DOWN_S = 60.0
V_REF = 150_000  # views over VIEW_DAYS that count as fully famous
S_REF = 120  # sitelinks that count as known everywhere
VIEWS_WEIGHT = 0.6
TOP = 0.7  # a city's must-see and above
NOTABLE = 0.5  # a city's second line: known, not a must
# One malformed id (an OSM "Q1;Q2", a typo) fails the whole wbgetentities request, and so
# does an id of ten digits or more (checked 2026-10-20; items are at nine digits), while an
# unknown id of nine digits only comes back as missing. Such ids are not sent.
_QID = re.compile(r"Q[1-9]\d{0,8}")
_down_until = 0.0  # time.monotonic() before which lookups are skipped (an outage)


@dataclass(frozen=True)
class _Entity:
    sitelinks: int
    titles: dict[str, str]  # wiki language -> article title


async def popularity(places: Sequence[Place], timeout_s: float | None = None) -> dict[str, float]:
    """Place id -> score 0..1, for the places whose Wikidata id could be scored."""
    by_qid = await scores((p.tags.get("wikidata", "") for p in places), timeout_s)
    return {
        p.id: by_qid[qid] for p in places if (qid := p.tags.get("wikidata", "").strip()) in by_qid
    }


async def scores(qids: Iterable[str], timeout_s: float | None = None) -> dict[str, float]:
    """Wikidata id -> score 0..1. Junk ids are ignored; ids that could not be scored are
    absent. Sorted before batching, so the same set of ids makes the same requests.

    `timeout_s` is the time the caller can give, TIMEOUT_S at most. When it runs out the
    ids whose answers are all in are still scored: a city's 'see' section takes about 50
    requests on a cold cache, and one slow batch must not cost the scores of the rest.
    """
    global _down_until
    wanted = sorted(
        {q.strip() for q in qids if _QID.fullmatch(q.strip())}, key=lambda q: int(q[1:])
    )
    if not wanted:
        return {}
    if time.monotonic() < _down_until:
        log.warning("popularity of %d places skipped: Wikimedia failed a moment ago", len(wanted))
        return {}
    limit = TIMEOUT_S if timeout_s is None else min(timeout_s, TIMEOUT_S)
    entities: dict[str, _Entity] = {}
    views: dict[tuple[str, str], int] = {}
    try:
        async with asyncio.timeout(limit):
            await _lookup(wanted, entities, views)
    except TimeoutError:
        log.warning("popularity of %d places: cut off after %.1f s", len(wanted), limit)
        if limit == TIMEOUT_S:  # its own limit, not the rest of a slow search's time
            _down_until = time.monotonic() + DOWN_S
    out = {}
    for qid, entity in entities.items():
        counted = [views.get((lang, title)) for lang, title in entity.titles.items()]
        if None not in counted:  # a failed or unfinished view batch: unknown, not unpopular
            out[qid] = score(sum(counted), entity.sitelinks)
    return out


def score(views: int, sitelinks: int) -> float:
    """The formula of the module docstring: absolute, so comparable across searches."""
    fame = min(1.0, math.log10(1 + views) / math.log10(1 + V_REF))
    reach = min(1.0, math.log1p(sitelinks) / math.log1p(S_REF))
    return round(VIEWS_WEIGHT * fame + (1 - VIEWS_WEIGHT) * reach, 2)


def tier(pop: float | None) -> int:
    """0 for a must-see, 1 for a notable place, 2 for the rest and the unknown: the
    research tools list the first two tiers near the start first."""
    if pop is None:
        return 2
    return 0 if pop >= TOP else 1 if pop >= NOTABLE else 2


async def _lookup(
    qids: list[str], entities: dict[str, _Entity], views: dict[tuple[str, str], int]
) -> None:
    """Fills `entities` and `views` as the answers come, so a timeout keeps them."""
    await asyncio.gather(
        *(
            _entities(qids[start : start + ENTITY_BATCH], entities)
            for start in range(0, len(qids), ENTITY_BATCH)
        )
    )
    titles = {
        lang: sorted({e.titles[lang] for e in entities.values() if lang in e.titles})
        for lang in VIEW_WIKIS
    }
    slots = asyncio.Semaphore(PARALLEL_VIEW_REQUESTS)

    async def batch(lang: str, chunk: list[str]) -> None:
        async with slots:
            await _batch_views(lang, chunk, views)

    await asyncio.gather(
        *(
            batch(lang, names[start : start + VIEW_BATCH])
            for lang, names in titles.items()
            for start in range(0, len(names), VIEW_BATCH)
        )
    )


async def _entities(qids: list[str], entities: dict[str, _Entity]) -> None:
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "sitelinks",
        "format": "json",
        "formatversion": 2,
    }
    try:
        payload = await query_mediawiki(
            WIKIDATA_API, params, timeout_s=REQUEST_TIMEOUT_S, attempts=REQUEST_ATTEMPTS
        )
        found = {}
        for key, entity in payload["entities"].items():
            if "missing" in entity:
                continue
            links = entity.get("sitelinks") or {}
            titles = {
                lang: links[f"{lang}wiki"]["title"] for lang in VIEW_WIKIS if f"{lang}wiki" in links
            }
            # A merged item may come back under its new id: the place carries the old one.
            asked = (entity.get("redirects") or {}).get("from", key)
            found[asked] = _Entity(len(links), titles)
        entities |= found
    except (SourceError, KeyError, TypeError, AttributeError) as e:
        _failed(e)
        log.warning("wikidata entities %s..%s: %s", qids[0], qids[-1], e)


async def _batch_views(lang: str, titles: list[str], views: dict[tuple[str, str], int]) -> None:
    """Views of each title over VIEW_DAYS, into `views` as each answer comes, so the
    titles answered before a failed continuation keep theirs; days without data (null)
    count as none."""
    params: dict[str, Any] = {
        "action": "query",
        "prop": "pageviews",
        "titles": "|".join(titles),
        "pvipdays": VIEW_DAYS,
        "format": "json",
        "formatversion": 2,
    }
    try:
        for _ in range(1 + MAX_CONTINUE):
            payload = await query_mediawiki(
                f"https://{lang}.wikipedia.org/w/api.php",
                params,
                timeout_s=REQUEST_TIMEOUT_S,
                attempts=REQUEST_ATTEMPTS,
            )
            for page in payload["query"]["pages"]:
                if "pageviews" in page:  # absent: left for the continuation
                    days = page["pageviews"] or {}
                    views[(lang, page["title"])] = sum(n for n in days.values() if n)
            if "continue" not in payload:
                break
            params = params | payload["continue"]
    except (SourceError, KeyError, TypeError, AttributeError) as e:
        _failed(e)
        log.warning("%swiki page views of %s..: %s", lang, titles[0], e)


def _failed(error: Exception) -> None:
    """An outage (no answer, 429 or 5xx after the retry) skips the lookups for DOWN_S; an
    answer the host did give, malformed or a MediaWiki error, does not."""
    global _down_until
    if isinstance(error, SourceError) and (
        error.status is None or error.status == 429 or error.status >= 500
    ):
        _down_until = time.monotonic() + DOWN_S
