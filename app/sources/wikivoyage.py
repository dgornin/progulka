"""Wikivoyage listings: places that travellers recommend, with a short review.

A Wikivoyage city guide stores its places as listing templates
({{see|name=...|lat=...|long=...|content=...}} on en, {{listing|type=see|...|description=...}}
on ru). We read the raw wikitext and parse those templates ourselves; the
rendered HTML would lose the coordinates.

Big cities keep their listings on district pages ("Санкт-Петербург/Центр"), and
the city page links them in its district list (the Regionlist template), in the
guide's own order: the centre first. Links before that list, such as the en map
shapes, come in no useful order (en "Moscow" names Zelenograd first). When the
city page has few listings, we fetch the first districts in one request.

Free-text fields go through a small markup stripper: links become their text,
templates are dropped (the ru {{hours}} template is rendered, since it carries
the opening times). Free-text hours stay in the description and in tags, never
in opening_hours, which must be machine-readable OSM syntax.
"""

import html
import re
from urllib.parse import quote

from app.models import Place
from app.sources.http import http

SECTION_KIND = {"see": "sight", "do": "activity", "eat": "food", "drink": "bar", "buy": "shop"}
MIN_LISTINGS = 30  # fewer on the city page: an overview, the places live on district pages
MAX_DISTRICTS = 4

_LISTING_NAMES = "see|do|eat|drink|buy|sleep|go|listing|marker"
_REGIONLIST = re.compile(r"\{\{\s*regionlist", re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FILE_LINK = re.compile(r"\[\[(?:File|Image|Файл|Изображение):[^\[\]]*\]\]", re.I)
_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]|]*)\]\]")
_HOURS_TEMPLATE = re.compile(r"\{\{\s*hours\s*\|([^{}]*)\}\}", re.I)
_OTHER_TEMPLATE = re.compile(rf"\{{\{{(?!\s*(?:{_LISTING_NAMES})\s*\|)[^{{}}]*\}}\}}", re.I)
_LISTING = re.compile(rf"\{{\{{\s*({_LISTING_NAMES})\s*\|([^{{}}]*)\}}\}}", re.I)
_EXTERNAL_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]")
_HTML_TAG = re.compile(r"<[^>]+>")
_EMPHASIS = re.compile(r"'{2,}")


async def listings(city: str, lang: str = "ru", sections: list[str] | None = None) -> list[Place]:
    wanted = set(sections or SECTION_KIND)
    if not wanted <= SECTION_KIND.keys():
        unknown = sorted(wanted - SECTION_KIND.keys())
        raise ValueError(f"Unknown sections {unknown}; valid: {', '.join(SECTION_KIND)}")
    pages = await _wikitexts(lang, [city])
    if not pages:
        return []
    title, text = next(iter(pages.items()))
    places = _parse_listings(text, title, lang, wanted)
    if len(places) < MIN_LISTINGS:
        districts = _district_titles(text, title)[:MAX_DISTRICTS]
        if districts:
            district_pages = await _wikitexts(lang, districts)
            for district in districts:
                places += _parse_listings(district_pages.get(district, ""), district, lang, wanted)
    return _dedupe_by_name(places)


async def _wikitexts(lang: str, titles: list[str]) -> dict[str, str]:
    """Current wikitext of several pages in one request, keyed by resolved title."""
    payload = await http.get_json(
        f"https://{lang}.wikivoyage.org/w/api.php",
        {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "redirects": 1,
            "format": "json",
            "formatversion": 2,
        },
    )
    return {
        page["title"]: page["revisions"][0]["slots"]["main"]["content"]
        for page in payload.get("query", {}).get("pages", [])
        if page.get("revisions")
    }


def _district_titles(text: str, title: str) -> list[str]:
    """Subpages of the city linked from its page: from the district list on, then the rest."""
    regionlist = _REGIONLIST.search(text)
    if regionlist:
        text = text[regionlist.start() :] + text[: regionlist.start()]
    prefix = re.escape(title + "/")
    found = re.findall(rf"\[\[\s*({prefix}[^\]|#]+)", text.replace("_", " "))
    return list(dict.fromkeys(t.strip() for t in found))


def _parse_listings(text: str, page: str, lang: str, wanted: set[str]) -> list[Place]:
    hours_label = "Часы" if lang == "ru" else "Hours"
    places = []
    # n counts every listing on the page, so an id does not depend on `wanted`.
    for n, (template, body) in enumerate(_LISTING.findall(_simplify(text))):
        params = _params(body)
        generic = template.lower() in ("listing", "marker")
        section = (params.get("type", "") if generic else template).lower()
        name = _clean(params.get("name", ""))
        try:
            lat, lon = float(params["lat"]), float(params["long"])
        except (KeyError, ValueError):
            continue
        if section not in wanted or not name:
            continue
        content = _clean(params.get("content") or params.get("description") or "")
        hours = _clean(params.get("hours", ""))
        description = " ".join(filter(None, [content, f"{hours_label}: {hours}" if hours else ""]))
        tags = {
            "address": _clean(params.get("address", "")),
            "hours": hours,
            "price": _clean(params.get("price", "")),
            "website": params.get("url", ""),
            "wikidata": params.get("wikidata") or params.get("wdid", ""),
        }
        places.append(
            Place(
                id=f"wikivoyage:{lang}:{page}#{n}",
                name=name,
                lat=lat,
                lon=lon,
                kind=SECTION_KIND[section],
                source="wikivoyage",
                url=f"https://{lang}.wikivoyage.org/wiki/{quote(page.replace(' ', '_'))}",
                description=description or None,
                tags={k: v.strip() for k, v in tags.items() if v.strip()},
            )
        )
    return places


def _simplify(text: str) -> str:
    """Strip everything around listing templates so that they contain no nested markup.

    Links are replaced by their text (they contain "|", which would break the
    parameter split) and all non-listing templates are removed innermost first.
    """
    text = _COMMENT.sub("", text)
    text = _FILE_LINK.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _HOURS_TEMPLATE.sub(_render_hours, text)
    while (stripped := _OTHER_TEMPLATE.sub("", text)) != text:
        text = stripped
    return text


def _render_hours(match: re.Match) -> str:
    """ru {{hours|Д1|Д2|Ч1|М1|Ч2|М2}} -> "Д1–Д2 Ч1:М1–Ч2:М2", minutes default to 00."""
    args = [a.strip() for a in match.group(1).split("|")] + [""] * 6
    days = "–".join(a for a in args[:2] if a)
    return f"{days} {args[2]}:{args[3] or '00'}–{args[4]}:{args[5] or '00'}".strip()


def _params(body: str) -> dict[str, str]:
    params = {}
    for part in body.split("|"):
        key, sep, value = part.partition("=")
        if sep:
            params[key.strip().lower()] = value.strip()
    return params


def _clean(value: str) -> str:
    value = _EXTERNAL_LINK.sub(r"\1", value)
    value = _HTML_TAG.sub(" ", value)
    value = _EMPHASIS.sub("", value)
    return " ".join(html.unescape(value).split())


def _dedupe_by_name(places: list[Place]) -> list[Place]:
    """The city page often repeats the top sights of its districts."""
    seen: set[str] = set()
    unique = []
    for place in places:
        key = place.name.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(place)
    return unique
