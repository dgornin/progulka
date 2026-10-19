"""Data-source parsers and ranking, against responses recorded from the real APIs.

The shared HTTP client is replaced by a fake that returns fixtures and records the
requests, so the tests run offline and also check what we ask the APIs for.
"""

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from app.config import settings
from app.models import GeoPoint, HourWeather
from app.planning.geo import haversine_m
from app.sources import geocode, osm, weather, wikipedia, wikivoyage
from app.sources.http import SourceError, http

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str):
    return json.loads((FIXTURES / f"sources_{name}.json").read_text())


@pytest.fixture
def api(monkeypatch):
    """Serve queued responses to get_json/post_form_json and record each request."""

    class FakeApi:
        def __init__(self):
            self.responses: list = []
            self.requests: list[tuple[str, dict]] = []

        async def respond(self, url, params=None, **options):
            self.requests.append((url, params))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    fake = FakeApi()
    monkeypatch.setattr(osm, "_preferred", 0)
    monkeypatch.setattr(http, "get_json", fake.respond)
    monkeypatch.setattr(http, "post_form_json", fake.respond)
    return fake


async def test_locate_city_takes_the_settlement_not_the_federal_subject(api):
    # Nominatim lists the federal-subject boundary (centre 59.96, 30.16) first.
    api.responses = [fixture("nominatim_spb")]

    city = await geocode.locate_city("Санкт-Петербург")

    assert city.center.lat == pytest.approx(59.94, abs=0.01)
    assert city.center.lon == pytest.approx(30.32, abs=0.01)
    south, west, north, east = city.bbox
    assert south < city.center.lat < north and west < city.center.lon < east
    assert city.country_code == "ru"


async def test_geocode_is_bounded_to_the_city_and_prefers_the_notable_match(api):
    # Bounded search: a bar named "Новая Голландия" comes first, the island is more important.
    api.responses = [fixture("nominatim_spb"), fixture("nominatim_new_holland")]

    point = await geocode.geocode("Новая Голландия", city="Санкт-Петербург")

    assert point.label == "остров Новая Голландия"
    params = api.requests[1][1]
    assert params["bounded"] == 1
    assert params["viewbox"] == "30.0433427,59.7441481,30.5671662,60.0907368"


async def test_geocode_searches_beyond_the_city_boundary_when_the_bbox_has_nothing(api):
    # Петергоф is outside the bbox of the city proper.
    peterhof = {"lat": "59.8835", "lon": "29.9070", "name": "Петергоф", "importance": 0.5}
    api.responses = [fixture("nominatim_spb"), [], [peterhof]]

    point = await geocode.geocode("Петергоф", city="Санкт-Петербург")

    assert point.label == "Петергоф"
    fallback = api.requests[2][1]
    assert fallback["q"] == "Петергоф, Санкт-Петербург" and "bounded" not in fallback


async def test_search_places_parses_ranks_and_names_artworks(api):
    api.responses = [fixture("overpass_pushkinskaya")]
    center = GeoPoint(lat=59.9330, lon=30.3570)

    places = await osm.search_places(center, 500, ["museum", "artwork"])

    by_id = {p.id: p for p in places}
    # Categories alternate, so the many artworks do not push museums out.
    assert [p.kind for p in places[:4]] == ["museum", "artwork", "museum", "artwork"]
    # Within a category: notable (wikidata) first, then by distance.
    museums = [p for p in places if p.kind == "museum"]
    assert [bool(p.tags.get("wikidata")) for p in museums[:3]] == [True, True, False]
    rest = [haversine_m(center.lat, center.lon, p.lat, p.lon) for p in museums[2:]]
    assert rest == sorted(rest)
    # An unnamed mural is kept under a generated title.
    mural = by_id["osm:node/6627300587"]
    assert mural.name == "Мурал" and mural.indoor is False
    tsoi = by_id["osm:node/4140882291"]
    assert tsoi.url == "https://www.openstreetmap.org/node/4140882291"
    assert "Автор: HoodGraff" in tsoi.description
    assert len(by_id) == len(places) == 18


async def test_search_places_dedupes_by_name_and_caps_at_limit(api):
    # The same museum mapped twice: the notable node and, 55 m away, a building outline.
    payload = fixture("overpass_pushkinskaya")
    node = next(e for e in payload["elements"] if e["id"] == 13092952615)
    outline = {
        "type": "way",
        "id": 1,
        "center": {"lat": node["lat"] + 0.0005, "lon": node["lon"]},
        "tags": {"tourism": "museum", "name": node["tags"]["name"]},
    }
    payload["elements"].append(outline)
    api.responses = [payload]

    places = await osm.search_places(GeoPoint(lat=59.9330, lon=30.3570), 500, ["museum"], limit=5)

    ids = [p.id for p in places]
    assert len(ids) == 5
    assert ids[0] == "osm:node/13092952615" and "osm:way/1" not in ids


async def test_search_places_rejects_unknown_categories_before_any_request(api):
    with pytest.raises(ValueError, match="valid: museum, gallery"):
        await osm.search_places(GeoPoint(lat=0, lon=0), 500, ["museum", "skyscraper"])
    assert api.requests == []


def test_every_category_except_artwork_requires_a_name():
    for spec in osm.CATEGORIES.values():
        named = all('["name"]' in f for f in spec.filters)
        assert named or spec.kind == "artwork", spec.kind
        assert 5 <= spec.visit_minutes <= 240


async def test_search_places_falls_back_to_the_mirror_and_then_prefers_it(api):
    urls = settings.overpass_urls
    api.responses = [
        SourceError("overpass-api.de: HTTP 504"),
        fixture("overpass_pushkinskaya"),
        fixture("overpass_pushkinskaya"),
    ]

    places = await osm.search_places(GeoPoint(lat=59.9330, lon=30.3570), 50_000, ["museum"])
    await osm.search_places(GeoPoint(lat=59.9330, lon=30.3570), 500, ["museum"])

    assert [url for url, _ in api.requests] == [urls[0], urls[1], urls[1]]
    # The radius is capped at 5 km: the box spans 2 x 5 km = 0.0898 degrees of latitude.
    assert "(59.88808,30.26735,59.97792,30.44665)" in api.requests[1][1]["data"]
    assert places


async def test_search_places_skips_an_instance_that_timed_out_inside_the_query(api):
    api.responses = [
        {"remark": "runtime error: Query timed out", "elements": []},
        fixture("overpass_pushkinskaya"),
    ]

    places = await osm.search_places(GeoPoint(lat=59.9330, lon=30.3570), 500, ["museum"])

    assert len(api.requests) == 2 and places


async def test_wikipedia_nearby_keeps_order_and_falls_back_to_short_description(api):
    api.responses = [fixture("wikipedia_nearby")]

    places = await wikipedia.nearby(GeoPoint(lat=59.9343, lon=30.3351), 600, limit=25)

    assert len(places) == 25
    first = places[0]
    assert first.name == "Российская национальная библиотека"
    assert first.id == f"wikipedia:ru:{first.url.rsplit('=', 1)[1]}"
    assert first.kind == "landmark" and first.tags["wikidata"].startswith("Q")
    assert all(len(p.description or "") <= wikipedia.MAX_DESCRIPTION_CHARS for p in places)
    # Pages past the 20th get no extract, only the one-line Wikidata description.
    assert places[21].description == "театр в Санкт-Петербурге"


async def test_wikipedia_search_is_limited_to_the_area(api):
    api.responses = [fixture("wikipedia_search")]

    places = await wikipedia.search("конструктивизм", GeoPoint(lat=59.9343, lon=30.3351), 5000)

    assert api.requests[0][1]["gsrsearch"] == "конструктивизм nearcoord:5000m,59.9343,30.3351"
    assert places[0].name == "Дом-коммуна инженеров и писателей"
    assert all(59.8 < p.lat < 60.0 for p in places)


async def test_page_extract_is_cut_at_a_sentence(api):
    api.responses = [fixture("wikipedia_extract")]

    text = await wikipedia.page_extract("wikipedia:ru:542", max_chars=600)

    assert api.requests[0][1]["pageids"] == "542"
    assert text.startswith("Госуда́рственный Ру́сский музе́й")
    assert len(text) <= 600 and text.endswith(".")


async def test_wikivoyage_falls_back_to_district_pages(api):
    # The ru city page has no listings with coordinates, only the district list.
    api.responses = [fixture("wikivoyage_spb_city"), fixture("wikivoyage_spb_districts")]

    places = await wikivoyage.listings("Санкт-Петербург", sections=["see"])

    assert api.requests[1][1]["titles"] == (
        "Санкт-Петербург/Центр|Санкт-Петербург/Коломна"
        "|Санкт-Петербург/Васильевский остров|Санкт-Петербург/Петроградская сторона"
    )
    assert places[0].id == "wikivoyage:ru:Санкт-Петербург/Центр#0"
    assert places[0].name == "Здание Главного штаба" and places[0].kind == "sight"
    names = {p.name for p in places}
    assert "Зимний дворец" not in names  # listed without coordinates
    assert {"Ростральные колонны", "Петропавловская крепость"} <= names
    garden = next(p for p in places if p.name == "Летний сад")
    assert garden.opening_hours is None
    assert "Часы: май–сентябрь: 10:00–22:00, октябрь–март: 10:00–20:00" in garden.description


async def test_wikivoyage_takes_districts_from_the_district_list_first(api):
    # Abridged en "Moscow": a map shape links an outlying district before the list.
    city_page = (
        "{{Mapshape|type=geoshape|title=[[Moscow/Zelenograd|Zelenograd]]}}\n"
        "{{Regionlist|\nregion1name=[[Moscow/Central|Central Moscow]] |\n"
        "region2name=[[Moscow/Central-North|Central-North]] |\n}}\n"
        "The Bolshoi Theatre is in [[Moscow/Central-North#Q138908|Central-North]]."
    )
    page = {"title": "Moscow", "revisions": [{"slots": {"main": {"content": city_page}}}]}
    api.responses = [{"query": {"pages": [page]}}, {"query": {"pages": []}}]

    await wikivoyage.listings("Moscow", lang="en")

    assert api.requests[1][1]["titles"] == "Moscow/Central|Moscow/Central-North|Moscow/Zelenograd"


async def test_wikivoyage_strips_markup_and_keeps_requested_sections(api):
    api.responses = [fixture("wikivoyage_paris")]

    places = await wikivoyage.listings("Paris", lang="en", sections=["see", "do"])

    assert len(api.requests) == 1  # enough listings on the city page: no district fetch
    eiffel = places[0]
    assert eiffel.name == "Eiffel Tower"  # was [[Paris/7th arrondissement#Q243|Eiffel Tower]]
    assert eiffel.url == "https://en.wikivoyage.org/wiki/Paris"
    assert eiffel.tags["wikidata"] == "Q243"
    notre_dame = next(p for p in places if p.name == "Notre Dame Cathedral")
    assert notre_dame.description == (
        "Impressive Gothic cathedral that was the inspiration for Victor Hugo's novel "
        "The Hunchback of Notre Dame."
    )
    for place in places:
        text = place.name + (place.description or "")
        assert not any(markup in text for markup in ("[[", "{{", "''", "<!--"))
    assert {p.kind for p in places} == {"sight", "activity"}


async def test_wikivoyage_rejects_unknown_sections(api):
    with pytest.raises(ValueError, match="sleep"):
        await wikivoyage.listings("Paris", lang="en", sections=["see", "sleep"])


async def test_weather_hourly_parses_naive_local_time(api):
    api.responses = [fixture("open_meteo")]
    day = date.today() + timedelta(days=1)

    hours = await weather.hourly(GeoPoint(lat=59.94, lon=30.32), day)

    assert api.requests[0][1]["start_date"] == api.requests[0][1]["end_date"] == day.isoformat()
    assert len(hours) == 24
    assert hours[0].time.tzinfo is None and hours[0].time.hour == 0
    assert hours[14].temperature_c is not None


async def test_weather_hours_past_the_forecast_horizon_are_dropped(api):
    # On the last forecast day Open-Meteo returns nulls after 06:00.
    api.responses = [fixture("open_meteo_last_day")]

    hours = await weather.hourly(GeoPoint(lat=59.94, lon=30.32), date.today())

    assert [h.time.hour for h in hours] == [0, 1, 2, 3, 4, 5]
    assert weather.summarize(hours, time(10), time(18)) == "прогноза нет"


@pytest.mark.parametrize("days_ahead", [-1, 16])
async def test_weather_outside_forecast_range_is_empty(api, days_ahead):
    day = date.today() + timedelta(days=days_ahead)

    assert await weather.hourly(GeoPoint(lat=59.94, lon=30.32), day) == []
    assert api.requests == []


def _hour(h: int, temp: float, prob: int, mm: float = 0.0) -> HourWeather:
    return HourWeather(
        time=datetime(2026, 10, 20, h),
        temperature_c=temp,
        precipitation_probability=prob,
        precipitation_mm=mm,
    )


def test_summarize_reports_temperature_and_rain_spans():
    hours = [_hour(h, 8 + (h - 10) * 0.5, 10) for h in range(10, 18)]
    hours[4] = _hour(14, 10, 70)
    hours[5] = _hour(15, 11, 40, mm=0.5)  # wet by amount, not by probability
    hours[7] = _hour(17, 9, 60)

    summary = weather.summarize(hours, time(10, 30), time(18, 0))

    assert summary == "+8…+11 °C, дождь вероятен 14:00–16:00, 17:00–18:00 (до 70%)"
    dry = weather.summarize(hours[:3], time(10), time(13), lang="en")
    assert dry == "+8…+9 °C, no precipitation"
    cold = [_hour(10, -0.4, 0), _hour(11, 2, 0)]
    assert weather.summarize(cold, time(10), time(12)) == "0…+2 °C, без осадков"
    assert weather.summarize([], time(10), time(18)) == "прогноза нет"
