"""Trips of several days offline: the day plan (outline), the days researched in
parallel over one registry, the trip planner agent that allocates their places to the
days (or code, when it is off or fails), the days optimized one by one without repeats,
and feedback on one day that leaves the others' places alone."""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any

import pytest

from app import pipeline
from app.agent import outline
from app.agent.outline import draft_days, locate_areas
from app.agent.tools import ResearchState
from app.llm import ChatTurn, LLMError
from app.models import Candidate, DayOutline, GeoPoint, HourWeather, Interest, Place, TripBrief
from app.planning.geo import haversine_m
from app.sources import geocode, osm, weather
from app.sources.geocode import CityInfo
from app.sources.http import SourceError
from tests.test_agent import Recorder, call, turn

CENTER = GeoPoint(lat=55.7963, lon=49.1088, label="Казанский кремль")
# Six kilometres south: within a day's reach of its own start, but too far for the
# days that start at the centre to borrow its places once their own fill the window.
OLD_TOWN = GeoPoint(lat=55.7423, lon=49.1088, label="Старо-Татарская слобода")
DAY = date(2026, 10, 24)
RAINY = DAY + timedelta(days=1)


def _place(n: int, kind: str, near: GeoPoint) -> Place:
    return Place(
        id=f"osm:node/{n}",
        name=f"Место {n}",
        lat=near.lat + 0.001 * (n % 10),
        lon=near.lon,
        kind=kind,
        source="osm",
        indoor=kind == "museum",
    )


KINDS = ["museum", "church", "artwork", "cafe", "cafe", "gallery", "cafe", "bakery"]
CENTRE_PLACES = [_place(n, kind, CENTER) for n, kind in enumerate(KINDS, start=1)]
OLD_TOWN_PLACES = [_place(20 + n, "artwork", OLD_TOWN) for n in range(1, 5)]
P = {p.id: p for p in [*CENTRE_PLACES, *OLD_TOWN_PLACES]}
SEARCH = {"categories": ["museum", "cafe"], "radius_m": 1000}


class TripLLM:
    """A scripted LLM for days that run in parallel: the agent's turns are scripted per
    day (read from the date in the research task) and the trip planner's on their own,
    an exception among them is raised, structured replies per schema, and the writer
    gets texts for exactly the stops it was sent."""

    def __init__(
        self,
        turns: dict[date, list[ChatTurn | Exception]],
        structured: dict[str, list[Any]],
        planner: list[ChatTurn | Exception] = (),
    ):
        self.turns = turns
        self.structured_replies = structured
        self.planner = list(planner)
        self.days: list[date] = []  # the day of every agent turn, in call order
        self.seen: list[tuple[str, list[dict[str, Any]]]] = []

    async def chat(self, messages, tools=None, **_):
        if any(tool["function"]["name"] == "finish_plan" for tool in tools or []):
            self.seen.append(("planner", json.loads(json.dumps(messages))))
            reply = self.planner.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        day = date.fromisoformat(re.search(r"Date: (\d{4}-\d{2}-\d{2})", messages[1]["content"])[1])
        self.days.append(day)
        self.seen.append(("chat", json.loads(json.dumps(messages))))
        await asyncio.sleep(0)  # a real call yields to the other days
        reply = self.turns[day].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def structured(self, messages, schema, **_):
        self.seen.append((schema.__name__, messages))
        if schema.__name__ == "PlanTexts":
            stops = json.loads(messages[1]["content"])["stops"]
            texts = [{"place_id": s["place_id"], "blurb": "Текст.", "tip": None} for s in stops]
            return schema(title=f"С {stops[0]['name']}", summary="Коротко.", stops=texts)
        return schema.model_validate(self.structured_replies[schema.__name__].pop(0))

    def calls(self, kind: str) -> list[list[dict[str, Any]]]:
        return [messages for name, messages in self.seen if name == kind]


@pytest.fixture(autouse=True)
def trip_planner(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Code assembles the days unless a test turns the trip planner agent on: most tests
    here pin how code splits the places between the days, which is also what runs
    whenever the planner fails."""

    def use(on: bool) -> None:
        monkeypatch.setattr(pipeline, "settings", replace(pipeline.settings, trip_planner_agent=on))

    use(False)
    return use


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> list[GeoPoint]:
    """Fake sources; returns the centres of the OSM searches that reached the source."""
    searches: list[GeoPoint] = []

    async def locate_city(city, country=None):
        return CityInfo(name="Казань", center=CENTER, bbox=(55.6, 49.0, 55.9, 49.3))

    async def geocoder(query, city=None, country=None):
        found = {
            "Старо-Татарская слобода": OLD_TOWN,
            "Иннополис": GeoPoint(lat=55.75, lon=48.74, label="Иннополис"),  # 23 km away
        }
        return found.get(query)

    async def hourly(point, day, today):
        rain = 90 if day == RAINY else 0
        return [
            HourWeather(
                time=datetime.combine(day, time(11)),
                temperature_c=10,
                precipitation_probability=rain,
            )
        ]

    async def search_places(center, radius_m, categories, limit=40):
        searches.append(center)
        await asyncio.sleep(0)
        return [p for p in P.values() if haversine_m(center.lat, center.lon, p.lat, p.lon) < 2000]

    async def no_route(points):
        return None

    monkeypatch.setattr(geocode, "locate_city", locate_city)
    monkeypatch.setattr(geocode, "geocode", geocoder)
    monkeypatch.setattr(weather, "hourly", hourly)
    monkeypatch.setattr(osm, "search_places", search_places)
    monkeypatch.setattr(pipeline, "walking_route", no_route)
    return searches


INTERESTS = [
    {"label": "история", "weight": 1.0},
    {"label": "кофе", "weight": 0.7},
    {"label": "стрит-арт", "weight": 0.4},
]
BRIEF_REPLY = {
    "city": "Казань",
    "date": DAY.isoformat(),
    "days": 3,
    "start_time": "10:00",
    "end_time": "13:00",
    "interests": INTERESTS,
}
OUTLINE_REPLY = {
    "title": "Казань за три дня",
    "days": [
        {"theme": "Кремль и музеи", "focus": ["История", "космос"], "area": None},
        {"theme": "Кофейни под дождём", "focus": ["кофе"], "area": "Атлантида"},
        {"theme": "Слобода", "focus": ["стрит-арт"], "area": "Старо-Татарская слобода"},
    ],
}


def _shortlist(ids: list[int], interest: str) -> list:
    return [
        call(
            "shortlist_place",
            place_id=f"osm:node/{n}",
            score=0.8,
            interests=[interest],
            reason="по теме дня",
            visit_minutes=40,
        )
        for n in ids
    ]


def _day_turns(ids: list[int], interest: str) -> list[ChatTurn]:
    return [
        turn(call("search_osm", **SEARCH)),
        turn(*_shortlist(ids, interest), call("finish_research", summary="готово")),
    ]


async def _three_days(events: Recorder) -> tuple[pipeline.Journey, TripLLM]:
    llm = TripLLM(
        turns={
            DAY: _day_turns([1, 2, 3, 4], "история"),
            # Places 3 and 4 were found by both of the first two days.
            RAINY: _day_turns([3, 4, 5, 6], "кофе"),
            DAY + timedelta(days=2): _day_turns([21, 22, 23, 24], "стрит-арт"),
        },
        structured={"BriefDraft": [BRIEF_REPLY], "OutlineDraft": [OUTLINE_REPLY]},
    )
    req = pipeline.PlanRequest(request="Казань, три дня с 24 октября, с 10 до 13")
    return await pipeline.plan_journey(req, llm, events, DAY), llm


async def test_three_days_are_planned_in_parallel_without_repeating_a_place(sources):
    events = Recorder()
    journey, llm = await _three_days(events)

    # The plan: one outline call that sees each day's forecast, checked by code.
    [outline_call] = llm.calls("OutlineDraft")
    days_in = json.loads(outline_call[1]["content"])["days"]
    assert [d["date"] for d in days_in] == ["2026-10-24", "2026-10-25", "2026-10-26"]
    assert "дождь" in days_in[1]["weather"] and "дождь" not in days_in[0]["weather"]
    assert journey.title == "Казань за три дня"
    assert (journey.brief.date, journey.brief.date_to) == (DAY, DAY + timedelta(days=2))
    assert [d.focus for d in journey.outline] == [["история"], ["кофе"], ["стрит-арт"]]
    # «Атлантида» is not on the map: that day starts from the trip's start.
    assert [d.area for d in journey.outline] == [None, None, "Старо-Татарская слобода"]
    assert [d.start for d in journey.outline] == [journey.start, journey.start, OLD_TOWN]

    # Research: the days' first turns come before any second turn, and the two days
    # that start at the centre sent the same search, which reached OSM once.
    assert llm.days[:3] == [DAY, RAINY, DAY + timedelta(days=2)]
    assert sorted((p.lat, p.lon) for p in sources) == [
        (OLD_TOWN.lat, OLD_TOWN.lon),
        (CENTER.lat, CENTER.lon),
    ]
    states = [day.state for day in journey.days]
    assert states[0].registry is states[1].registry is states[2].registry
    assert all(state.searches is states[0].searches for state in states)

    # Each day works from its own date, forecast, start and emphasis.
    assert [s.brief.date for s in states] == [DAY, RAINY, DAY + timedelta(days=2)]
    assert [s.brief.date_to for s in states] == [None, None, None]
    assert "дождь" in states[1].weather_summary and "дождь" not in states[0].weather_summary
    assert journey.days[1].weather[0].time.date() == RAINY
    assert [(i.label, i.weight) for i in states[1].brief.interests] == [
        ("история", 0.6),
        ("кофе", 0.9),
        ("стрит-арт", 0.24),
    ]
    assert states[2].start == OLD_TOWN and states[2].brief.start_point == OLD_TOWN.label
    tasks = [messages[1]["content"] for messages in llm.calls("chat")]
    assert any("plans day 2 of 3 of the trip (2026-10-25): Кофейни под дождём" in t for t in tasks)

    # No place on two days; every stop is on its day's shortlist, so a later edit of
    # the day keeps it.
    stops = [[s.place.id for s in day.plan.stops] for day in journey.days]
    assert all(len(day) >= 2 for day in stops)
    flat = [pid for day in stops for pid in day]
    assert len(flat) == len(set(flat))
    for day, ids in zip(journey.days, stops, strict=True):
        assert set(ids) <= set(day.state.shortlist)
        assert not [v for v in day.plan.violations if v.severity == "error"]
    # The first day took a cafe the second day's research found, at a discount: its own
    # three history stops already cover history.
    assert "osm:node/5" in stops[0]
    assert journey.days[0].state.shortlist["osm:node/5"].score == 0.48
    assert [day.plan.title for day in journey.days] == [f"С {P[ids[0]].name}" for ids in stops]

    # Every event of a day's work carries the day; the trip's own stages do not.
    for name in ("brief", "weather", "outline"):
        assert {e.data.get("day") for e in events if e.data.get("name") == name} == {None}
    for kind in ("tool_call", "candidate"):
        assert {e.data["day"] for e in events if e.type == kind} == {0, 1, 2}
    research = [e.data["day"] for e in events if e.data.get("name") == "research"]
    assert sorted(research) == [0, 0, 1, 1, 2, 2]
    outline_done = next(
        e.data for e in events if e.data.get("name") == "outline" and e.data["status"] == "done"
    )
    assert outline_done["detail"]["title"] == "Казань за три дня"
    assert [d["index"] for d in outline_done["detail"]["days"]] == [0, 1, 2]


async def test_feedback_on_one_day_leaves_the_other_days_places_alone(sources):
    journey, llm = await _three_days(Recorder())
    first_day = [s.place.id for s in journey.days[0].plan.stops]
    last_day = [s.place.id for s in journey.days[2].plan.stops]
    llm.structured_replies["PlanEdit"] = [
        {"research_goal": "найти ещё кофейню", "reply": "Ищу ещё кофейню."}
    ]
    llm.turns[RAINY] = [
        turn(call("search_osm", categories=["cafe"], radius_m=1500)),
        turn(
            *_shortlist([int(first_day[0].split("/")[1]), 7, 8], "кофе"),
            call("finish_research", summary="готово"),
        ),
        turn(call("finish_research", summary="готово")),
        turn(call("finish_research", summary="готово")),
    ]
    events = Recorder()

    journey, reply = await pipeline.apply_day_feedback(
        journey, 1, pipeline.FeedbackRequest(feedback="хочу ещё кофе"), llm, events
    )

    assert reply == "Ищу ещё кофейню."
    assert journey.days[1].state.taken == {*first_day, *last_day}
    # The agent never sees another day's stop, and cannot shortlist it.
    found = next(e for e in events if e.type == "tool_result" and e.data["name"] == "search_osm")
    search_reply = llm.calls("chat")[-1][3]["content"]
    assert found.data["error"] is None and not any(pid in search_reply for pid in first_day)
    rejected = [
        e.data["error"]
        for e in events
        if e.type == "tool_result" and e.data["error"] and "another day" in e.data["error"]
    ]
    assert len(rejected) == 1
    day = [s.place.id for s in journey.days[1].plan.stops]
    assert not set(day) & {*first_day, *last_day}
    assert {"osm:node/7", "osm:node/8"} <= set(journey.days[1].state.shortlist)
    assert [s.place.id for s in journey.days[0].plan.stops] == first_day
    assert {e.data.get("day") for e in events} == {1}


async def test_a_place_removed_from_one_day_is_not_planned_on_another(sources, make_journey):
    journey = make_journey(days=2)
    first, second = journey.days
    rejected = first.state.registry["osm:node/13"]  # the traveller removed it from day 2
    assert rejected.id in second.state.excluded
    spare = Place(
        id="osm:node/50", name="Запасное", lat=59.935, lon=30.36, kind="park", source="osm"
    )
    first.state.registry[spare.id] = spare
    # The first day's research found both too.
    for place in (rejected, spare):
        first.state.shortlist[place.id] = Candidate(
            place=place, score=0.9, reason="рядом", visit_minutes=20
        )
    llm = TripLLM(
        turns={},
        structured={"PlanEdit": [{"exclude_place_ids": ["osm:node/1"], "reply": "Убираю."}]},
    )

    journey, _ = await pipeline.apply_day_feedback(
        journey, 0, pipeline.FeedbackRequest(feedback="без первого места"), llm, Recorder()
    )

    state = journey.days[0].state
    assert state.taken == {"osm:node/11", "osm:node/12"}  # the other day's stops only
    assert state.rejected_elsewhere == {rejected.id}
    assert {s.place.id for s in journey.days[0].plan.stops} == {"osm:node/2", spare.id}


async def test_a_place_removed_from_one_day_moves_to_another_when_the_traveller_asks(
    sources, make_journey
):
    journey = make_journey(days=2)
    first, second = journey.days
    removed = first.state.registry["osm:node/13"]  # the traveller removed it from day 2
    # Wikipedia's record of it, which the first day's research found.
    article = removed.model_copy(update={"id": "wikipedia:ru:13", "source": "wikipedia"})
    first.state.registry[article.id] = article
    first.state.shortlist[article.id] = Candidate(
        place=article, score=0.9, reason="рядом", visit_minutes=20
    )
    llm = TripLLM(
        turns={},
        structured={
            "PlanEdit": [
                {"reply": "Хорошо."},
                {"add_places": ["Место 13"], "reply": "Добавляю Место 13."},
                {"reply": "Хорошо."},  # a later edit keeps the moved place
            ]
        },
    )

    async def edit(text: str) -> list[str]:
        await pipeline.apply_day_feedback(
            journey, 0, pipeline.FeedbackRequest(feedback=text), llm, Recorder()
        )
        return [s.place.id for s in journey.days[0].plan.stops]

    assert article.id not in await edit("подольше")  # removed elsewhere: not picked by itself
    assert article.id in await edit("добавь Место 13")
    assert article.id in await edit("подольше")
    plan = journey.days[0].plan
    assert "Место 13" in plan.brief.must_include
    assert not [v for v in plan.violations if v.code == "missing_must_include"]
    assert journey.days[0].state.rejected_elsewhere == set()
    assert journey.days[0].state.taken == {"osm:node/11", "osm:node/12"}
    assert second.state.excluded == {removed.id}  # the other day is not touched


async def test_a_day_the_provider_fails_is_built_from_the_other_days_places(sources):
    failure = LLMError("rate limited", 429)
    llm = TripLLM(
        turns={
            DAY: _day_turns([1, 2, 3, 4, 5], "история"),  # one more than the day takes
            RAINY: [failure, failure],  # its research, then its repair round
            DAY + timedelta(days=2): _day_turns([21, 22, 23, 24], "стрит-арт"),
        },
        structured={"BriefDraft": [BRIEF_REPLY], "OutlineDraft": [OUTLINE_REPLY]},
    )
    events = Recorder()

    journey = await pipeline.plan_journey(
        pipeline.PlanRequest(request="Казань, три дня"), llm, events, DAY
    )

    stops = [[s.place.id for s in day.plan.stops] for day in journey.days]
    assert all(stops) and len(sum(stops, [])) == len(set(sum(stops, [])))
    # The second day got the place the first day had no room for; its repair round
    # failed too, and it keeps the route as built.
    assert len(stops[1]) == 1 and stops[1][0] not in stops[0]
    assert "too_few_stops" in {v.code for v in journey.days[1].plan.violations}
    warnings = [e.data for e in events if e.type == "warning"]
    assert warnings == [
        {"message": "День 2: LLM-провайдер не ответил, соберу день из мест других дней", "day": 1}
    ]
    done = [e.data for e in events if e.type == "stage" and e.data["status"] == "done"]
    assert sorted(e["day"] for e in done if e["name"] == "research") == [0, 1, 2]
    assert [e["label"] for e in done if e["name"] == "repair"] == [
        "Исправить не удалось: LLM-провайдер не ответил"
    ]


async def test_a_trip_whose_every_day_fails_reports_the_providers_error(sources):
    failure = LLMError("rate limited", 429)
    llm = TripLLM(
        turns={DAY + timedelta(days=i): [failure] for i in range(3)},
        structured={"BriefDraft": [BRIEF_REPLY], "OutlineDraft": [OUTLINE_REPLY]},
    )

    with pytest.raises(LLMError, match="rate limited"):
        await pipeline.plan_journey(
            pipeline.PlanRequest(request="Казань, три дня"), llm, Recorder(), DAY
        )


async def test_a_single_walk_makes_no_outline_call_nor_a_planner_call(sources, trip_planner):
    trip_planner(True)
    llm = TripLLM(
        turns={DAY: _day_turns([1, 2, 3, 4], "история")},
        structured={"BriefDraft": [{**BRIEF_REPLY, "days": None, "interests": INTERESTS[::2]}]},
    )
    req = pipeline.PlanRequest(request="Казань, 24 октября, с 10 до 13")

    journey = await pipeline.plan_journey(req, llm, Recorder(), DAY)

    # With the trip planner on: a single walk has no days to share places between.
    assert [name for name, _ in llm.seen if name != "chat"] == ["BriefDraft", "PlanTexts"]
    assert len(journey.days) == len(journey.outline) == 1
    # Nothing is shifted between days: the brief's own weights, no day goal.
    assert [i.weight for i in journey.days[0].state.brief.interests] == [1.0, 0.4]
    assert "plans day" not in llm.calls("chat")[0][1]["content"]
    assert journey.title == journey.days[0].plan.title


async def test_dates_from_the_form_make_a_trip_without_a_brief_call(sources):
    llm = TripLLM(
        turns={
            DAY: _day_turns([1, 2, 3, 4], "История"),
            RAINY: _day_turns([5, 6, 7, 8], "История"),
        },
        structured={"OutlineDraft": [{"title": "Два дня", "days": []}]},
    )
    req = pipeline.PlanRequest(
        city="Казань",
        date_from=DAY,
        date_to=RAINY,
        start_time=time(10),
        end_time=time(13),
        preferences={"more": ["history"]},
    )

    journey = await pipeline.plan_journey(req, llm, Recorder(), DAY)

    assert "BriefDraft" not in [name for name, _ in llm.seen]
    assert [d.date for d in journey.outline] == [DAY, RAINY]
    # An outline without days: both days get no theme, and the must-have interest
    # still becomes the focus of one of them.
    assert [(d.theme, d.focus) for d in journey.outline] == [("", ["История"]), ("", [])]


async def test_a_trip_longer_than_a_week_is_refused(sources):
    llm = TripLLM(turns={}, structured={"BriefDraft": [{**BRIEF_REPLY, "days": 10}]})

    with pytest.raises(pipeline.PipelineError, match="Не больше 7 дней"):
        await pipeline.plan_journey(
            pipeline.PlanRequest(request="Казань на 10 дней"), llm, Recorder(), DAY
        )
    assert [name for name, _ in llm.seen] == ["BriefDraft"]


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        ({"date_to": DAY}, "Укажите первый день"),
        ({"date_from": RAINY, "date_to": DAY}, "раньше первого"),
        ({"date_from": DAY, "date_to": DAY + timedelta(days=7)}, "Не больше 7 дней"),
    ],
)
def test_the_forms_dates_are_checked(dates, message):
    with pytest.raises(ValueError, match=message):
        pipeline.PlanRequest(city="Казань", **dates)
    assert pipeline.PlanRequest(city="Казань", date_from=DAY, date_to=DAY + timedelta(days=6))


# --- the outline ---------------------------------------------------------------------


def _brief(**changes: Any) -> TripBrief:
    fields: dict[str, Any] = {
        "city": "Казань",
        "date": DAY,
        "date_to": DAY + timedelta(days=2),
        "start_time": time(10),
        "end_time": time(13),
        "interests": [Interest(**i) for i in INTERESTS],
        "must_include": ["Кремль", "Кул-Шариф", "Сююмбике"],
    }
    return TripBrief(**{**fields, **changes})


async def test_the_outline_keeps_only_what_the_brief_knows():
    reply = {
        "title": "Три дня в Казани, где история встречается с кофе и уличным искусством",
        "days": [
            {"theme": "Кремль", "focus": ["история", "гастрономия"], "must_include": ["кремль"]},
            {"theme": "Мечети", "focus": [], "must_include": ["Кремль", "Кул-Шариф"]},
        ],
    }
    llm = TripLLM(turns={}, structured={"OutlineDraft": [reply]})

    title, days = await draft_days(llm, _brief(), CENTER, ["сухо"] * 3)

    assert len(title) <= outline.TITLE_CHARS and title.endswith("…")
    assert [d.date for d in days] == [DAY, DAY + timedelta(days=1), DAY + timedelta(days=2)]
    # «гастрономия» is no interest of the brief; «кофе» (0.7) was forgotten and goes to
    # the day with the fewest focus labels; «стрит-арт» (0.4) may stay without a day.
    assert [d.focus for d in days] == [["история"], ["кофе"], []]
    # Each named place on one day, in the brief's spelling; the forgotten one goes to the
    # day with the fewest named places.
    assert [d.must_include for d in days] == [["Кремль"], ["Кул-Шариф"], ["Сююмбике"]]
    assert days[2].theme == ""


async def test_an_area_off_the_map_or_far_away_starts_from_the_trips_start(sources, monkeypatch):
    days = [
        DayOutline(index=0, date=DAY, area="Старо-Татарская слобода"),
        DayOutline(index=1, date=DAY, area="Иннополис"),
        DayOutline(index=2, date=DAY, area="Атлантида"),
        DayOutline(index=3, date=DAY, area="Кремль"),
    ]

    async def geocoder(query, city=None, country=None):
        if query == "Кремль":
            raise SourceError("HTTP 503")
        return {
            "Старо-Татарская слобода": OLD_TOWN,
            "Иннополис": GeoPoint(lat=55.75, lon=48.74),
        }.get(query)

    monkeypatch.setattr(geocode, "geocode", geocoder)

    located = await locate_areas(days, _brief(), CENTER)

    assert [d.area for d in located] == ["Старо-Татарская слобода", None, None, None]
    assert [d.start for d in located] == [OLD_TOWN, CENTER, CENTER, CENTER]


def test_other_days_candidates_are_discounted_and_named_places_wait_for_their_day():
    def run(named: str, *candidates: tuple[int, bool]) -> pipeline.DayRun:
        shortlist = {
            f"osm:node/{n}": Candidate(
                place=P[f"osm:node/{n}"], score=0.5, reason="ок", visit_minutes=30, required=req
            )
            for n, req in candidates
        }
        brief = _brief(must_include=[named])
        state = ResearchState(brief=brief, start=CENTER, weather_summary="", shortlist=shortlist)
        return pipeline.DayRun(id="d", state=state, weather=[])

    # «Место 2» is named for the first day, «Место 3» for the second; the second day's
    # research found both.
    days = [
        run("Место 2", (1, False)),
        run("Место 3", (1, False), (2, False), (3, True), (4, False), (5, False)),
    ]

    taken = pipeline._named_elsewhere(days, 0) | {"osm:node/4"}  # 4: an earlier day's stop
    borrowed = pipeline._borrowed(days, 0, taken)

    assert taken == {"osm:node/3", "osm:node/4"}
    assert pipeline._named_elsewhere(days, 1) == {"osm:node/2"}
    assert {pid: (c.score, c.required) for pid, c in borrowed.items()} == {
        "osm:node/2": (0.5, True),  # named for this day: kept, at full score
        "osm:node/5": (0.3, False),
    }
    assert days[1].state.shortlist["osm:node/5"].score == 0.5  # the other day keeps its own


async def test_another_sources_record_of_a_planned_place_is_not_planned_again(sources):
    cathedral = Place(
        id="osm:way/1",
        name="Исаакиевский собор",
        lat=CENTER.lat + 0.005,
        lon=CENTER.lon,
        kind="church",
        source="osm",
        tags={"wikidata": "Q1"},
    )
    # Wikipedia's record of the same cathedral, six metres away.
    article = cathedral.model_copy(
        update={"id": "wikipedia:ru:2", "source": "wikipedia", "lon": CENTER.lon + 0.0001}
    )
    others = [P["osm:node/2"], P["osm:node/3"]]

    def shortlist(*places: Place) -> dict[str, Candidate]:
        return {p.id: Candidate(place=p, score=0.8, reason="ок", visit_minutes=30) for p in places}

    state = ResearchState(
        brief=_brief(date_to=None, must_include=[]),
        start=CENTER,
        weather_summary="",
        registry={p.id: p for p in [cathedral, article, *others]},
        shortlist=shortlist(cathedral, *others),
    )
    day = pipeline.DayRun(id="d", state=state, weather=[])

    # Another day's research found the article: the day keeps its own record only.
    plan = await pipeline._build(day, Recorder(), borrowed=shortlist(article))
    assert [s.place.id for s in plan.stops].count(cathedral.id) == 1
    assert article.id not in {s.place.id for s in plan.stops}

    # Another day planned the cathedral: its article on this day's shortlist stays out.
    state.shortlist = shortlist(article, *others)
    state.taken = {cathedral.id}
    plan = await pipeline._build(day, Recorder())
    assert {s.place.id for s in plan.stops} == {p.id for p in others}


async def test_a_named_place_found_by_an_earlier_day_stays_with_its_own_day(sources):
    llm = TripLLM(
        turns={DAY: _day_turns([1, 2, 3, 4], "история"), RAINY: _day_turns([3, 5, 6, 7], "кофе")},
        structured={
            "BriefDraft": [{**BRIEF_REPLY, "days": 2, "must_include": ["Место 3"]}],
            "OutlineDraft": [
                {
                    "title": "Два дня",
                    "days": [
                        {"theme": "История", "focus": ["история"], "must_include": []},
                        {"theme": "Кофе", "focus": ["кофе"], "must_include": ["Место 3"]},
                    ],
                }
            ],
        },
    )
    req = pipeline.PlanRequest(request="Казань, два дня, обязательно Место 3")

    journey = await pipeline.plan_journey(req, llm, Recorder(), DAY)

    first, second = ([s.place.id for s in day.plan.stops] for day in journey.days)
    assert "osm:node/3" not in first and "osm:node/3" in second
    assert not [v for v in journey.days[1].plan.violations if v.code == "missing_must_include"]


# --- the trip planner ----------------------------------------------------------------

TWO_DAYS_OUTLINE = {
    "title": "Казань за два дня",
    "days": [
        {"theme": "Кремль и музеи", "focus": ["история"], "area": None},
        {"theme": "Кофейни под дождём", "focus": ["кофе"], "area": None},
    ],
}


def _ids(*numbers: int) -> list[str]:
    return [f"osm:node/{n}" for n in numbers]


# Places 3 and 4 were found by both days, which code would leave on the first; the planner
# gives them to the second and the first day place 5, which only the second day found.
PLANNED = [
    turn(
        call("assign_day", day=1, place_ids=_ids(1, 2, 5)),
        call("assign_day", day=2, place_ids=_ids(3, 4, 6)),
        call("check_day", day=1),
        call("check_day", day=2),
        content="Делю места между днями.",
    ),
    turn(call("finish_plan", summary="Храмы и музеи в первый день, кофейни во второй.")),
]


def _two_days_llm(planner: list[ChatTurn | Exception]) -> TripLLM:
    return TripLLM(
        turns={DAY: _day_turns([1, 2, 3, 4], "история"), RAINY: _day_turns([3, 4, 5, 6], "кофе")},
        structured={"BriefDraft": [{**BRIEF_REPLY, "days": 2}], "OutlineDraft": [TWO_DAYS_OUTLINE]},
        planner=planner,
    )


async def _plan_two_days(llm: TripLLM, events: Recorder) -> pipeline.Journey:
    req = pipeline.PlanRequest(request="Казань, два дня с 24 октября, с 10 до 13")
    return await pipeline.plan_journey(req, llm, events, DAY)


def _stops(journey: pipeline.Journey) -> list[list[str]]:
    return [[s.place.id for s in day.plan.stops] for day in journey.days]


async def test_the_trip_planner_allocates_the_places_and_code_builds_its_days(
    sources, trip_planner
):
    trip_planner(True)
    llm = _two_days_llm(PLANNED)
    events = Recorder()

    journey = await _plan_two_days(llm, events)

    # Each day is built from the places the planner gave it, and from nothing else.
    stops = _stops(journey)
    assert [set(day) for day in stops] == [set(_ids(1, 2, 5)), set(_ids(3, 4, 6))]
    assert all(not [v for v in d.plan.violations if v.severity == "error"] for d in journey.days)
    # Place 5 moved to the first day's shortlist at its full score: the planner, not a
    # discount, decided it belongs there.
    assert journey.days[0].state.shortlist["osm:node/5"].score == 0.8

    # The planner saw the pooled findings and the forecast, and the real code's verdict.
    first, second = llm.calls("planner")
    task = first[1]["content"]
    assert '"found_by":[1,2]' in task and '"rainy_hours":["11:00"]' in task
    checks = [json.loads(m["content"]) for m in second if m["role"] == "tool"][2:]
    assert [c["verdict"] for c in checks] == ["no errors", "no errors"]
    assert {s["id"] for s in checks[1]["stops"]} == set(_ids(3, 4, 6))

    # The planner works on the whole trip: its stage and its events carry no day; it runs
    # after every day's research and before any day is built.
    mine = [e for e in events if e.data.get("agent") == "planner"]
    assert {e.type for e in mine} == {"thought", "tool_call", "tool_result"}
    assert all("day" not in e.data for e in mine)
    synthesize = [e.data for e in events if e.data.get("name") == "synthesize"]
    assert [(s["status"], s.get("day")) for s in synthesize] == [("start", None), ("done", None)]
    assert synthesize[1]["label"].endswith("Храмы и музеи в первый день, кофейни во второй.")
    assert synthesize[1]["detail"]["places"] == [3, 3]
    started = [e.data["name"] for e in events if e.type == "stage" and e.data["status"] == "start"]
    research = max(i for i, name in enumerate(started) if name == "research")
    assert research < started.index("synthesize") < started.index("optimize")
    assert not [e for e in events if e.type == "warning"]


async def test_trip_feedback_can_move_a_stop_previously_assigned_by_the_planner(
    sources, trip_planner
):
    trip_planner(True)
    llm = _two_days_llm(PLANNED)
    journey = await _plan_two_days(llm, Recorder())
    before = _stops(journey)
    research_calls, planner_calls = len(llm.calls("chat")), len(llm.calls("planner"))
    # The planner borrowed place 5 from the second day's research. Feedback must be
    # able to move it back using the stored shortlist, without rerunning the planner.
    llm.structured_replies["JourneyEdit"] = [
        {
            "reply": "Меняю кофейни местами.",
            "moves": [
                {"source_day": 0, "target_day": 1, "place_id": "osm:node/5"},
                {"source_day": 1, "target_day": 0, "place_id": "osm:node/4"},
            ],
        }
    ]
    events = Recorder()

    updated, _, changed = await pipeline.apply_journey_feedback(
        journey,
        pipeline.JourneyFeedbackRequest(feedback="Поменяй Место 5 и Место 4 местами"),
        llm,
        events,
    )

    assert changed == [0, 1]
    assert [set(day) for day in _stops(updated)] == [set(_ids(1, 2, 4)), set(_ids(3, 5, 6))]
    assert _stops(journey) == before
    assert all(not [v for v in d.plan.violations if v.severity == "error"] for d in updated.days)
    assert len(llm.calls("chat")) == research_calls
    assert len(llm.calls("planner")) == planner_calls
    assert not [e for e in events if e.data.get("name") == "synthesize"]


@pytest.mark.parametrize(
    "planner",
    [
        [LLMError("rate limited", 429)],
        [RuntimeError("bug")],
        [turn(call("assign_day", day=1, place_ids=_ids(1, 2, 5)))],
    ],
    ids=["provider error", "unexpected exception", "out of steps"],
)
async def test_when_the_trip_planner_does_not_finish_code_assembles_the_days(
    sources, trip_planner, monkeypatch, planner
):
    by_code = _stops(await _plan_two_days(_two_days_llm([]), Recorder()))
    trip_planner(True)
    monkeypatch.setattr(pipeline, "PLANNER_STEPS", 1)
    events = Recorder()

    journey = await _plan_two_days(_two_days_llm(planner), events)

    assert _stops(journey) == by_code
    assert [e.data for e in events if e.type == "warning"] == [
        {"message": pipeline.PLANNER_FALLBACK}
    ]
    done = [e.data for e in events if e.data.get("name") == "synthesize"][-1]
    assert done["status"] == "done" and done["label"].startswith("Планировщик остановился")


async def test_with_the_trip_planner_off_code_assembles_the_days(sources):
    llm = _two_days_llm([])
    events = Recorder()

    journey = await _plan_two_days(llm, events)

    assert not llm.calls("planner")
    assert not [e for e in events if e.data.get("name") == "synthesize"]
    # The first day comes first: it keeps place 3, which both days found, and takes the
    # second day's place 5 at a discount.
    assert {"osm:node/3", "osm:node/5"} <= set(_stops(journey)[0])
    assert journey.days[0].state.shortlist["osm:node/5"].score == 0.48


async def test_a_planners_plan_that_does_not_build_is_undone_and_code_assembles_the_days(
    sources, trip_planner, monkeypatch
):
    trip_planner(True)
    build_checked = pipeline._build_checked

    async def second_day_fails(day, llm, emit, facts, borrowed=None, allowed=None):
        if allowed is not None and day.state.brief.date == RAINY:
            raise pipeline.PipelineError("Не удалось собрать маршрут")
        return await build_checked(day, llm, emit, facts, borrowed, allowed)

    monkeypatch.setattr(pipeline, "_build_checked", second_day_fails)
    events = Recorder()

    journey = await _plan_two_days(_two_days_llm(PLANNED), events)

    # The first day was built from the plan and place 5 moved to its shortlist at full
    # score; undone, code assembles it again and takes place 5 at a discount.
    assert {"osm:node/3", "osm:node/5"} <= set(_stops(journey)[0])
    assert journey.days[0].state.shortlist["osm:node/5"].score == 0.48
    assert [e.data for e in events if e.type == "warning"] == [
        {"message": pipeline.PLANNER_FALLBACK}
    ]


async def test_a_place_the_traveller_named_stays_on_its_day_when_the_planner_leaves_it_out(
    sources, trip_planner
):
    trip_planner(True)
    llm = TripLLM(
        turns={DAY: _day_turns([1, 2, 3, 4], "история"), RAINY: _day_turns([3, 5, 6, 7], "кофе")},
        structured={
            "BriefDraft": [{**BRIEF_REPLY, "days": 2, "must_include": ["Место 3"]}],
            "OutlineDraft": [
                {
                    "title": "Два дня",
                    "days": [
                        {"theme": "История", "focus": ["история"], "must_include": []},
                        {"theme": "Кофе", "focus": ["кофе"], "must_include": ["Место 3"]},
                    ],
                }
            ],
        },
        planner=[
            turn(
                call("assign_day", day=1, place_ids=_ids(1, 2, 4)),
                call("assign_day", day=2, place_ids=_ids(5, 6, 7)),  # without «Место 3»
                call("check_day", day=1),
                call("check_day", day=2),
            ),
            turn(call("finish_plan", summary="Два дня.")),
        ],
    )
    events = Recorder()
    req = pipeline.PlanRequest(request="Казань, два дня, обязательно Место 3")

    journey = await pipeline.plan_journey(req, llm, events, DAY)

    # The planner's plan was built (no fallback), with the named place on its day.
    assert not [e for e in events if e.type == "warning"]
    first, second = _stops(journey)
    assert set(first) <= set(_ids(1, 2, 4)) and "osm:node/3" in second
    assert not [v for v in journey.days[1].plan.violations if v.code == "missing_must_include"]


async def test_with_fewer_places_than_days_the_planner_is_not_started(sources, trip_planner):
    trip_planner(True)
    llm = _two_days_llm([])
    # Both days found only place 1: no allocation can give each day a place. The repair
    # rounds of the days code builds find nothing more.
    give_up = [turn(call("finish_research", summary="больше ничего"))] * 6
    llm.turns = {DAY: [*_day_turns([1], "история"), *give_up], RAINY: _day_turns([1], "кофе")}
    llm.turns[RAINY] += give_up
    events = Recorder()

    with pytest.raises(pipeline.PipelineError, match="День 2"):
        await _plan_two_days(llm, events)

    assert not llm.calls("planner")
    assert not [e for e in events if e.type == "warning"]


async def test_a_place_the_planner_moves_replaces_the_days_own_record_of_it(sources):
    museum = Place(
        id="osm:node/90",
        name="Музей Казани",
        lat=CENTER.lat + 0.002,
        lon=CENTER.lon,
        kind="museum",
        source="osm",
        opening_hours="Mo-Su 09:00-20:00",
        indoor=True,
        tags={"wikidata": "Q90"},
    )
    # Its Wikipedia article, on the second day's shortlist: no hours, no indoor flag.
    article = museum.model_copy(
        update={"id": "wikipedia:ru:90", "source": "wikipedia", "kind": "landmark"}
        | {"opening_hours": None, "indoor": None}
    )

    def cand(p: Place) -> Candidate:
        return Candidate(place=p, score=0.8, reason="ок", visit_minutes=30)

    other = [P[pid] for pid in _ids(2, 3, 4, 5)]
    registry = {p.id: p for p in [museum, article, *other]}
    days = [
        pipeline.DayRun(
            id=str(i),
            state=ResearchState(
                brief=_brief(date=DAY, date_to=None, must_include=[]),
                start=CENTER,
                weather_summary="",
                registry=registry,
                shortlist={p.id: cand(p) for p in places},
            ),
            weather=[],
        )
        for i, places in enumerate([[museum, *other[:2]], [article, *other[2:]]])
    ]
    # The planner gave the museum's fullest record, the first day's, to the second day.
    selections = [{p.id: cand(p) for p in other[:2]}, {p.id: cand(p) for p in [museum, *other[2:]]}]

    plans = await pipeline._assemble(days, None, Recorder(), [], selections)

    # Built from the OSM record and kept by the day in its place of the article: a later
    # rebuild of the day (feedback) still knows the museum's hours.
    assert {s.place.id for s in plans[1].stops} == {museum.id, *_ids(4, 5)}
    assert set(days[1].state.shortlist) == {museum.id, *_ids(4, 5)}


async def test_a_bug_in_pooling_the_finds_costs_the_trip_nothing(
    sources, trip_planner, monkeypatch
):
    by_code = _stops(await _plan_two_days(_two_days_llm([]), Recorder()))
    trip_planner(True)

    def broken(*args: Any) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline, "pool_trip", broken)
    events = Recorder()

    journey = await _plan_two_days(_two_days_llm([]), events)

    assert _stops(journey) == by_code
    assert [e.data for e in events if e.type == "warning"] == [
        {"message": pipeline.PLANNER_FALLBACK}
    ]
