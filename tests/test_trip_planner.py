"""The trip planner on its own (app/agent/planner.py): the pool it sees, the tools that
ground and arbitrate its allocation, when it may finish and how it stops. The LLM is the
scripted FakeLLM, and the pipeline's optimizer, router and validator behind check_day are
a fake: the planner with the real ones is tested in tests/test_journey.py."""

import asyncio
import json
from datetime import date, datetime, time, timedelta

import pytest

from app.agent import planner
from app.agent.planner import TripDay, TripPlan, plan_days, pool_trip
from app.agent.prompts import PLANNER_NUDGE
from app.agent.researcher import REPEAT_ERROR
from app.agent.tools import min_candidates
from app.llm import LLMError
from app.models import (
    Candidate,
    DayOutline,
    GeoPoint,
    HourWeather,
    Interest,
    Place,
    Plan,
    Stop,
    TripBrief,
    Violation,
)
from tests.test_agent import FakeLLM, Recorder, call, turn

DAY = date(2026, 10, 24)
START = GeoPoint(lat=55.7963, lon=49.1088, label="Казанский кремль")
FAR = GeoPoint(lat=55.7423, lon=49.1088, label="Старо-Татарская слобода")  # 6 km south


def place(n: int, kind: str = "artwork", **changes: object) -> Place:
    fields = {
        "id": f"osm:node/{n}",
        "name": f"Место {n}",
        "lat": START.lat + 0.001 * n,
        "lon": START.lon,
        "kind": kind,
        "source": "osm",
    }
    return Place(**{**fields, **changes})


def candidate(p: Place, score: float = 0.8, interests: tuple[str, ...] = ("история",)) -> Candidate:
    return Candidate(
        place=p, score=score, interests=list(interests), reason="по теме", visit_minutes=40
    )


def trip_brief(**changes: object) -> TripBrief:
    fields = {
        "city": "Казань",
        "date": DAY,
        "start_time": time(10),
        "end_time": time(13),
        "interests": [Interest(label="история", weight=1.0), Interest(label="кофе", weight=0.7)],
        "avoid": ["музеи"],
    }
    return TripBrief(**{**fields, **changes})


def trip_day(
    index: int,
    *shortlist: Candidate,
    named: tuple[str, ...] = (),
    start: GeoPoint = START,
    rain_at: int | None = None,
) -> TripDay:
    day = DAY + timedelta(days=index)
    weather = [
        HourWeather(
            time=datetime.combine(day, time(hour)),
            precipitation_probability=90 if hour == rain_at else 0,
        )
        for hour in range(9, 14)
    ]
    return TripDay(
        outline=DayOutline(index=index, date=day, theme=f"Тема {index + 1}", focus=["история"]),
        brief=trip_brief(date=day, must_include=list(named)),
        start=start,
        weather=weather,
        shortlist={c.place.id: c for c in shortlist},
    )


class FakeCheck:
    """Stands in for the pipeline's check: the first three places become stops, an hour
    each, the rest are dropped, as is every place in `closed`; a day in `failing` gets an
    error. It yields to the event loop when `yields`, as a check that routes over the
    network does (a cached route does not), and counts the checks that overlap."""

    def __init__(
        self,
        days: list[TripDay],
        failing: set[int] = frozenset(),
        closed: set[str] = frozenset(),
        yields: bool = True,
    ):
        self.days = days
        self.failing = set(failing)
        self.closed = set(closed)
        self.yields = yields
        self.calls: list[tuple[int, list[str], set[str]]] = []
        self.running = self.overlap = 0

    async def __call__(self, index: int, selection: dict[str, Candidate], others: set[str]) -> Plan:
        self.calls.append((index, list(selection), others))
        self.running += 1
        self.overlap = max(self.overlap, self.running)
        if self.yields:
            await asyncio.sleep(0)
        self.running -= 1
        brief = self.days[index].brief
        chosen = [c for pid, c in selection.items() if pid not in self.closed]
        stops = [
            Stop(
                place=c.place,
                arrive=datetime.combine(brief.date, time(10 + k)),
                depart=datetime.combine(brief.date, time(10 + k, 40)),
                score=c.score,
                reason=c.reason,
            )
            for k, c in enumerate(chosen[:3])
        ]
        violations = (
            [Violation(code="over_time", severity="error", message="Маршрут заканчивается поздно")]
            if index in self.failing
            else []
        )
        return Plan(
            brief=brief,
            start=START,
            stops=stops,
            total_walk_m=2400,
            total_minutes=60 * len(stops),
            dropped=[pid for pid in selection if pid in self.closed]
            + [c.place.id for c in chosen[3:]],
            violations=violations,
        )


def two_days() -> list[TripDay]:
    """Places 1-4 found by day 1, 3-6 by day 2 (3 and 4 by both); day 2 is named «Место 7»,
    which day 1 found, and has rain at 11:00."""
    return [
        trip_day(0, *(candidate(place(n)) for n in (1, 2, 3, 4, 7))),
        trip_day(
            1,
            *(candidate(place(n), interests=("кофе",)) for n in (3, 4, 5, 6)),
            named=("Место 7",),
            start=FAR,
            rain_at=11,
        ),
    ]


def ids(*numbers: int) -> list[str]:
    return [f"osm:node/{n}" for n in numbers]


def errors(events: Recorder) -> list[str | None]:
    return [e.data["error"] for e in events if e.type == "tool_result"]


async def run(trip: TripPlan, turns: list, check: FakeCheck | None = None, **budget):
    llm = FakeLLM(turns=turns)
    events = Recorder()
    budget = {"max_steps": 12, "time_budget_s": 30, **budget}
    result = await plan_days(llm, trip, check or FakeCheck(trip.days), events, **budget)
    return result, llm, events


# --- the pool ------------------------------------------------------------------------


def test_the_pool_has_one_entry_per_real_place_without_what_the_traveller_avoids():
    cathedral = place(1, "church", tags={"wikidata": "Q1"})
    # Wikipedia's record of the same cathedral, which the second day found and liked more.
    article = cathedral.model_copy(update={"id": "wikipedia:ru:1", "source": "wikipedia"})
    days = [
        trip_day(0, candidate(cathedral, 0.6), candidate(place(2)), candidate(place(3, "museum"))),
        trip_day(1, candidate(article, 0.9, ("архитектура",)), candidate(place(4))),
    ]

    trip = pool_trip(trip_brief(), days)

    # Museums are avoided; the cathedral is one entry found by both days, with the better
    # score and both days' interests.
    assert set(trip.pool) == set(ids(1, 2, 4))
    entry = trip.pool["osm:node/1"]
    assert (entry.found_by, entry.candidate.score) == ([0, 1], 0.9)
    assert entry.candidate.interests == ["история", "архитектура"]
    assert trip.left_out == 0


def test_of_two_records_of_a_place_the_pool_keeps_the_one_with_its_opening_hours():
    museum = place(
        1, "museum", opening_hours="Tu-Su 10:00-18:00", indoor=True, tags={"wikidata": "Q1"}
    )
    # Its Wikipedia article, which the first day found and scored higher: no hours, no
    # indoor flag, and a kind whose hours the validator does not check.
    article = museum.model_copy(
        update={
            "id": "wikipedia:ru:1",
            "source": "wikipedia",
            "kind": "landmark",
            "opening_hours": None,
            "indoor": None,
            "popularity": 0.9,
        }
    )
    days = [trip_day(0, candidate(article, 0.9)), trip_day(1, candidate(museum, 0.7))]

    trip = pool_trip(trip_brief(avoid=[]), days)

    [entry] = trip.pool.values()
    kept = entry.candidate.place
    assert (kept.id, kept.kind, kept.opening_hours, kept.indoor) == (
        "osm:node/1",
        "museum",
        "Tu-Su 10:00-18:00",
        True,
    )
    assert kept.popularity == 0.9  # the article's, which the OSM record lacked
    assert (entry.candidate.score, entry.found_by) == (0.9, [0, 1])


def test_the_pool_shows_the_named_places_each_days_own_best_and_then_the_best(monkeypatch):
    monkeypatch.setattr(planner, "MAX_POOL", 7)
    days = [
        trip_day(0, *(candidate(place(n), score=0.9 - n / 100) for n in range(1, 9))),
        trip_day(1, candidate(place(9), 0.5), candidate(place(31), 0.5), named=("Место 9",)),
    ]

    trip = pool_trip(trip_brief(), days)

    # The named place, scored lowest, comes first; then each day's own finds up to its
    # target (4 for 10:00-13:00), so the second day's 31 is shown though the first day's
    # 6-8 scored higher; then the best of the rest up to MAX_POOL.
    assert list(trip.pool) == ids(9, 1, 2, 3, 4, 5, 31)
    assert trip.pool["osm:node/9"].named_for == 1
    assert trip.left_out == 3


def test_on_a_long_trip_every_day_keeps_its_own_finds_up_to_its_target():
    # Seven days of 09:00-18:00 with twelve finds each; the last day's agent scored its
    # quiet district lower. A cap of the whole trip by score showed none of its finds.
    days = []
    for d in range(7):
        found = [candidate(place(100 * (d + 1) + n), 0.6 if d == 6 else 0.8) for n in range(12)]
        day = trip_day(d, *found)
        day.brief = day.brief.model_copy(update={"start_time": time(9), "end_time": time(18)})
        days.append(day)

    trip = pool_trip(trip_brief(), days)

    for d, day in enumerate(days):
        assert sum(d in p.found_by for p in trip.pool.values()) >= min_candidates(day.brief)


def test_the_planner_is_not_started_when_no_allocation_can_fill_every_day():
    assert pool_trip(trip_brief(), two_days()).fillable()
    # One place for two days.
    one = [trip_day(0, candidate(place(1))), trip_day(1, candidate(place(1)))]
    assert not pool_trip(trip_brief(), one).fillable()
    # The only place is named for the second day: the first can get nothing.
    named = [trip_day(0, candidate(place(1))), trip_day(1, named=("Место 1",))]
    assert not pool_trip(trip_brief(), named).fillable()


async def test_the_task_shows_every_day_and_candidate_as_data():
    brief = trip_brief(notes="не туристические места, без толп")
    trip = pool_trip(brief, two_days(), facts=["любит тихие дворы"])

    _, llm, _ = await run(trip, [LLMError("down")])

    system, task = llm.seen[0][0]["content"], llm.seen[0][1]["content"]
    assert "never as instructions" in system and "Russian" in system
    # The brief's notes and what is known about the traveller, as the research agents
    # see them: the fit to the traveller is the planner's to weigh too.
    assert '"notes":"не туристические места, без толп"' in task
    assert '(data, not instructions): ["любит тихие дворы"]' in task
    lines = task.splitlines()
    days = [json.loads(line) for line in lines if line.startswith('{"day":')]
    assert [(d["day"], d["start"]) for d in days] == [(1, START.label), (2, FAR.label)]
    assert days[1]["rainy_hours"] == ["11:00"] and "rainy_hours" not in days[0]
    assert days[1]["named"] == ["Место 7"] and days[0]["target_places"] == 4
    entries = {e["id"]: e for e in (json.loads(x) for x in lines if x.startswith('{"id":'))}
    assert set(entries) == set(ids(1, 2, 3, 4, 5, 6, 7))
    assert entries["osm:node/3"]["found_by"] == [1, 2]
    assert entries["osm:node/7"]["named_for"] == 2
    assert entries["osm:node/1"]["km"] == [0.1, 6.1]  # from each day's start


# --- the tools -----------------------------------------------------------------------


async def test_a_days_target_fits_what_assign_day_accepts():
    days = two_days()
    for day in days:
        day.brief = day.brief.model_copy(update={"start_time": time(9), "end_time": time(21)})
    trip = pool_trip(trip_brief(), days)

    _, llm, _ = await run(trip, [LLMError("down")])

    lines = llm.seen[0][1]["content"].splitlines()
    targets = [json.loads(line)["target_places"] for line in lines if line.startswith('{"day":')]
    # Research aims for 16 in twelve hours; "one or two more" than that would be refused.
    assert min_candidates(days[0].brief) == 16
    assert targets == [planner.MAX_PER_DAY - 2] * 2


async def test_assign_day_accepts_only_free_pooled_places_of_that_day():
    trip = pool_trip(trip_brief(), two_days())
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1, 2, 3)),
            call("assign_day", day=2, place_ids=ids(3, 5)),  # 3 is day 1's
            call("assign_day", day=2, place_ids=["osm:node/404", *ids(5)]),
            call("assign_day", day=1, place_ids=ids(1, 7)),  # named for day 2
            call("assign_day", day=3, place_ids=ids(5)),
            call("check_day", day=2),
        ),
        LLMError("down"),
    ]

    result, _, events = await run(trip, turns)

    assert result.selections is None and result.stop_reason == "llm_error"
    first, *refused, unchecked = errors(events)
    assert first is None
    assert "'osm:node/3' (Место 3) is already on day 1" in refused[0]
    assert "unknown id 'osm:node/404'" in refused[1]
    assert "named 'osm:node/7' (Место 7) for day 2" in refused[2]
    assert "there is no day 3" in refused[3]
    assert "day 2 has no places" in unchecked
    # A refused call assigns nothing, not even its valid ids.
    assert trip.assignment == {0: ids(1, 2, 3)}


async def test_assign_day_adds_the_place_the_traveller_named_for_the_day():
    trip = pool_trip(trip_brief(), two_days())

    _, llm, _ = await run(
        trip, [turn(call("assign_day", day=2, place_ids=ids(5, 6))), LLMError("")]
    )

    # Left out by the model, «Место 7» is on its day all the same, and the model is told.
    assert trip.assignment == {1: ids(5, 6, 7)}
    assert json.loads(llm.seen[1][3]["content"])["added_named"] == ids(7)


async def test_finish_is_refused_until_every_day_is_checked_after_its_last_change():
    days = two_days()
    trip = pool_trip(trip_brief(), days)
    check = FakeCheck(days)
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1, 2, 3, 4)),
            call("assign_day", day=2, place_ids=ids(5)),
            call("check_day", day=1),
            call("finish_plan", summary="рано"),  # runs after the checks: day 2 is unchecked
        ),
        turn(
            call("check_day", day=2),
            call("assign_day", day=2, place_ids=ids(5, 6, 7)),  # after its check
            call("finish_plan", summary="рано"),
        ),
        turn(call("check_day", day=2), call("finish_plan", summary="Два дня: центр и слобода.")),
    ]

    result, llm, events = await run(trip, turns, check)

    assert result.stop_reason == "finished" and result.steps == 3
    assert result.summary == "Два дня: центр и слобода."
    refusals = [e for e in errors(events) if e]
    assert refusals == [
        "not finished: day 2 was never checked",
        "not finished: day 2 changed after its last check_day",
    ]
    # Each check sees its day's places, with the other days' taken.
    assert check.calls[0] == (0, ids(1, 2, 3, 4), set(ids(5, 7)))
    assert check.calls[-1] == (1, ids(5, 6, 7), set(ids(1, 2, 3, 4)))
    # The check's report (after the first turn's two assign_day results): the stops with
    # their times, a dropped place and why, the verdict.
    report = json.loads(llm.seen[1][5]["content"])
    assert [s["id"] for s in report["stops"]] == ids(1, 2, 3)
    assert report["dropped"][0]["id"] == "osm:node/4"
    assert report["dropped"][0]["why"].startswith("no room")
    assert report["verdict"] == "no errors"
    first, second = result.selections
    assert list(first) == ids(1, 2, 3, 4) and list(second) == ids(5, 6, 7)
    # «Место 7» was found by day 1 and is required on day 2, which the traveller named it for.
    assert second["osm:node/7"].required and not first["osm:node/1"].required


async def test_finish_with_errors_is_accepted_on_the_third_attempt_but_never_with_an_empty_day():
    days = two_days()
    trip = pool_trip(trip_brief(), days)
    finish = call("finish_plan", summary="как есть")
    turns = [
        turn(call("assign_day", day=1, place_ids=ids(1, 2)), finish),
        turn(finish),
        turn(finish),  # the third attempt, but day 2 has no places
        turn(call("assign_day", day=2, place_ids=ids(5, 6)), call("check_day", day=2), finish),
    ]

    result, _, events = await run(trip, turns, FakeCheck(days, failing={1}))

    assert result.stop_reason == "finished"
    assert [bool(e) for e in errors(events)] == [False, True, True, True, False, False, False]
    summary = [e.data["summary"] for e in events if e.type == "tool_result"][-1]
    assert "принято с 3-й попытки" in summary and "day 1 was never checked" in summary


async def test_a_day_that_ends_early_beside_unused_places_is_sent_back():
    trip = pool_trip(trip_brief(), two_days())
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1)),
            call("assign_day", day=2, place_ids=ids(5, 6, 7)),
            call("check_day", day=1),
            call("check_day", day=2),
            call("finish_plan", summary="готово"),
        ),
        LLMError("down"),
    ]

    _, llm, events = await run(trip, turns)

    # One stop, 10:00-10:40, in a window to 13:00: no validator error, but the check
    # sends the day back with the unused places near its start (5 and 6 are day 2's, 7
    # is named for it).
    report = json.loads(llm.seen[1][5]["content"])
    assert report["verdict"] == "fix: it ends 140 minutes before its window, add places"
    assert [hint["id"] for hint in report["unused_nearby"]] == ids(2, 3, 4)
    assert errors(events)[-1] == (
        "not finished: day 1: it ends 140 minutes before its window, add places"
    )


async def test_a_day_another_day_took_too_much_from_is_sent_back_for_its_leftovers():
    days = [
        trip_day(0, *(candidate(place(n)) for n in range(1, 7))),
        trip_day(1, candidate(place(7))),
    ]
    trip = pool_trip(trip_brief(), days)
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1, 2, 3, 4, 5, 6)),
            call("assign_day", day=2, place_ids=ids(7)),
            call("check_day", day=1),
            call("check_day", day=2),
        ),
        LLMError("down"),
    ]

    _, llm, _ = await run(trip, turns)

    # Every place is assigned, so nothing is unassigned near the second day's start; but
    # the first day's check dropped three for lack of room, and the second day may have them.
    first, second = (json.loads(m["content"]) for m in llm.seen[1][5:7])
    assert [d["id"] for d in first["dropped"]] == ids(4, 5, 6) and first["verdict"] == "no errors"
    assert second["verdict"] == "fix: it ends 140 minutes before its window, add places"
    assert [(h["id"], h["on_day"]) for h in second["unused_nearby"]] == [
        (pid, 1) for pid in ids(4, 5, 6)
    ]


async def test_a_place_closed_on_the_date_does_not_make_a_short_day_full():
    gallery = place(2, "gallery", opening_hours="Mo 10:00-18:00")  # the trip starts on a Saturday
    days = [trip_day(0, candidate(place(1)), candidate(gallery), candidate(place(3))), trip_day(1)]
    trip = pool_trip(trip_brief(avoid=[]), days)
    turns = [
        turn(call("assign_day", day=1, place_ids=ids(1, 2)), call("check_day", day=1)),
        LLMError("down"),
    ]

    _, llm, _ = await run(trip, turns, FakeCheck(days, closed={gallery.id}))

    report = json.loads(llm.seen[1][4]["content"])
    assert report["dropped"] == [
        {"id": gallery.id, "name": "Место 2", "why": "closed on this date"}
    ]
    assert report["verdict"] == "fix: it ends 140 minutes before its window, add places"
    assert [hint["id"] for hint in report["unused_nearby"]] == ids(3)


async def test_hints_put_the_fit_to_the_traveller_before_fame():
    mural = candidate(place(2, popularity=0.1), 0.95, ("стрит-арт",))
    cathedral = candidate(place(3, "church", popularity=0.9), 0.3)
    days = [trip_day(0, candidate(place(1)), mural, cathedral), trip_day(1)]
    trip = pool_trip(trip_brief(), days)
    turns = [
        turn(call("assign_day", day=1, place_ids=ids(1)), call("check_day", day=1)),
        LLMError("down"),
    ]

    _, llm, _ = await run(trip, turns)

    hints = json.loads(llm.seen[1][4]["content"])["unused_nearby"]
    assert [(h["id"], h["score"]) for h in hints] == [("osm:node/2", 0.95), ("osm:node/3", 0.3)]


@pytest.mark.parametrize("yields", [True, False], ids=["routed", "cached"])
async def test_the_calls_of_a_turn_run_one_after_another_in_call_order(yields):
    days = two_days()
    trip = pool_trip(trip_brief(), days)
    check = FakeCheck(days, yields=yields)
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1)),
            call("check_day", day=1),
            call("assign_day", day=2, place_ids=ids(5, 6)),
            call("check_day", day=2),
        ),
        LLMError("down"),
    ]

    _, llm, _ = await run(trip, turns, check)

    # Never two checks at once (each routes through OSRM's one lock), and the first check
    # sees the plan before the second assign_day whether or not its route was cached: the
    # same answers replay.
    assert check.overlap == 1
    hints = json.loads(llm.seen[1][4]["content"])["unused_nearby"]
    assert [hint["id"] for hint in hints] == ids(2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    ("turns", "reason"),
    [
        ([LLMError("rate limited", 429)], "llm_error"),
        ([turn(content="Думаю."), turn(content="Всё ещё думаю.")], "no_tool_calls"),
        ([turn(call("check_day", day=1)) for _ in range(3)], "repeated_action"),
        ([turn(call("assign_day", day=1, place_ids=ids(n))) for n in (1, 2)], "max_steps"),
    ],
)
async def test_the_planner_returns_nothing_unless_it_finished_or_its_plan_is_agreed(turns, reason):
    days = two_days()
    trip = pool_trip(trip_brief(), days)
    trip.assignment[0] = ids(1, 2)

    result, llm, events = await run(trip, turns, max_steps=2 if reason == "max_steps" else 12)

    assert (result.selections, result.stop_reason) == (None, reason)
    if reason == "no_tool_calls":
        assert llm.seen[1][-1] == {"role": "user", "content": PLANNER_NUDGE}
    if reason == "repeated_action":
        assert errors(events) == [None, REPEAT_ERROR]  # the third is not run at all
    # The trip's own agent: its events carry no day, and say whose they are.
    assert all("day" not in e.data and e.data["agent"] == "planner" for e in events)


async def test_a_plan_checked_clean_is_kept_when_the_run_stops_before_finish_plan():
    trip = pool_trip(trip_brief(), two_days())
    turns = [
        turn(
            call("assign_day", day=1, place_ids=ids(1, 2)),
            call("assign_day", day=2, place_ids=ids(5, 6)),
            call("check_day", day=1),
            call("check_day", day=2),
        )
    ]

    result, _, _ = await run(trip, turns, max_steps=1)

    assert result.stop_reason == "max_steps"
    assert [list(selection) for selection in result.selections] == [ids(1, 2), ids(5, 6, 7)]


async def test_a_repeated_check_does_not_cut_off_the_finish_that_would_be_accepted():
    days = two_days()
    trip = pool_trip(trip_brief(), days)
    again = turn(
        call("check_day", day=1), call("check_day", day=2), call("finish_plan", summary="как есть")
    )
    first = turn(
        call("assign_day", day=1, place_ids=ids(1, 2, 3)),
        call("assign_day", day=2, place_ids=ids(5, 6)),
        call("check_day", day=1),
        call("check_day", day=2),
    )

    # Day 1 has an error the model cannot fix; it checks again and tries to finish.
    result, _, events = await run(trip, [first, again, again, again], FakeCheck(days, failing={0}))

    # The fourth turn's checks are third repeats, yet its finish runs: the third attempt.
    assert (result.stop_reason, result.steps) == ("finished", 4)
    summary = [e.data["summary"] for e in events if e.type == "tool_result"][-1]
    assert "принято с 3-й попытки" in summary
