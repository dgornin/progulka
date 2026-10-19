from datetime import date, time

from app.agent.replanner import PlanEdit
from app.models import TripBrief


def test_times_from_the_model_lose_their_timezone():
    # Some models answer "11:00+03:00"; aware and naive times cannot be compared.
    brief = TripBrief(
        city="Казань",
        date=date(2026, 10, 24),
        start_time="11:00:00+03:00",
        end_time="18:00:00Z",
    )
    edit = PlanEdit(rain_from="15:00+03:00", reply="ok")

    assert brief.start_time == time(11, 0) and brief.start_time.tzinfo is None
    assert brief.end_time.tzinfo is None
    assert edit.rain_from == time(15, 0) and edit.rain_from.tzinfo is None
