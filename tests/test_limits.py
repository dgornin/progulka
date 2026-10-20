import pytest

from app import limits as limits_module
from app.limits import Limits, SlidingWindow


def test_all_limits_off_by_default():
    limits = Limits(per_hour=0, daily_budget_usd=0, max_concurrent=0)
    assert all(limits.admit("1.2.3.4") is None for _ in range(100))


def test_rate_limit_is_per_client_and_slides(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(limits_module.time, "monotonic", lambda: clock[0])
    limits = Limits(per_hour=2, daily_budget_usd=0, max_concurrent=0)

    assert limits.admit("a") is None and limits.admit("a") is None
    assert "2 маршрутов в час" in limits.admit("a")
    assert limits.admit("b") is None  # another visitor is not affected
    clock[0] += 3601
    assert limits.admit("a") is None


def test_the_admins_runs_pass_every_limit_and_still_count():
    limits = Limits(per_hour=1, daily_budget_usd=0.02, max_concurrent=1)
    assert limits.admit("a") is None  # the address's one run takes the one slot
    assert "слишком много" in limits.admit("b")

    assert limits.admit("a", exempt=True) is None and limits.running == 2
    limits.release(cost_usd=0.02)  # the admin's run spends the budget
    assert limits.running == 1 and limits.admit("b") == limits_module.BUDGET_SPENT
    assert limits.admit("a", exempt=True) is None


def test_concurrency_slot_is_returned_on_release():
    limits = Limits(per_hour=0, daily_budget_usd=0, max_concurrent=1)

    assert limits.admit("a") is None
    assert "слишком много" in limits.admit("b")
    limits.release(cost_usd=0.01)
    assert limits.admit("b") is None


def test_daily_budget_counts_released_runs():
    limits = Limits(per_hour=0, daily_budget_usd=0.02, max_concurrent=0)

    assert limits.admit("a") is None
    limits.release(cost_usd=0.015)
    assert limits.admit("a") is None
    limits.release(cost_usd=0.01)
    assert "Дневной лимит" in limits.admit("a")


@pytest.mark.parametrize("cost", [0.0, 0.5])
def test_release_never_goes_negative(cost):
    limits = Limits(per_hour=0, daily_budget_usd=0, max_concurrent=1)
    limits.release(cost)
    assert limits.running == 0


def test_sliding_window_counts_per_key_and_forgets_old_events(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(limits_module.time, "monotonic", lambda: clock[0])
    window = SlidingWindow(limit=2, window_s=300)

    window.add("a")
    clock[0] += 100
    window.add("a")
    assert window.full("a") and not window.full("b")
    clock[0] += 201  # the first event left the window
    assert not window.full("a")


def test_sliding_window_takes_back_an_event():
    window = SlidingWindow(limit=2, window_s=300)
    window.add("a")
    window.add("a")

    window.undo("a")
    window.undo("b")  # nothing recorded: nothing to take back

    assert not window.full("a")
    window.add("a")
    assert window.full("a")


def test_sliding_window_off_at_zero():
    window = SlidingWindow(limit=0, window_s=300)
    for _ in range(100):
        window.add("a")
    assert not window.full("a")


def test_spend_before_a_restart_counts_against_the_budget():
    limits = Limits(per_hour=0, daily_budget_usd=0.02, max_concurrent=0)
    limits.spend(0.02)
    assert "Дневной лимит" in limits.admit("a")


def test_a_spent_budget_is_reported_without_admitting_a_run():
    limits = Limits(per_hour=0, daily_budget_usd=0.02, max_concurrent=1)
    assert not limits.out_of_budget()

    limits.spend(0.02)  # a transcription, which takes no run slot

    assert limits.out_of_budget() and limits.running == 0
    assert Limits(per_hour=0, daily_budget_usd=0, max_concurrent=0).out_of_budget() is False
