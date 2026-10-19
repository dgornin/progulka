import pytest

from app import limits as limits_module
from app.limits import Limits


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
