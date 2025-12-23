"""Tests for the Time dataclass."""

from lcm.typing import Time


def test_time_last_period_property():
    """Test that last_period is computed correctly as n_periods - 1."""
    time = Time(period=5, n_periods=10)
    assert time.period == 5
    assert time.n_periods == 10
    assert time.last_period == 9
