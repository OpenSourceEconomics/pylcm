"""BQSEGM's continuation is interval-constant when a co-state law reads liquid.

When a carried state's law of motion reads the current liquid (Euler) state, BQSEGM
binds the liquid state to each interval's node and reuses that continuation row
across the interval. That is exact only when the law's liquid dependence is
piecewise-constant — a level switched at a threshold, whose derivative between
breakpoints is zero. A smoothly varying dependence makes the midpoint-bound row
wrong for the interval's other liquid points, so it is refused at model build.
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import bqsegm_ride_discrete_toy as ride_toy


def test_costate_law_varying_smoothly_in_liquid_is_rejected_at_build() -> None:
    """A co-state whose law varies smoothly in the liquid state fails model build."""
    with pytest.raises(
        RegimeInitializationError, match=r"liquid|interval|continuation"
    ):
        ride_toy.build_model(
            variant="bqsegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            costate_reads_liquid=True,
            costate_smooth=True,
        )


def test_costate_law_piecewise_constant_in_liquid_builds() -> None:
    """A co-state whose law switches at a liquid threshold builds without error."""
    ride_toy.build_model(
        variant="bqsegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        costate_reads_liquid=True,
        costate_smooth=False,
    )
