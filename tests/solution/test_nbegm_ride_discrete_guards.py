"""Build-time guards on NBEGM's ride-along discrete envelope.

The envelope solves the continuous subproblem per discrete branch and combines
branches by the discrete upper envelope, which supports one action feeding the
budget, a co-state law, the off-budget liquid law, period utility, the regime
transition, and the schedule variable (per-branch breakpoints). The cases outside
that contract are refused at model build:

- an action entering the discount factor (evaluated per cell, not per branch),
- an action entering a *jumped* schedule variable under the one-sided cliff read
  (branches would not share the published parent query grid).
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import nbegm_ride_discrete_toy as ride_toy


def test_action_in_discount_factor_is_rejected_at_build() -> None:
    """A discount factor reading the discrete action fails model build."""
    with pytest.raises(RegimeInitializationError, match=r"discount factor"):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            action_in_discount=True,
        )


def test_action_in_jumped_schedule_variable_under_one_sided_is_rejected() -> None:
    """An action entering a jumped schedule variable is refused under `one_sided`."""
    with pytest.raises(RegimeInitializationError, match=r"one_sided|jump"):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            action_in_schedule_variable=True,
            jump_schedule=True,
        )
