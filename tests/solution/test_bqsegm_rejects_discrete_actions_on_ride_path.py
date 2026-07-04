"""BQSEGM's ride-along discrete envelope rejects the cases it cannot solve.

The ride-along path envelopes a single discrete action over a kink budget: each
branch solves the continuous subproblem with the action bound into cash-on-hand,
and the discrete choice is taken by the upper envelope. A schedule carrying a
jump breakpoint falls outside that contract — its published one-sided value
limits would have to be taken over branches (topology through the envelope) — so
a jump schedule with a discrete action is refused at model build.
"""

import pytest

from lcm import DiscreteGrid, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ScalarInt
from tests.test_models import bqsegm_jump_ride_along_toy as jump_toy


@categorical(ordered=False)
class WorkChoice:
    no: ScalarInt
    yes: ScalarInt


def test_ride_along_regime_with_a_jump_and_discrete_action_is_rejected_at_build():
    """A ride regime with a jump schedule plus a discrete action fails build."""
    with pytest.raises(RegimeInitializationError, match=r"jump.*discrete action"):
        jump_toy.build_model(
            variant="bqsegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            extra_actions={"work_choice": DiscreteGrid(WorkChoice)},
        )
