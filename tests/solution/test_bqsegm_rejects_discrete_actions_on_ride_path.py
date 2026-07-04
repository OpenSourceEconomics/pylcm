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
from tests.test_models import bqsegm_ride_discrete_toy as ride_toy


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


def test_discrete_action_feeding_a_co_state_law_is_rejected_at_build():
    """A discrete action that shifts a co-state's law of motion fails build.

    The discrete envelope shares one next-period continuation across branches, so
    an action feeding a non-liquid state's transition — where the branches would
    evolve to different co-states and read different continuations — is refused.
    """
    with pytest.raises(RegimeInitializationError, match=r"streak.*|continuation"):
        ride_toy.build_model(
            variant="bqsegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            action_in_costate=True,
        )
