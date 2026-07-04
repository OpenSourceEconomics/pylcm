"""BQSEGM's ride-along discrete envelope rejects the cases it cannot solve.

The ride-along path envelopes a single discrete action over a cliffed budget: each
branch solves the continuous subproblem with the action bound into cash-on-hand,
and the discrete choice is taken by the upper envelope over the branch values
(including, under a jump, each branch's published one-sided cliff limits). That
composition is valid only when the action shifts the current budget alone. An
action that instead shifts the next-period continuation — by feeding a non-liquid
state's law of motion — is refused at model build.
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import bqsegm_ride_discrete_toy as ride_toy


def test_discrete_action_feeding_a_co_state_law_is_rejected_at_build() -> None:
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


def test_discrete_action_shifting_next_liquid_off_budget_is_rejected_at_build() -> None:
    """A discrete action that shifts next liquid off the budget channel fails build.

    Binding the action into cash-on-hand is the allowed budget channel — it reaches
    next liquid only through the post-decision savings. An action that instead feeds
    next liquid directly (an out-of-pocket cost that lands on next assets) gives each
    branch a different continuation, so the shared-continuation envelope refuses it.
    """
    with pytest.raises(RegimeInitializationError, match=r"liquid|continuation"):
        ride_toy.build_model(
            variant="bqsegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            action_in_liquid_law=True,
        )
