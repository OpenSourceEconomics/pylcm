"""BQSEGM's ride-along discrete envelope rejects the cases it cannot solve.

The ride-along path envelopes a single discrete action over a cliffed budget: each
branch solves the continuous subproblem with the action bound into cash-on-hand, and
the discrete choice is taken by the upper envelope over the branch values (including,
under a jump, each branch's published one-sided cliff limits). An action that feeds a
non-liquid co-state's law of motion is supported — the continuation carries a leading
branch axis so each branch reads its own next-co-state (see
`test_bqsegm_action_in_costate_agreement`). Two channels stay refused at model build:
the regime transition, and the liquid law off the budget channel (an out-of-pocket
cost landing directly on next assets).
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import bqsegm_ride_discrete_toy as ride_toy


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
