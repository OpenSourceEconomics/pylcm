"""BQSEGM's schedule+ride-along path rejects unsupported discrete actions.

The ride-along kernels solve one continuous consumption problem per ride cell;
they carry no loop or envelope over discrete actions. A regime that declares a
discrete action on this path would have that action silently ignored — the
solver would publish the value of one arbitrary branch instead of the max over
branches. The model build must refuse such a regime so the restriction is
explicit: fix the action in the regime's functions (and drop it from
`actions`), or use a solver that aggregates discrete branches.
"""

import pytest

from lcm import DiscreteGrid, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ScalarInt
from tests.test_models import bqsegm_jump_ride_along_toy as toy


@categorical(ordered=False)
class WorkChoice:
    no: ScalarInt
    yes: ScalarInt


def test_ride_along_regime_with_discrete_action_is_rejected_at_build():
    """A schedule+ride regime declaring a discrete action fails model build."""
    with pytest.raises(RegimeInitializationError, match="discrete action"):
        toy.build_model(
            variant="bqsegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            extra_actions={"work_choice": DiscreteGrid(WorkChoice)},
        )
