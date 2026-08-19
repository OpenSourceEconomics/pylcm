"""Declaring the identity no-adjustment map with `lcm.outer_unchanged`.

`OuterContinuousMargin.no_adjustment=lcm.outer_unchanged` declares that the outer
state is carried unchanged when the agent does not adjust. It is a declaration
rather than a function name, so a regime using it needs no identity function of
its own, and the model it produces solves to the value function of an otherwise
identical model that names one.
"""

import numpy as np
import pytest

from lcm import AgeGrid, Model, OuterContinuousMargin, outer_unchanged
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.test_models import negm_kinked_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}
_FINAL_AGE_ALIVE = 20 + (negm_kinked_toy.N_PERIODS - 2) * 5


def _model(*, no_adjustment: str) -> Model:
    """The kinked toy with its outer no-adjustment map declared as given."""
    alive = negm_kinked_toy.build_alive_regime()
    outer = alive.outer_continuous
    return Model(
        regimes={
            "alive": alive.replace(
                outer_continuous=OuterContinuousMargin(
                    state=outer.state,
                    action=outer.action,
                    post_decision_state=outer.post_decision_state,
                    no_adjustment=no_adjustment,
                )
            ),
            "dead": negm_kinked_toy.build_dead_regime(),
        },
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=AgeGrid(
            start=20, stop=20 + (negm_kinked_toy.N_PERIODS - 1) * 5, step="5Y"
        ),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_identity_sentinel_builds_a_model() -> None:
    """A regime declaring the sentinel constructs without an identity function."""
    assert _model(no_adjustment=outer_unchanged) is not None


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_identity_sentinel_solves_to_the_named_identity_function() -> None:
    """The sentinel and a regime function returning the outer state agree."""
    # `keep_illiquid` is `illiquid -> illiquid`, so the two declarations name the
    # same map and any difference in the solved value is a defect, not rounding.
    declared = _model(no_adjustment=outer_unchanged).solve(
        params=_PARAMS, log_level="debug"
    )
    named = _model(no_adjustment="keep_illiquid").solve(
        params=_PARAMS, log_level="debug"
    )

    for period in named:
        for regime in named[period]:
            np.testing.assert_array_almost_equal(
                np.asarray(declared[period][regime]),
                np.asarray(named[period][regime]),
                decimal=DECIMAL_PRECISION,
            )
