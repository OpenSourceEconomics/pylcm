"""DC-EGM splaying: `batch_size` on the exogenous savings grid is a memory knob only.

The dominant egm_step working buffer is the per-savings-node continuation
computation — the savings nodes times the child stochastic mesh times the combo
block. Splaying the savings grid processes those nodes in `lax.map` blocks rather
than one fused vmap, shedding peak working-set memory. The upper envelope still
runs on the gathered full endogenous grid, so the solved value function is
identical to the unsplayed (`batch_size=0`) solve, whatever the block size —
including block sizes that do not divide the grid.
"""

import functools

import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import LinSpacedGrid, MarkovTransition, Model
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from tests.solution.test_egm_batch_size_euler import (
    CONSUMPTION_GRID,
    N_WEALTH,
    RegimeId,
    _ages,
    _params,
    bequest,
    death_prob,
    inverse_marginal_utility,
    next_wealth,
    resources,
    savings,
    stay_prob,
    utility,
)

N_SAVINGS = 149  # prime: no batch size below divides it evenly


@functools.cache
def _model(savings_batch_size: int) -> Model:
    """Asset-row DC-EGM model with `batch_size` splaying on the savings grid."""
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = UserRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH)},
        state_transitions={"wealth": next_wealth},
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=LinSpacedGrid(
                start=0.0,
                stop=110.0,
                n_points=N_SAVINGS,
                batch_size=savings_batch_size,
            ),
            n_constrained_points=32,
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=200)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def _solve(savings_batch_size: int) -> PeriodToRegimeToVArr:
    return _model(savings_batch_size).solve(params=_params(), log_level="debug")


@pytest.mark.parametrize("savings_batch_size", [1, 7, 16, N_SAVINGS])
def test_savings_grid_batch_size_leaves_value_function_unchanged(savings_batch_size):
    """Splaying the savings grid into blocks does not change the solved V.

    `batch_size` on the savings grid only changes how the per-savings-node
    continuation is scheduled (blocks via `lax.map` instead of one fused vmap),
    so the value function at every period matches the unsplayed `batch_size=0`
    solve exactly — including block sizes that do not divide the grid, and the
    boundary size equal to the grid length (which falls back to the vmap).
    """
    reference = _solve(0)
    splayed = _solve(savings_batch_size)
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"period={period}, regime={regime_name}",
            )
