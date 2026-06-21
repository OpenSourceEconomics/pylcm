"""DC-EGM feasibility of the Dobrescu-Shanker housing keeper regime.

The keeper (no-house-trade) branch of the Dobrescu-Shanker housing model is a
plain 1-D DC-EGM problem: liquid assets is the Euler state, consumption the
continuous action, and housing a passive continuous state. Utility reads the
passive housing state directly through the `alpha * log(housing)` service flow.

The cornerstone here is that the DC-EGM envelope-condition guard rejects
utility reading the *Euler* state but allows utility reading a *passive*
state — so the keeper is expressible today. That guard is asserted against the
validator directly (no kernel trace, no solve). The full GPU solve is gated
off the local box, which OOMs on a DC-EGM solve.
"""

from types import MappingProxyType

import numpy as np
import pytest

from _lcm.egm.validation import validate_dcegm_regimes
from _lcm.regime_building.finalize import finalize_regimes
from tests.test_models.ds_housing_keeper import (
    HOUSING_GRID,
    LIQUID_ASSETS_GRID,
    build_model,
    build_params,
    build_working_regime,
    dead,
)


def _finalized_keeper_regimes() -> MappingProxyType:
    """Finalize the keeper regimes as the model build does, without solving.

    Finalization injects the default Bellman aggregator `H` and validates
    completeness — the exact form the DC-EGM validator runs against inside
    `Model` construction.
    """
    return finalize_regimes(
        user_regimes={"keeper": build_working_regime(), "dead": dead},
        derived_categoricals=MappingProxyType({}),
    )


def test_dcegm_accepts_utility_reading_passive_housing():
    """DC-EGM admits utility reading a passive continuous state directly.

    The keeper's utility reads the passive housing state through
    `alpha * log(housing)`. The DC-EGM envelope condition forbids utility
    reaching the *Euler* state (here `liquid_assets`), but a passive state is
    allowed, so `validate_dcegm_regimes` accepts the keeper regimes.
    """
    validate_dcegm_regimes(user_regimes=_finalized_keeper_regimes())


@pytest.mark.skip(reason="gpu-01 only: DC-EGM solve OOMs the local box")
def test_housing_keeper_solves_to_finite_values():
    """The keeper model solves to finite, housing-monotone value functions.

    A larger housing stock yields a larger within-period service flow
    `alpha * log(housing)`, so for every income node and liquid-asset node the
    solved value function is strictly increasing in the passive housing state.
    """
    solution = build_model().solve(params=build_params(), log_level="debug")

    n_liquid = LIQUID_ASSETS_GRID.to_jax().shape[0]
    n_housing = HOUSING_GRID.to_jax().shape[0]
    n_income = 2

    for period in sorted(solution)[:-1]:
        keeper_V = np.asarray(solution[period]["keeper"])
        # V axes are (income, liquid_assets, housing); housing is the passive
        # trailing axis.
        assert keeper_V.shape == (n_income, n_liquid, n_housing)
        assert np.all(np.isfinite(keeper_V)), f"period={period}"
        housing_increments = np.diff(keeper_V, axis=-1)
        assert np.all(housing_increments > 0.0), f"period={period}"
