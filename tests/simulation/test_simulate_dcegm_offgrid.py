"""Off-grid DC-EGM forward simulation: the continuous action is interpolated.

Standard EGM (Carroll 2006; Iskhakov-Jørgensen-Rust-Schjerning 2017; Druedahl
2021) simulates the continuous choice by interpolating the refined consumption
function at the simulated state, not by an argmax over the action grid. The
solve publishes that function (`EGMSimPolicy`, per `model.solve(
return_simulation_policy=True)`); simulation interpolates it at each subject's
resources.

The spec below is the closed-form two-period retirement model with log utility,
zero interest, and an age-50 bequest `(age/50) log(wealth)`: optimal consumption
is `c* = wealth / (1 + beta)` at every resources level. Seeding subjects at
wealth strictly between consumption-grid nodes, the simulated consumption must
hit `c*` — which a grid-restricted argmax cannot.

Marked xfail until Increment 2 wires the off-grid interpolation into `simulate`
(the solve-side publication landed first); flip when it does.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LogSpacedGrid, Model
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousState, FloatND
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_params,
)

_DISCOUNT_FACTOR = 0.98


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _closed_form_model() -> Model:
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={
            "retirement": dcegm_retirement.replace(active=lambda age: age < 50),
            "dead": bequest_dead,
        },
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


@pytest.mark.xfail(
    reason="off-grid continuous-action interpolation not yet wired into simulate "
    "(Increment 2 pending; the solve-side EGMSimPolicy publication landed first)",
    strict=True,
)
def test_dcegm_simulated_consumption_is_off_grid_closed_form():
    """Simulated consumption equals `wealth / (1 + beta)` at off-grid wealth.

    A grid-restricted argmax snaps consumption to the action grid and cannot hit
    the closed form between nodes; off-grid interpolation of the published policy
    must.
    """
    model = _closed_form_model()
    params = get_retirement_only_params(2, discount_factor=_DISCOUNT_FACTOR)

    # Seed subjects at wealth strictly between consumption-grid nodes (the
    # consumption grid shares spacing with the wealth grid in this example).
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid_wealth = 0.5 * (wealth_nodes[3:-1] + wealth_nodes[4:])
    initial_conditions = {
        "wealth": jnp.asarray(off_grid_wealth),
        "regime_id": jnp.full(
            off_grid_wealth.shape,
            retirement_only.RetirementOnlyRegimeId.retirement,
        ),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe(additional_targets=["consumption"])
    period_0 = df.query("period == 0")
    consumption = period_0["consumption"].to_numpy()
    expected = off_grid_wealth / (1.0 + _DISCOUNT_FACTOR)
    np.testing.assert_allclose(consumption, expected, rtol=2e-2)
