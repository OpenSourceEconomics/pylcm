"""The DC-EGM solve publishes the refined consumption function for simulation.

The Euler inversion plus upper envelope recover the exact off-grid optimal
continuous action on the resources grid. The solve hands that policy to
`simulate` as a per-period `EGMSimPolicy`, so a simulated subject's continuous
action can be interpolated at its resources rather than snapped to the action
grid. This test pins the published artifact directly: interpolating it must
reproduce the closed-form consumption, including at resources strictly between
action-grid nodes (where a grid argmax cannot land).
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.published_policy import EGMSimPolicy
from lcm import AgeGrid, LogSpacedGrid, Model
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousState, FloatND
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_params,
)


def _bequest_utility(wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _two_period_bequest_model() -> Model:
    """Two-period log-utility retirement model with a terminal bequest."""
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


def test_solve_publishes_policy_matching_closed_form_consumption():
    """Interpolating the published policy reproduces `c* = wealth / (1 + beta)`.

    With log utility, zero interest, a two-period horizon, and a terminal
    bequest `(age/50) log(wealth)` at age 50, the decision period's optimal
    consumption is `wealth / (1 + beta)` at every resources level. The published
    policy interpolated at off-grid resources must hit it.
    """
    discount_factor = 0.98
    params = get_retirement_only_params(2, discount_factor=discount_factor)

    _v, sim_policy = _two_period_bequest_model().solve(
        params=params, log_level="debug", return_simulation_policy=True
    )

    pol = sim_policy[0]["retirement"]
    assert isinstance(pol, EGMSimPolicy)

    # Resources = wealth here (zero interest, no labor income). Query strictly
    # between wealth-grid nodes to exercise the off-grid interpolation.
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid = 0.5 * (wealth_nodes[3:-1] + wealth_nodes[4:])
    consumption = interp_on_padded_grid(
        x_query=jnp.asarray(off_grid), xp=pol.endog_grid, fp=pol.policy
    )
    expected = off_grid / (1.0 + discount_factor)
    np.testing.assert_allclose(np.asarray(consumption), expected, rtol=2e-2)


def test_published_policies_are_host_resident():
    """Solve evicts simulation policies to host, not device.

    The policies are a solve output no backward step reads; keeping them on the
    accelerator would pin one carry-sized buffer per period for the whole
    induction. So the returned policy arrays live on the host (CPU) device.
    """
    params = get_retirement_only_params(2, discount_factor=0.98)
    _v, sim_policy = _two_period_bequest_model().solve(
        params=params, log_level="debug", return_simulation_policy=True
    )

    pol = sim_policy[0]["retirement"]
    assert isinstance(pol, EGMSimPolicy)
    assert all(
        device.platform == "cpu"
        for array in (pol.endog_grid, pol.policy)
        for device in array.devices()
    )
