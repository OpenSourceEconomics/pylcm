"""End-to-end continuous-outer NNBEGM simulation through the nested read.

The merge gate for the nested simulation policy: simulate must replay the
solve's continuous policy class — off-grid outer actions (no silent snapping
to the legacy finite action grid), consumption from the conditional inner
policies, both under the published acceptance rules — and stay deterministic
without taste shocks.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import config as jax_config

from lcm import AdaptiveOuterMesh
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}
# The settings the solve battery converges under (its 120 cells spread their
# optimum basins over the whole outer axis).
_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)
# Subjects strictly between grid nodes on both asset axes, plus two exactly
# on-grid corners: the off-grid subjects are the case the grid argmax cannot
# represent.
_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 27.2727272727272727, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 8.8888888888888889, 17.5]),
    "age": jnp.full(5, 20.0),
    "regime_id": jnp.zeros(5, dtype=jnp.int32),
}


def _simulate(*, seed: int) -> pd.DataFrame:
    model = toy.build_model(variant="n_nbegm", n_periods=3, outer_search=_MESH)
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=seed,
    ).to_dataframe()


@pytest.fixture(scope="module")
def simulated() -> pd.DataFrame:
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    return _simulate(seed=42)


def test_outer_actions_leave_the_finite_action_grid(simulated: pd.DataFrame) -> None:
    """No silent grid snapping: some accepted outer actions are off-grid.

    The legacy finite path can only record nodes of the investment action
    grid; the continuous read must produce interior refined actions for
    off-grid subjects.
    """
    alive = simulated[simulated["regime_name"] == "alive"]
    actions = np.asarray(alive["illiquid_investment"], dtype=float)
    action_grid = np.asarray(toy.ILLIQUID_INVESTMENT_GRID.to_jax())
    distance = np.min(np.abs(actions[:, None] - action_grid[None, :]), axis=1)
    assert np.any(distance > 1e-6), "every outer action snapped to the grid"


def test_consumption_leaves_the_finite_action_grid(simulated: pd.DataFrame) -> None:
    alive = simulated[simulated["regime_name"] == "alive"]
    consumption = np.asarray(alive["consumption"], dtype=float)
    grid = np.asarray(toy.CONSUMPTION_GRID.to_jax())
    distance = np.min(np.abs(consumption[:, None] - grid[None, :]), axis=1)
    assert np.any(distance > 1e-6)


def test_recorded_pairs_respect_the_intrinsic_budget(simulated: pd.DataFrame) -> None:
    """Consumption positive and within resources at the recorded outer action."""
    alive = simulated[simulated["regime_name"] == "alive"]
    wealth = np.asarray(alive["wealth"], dtype=float)
    illiquid = np.asarray(alive["illiquid"], dtype=float)
    investment = np.asarray(alive["illiquid_investment"], dtype=float)
    consumption = np.asarray(alive["consumption"], dtype=float)
    resources = wealth + toy.LABOUR_INCOME - investment
    assert np.all(consumption > 0.0)
    assert np.all(consumption <= resources + 1e-9)
    # The chosen next durable stays inside the outer search domain.
    next_illiquid = illiquid + investment
    assert np.all(next_illiquid >= toy.OUTER_GRID.start - 1e-9)
    assert np.all(next_illiquid <= toy.OUTER_GRID.stop + 1e-9)


def test_simulation_is_deterministic(simulated: pd.DataFrame) -> None:
    again = _simulate(seed=42)
    for column in ("wealth", "illiquid", "consumption", "illiquid_investment"):
        np.testing.assert_array_equal(
            np.asarray(simulated[column], dtype=float),
            np.asarray(again[column], dtype=float),
        )
