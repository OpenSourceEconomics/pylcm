"""The lognormal process fixture is the normal one with income kept in levels."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from tests.conftest import X64_ENABLED
from tests.test_models.processes import (
    RegimeId,
    get_initial_income,
    get_model,
    get_params,
)

pytestmark = pytest.mark.skipif(not X64_ENABLED, reason="Parity is a float64 check.")


def _solve_and_simulate(distribution_type):
    model = get_model(n_periods=4, distribution_type=distribution_type)
    params = get_params(distribution_type)
    solution = model.solve(params=params, log_level="off")
    panel = model.simulate(
        params=params,
        initial_conditions={
            "health": jnp.asarray([0, 0], dtype=jnp.int32),
            "income": jnp.full(2, get_initial_income(distribution_type)),
            "wealth": jnp.asarray([1.0, 1.0]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.array([RegimeId.alive] * 2),
        },
        solution=solution,
        seed=42,
        log_level="off",
    ).to_dataframe()
    return solution, panel


@pytest.mark.parametrize("period", [0, 1, 2])
def test_lognormal_values_equal_normal_values(period):
    """Both variants draw the same income levels, so their values coincide."""
    lognormal, _ = _solve_and_simulate("lognormal")
    normal, _ = _solve_and_simulate("normal")
    np.testing.assert_allclose(
        np.asarray(lognormal.values[period]["alive"]),
        np.asarray(normal.values[period]["alive"]),
        rtol=1e-12,
        atol=0,
    )


def test_lognormal_panel_is_normal_panel_with_income_in_levels():
    """Under one seed the lognormal draw is the exponential of the normal draw."""
    _, lognormal = _solve_and_simulate("lognormal")
    _, normal = _solve_and_simulate("normal")
    expected = normal.assign(income=np.exp(normal["income"]))
    pd.testing.assert_frame_equal(lognormal, expected, rtol=1e-12, atol=0)
