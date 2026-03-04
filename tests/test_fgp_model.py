"""FGP-style consumption-savings model tests.

Compare Rouwenhorst vs Tauchen discretization in a simplified lifecycle model
following the spirit of Fella, Gallipoli & Pan (2019).

"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models.fgp_consumption_savings import (
    RHO,
    SIGMA_EPS,
    get_model,
    get_params,
)

_N_SUBJECTS = 100
_SEED = 42


def _solve_and_simulate(shock_type):
    model = get_model(shock_type)
    params = get_params(shock_type)
    result = model.solve_and_simulate(
        params=params,
        initial_states={
            "wealth": jnp.full(_N_SUBJECTS, 5.0),
            "income": jnp.zeros(_N_SUBJECTS),
            "age": jnp.zeros(_N_SUBJECTS),
        },
        initial_regimes=["alive"] * _N_SUBJECTS,
        seed=_SEED,
    )
    return result.to_dataframe()


@pytest.mark.parametrize("shock_type", ["rouwenhorst", "tauchen"])
def test_model_solves(shock_type):
    """Model solves without error."""
    model = get_model(shock_type)
    params = get_params(shock_type)
    V = model.solve(params=params)
    assert V is not None
    # Value functions include all periods (n_periods + 1 ages from AgeGrid)
    assert len(V) == model.n_periods


@pytest.mark.parametrize("shock_type", ["rouwenhorst", "tauchen"])
def test_model_simulates(shock_type):
    """Model simulates without error."""
    df = _solve_and_simulate(shock_type)
    assert len(df) > 0
    assert "wealth" in df.columns
    assert "consumption" in df.columns
    assert "income" in df.columns


@pytest.mark.parametrize("shock_type", ["rouwenhorst", "tauchen"])
def test_simulated_income_moments(shock_type):
    """Simulated income moments are in the right ballpark."""
    df = _solve_and_simulate(shock_type)
    alive_df = df[df["regime"] == "alive"]

    # Income on the grid should have mean near 0 and std near sigma_y
    income_vals = alive_df["income"].to_numpy()
    expected_std = SIGMA_EPS / np.sqrt(1 - RHO**2)

    assert abs(np.mean(income_vals)) < 0.5 * expected_std
    assert np.std(income_vals) < 2 * expected_std


def test_rouwenhorst_income_moments_closer_to_theory():
    """Rouwenhorst income moments are at least as close to theory as Tauchen's."""
    expected_var = SIGMA_EPS**2 / (1 - RHO**2)

    df_r = _solve_and_simulate("rouwenhorst")
    df_t = _solve_and_simulate("tauchen")

    r_var = df_r[df_r["regime"] == "alive"]["income"].var()
    t_var = df_t[df_t["regime"] == "alive"]["income"].var()

    # Both should be in the right ballpark; Rouwenhorst should be at least as close
    r_err = abs(r_var - expected_var)
    t_err = abs(t_var - expected_var)

    # Rouwenhorst error should be no worse than Tauchen, with generous tolerance
    assert r_err < t_err + 0.1 * expected_var
