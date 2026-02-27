"""Economic validation tests for shock implementations.

Verify that different shock parametrizations produce economically meaningful
differences in a consumption-savings model — specifically, the precautionary
savings motive (Carroll 1997, Deaton 1991).

"""

import jax.numpy as jnp
import pytest

from tests.conftest import X64_ENABLED
from tests.test_models.precautionary_savings import get_model, get_params

_N_PERIODS = 7
_N_SUBJECTS = 50
_SEED = 42
_SIGMA_ZERO = 1e-8  # Effectively zero; exact 0 causes degenerate grids


def _solve_and_simulate(shock_type, *, sigma, rho=0.0, mu=0.0):
    model = get_model(_N_PERIODS, shock_type)
    params = get_params(shock_type, sigma=sigma, mu=mu, rho=rho)
    unconditional_mean = mu / (1 - rho)
    result = model.solve_and_simulate(
        params=params,
        initial_states={
            "wealth": jnp.full(_N_SUBJECTS, 5.0),
            "income": jnp.full(_N_SUBJECTS, unconditional_mean),
            "age": jnp.zeros(_N_SUBJECTS),
        },
        initial_regimes=["alive"] * _N_SUBJECTS,
        seed=_SEED,
    )
    return result.to_dataframe()


def _mean_wealth_in_final_alive_period(df):
    return df[(df["period"] == _N_PERIODS - 1) & (df["regime"] == "alive")][
        "wealth"
    ].mean()


# ======================================================================================
# Section A: Deterministic behavior (sigma ~ 0)
# ======================================================================================


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst"])
def test_deterministic_when_sigma_zero(shock_type):
    rho = 0.5 if shock_type == "rouwenhorst" else 0.0
    df = _solve_and_simulate(shock_type, sigma=_SIGMA_ZERO, rho=rho)

    alive_df = df[df["regime"] == "alive"]
    for period in alive_df["period"].unique():
        period_data = alive_df[alive_df["period"] == period]
        assert period_data["wealth"].std() == pytest.approx(0, abs=1e-6)
        assert period_data["consumption"].std() == pytest.approx(0, abs=1e-6)


# ======================================================================================
# Section B: Precautionary savings, IID
# ======================================================================================


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
def test_higher_sigma_increases_mean_wealth_normal():
    df_low = _solve_and_simulate("normal_gh", sigma=0.1)
    df_high = _solve_and_simulate("normal_gh", sigma=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


# ======================================================================================
# Section C: Precautionary savings, AR(1)
# ======================================================================================


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
def test_higher_sigma_increases_mean_wealth_rouwenhorst():
    df_low = _solve_and_simulate("rouwenhorst", sigma=0.1, rho=0.5)
    df_high = _solve_and_simulate("rouwenhorst", sigma=0.5, rho=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
def test_higher_rho_increases_mean_wealth_rouwenhorst():
    df_low = _solve_and_simulate("rouwenhorst", sigma=0.3, rho=0.2)
    df_high = _solve_and_simulate("rouwenhorst", sigma=0.3, rho=0.8)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
def test_precautionary_savings_with_nonzero_mu():
    """Precautionary savings motive holds with non-zero drift (mu != 0)."""
    df_low = _solve_and_simulate("rouwenhorst", sigma=0.1, rho=0.5, mu=0.5)
    df_high = _solve_and_simulate("rouwenhorst", sigma=0.5, rho=0.5, mu=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


# ======================================================================================
# Section D: Stochastic vs deterministic baseline
# ======================================================================================


@pytest.mark.skipif(not X64_ENABLED, reason="Requires 64-bit precision")
@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst"])
def test_precautionary_savings_versus_deterministic_baseline(shock_type):
    rho = 0.5 if shock_type == "rouwenhorst" else 0.0
    df_det = _solve_and_simulate(shock_type, sigma=_SIGMA_ZERO, rho=rho)
    df_stoch = _solve_and_simulate(shock_type, sigma=0.5, rho=rho)

    assert _mean_wealth_in_final_alive_period(
        df_stoch
    ) > _mean_wealth_in_final_alive_period(df_det)
