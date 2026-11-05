"""Economic validation tests for shock implementations.

Verify that different shock parametrizations produce economically meaningful
differences in a consumption-savings model — specifically, the precautionary
savings motive (Carroll 1997, Deaton 1991).

"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.shocks import GaussHermite, Normal, PhysicalValues, Product
from tests.conftest import X64_ENABLED
from tests.test_models.precautionary_savings import RegimeId, get_model, get_params

_N_PERIODS = 7
_N_SUBJECTS = 50
_SEED = 42
_SIGMA_ZERO = 1e-8  # Effectively zero; exact 0 causes degenerate grids
_DETERMINISTIC_ATOL = 1e-6 if X64_ENABLED else 1e-3


def _simulate(*, shock_type, sigma, rho=0.0, mu=0.0):
    model = get_model(n_periods=_N_PERIODS, shock_type=shock_type)
    params = get_params(shock_type=shock_type, sigma=sigma, mu=mu, rho=rho)
    unconditional_mean = mu / (1 - rho)
    result = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "wealth": jnp.full(_N_SUBJECTS, 5.0),
            "income": (
                PhysicalValues(values=jnp.full(_N_SUBJECTS, unconditional_mean))
                if shock_type == "normal_gh"
                else jnp.full(_N_SUBJECTS, unconditional_mean)
            ),
            "age": jnp.full(_N_SUBJECTS, 20.0),
            "regime_id": jnp.array([RegimeId.alive] * _N_SUBJECTS),
        },
        seed=_SEED,
    )
    return result.to_dataframe()


def _mean_wealth_in_final_alive_period(df):
    rows = df[(df["period"] == _N_PERIODS - 2) & (df["regime_name"] == "alive")]
    assert len(rows) > 0, "No rows in final alive period"
    return rows["wealth"].mean()


@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst", "tauchen"])
def test_deterministic_when_sigma_zero(shock_type):
    rho = 0.5 if shock_type in ("rouwenhorst", "tauchen") else 0.0
    df = _simulate(shock_type=shock_type, sigma=_SIGMA_ZERO, rho=rho)

    alive_df = df[df["regime_name"] == "alive"]
    for period in alive_df["period"].unique():
        period_data = alive_df[alive_df["period"] == period]
        assert period_data["wealth"].std() == pytest.approx(0, abs=_DETERMINISTIC_ATOL)
        assert period_data["consumption"].std() == pytest.approx(
            0, abs=_DETERMINISTIC_ATOL
        )


def test_higher_sigma_increases_mean_wealth_normal():
    df_low = _simulate(shock_type="normal_gh", sigma=0.1)
    df_high = _simulate(shock_type="normal_gh", sigma=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


def test_higher_sigma_increases_mean_wealth_rouwenhorst():
    df_low = _simulate(shock_type="rouwenhorst", sigma=0.1, rho=0.5)
    df_high = _simulate(shock_type="rouwenhorst", sigma=0.5, rho=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


def test_higher_rho_increases_mean_wealth_rouwenhorst():
    df_low = _simulate(shock_type="rouwenhorst", sigma=0.3, rho=0.2)
    df_high = _simulate(shock_type="rouwenhorst", sigma=0.3, rho=0.8)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


def test_precautionary_savings_with_nonzero_mu():
    """Precautionary savings motive holds with non-zero drift (mu != 0)."""
    df_low = _simulate(shock_type="rouwenhorst", sigma=0.1, rho=0.5, mu=0.5)
    df_high = _simulate(shock_type="rouwenhorst", sigma=0.5, rho=0.5, mu=0.5)

    assert _mean_wealth_in_final_alive_period(
        df_high
    ) > _mean_wealth_in_final_alive_period(df_low)


@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst", "tauchen"])
def test_precautionary_savings_versus_deterministic_baseline(shock_type):
    rho = 0.5 if shock_type in ("rouwenhorst", "tauchen") else 0.0
    df_det = _simulate(shock_type=shock_type, sigma=_SIGMA_ZERO, rho=rho)
    df_stoch = _simulate(shock_type=shock_type, sigma=0.5, rho=rho)

    assert _mean_wealth_in_final_alive_period(
        df_stoch
    ) > _mean_wealth_in_final_alive_period(df_det)


def test_normal_gh_declares_income_as_a_named_current_shock() -> None:
    """Income is observed at choice time without a stored income-state axis."""
    model = get_model(
        n_periods=4,
        shock_type="normal_gh",
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    assert set(model.user_regimes["alive"].states) == {"wealth"}
    assert model.user_regimes["alive"].shocks == ("income_risk",)
    group = model.shocks["income_risk"]
    assert set(group.shocks) == {"income"}
    assert isinstance(group.shocks["income"], Normal)
    assert isinstance(group.nodes, Product)
    assert group.nodes.rules == {"income": GaussHermite(n=3)}
    assert model.get_params_template()["income_risk"] == {
        "income__mu": "float",
        "income__sigma": "float",
    }


def test_normal_gh_replays_known_entry_income_and_dated_physical_paths() -> None:
    """The same solution prices known current income before updating wealth."""
    model = get_model(
        n_periods=4,
        shock_type="normal_gh",
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    params = get_params(shock_type="normal_gh", mu=0.1, sigma=0.2, interest_rate=0.03)
    solution = model.solve(params=params, log_level="off")
    entry_income = jnp.array([0.37, -0.29])
    initial = {
        "age": jnp.array([20.0, 20.0]),
        "wealth": jnp.array([10.0, 12.0]),
        "income": PhysicalValues(values=entry_income),
        "regime_id": jnp.array([RegimeId.alive, RegimeId.alive]),
    }
    entry_frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        seed=17,
        log_level="off",
    ).to_dataframe()
    entry = entry_frame.loc[entry_frame["period"] == 0].sort_values("subject_id")
    landed = entry_frame.loc[entry_frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(entry["subject_id"], [0, 1])
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(entry["age"], [20.0, 20.0])
    np.testing.assert_array_equal(landed["age"], [30.0, 30.0])
    np.testing.assert_allclose(entry["income"], [0.37, -0.29], rtol=1e-6)
    np.testing.assert_allclose(
        landed["wealth"],
        1.03 * (np.array([10.0, 12.0]) - entry["consumption"].to_numpy())
        + np.exp(np.array([0.37, -0.29])),
        rtol=1e-6,
    )

    physical_path = jnp.array([[0.37, 0.51, 0.11, 0.0], [-0.29, -0.43, 0.12, 0.0]])
    path_frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        shock_paths={"income": PhysicalValues(values=physical_path)},
        seed=23,
        log_level="off",
    ).to_dataframe()
    for period, expected_income in enumerate(([0.37, -0.29], [0.51, -0.43])):
        current = path_frame.loc[path_frame["period"] == period].sort_values(
            "subject_id"
        )
        following = path_frame.loc[path_frame["period"] == period + 1].sort_values(
            "subject_id"
        )
        np.testing.assert_array_equal(current["subject_id"], [0, 1])
        np.testing.assert_array_equal(following["subject_id"], [0, 1])
        np.testing.assert_array_equal(current["age"], [20.0 + 10 * period] * 2)
        np.testing.assert_allclose(current["income"], expected_income, rtol=1e-6)
        np.testing.assert_allclose(
            following["wealth"],
            1.03 * (current["wealth"].to_numpy() - current["consumption"].to_numpy())
            + np.exp(np.array(expected_income)),
            rtol=1e-6,
        )
    np.testing.assert_array_equal(initial["wealth"], [10.0, 12.0])
    np.testing.assert_allclose(entry_income, [0.37, -0.29], rtol=1e-6)
    np.testing.assert_allclose(
        physical_path,
        [[0.37, 0.51, 0.11, 0.0], [-0.29, -0.43, 0.12, 0.0]],
        rtol=1e-6,
    )
