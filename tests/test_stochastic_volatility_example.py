"""Smoke + behavior test for the stochastic-volatility example model."""

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import lcm.processes
from lcm.exceptions import GridInitializationError
from lcm.shocks import BaseInnovations, GaussHermite, Normal, PhysicalValues, Product
from lcm_examples.stochastic_volatility import (
    RegimeId,
    Uncertainty,
    get_model,
    get_params,
)


def test_state_conditioned_is_public():
    assert lcm.StateConditioned is not None
    assert lcm.processes.StateConditioned is lcm.StateConditioned


@pytest.mark.parametrize(
    ("sigma_low", "sigma_high"),
    [(0.0, 0.3), (0.1, -0.3), (float("nan"), 0.3), (0.1, float("inf"))],
    ids=("zero-low", "negative-high", "nan-low", "infinite-high"),
)
def test_example_rejects_invalid_income_sigmas(
    *, sigma_low: float, sigma_high: float
) -> None:
    with pytest.raises(GridInitializationError, match="finite positive values"):
        get_model(sigma_low=sigma_low, sigma_high=sigma_high)


def test_example_solves_and_uncertainty_matters():
    model = get_model(n_periods=6)
    solution = model.solve(params=get_params(), log_level="debug")
    V = solution.values
    # finite value everywhere
    for regime_to_value in V.values():
        for value in regime_to_value.values():
            assert np.all(np.isfinite(np.asarray(value)))
    # the value depends on the uncertainty regime (distinct per-regime sigmas)
    maxdiff = 0.0
    for period, regime_to_value in V.items():
        for regime, value in regime_to_value.items():
            a = np.asarray(value)
            schema = solution.metadata.value_schemas[period, regime]
            if "uncertainty" in schema.axis_names:
                ax = schema.axis_names.index("uncertainty")
                maxdiff = max(
                    maxdiff,
                    float(np.abs(np.take(a, 0, ax) - np.take(a, 1, ax)).max()),
                )
    assert maxdiff > 1e-3


def test_example_declares_source_dated_income_with_an_explicit_lag() -> None:
    """The current income law retains the preceding uncertainty state."""
    model = get_model(
        n_periods=4,
        sigma_low=0.1,
        sigma_high=0.3,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    assert set(model.user_regimes["alive"].states) == {
        "wealth",
        "uncertainty",
        "lagged_uncertainty",
    }
    assert model.user_regimes["alive"].shocks == ("income_risk",)
    group = model.shocks["income_risk"]
    assert set(group.shocks) == {"income"}
    assert isinstance(group.shocks["income"], Normal)
    assert isinstance(group.nodes, Product)
    assert group.nodes.rules == {"income": GaussHermite(n=3)}


def test_example_replays_known_income_and_source_dated_switching_scales() -> None:
    """Known entry and a nonzero base path use the original solved model."""
    model = get_model(
        n_periods=4,
        sigma_low=0.1,
        sigma_high=0.3,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    params = get_params(persistence=0.0)
    solution = model.solve(params=params, log_level="off")
    initial = {
        "age": jnp.array([20.0, 20.0]),
        "wealth": jnp.array([12.0, 14.0]),
        "uncertainty": jnp.array([Uncertainty.low, Uncertainty.high]),
        "lagged_uncertainty": jnp.array([Uncertainty.high, Uncertainty.low]),
        "regime_id": jnp.array([RegimeId.alive, RegimeId.alive]),
    }
    entry_income = jnp.array([0.37, -0.29])
    entry_frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial | {"income": PhysicalValues(values=entry_income)},
        seed=17,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    entry = entry_frame.loc[entry_frame["period"] == 0].sort_values("subject_id")
    following = entry_frame.loc[entry_frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(entry["subject_id"], [0, 1])
    np.testing.assert_array_equal(following["subject_id"], [0, 1])
    np.testing.assert_array_equal(entry["age"], [20.0, 20.0])
    np.testing.assert_array_equal(following["age"], [30.0, 30.0])
    np.testing.assert_allclose(entry["income"], [0.37, -0.29], rtol=1e-6)
    np.testing.assert_allclose(
        following["wealth"],
        1.03 * (np.array([12.0, 14.0]) - entry["consumption"].to_numpy())
        + np.exp(np.array([0.37, -0.29])),
        rtol=1e-6,
    )

    bases = jnp.ones((2, 4))
    switched_frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        shock_paths={"income": BaseInnovations(values=bases)},
        seed=23,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    expected_uncertainty = (
        [Uncertainty.low, Uncertainty.high],
        [Uncertainty.high, Uncertainty.low],
    )
    expected_lag = (
        [Uncertainty.high, Uncertainty.low],
        [Uncertainty.low, Uncertainty.high],
    )
    for period, expected_income in enumerate(([0.3, 0.1], [0.1, 0.3])):
        current = switched_frame.loc[switched_frame["period"] == period].sort_values(
            "subject_id"
        )
        following = switched_frame.loc[
            switched_frame["period"] == period + 1
        ].sort_values("subject_id")
        np.testing.assert_array_equal(current["subject_id"], [0, 1])
        np.testing.assert_array_equal(following["subject_id"], [0, 1])
        np.testing.assert_array_equal(current["age"], [20.0 + 10 * period] * 2)
        np.testing.assert_array_equal(
            current["uncertainty"], expected_uncertainty[period]
        )
        np.testing.assert_array_equal(
            current["lagged_uncertainty"], expected_lag[period]
        )
        np.testing.assert_array_equal(
            following["lagged_uncertainty"], current["uncertainty"]
        )
        np.testing.assert_allclose(current["income"], expected_income, rtol=1e-6)
        np.testing.assert_allclose(
            following["wealth"],
            1.03 * (current["wealth"].to_numpy() - current["consumption"].to_numpy())
            + np.exp(np.array(expected_income)),
            rtol=1e-6,
        )
    np.testing.assert_array_equal(initial["wealth"], [12.0, 14.0])
    np.testing.assert_array_equal(bases, np.ones((2, 4)))
    np.testing.assert_allclose(entry_income, [0.37, -0.29], rtol=1e-6)
