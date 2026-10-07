"""Example and fixture parameter helpers fit the `edges` parameter layout.

A declared regime-transition law keeps its parameter slot at every horizon, so a
helper writes the law's parameters at one place, `params["edges"][source]` (or at
the model level), whatever the horizon. The checks build models at tiny grids,
inspect the public template, and validate one or two initial conditions; they do
not solve or simulate.
"""

from collections.abc import Callable

import jax.numpy as jnp
import pytest

from lcm import Model
from lcm_examples import (
    epstein_zin,
    iskhakov_et_al_2017,
    mortality,
    precautionary_savings,
    precautionary_savings_health,
    stochastic_volatility,
    tiny,
)
from tests.test_models.deterministic import base, ds_pension, housing

_SMALL_HOUSING = {"n_liquid": 3, "n_housing": 3, "n_consumption": 3, "n_new_housing": 3}

# Each builder takes a horizon index from the cases below and returns a model.
_EDGES_TEMPLATES: dict[str, tuple[Callable[[int], Model], tuple[int, ...], dict]] = {
    "tiny": (
        lambda n: tiny.get_model(n_periods=n),
        (2, 3, 4),
        {"working_life": {"last_working_age": "float"}},
    ),
    "health": (
        precautionary_savings_health.get_model,
        (19, 20, 24),
        {"working_life": {"n_periods": "int"}},
    ),
    "epstein_zin": (
        lambda n: epstein_zin.get_model(certainty_equivalent=None, n_periods=n),
        (2, 4),
        {"alive": {"survival_probs": "Float1D"}},
    ),
    "mortality": (
        mortality.get_model,
        (2, 3, 6),
        {
            "working_life": {"survival_probs": "FloatND"},
            "retirement": {"survival_probs": "FloatND"},
        },
    ),
    "iskhakov": (
        iskhakov_et_al_2017.get_model,
        (2, 3, 6),
        {
            "working_life": {"final_age_alive": "float"},
            "retirement": {"final_age_alive": "float"},
        },
    ),
    "housing": (
        lambda n: housing.get_model(n_periods=n, **_SMALL_HOUSING),
        (2, 3, 4),
        {"working": {"final_age_alive": "float"}},
    ),
    "ds_pension": (
        lambda n: ds_pension.get_model(n_periods=n),
        (5, 6),
        {
            "retired": {
                "retired": {"final_age_alive": "float"},
                "dead": {"final_age_alive": "float"},
            }
        },
    ),
}


@pytest.mark.parametrize(
    ("builder", "horizon", "expected"),
    [
        pytest.param(builder, horizon, expected, id=f"{name}-{horizon}")
        for name, (builder, horizons, expected) in _EDGES_TEMPLATES.items()
        for horizon in horizons
    ],
)
def test_declared_law_keeps_its_edges_slots_at_every_horizon(
    *, builder: Callable[[int], Model], horizon: int, expected: dict
) -> None:
    """The `edges` template of an example equals its law's slots at any horizon."""
    assert builder(horizon).get_params_template()["edges"] == expected


@pytest.mark.parametrize("n_periods", [2, 3, 6])
@pytest.mark.parametrize(
    "build",
    [
        lambda n: precautionary_savings.create_model(
            n_periods=n,
            shock_type="rouwenhorst",
            wealth_n_points=3,
            consumption_n_points=3,
            income_n_points=3,
        ),
        lambda n: stochastic_volatility.get_model(
            n_periods=n, wealth_n_points=3, consumption_n_points=3, income_n_points=3
        ),
    ],
    ids=["precautionary_savings", "stochastic_volatility"],
)
def test_fixed_law_input_leaves_no_edges_slot(
    *, build: Callable[[int], Model], n_periods: int
) -> None:
    """A law whose only input is fixed at build time asks for no `edges` value."""
    assert "edges" not in build(n_periods).get_params_template()


@pytest.mark.parametrize("n_periods", [2, 3, 6])
@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst", "tauchen"])
def test_precautionary_params_fit_every_horizon(
    *, n_periods: int, shock_type: precautionary_savings.ShockType
) -> None:
    """One parameter set serves the precautionary-savings model at every horizon."""
    model = precautionary_savings.create_model(
        n_periods=n_periods,
        shock_type=shock_type,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    model.validate_initial_conditions(
        params=precautionary_savings.get_params(
            shock_type=shock_type, sigma=0.1, rho=0.4
        ),
        initial_conditions={
            "age": jnp.array([20.0]),
            "wealth": jnp.array([10.0]),
            "income": jnp.array([0.0]),
            "regime_id": jnp.array([precautionary_savings.RegimeId.alive]),
        },
    )


@pytest.mark.parametrize("n_periods", [2, 3, 6])
def test_stochastic_volatility_params_fit_every_horizon(n_periods: int) -> None:
    """One parameter set serves the volatility model at every horizon."""
    model = stochastic_volatility.get_model(
        n_periods=n_periods,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    model.validate_initial_conditions(
        params=stochastic_volatility.get_params(),
        initial_conditions={
            "age": jnp.array([20.0]),
            "wealth": jnp.array([10.0]),
            "income": jnp.array([0.0]),
            "uncertainty": jnp.array([stochastic_volatility.Uncertainty.low]),
            "regime_id": jnp.array([stochastic_volatility.RegimeId.alive]),
        },
    )


@pytest.mark.parametrize("retirement_age", [19, 20, 24])
def test_health_params_carry_the_horizon_under_edges(retirement_age: int) -> None:
    """The health example writes its law's `n_periods` under `edges`."""
    params = precautionary_savings_health.get_params(retirement_age=retirement_age)
    assert params["edges"] == {"working_life": {"n_periods": retirement_age - 17}}


@pytest.mark.parametrize("retirement_age", [19, 20, 24])
def test_health_params_fit_every_horizon(retirement_age: int) -> None:
    """The health example's parameters fit its model at every retirement age."""
    model = precautionary_savings_health.get_model(retirement_age=retirement_age)
    model.validate_initial_conditions(
        params=precautionary_savings_health.get_params(retirement_age=retirement_age),
        initial_conditions={
            "age": jnp.array([18.0]),
            "wealth": jnp.array([10.0]),
            "health": jnp.array([0.5]),
            "regime_id": jnp.array(
                [precautionary_savings_health.RegimeId.working_life]
            ),
        },
    )


@pytest.mark.parametrize(("n_periods", "last_working_age"), [(2, 25), (3, 45), (4, 65)])
def test_tiny_params_carry_the_last_working_age_under_edges(
    *, n_periods: int, last_working_age: int
) -> None:
    """The tiny example writes its law's last working age under `edges`."""
    params = tiny.get_params(n_periods=n_periods)
    assert params["edges"] == {"working_life": {"last_working_age": last_working_age}}


@pytest.mark.parametrize("n_periods", [2, 3, 4])
def test_tiny_params_fit_every_horizon(n_periods: int) -> None:
    """The tiny example's parameters fit its model at every horizon."""
    tiny.get_model(n_periods=n_periods).validate_initial_conditions(
        params=tiny.get_params(n_periods=n_periods),
        initial_conditions={
            "age": jnp.array([25.0]),
            "wealth": jnp.array([10.0]),
            "regime_id": jnp.array([tiny.RegimeId.working_life]),
        },
    )


@pytest.mark.parametrize(
    ("n_periods", "survival_probs"), [(2, (0.0,)), (4, epstein_zin.SURVIVAL_PROBS)]
)
def test_epstein_zin_params_fit_every_horizon(
    *, n_periods: int, survival_probs: tuple[float, ...]
) -> None:
    """The Epstein-Zin example's survival schedule fits its law at every horizon."""
    model = epstein_zin.get_model(certainty_equivalent=None, n_periods=n_periods)
    model.validate_initial_conditions(
        params=epstein_zin.get_params(
            risk_aversion=None, survival_probs=survival_probs
        ),
        initial_conditions={
            "age": jnp.array([25.0]),
            "wealth": jnp.array([5.0]),
            "health": jnp.array([epstein_zin.HealthStatus.good]),
            "regime_id": jnp.array([epstein_zin.EZRegimeId.alive]),
        },
    )


@pytest.mark.parametrize("n_periods", [2, 3, 6])
def test_both_initial_regimes_accept_the_example_params(n_periods: int) -> None:
    """Both declared starts accept the shared parameters at every horizon."""
    base.get_model(n_periods).validate_initial_conditions(
        params=base.get_params(n_periods=n_periods),
        initial_conditions={
            "age": jnp.array([40.0, 40.0]),
            "wealth": jnp.array([100.0, 100.0]),
            "regime_id": jnp.array(
                [base.RegimeId.working_life, base.RegimeId.retirement]
            ),
        },
    )


@pytest.mark.parametrize(("n_periods", "final_age_alive"), [(2, 1.0), (4, 3.0)])
def test_housing_params_fit_every_horizon(
    *, n_periods: int, final_age_alive: float
) -> None:
    """The housing helper's parameters fit its model at every horizon."""
    housing.get_model(
        n_periods=n_periods, **_SMALL_HOUSING
    ).validate_initial_conditions(
        params=housing.get_params(final_age_alive=final_age_alive),
        initial_conditions={
            "age": jnp.array([0.0]),
            "liquid": jnp.array([10.0]),
            "housing": jnp.array([10.0]),
            "regime_id": jnp.array([housing.RegimeId.working]),
        },
    )


def test_ds_pension_law_input_lives_under_its_source_edge() -> None:
    """The pension helper writes one `final_age_alive` for both retired-law cells."""
    assert ds_pension.get_params()["edges"] == {"retired": {"final_age_alive": 4.0}}
