"""Example and fixture parameter helpers match the regime graph they describe.

A source with a single destination carries no numerical transition law, so a
helper supplies a law-only input (`final_age_alive`, a selector's `n_periods`)
exactly when the model's parameter template still asks for it. Strict parameter
matching rejects anything else. The checks build models at tiny grids, inspect
the public graph and template, and validate one or two initial conditions; they
do not solve or simulate.
"""

import jax.numpy as jnp
import pytest

from lcm_examples import (
    precautionary_savings,
    precautionary_savings_health,
    stochastic_volatility,
)
from tests.test_models.deterministic import base, dcegm_variants, housing


@pytest.mark.parametrize("n_periods", [2, 3, 6])
@pytest.mark.parametrize("shock_type", ["normal_gh", "rouwenhorst", "tauchen"])
def test_precautionary_horizon_builds(
    *, n_periods: int, shock_type: precautionary_savings.ShockType
) -> None:
    """A horizon without a regime selector also drops the selector's fixed input."""
    model = precautionary_savings.create_model(
        n_periods=n_periods,
        shock_type=shock_type,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    assert model.n_periods == n_periods
    if n_periods == 2:
        # One alive decision at 20 lands in the terminal dead regime at 30.
        expected = {"dead": frozenset({20})}
        assert model.graph.edges.solve["alive"] == expected
        assert model.graph.edges.simulate["alive"] == expected
        assert model.graph.nodes == frozenset({(20, "alive"), (30, "dead")})
        assert not model.get_params_template()["alive"].get("next_regime", {})
    params = precautionary_savings.get_params(shock_type=shock_type, sigma=0.1, rho=0.4)
    model.validate_initial_conditions(
        params=params,
        initial_conditions={
            "age": jnp.array([20.0]),
            "wealth": jnp.array([10.0]),
            "income": jnp.array([0.0]),
            "regime_id": jnp.array([precautionary_savings.RegimeId.alive]),
        },
    )


@pytest.mark.parametrize("n_periods", [2, 3, 6])
def test_stochastic_volatility_horizon_builds(n_periods: int) -> None:
    """The volatility example builds at every horizon with its default parameters."""
    model = stochastic_volatility.get_model(
        n_periods=n_periods,
        wealth_n_points=3,
        consumption_n_points=3,
        income_n_points=3,
    )
    assert model.n_periods == n_periods
    if n_periods == 2:
        expected = {"dead": frozenset({20})}
        assert model.graph.edges.solve["alive"] == expected
        assert model.graph.edges.simulate["alive"] == expected
        assert model.graph.nodes == frozenset({(20, "alive"), (30, "dead")})
        assert not model.get_params_template()["alive"].get("next_regime", {})
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
def test_health_params_match_retained_semantics(retirement_age: int) -> None:
    """The health example supplies the selector's `n_periods` iff the model reads it."""
    model = precautionary_savings_health.get_model(retirement_age=retirement_age)
    params = precautionary_savings_health.get_params(retirement_age=retirement_age)
    template = model.get_params_template()
    supplied = params["working_life"].get("next_regime", {})
    required = template["working_life"].get("next_regime", {})
    # `n_periods` has one consumer, the next-regime selector; the template
    # states whether that selector exists.
    assert ("n_periods" in supplied) == ("n_periods" in required)
    if retirement_age == 19:
        expected = {"retirement": frozenset({18})}
        assert model.graph.edges.solve["working_life"] == expected
        assert model.graph.edges.simulate["working_life"] == expected
    model.validate_initial_conditions(
        params=params,
        initial_conditions={
            "age": jnp.array([18.0]),
            "wealth": jnp.array([10.0]),
            "health": jnp.array([0.5]),
            "regime_id": jnp.array(
                [precautionary_savings_health.RegimeId.working_life]
            ),
        },
    )


@pytest.mark.parametrize("n_periods", [2, 3, 6])
def test_both_initial_regimes_accept_the_example_params(n_periods: int) -> None:
    """Both declared starts accept the shared parameters at every horizon."""
    model = base.get_model(n_periods)
    params = base.get_params(n_periods=n_periods)
    template = model.get_params_template()
    retained_consumer = any(
        "final_age_alive" in per_regime.get("next_regime", {})
        for per_regime in template.values()
    )
    assert ("final_age_alive" in params) == retained_consumer
    assert model.initial_nodes == frozenset({(40, "working_life"), (40, "retirement")})
    if n_periods == 2:
        expected = {
            "working_life": {"dead": frozenset({40})},
            "retirement": {"dead": frozenset({40})},
        }
        assert model.graph.edges.solve == expected
        assert model.graph.edges.simulate == expected
        assert model.graph.nodes == frozenset(
            {(40, "working_life"), (40, "retirement"), (50, "dead")}
        )
    model.validate_initial_conditions(
        params=params,
        initial_conditions={
            "age": jnp.array([40.0, 40.0]),
            "wealth": jnp.array([100.0, 100.0]),
            "regime_id": jnp.array(
                [base.RegimeId.working_life, base.RegimeId.retirement]
            ),
        },
    )


@pytest.mark.parametrize("n_periods", [2, 3, 6])
def test_solver_pair_helper_tracks_the_shared_parameter_contract(
    n_periods: int,
) -> None:
    """The DC-EGM pair helper returns the shared parameters at the same calibration."""
    calibration = {
        "discount_factor": 0.98,
        "disutility_of_work": 1.0,
        "interest_rate": 0.0,
        "wage": 20.0,
    }
    expected = base.get_params(n_periods=n_periods, **calibration)
    assert (
        dcegm_variants.get_full_params(n_periods=n_periods, **calibration) == expected
    )


@pytest.mark.parametrize(
    ("n_periods", "final_age_alive"),
    [(2, 1.0), (3, 2.0), (4, 3.0), (4, 1.0), (6, 5.0)],
)
def test_housing_params_follow_the_model_horizon(
    *, n_periods: int, final_age_alive: float
) -> None:
    """The housing helper supplies `final_age_alive` iff the horizon reads it."""
    model = housing.get_model(
        n_periods=n_periods,
        n_liquid=3,
        n_housing=3,
        n_consumption=3,
        n_new_housing=3,
    )
    params = housing.get_params(n_periods=n_periods, final_age_alive=final_age_alive)
    template = model.get_params_template()
    retained_consumer = "final_age_alive" in template["working"].get("next_regime", {})
    assert ("final_age_alive" in params) == retained_consumer
    assert retained_consumer == (n_periods > 2)
    assert model.initial_nodes == frozenset({(0, "working")})
    if n_periods == 2:
        expected = {"working": {"dead": frozenset({0})}}
        assert model.graph.edges.solve == expected
        assert model.graph.edges.simulate == expected
        assert model.graph.nodes == frozenset({(0, "working"), (1, "dead")})
    else:
        # An early death value in a longer graph is still an economic input.
        assert params["final_age_alive"] == final_age_alive
    model.validate_initial_conditions(
        params=params,
        initial_conditions={
            "age": jnp.array([0.0]),
            "liquid": jnp.array([10.0]),
            "housing": jnp.array([10.0]),
            "regime_id": jnp.array([housing.RegimeId.working]),
        },
    )


def test_default_housing_parameter_call_preserves_its_calibration() -> None:
    """The default housing parameters describe the default four-period model."""
    assert housing.get_params() == housing.get_params(n_periods=4)
    assert housing.get_params()["final_age_alive"] == 3.0
