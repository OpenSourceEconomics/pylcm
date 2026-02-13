"""Tests for runtime-supplied ShockGrid parameters."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

import lcm
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, ShockGrid, categorical
from lcm.typing import ContinuousAction, ContinuousState, FloatND

# ======================================================================================
# Model setup helpers
# ======================================================================================


@categorical
class RegimeIdShock:
    alive: int
    dead: int


def _shock_utility(
    wealth: ContinuousState,
    income: ContinuousState,
    consumption: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) + 0.01 * (wealth + jnp.exp(income))


def _shock_next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    return wealth - consumption


@lcm.mark.stochastic
def _shock_next_income() -> None:
    pass


def _shock_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


_TAUCHEN_PARAMS = {"rho": 0.9, "sigma_eps": 1.0, "mu_eps": 0.0, "n_std": 2}


def _make_shock_model(*, fixed_params=None):
    """Create a shock model with all shock params supplied at runtime."""
    alive = Regime(
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
            "income": ShockGrid(distribution_type="tauchen", n_points=3),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4)},
        functions={"utility": _shock_utility},
        constraints={"borrowing": _shock_constraint},
        transitions={
            "next_wealth": _shock_next_wealth,
            "next_income": _shock_next_income,
            "next_regime": lambda period: jnp.where(
                period >= 1, RegimeIdShock.dead, RegimeIdShock.alive
            ),
        },
        active=lambda age: age < 2,
    )
    dead = Regime(
        functions={"utility": lambda: 0.0},
        terminal=True,
        active=lambda age: age >= 2,
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeIdShock,
        fixed_params=fixed_params or {},
    )


# ======================================================================================
# Tests
# ======================================================================================


def test_runtime_shock_params_property():
    """ShockGrid without shock_params reports all params as runtime-supplied."""
    grid = ShockGrid(distribution_type="tauchen", n_points=5)
    for name in ("rho", "sigma_eps", "mu_eps", "n_std"):
        assert name in grid.params_to_pass_at_runtime
    assert not grid.is_fully_specified


def test_fully_specified_shock():
    """ShockGrid with all params should have no runtime-supplied params."""
    grid = ShockGrid(
        distribution_type="tauchen",
        n_points=5,
        shock_params=MappingProxyType(_TAUCHEN_PARAMS),
    )
    assert grid.params_to_pass_at_runtime == ()
    assert grid.is_fully_specified


@pytest.mark.parametrize("distribution_type", ["uniform", "normal"])
def test_shock_without_params_is_not_fully_specified(distribution_type):
    """All distributions require explicit params — nothing is defaulted."""
    grid = ShockGrid(distribution_type=distribution_type, n_points=5)
    assert not grid.is_fully_specified
    assert len(grid.params_to_pass_at_runtime) > 0


def test_runtime_shock_in_params_template():
    """Runtime-supplied ShockGrid params appear in params_template."""
    model = _make_shock_model()
    alive_template = model.params_template["alive"]
    assert "income" in alive_template
    for name in ("rho", "sigma_eps", "mu_eps", "n_std"):
        assert name in alive_template["income"]


def test_solve_with_runtime_shock():
    """Solve should work with runtime-supplied shock params."""
    model = _make_shock_model()
    params = {"discount_factor": 1.0, **_TAUCHEN_PARAMS}
    V_arr_dict = model.solve(params, debug_mode=False)
    assert len(V_arr_dict) > 0


def test_runtime_shock_with_fixed_params():
    """Shock params provided via fixed_params are removed from template."""
    model = _make_shock_model(fixed_params=_TAUCHEN_PARAMS)

    alive_template = model.params_template.get("alive", {})
    income_params = alive_template.get("income", {})
    for name in _TAUCHEN_PARAMS:
        assert name not in income_params

    params = {"discount_factor": 1.0}
    V_arr_dict = model.solve(params, debug_mode=False)
    assert len(V_arr_dict) > 0
