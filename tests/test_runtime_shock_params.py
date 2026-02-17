"""Tests for runtime-supplied ShockGrid parameters."""

import jax.numpy as jnp
import pytest

import lcm.shocks.ar1
import lcm.shocks.iid
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import ContinuousAction, ContinuousState, FloatND

# ======================================================================================
# Model setup helpers
# ======================================================================================


@categorical
class RegimeIdShock:
    alive: int
    dead: int


def _utility(
    wealth: ContinuousState,
    income: ContinuousState,
    consumption: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) + 0.01 * (wealth + jnp.exp(income))


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    return wealth - consumption


def _constraint(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return consumption <= wealth


_TAUCHEN_PARAMS = {"rho": 0.9, "sigma": 1.0, "mu": 0.0, "n_std": 2}


def _make_model(*, fixed_params=None):
    """Create a shock model with all shock params supplied at runtime."""
    alive = Regime(
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=10, n_points=5, transition=_next_wealth
            ),
            "income": lcm.shocks.ar1.Tauchen(n_points=3),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4)},
        functions={"utility": _utility},
        constraints={"borrowing": _constraint},
        transition=lambda period: jnp.where(
            period >= 1, RegimeIdShock.dead, RegimeIdShock.alive
        ),
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
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
    """Tauchen without params reports all params as runtime-supplied."""
    grid = lcm.shocks.ar1.Tauchen(n_points=5)
    for name in ("rho", "sigma", "mu", "n_std"):
        assert name in grid.params_to_pass_at_runtime
    assert not grid.is_fully_specified


def test_fully_specified_shock():
    """Tauchen with all params should have no runtime-supplied params."""
    grid = lcm.shocks.ar1.Tauchen(n_points=5, **_TAUCHEN_PARAMS)
    assert grid.params_to_pass_at_runtime == ()
    assert grid.is_fully_specified


@pytest.mark.parametrize("grid_cls", [lcm.shocks.iid.Uniform, lcm.shocks.iid.Normal])
def test_shock_without_params_is_not_fully_specified(grid_cls):
    """All distributions require explicit params — nothing is defaulted."""
    grid = grid_cls(n_points=5)
    assert not grid.is_fully_specified
    assert grid.params_to_pass_at_runtime


def test_runtime_shock_in_params_template():
    """Runtime-supplied ShockGrid params appear in params_template."""
    model = _make_model()
    alive_template = model.params_template["alive"]
    assert "income" in alive_template
    for name in ("rho", "sigma", "mu", "n_std"):
        assert name in alive_template["income"]


def test_solve_with_runtime_shock():
    """Solve should work with runtime-supplied shock params."""
    model = _make_model()
    params = {"discount_factor": 1.0, **_TAUCHEN_PARAMS}
    V_arr_dict = model.solve(params, debug_mode=False)
    assert len(V_arr_dict) > 0


def test_runtime_shock_with_fixed_params():
    """Shock params provided via fixed_params are removed from template."""
    model = _make_model(fixed_params=_TAUCHEN_PARAMS)

    alive_template = model.params_template.get("alive", {})
    income_params = alive_template.get("income", {})
    for name in _TAUCHEN_PARAMS:
        assert name not in income_params

    params = {"discount_factor": 1.0}
    V_arr_dict = model.solve(params, debug_mode=False)
    assert len(V_arr_dict) > 0
