"""Tests for runtime-supplied ShockGrid parameters."""

from types import MappingProxyType

import jax.numpy as jnp

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


def _make_shock_model(*, runtime_rho=True, fixed_rho=False):
    """Create a shock model, optionally with rho supplied at runtime."""
    if runtime_rho:
        income_grid = ShockGrid(distribution_type="tauchen", n_points=3)
    else:
        income_grid = ShockGrid(
            distribution_type="tauchen",
            n_points=3,
            shock_params=MappingProxyType({"rho": 0.9}),
        )

    alive = Regime(
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
            "income": income_grid,
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

    fixed_params: dict = {}
    if fixed_rho:
        fixed_params = {"rho": 0.9}

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeIdShock,
        fixed_params=fixed_params,
    )


# ======================================================================================
# Tests
# ======================================================================================


class TestRuntimeShockGrid:
    def test_runtime_shock_params_property(self):
        """ShockGrid without rho should report rho as runtime-supplied."""
        grid = ShockGrid(distribution_type="tauchen", n_points=5)
        assert "rho" in grid.params_to_pass_at_runtime
        assert not grid.is_fully_specified

    def test_fully_specified_shock(self):
        """ShockGrid with all params should have no runtime-supplied params."""
        grid = ShockGrid(
            distribution_type="tauchen",
            n_points=5,
            shock_params=MappingProxyType({"rho": 0.9}),
        )
        assert grid.params_to_pass_at_runtime == ()
        assert grid.is_fully_specified

    def test_uniform_shock_fully_specified(self):
        """Uniform shock has no required params, so none are runtime-supplied."""
        grid = ShockGrid(distribution_type="uniform", n_points=5)
        assert grid.params_to_pass_at_runtime == ()
        assert grid.is_fully_specified

    def test_runtime_shock_in_params_template(self):
        """Runtime-supplied ShockGrid params appear in params_template."""
        model = _make_shock_model(runtime_rho=True)
        alive_template = model.params_template["alive"]
        assert "income" in alive_template
        assert "rho" in alive_template["income"]

    def test_solve_with_runtime_shock(self):
        """Solve should work with runtime-supplied shock params."""
        model = _make_shock_model(runtime_rho=True)
        params = {
            "discount_factor": 1.0,
            "rho": 0.9,
        }
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0

    def test_runtime_shock_with_fixed_rho(self):
        """Runtime-supplied shock with rho provided via fixed_params at model init."""
        model = _make_shock_model(runtime_rho=True, fixed_rho=True)

        # rho should be removed from template since it's fixed
        alive_template = model.params_template.get("alive", {})
        income_params = alive_template.get("income", {})
        assert "rho" not in income_params

        params = {"discount_factor": 1.0}
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0
