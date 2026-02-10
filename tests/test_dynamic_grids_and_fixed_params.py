"""Tests for fixed_params partialling and dynamic IrregSpacedGrid / ShockGrid."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import lcm
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, ShockGrid, categorical
from lcm.grids import IrregSpacedGrid
from lcm.typing import ContinuousAction, ContinuousState, FloatND

# ======================================================================================
# Shared model setup helpers
# ======================================================================================


@categorical
class RegimeId:
    alive: int
    dead: int


def _simple_utility(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return jnp.log(consumption + 1) + 0.01 * wealth


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def _borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _simple_model(n_periods=3, *, wealth_grid=None, extra_fixed_params=None):
    """Create a simple 2-regime model for testing."""
    if wealth_grid is None:
        wealth_grid = LinSpacedGrid(start=1, stop=10, n_points=5)

    alive = Regime(
        functions={"utility": _simple_utility},
        states={"wealth": wealth_grid},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transitions={
            "next_wealth": _next_wealth,
            "next_regime": _next_regime,
        },
        active=lambda age, n=n_periods: age < n - 1,
    )
    dead = Regime(
        functions={"utility": lambda: 0.0},
        terminal=True,
        active=lambda age, n=n_periods: age >= n - 1,
    )

    fixed_params = extra_fixed_params or {}

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params,
    )


# ======================================================================================
# Feature 1: fixed_params partialling
# ======================================================================================


class TestFixedParamsPartialling:
    def test_fixed_param_removed_from_template(self):
        """Fixed params should disappear from params_template."""
        model = _simple_model(
            extra_fixed_params={"interest_rate": 0.05},
        )
        # interest_rate should NOT be in the template
        alive_template = model.params_template.get("alive", {})
        all_param_names = set()
        for fn_params in alive_template.values():
            all_param_names.update(fn_params.keys())
        assert "interest_rate" not in all_param_names
        # discount_factor should still be there
        assert "discount_factor" in all_param_names

    def test_solve_with_fewer_params(self):
        """Solve should work with only the non-fixed params."""
        model = _simple_model(
            extra_fixed_params={"interest_rate": 0.05},
        )
        # Should NOT need interest_rate in params, only discount_factor
        params = {"discount_factor": 0.95}
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0

    def test_solve_and_simulate_with_fixed_params(self):
        """Full solve and simulate with fixed params should produce valid results."""
        # Model without fixed_params
        model_full = _simple_model()
        params_full = {"discount_factor": 0.95, "interest_rate": 0.05}
        result_full = model_full.solve_and_simulate(
            params=params_full,
            initial_states={"wealth": jnp.array([5.0, 7.0])},
            initial_regimes=["alive", "alive"],
            debug_mode=False,
        )

        # Model with interest_rate as fixed param
        model_fixed = _simple_model(
            extra_fixed_params={"interest_rate": 0.05},
        )
        params_fixed = {"discount_factor": 0.95}
        result_fixed = model_fixed.solve_and_simulate(
            params=params_fixed,
            initial_states={"wealth": jnp.array([5.0, 7.0])},
            initial_regimes=["alive", "alive"],
            debug_mode=False,
        )

        # Results should be identical
        df_full = result_full.to_dataframe()
        df_fixed = result_fixed.to_dataframe()
        aaae(df_full["wealth"].values, df_fixed["wealth"].values)
        aaae(df_full["consumption"].values, df_fixed["consumption"].values)

    def test_regime_level_fixed_param(self):
        """Fixed params at regime level should work."""
        model = _simple_model(
            extra_fixed_params={"alive": {"interest_rate": 0.05}},
        )
        # interest_rate should be removed from alive's template
        alive_template = model.params_template.get("alive", {})
        all_param_names = set()
        for fn_params in alive_template.values():
            all_param_names.update(fn_params.keys())
        assert "interest_rate" not in all_param_names

        params = {"discount_factor": 0.95}
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0

    def test_all_params_fixed(self):
        """All params can be fixed, leaving an empty template."""
        model = _simple_model(
            extra_fixed_params={"interest_rate": 0.05, "discount_factor": 0.95},
        )
        # All regime templates should be empty
        for regime_template in model.params_template.values():
            assert len(regime_template) == 0

        # Solve with empty params
        V_arr_dict = model.solve({}, debug_mode=False)
        assert len(V_arr_dict) > 0


# ======================================================================================
# Feature 2: Dynamic IrregSpacedGrid
# ======================================================================================


class TestDynamicIrregSpacedGrid:
    def test_dynamic_grid_creation(self):
        """Dynamic IrregSpacedGrid without points, only n_points."""
        grid = IrregSpacedGrid(n_points=5)
        assert grid.is_dynamic
        assert grid.n_points == 5
        assert grid.to_jax().shape == (5,)  # placeholder zeros

    def test_static_grid_not_dynamic(self):
        """Static IrregSpacedGrid with points should not be dynamic."""
        grid = IrregSpacedGrid(points=[1.0, 2.0, 3.0])
        assert not grid.is_dynamic
        assert grid.n_points == 3

    def test_dynamic_grid_validation(self):
        """Dynamic grid requires n_points >= 2."""
        with pytest.raises(Exception, match="at least 2"):
            IrregSpacedGrid(n_points=1)

    def test_dynamic_grid_requires_points_or_n_points(self):
        """Must specify either points or n_points."""
        with pytest.raises(Exception, match="Either points or n_points"):
            IrregSpacedGrid()

    def test_dynamic_grid_in_params_template(self):
        """Dynamic IrregSpacedGrid adds 'points' entry to params_template."""
        model = _simple_model(
            wealth_grid=IrregSpacedGrid(n_points=5),
        )
        alive_template = model.params_template["alive"]
        assert "wealth" in alive_template
        assert "points" in alive_template["wealth"]

    def test_solve_with_dynamic_grid(self):
        """Solve should work when grid points are provided via params."""
        model = _simple_model(
            wealth_grid=IrregSpacedGrid(n_points=5),
        )
        params = {
            "discount_factor": 0.95,
            "interest_rate": 0.05,
            "alive": {"wealth": {"points": jnp.linspace(1, 10, 5)}},
        }
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0

    def test_dynamic_grid_matches_static(self):
        """Dynamic grid with same points should give same results as static."""
        points = jnp.linspace(1, 10, 5)

        # Static model
        model_static = _simple_model(
            wealth_grid=IrregSpacedGrid(points=list(points.tolist())),
        )
        params_static = {"discount_factor": 0.95, "interest_rate": 0.05}
        V_static = model_static.solve(params_static, debug_mode=False)

        # Dynamic model
        model_dynamic = _simple_model(
            wealth_grid=IrregSpacedGrid(n_points=5),
        )
        params_dynamic = {
            "discount_factor": 0.95,
            "interest_rate": 0.05,
            "alive": {"wealth": {"points": points}},
        }
        V_dynamic = model_dynamic.solve(params_dynamic, debug_mode=False)

        # V_arr for period 0, regime "alive" should match
        for period in V_static:
            if "alive" in V_static[period] and "alive" in V_dynamic[period]:
                aaae(V_static[period]["alive"], V_dynamic[period]["alive"])


# ======================================================================================
# Feature 3: Dynamic ShockGrid params
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


def _make_shock_model(*, dynamic=True, fixed_rho=False):
    """Create a shock model, optionally with dynamic rho."""
    if dynamic:
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
    if not dynamic:
        fixed_params = {"income": {"rho": 0.9}}
    elif fixed_rho:
        fixed_params = {"rho": 0.9}

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeIdShock,
        fixed_params=fixed_params,
    )


class TestDynamicShockGrid:
    def test_dynamic_shock_params_property(self):
        """ShockGrid without rho should report rho as dynamic."""
        grid = ShockGrid(distribution_type="tauchen", n_points=5)
        assert "rho" in grid.dynamic_shock_params
        assert not grid.is_fully_specified

    def test_fully_specified_shock(self):
        """ShockGrid with all params should have no dynamic params."""
        grid = ShockGrid(
            distribution_type="tauchen",
            n_points=5,
            shock_params=MappingProxyType({"rho": 0.9}),
        )
        assert grid.dynamic_shock_params == ()
        assert grid.is_fully_specified

    def test_uniform_shock_no_dynamic(self):
        """Uniform shock has no required params, so no dynamic params."""
        grid = ShockGrid(distribution_type="uniform", n_points=5)
        assert grid.dynamic_shock_params == ()
        assert grid.is_fully_specified

    def test_dynamic_shock_in_params_template(self):
        """Dynamic ShockGrid params appear in params_template."""
        model = _make_shock_model(dynamic=True)
        alive_template = model.params_template["alive"]
        assert "income" in alive_template
        assert "rho" in alive_template["income"]

    def test_solve_with_dynamic_shock(self):
        """Solve should work with dynamic shock params provided at solve time."""
        model = _make_shock_model(dynamic=True)
        params = {
            "discount_factor": 1.0,
            "rho": 0.9,
        }
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0

    def test_dynamic_shock_with_fixed_rho(self):
        """Dynamic shock with rho provided via fixed_params at model init."""
        model = _make_shock_model(dynamic=True, fixed_rho=True)

        # rho should be removed from template since it's fixed
        alive_template = model.params_template.get("alive", {})
        income_params = alive_template.get("income", {})
        assert "rho" not in income_params

        params = {"discount_factor": 1.0}
        V_arr_dict = model.solve(params, debug_mode=False)
        assert len(V_arr_dict) > 0
