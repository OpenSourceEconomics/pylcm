"""Tests for static params (fixed_params partialled at model initialization)."""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousAction, ContinuousState, FloatND

# ======================================================================================
# Model setup helpers
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


def _simple_model(n_periods=3, *, extra_fixed_params=None):
    """Create a simple 2-regime model for testing."""
    alive = Regime(
        functions={"utility": _simple_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
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
# Tests
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
