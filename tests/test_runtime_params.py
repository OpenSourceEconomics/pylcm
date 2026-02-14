"""Tests for runtime-supplied grid points (IrregSpacedGrid)."""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.grids import IrregSpacedGrid
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


def _simple_model(*, wealth_grid=None):
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
        regime_id_class=RegimeId,
    )


# ======================================================================================
# Tests
# ======================================================================================


class TestRuntimeIrregSpacedGrid:
    def test_runtime_grid_creation(self):
        """IrregSpacedGrid without points, only n_points."""
        grid = IrregSpacedGrid(n_points=5)
        assert grid.pass_points_at_runtime
        assert grid.n_points == 5
        assert grid.to_jax().shape == (5,)  # placeholder zeros

    def test_fixed_grid_not_runtime(self):
        """IrregSpacedGrid with fixed points should not need runtime points."""
        grid = IrregSpacedGrid(points=[1.0, 2.0, 3.0])
        assert not grid.pass_points_at_runtime
        assert grid.n_points == 3

    def test_runtime_grid_validation(self):
        """Grid with runtime-supplied points requires n_points >= 2."""
        with pytest.raises(Exception, match="at least 2"):
            IrregSpacedGrid(n_points=1)

    def test_runtime_grid_requires_points_or_n_points(self):
        """Must specify either points or n_points."""
        with pytest.raises(Exception, match="Either points or n_points"):
            IrregSpacedGrid()

    def test_runtime_grid_in_params_template(self):
        """IrregSpacedGrid with runtime-supplied points adds 'points' to template."""
        model = _simple_model(
            wealth_grid=IrregSpacedGrid(n_points=5),
        )
        alive_template = model.params_template["alive"]
        assert "wealth" in alive_template
        assert "points" in alive_template["wealth"]

    def test_solve_with_runtime_grid(self):
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

    def test_runtime_grid_matches_fixed(self):
        """Runtime-supplied grid with same points should give same results as fixed."""
        points = jnp.linspace(1, 10, 5)

        # Fixed-points model
        model_fixed = _simple_model(
            wealth_grid=IrregSpacedGrid(points=list(points.tolist())),
        )
        params_fixed = {"discount_factor": 0.95, "interest_rate": 0.05}
        V_fixed = model_fixed.solve(params_fixed, debug_mode=False)

        # Runtime-points model
        model_runtime = _simple_model(
            wealth_grid=IrregSpacedGrid(n_points=5),
        )
        params_runtime = {
            "discount_factor": 0.95,
            "interest_rate": 0.05,
            "alive": {"wealth": {"points": points}},
        }
        V_runtime = model_runtime.solve(params_runtime, debug_mode=False)

        # V_arr for period 0, regime "alive" should match
        for period in V_fixed:
            if "alive" in V_fixed[period] and "alive" in V_runtime[period]:
                aaae(V_fixed[period]["alive"], V_runtime[period]["alive"])
