"""Tests for runtime-supplied grid points (IrregSpacedGrid)."""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.grids import IrregSpacedGrid
from lcm.typing import ContinuousAction, ContinuousState, FloatND


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _utility(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
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


def _make_model(*, wealth_grid=None):
    """Create a simple 2-regime model for testing."""
    if wealth_grid is None:
        wealth_grid = LinSpacedGrid(start=1, stop=10, n_points=5)

    alive = Regime(
        functions={"utility": _utility},
        states={"wealth": wealth_grid},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
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
        regime_id_class=RegimeId,
    )


def test_runtime_grid_creation():
    """IrregSpacedGrid without points, only n_points."""
    grid = IrregSpacedGrid(n_points=5)
    assert grid.pass_points_at_runtime
    assert grid.n_points == 5


def test_runtime_grid_to_jax_raises_before_substitution():
    """`to_jax()` on a runtime grid is a bug — substitution happens at
    solve/simulate time via `InternalRegime.state_action_space(regime_params=...)`.
    The error message must point the caller at that API."""
    grid = IrregSpacedGrid(n_points=5)
    with pytest.raises(Exception, match="state_action_space"):
        grid.to_jax()


def test_fixed_grid_not_runtime():
    """IrregSpacedGrid with fixed points should not need runtime points."""
    grid = IrregSpacedGrid(points=[1.0, 2.0, 3.0])
    assert not grid.pass_points_at_runtime
    assert grid.n_points == 3


def test_runtime_grid_validation():
    """Grid with runtime-supplied points requires n_points >= 2."""
    with pytest.raises(Exception, match="at least 2"):
        IrregSpacedGrid(n_points=1)


def test_runtime_grid_requires_points_or_n_points():
    """Must specify either points or n_points."""
    with pytest.raises(Exception, match="Either points or n_points"):
        IrregSpacedGrid()


def test_runtime_grid_in_params_template():
    """IrregSpacedGrid with runtime-supplied points adds 'points' to template."""
    model = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    alive_template = model._params_template["alive"]
    assert "wealth" in alive_template
    assert "points" in alive_template["wealth"]


def test_solve_with_runtime_grid():
    """Solve should work when grid points are provided via params."""
    model = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"wealth": {"points": jnp.linspace(1, 10, 5)}},
    }
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


def test_runtime_grid_matches_fixed():
    """Runtime-supplied grid with same points should give same results as fixed."""
    points = jnp.linspace(1, 10, 5)

    # Fixed-points model
    model_fixed = _make_model(
        wealth_grid=IrregSpacedGrid(points=list(points.tolist())),
    )
    params_fixed = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_fixed = model_fixed.solve(params=params_fixed, log_level="off")

    # Runtime-points model
    model_runtime = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    params_runtime = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"wealth": {"points": points}},
    }
    V_runtime = model_runtime.solve(params=params_runtime, log_level="off")

    # V_arr for period 0, regime "alive" should match
    for period in V_fixed:
        if "alive" in V_fixed[period] and "alive" in V_runtime[period]:
            aaae(V_fixed[period]["alive"], V_runtime[period]["alive"])


def _make_action_grid_model(*, consumption_grid: IrregSpacedGrid) -> Model:
    """Create a 2-regime model where consumption is the runtime-points action grid."""
    alive = Regime(
        functions={"utility": _utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": consumption_grid},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
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
        regime_id_class=RegimeId,
    )


def test_runtime_action_grid_in_params_template():
    """IrregSpacedGrid action with runtime-supplied points adds 'points' to template."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    alive_template = model._params_template["alive"]
    assert "consumption" in alive_template
    assert "points" in alive_template["consumption"]


def test_solve_with_runtime_action_grid():
    """Solve should work when action grid points are provided via params."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}},
    }
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


def test_runtime_action_grid_matches_fixed():
    """Runtime action grid with same points gives same V as a fixed action grid."""
    points = jnp.linspace(0.1, 5.0, 5)

    model_fixed = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(points=list(points.tolist())),
    )
    params_fixed = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_fixed = model_fixed.solve(params=params_fixed, log_level="off")

    model_runtime = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params_runtime = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": points}},
    }
    V_runtime = model_runtime.solve(params=params_runtime, log_level="off")

    for period in V_fixed:
        if "alive" in V_fixed[period] and "alive" in V_runtime[period]:
            aaae(V_fixed[period]["alive"], V_runtime[period]["alive"])


def test_runtime_action_grid_changes_solution():
    """Different runtime action points should yield different V (sanity check)."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    base = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_low = model.solve(
        params=base | {"alive": {"consumption": {"points": jnp.linspace(0.1, 1.0, 5)}}},
        log_level="off",
    )
    V_high = model.solve(
        params=base | {"alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}}},
        log_level="off",
    )
    # Period 0 alive value should differ when the action support differs
    assert not jnp.allclose(V_low[0]["alive"], V_high[0]["alive"])


def _make_action_grid_model_with_stateful_dead(
    *, consumption_grid: IrregSpacedGrid
) -> Model:
    """Variant where `dead` has a `wealth` state so its utility depends on it.

    Used to surface NaN propagation when the simulate path forgets to
    substitute runtime-supplied action gridpoints.
    """

    def _alive_utility(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> FloatND:
        return jnp.log(consumption + 1) + 0.01 * wealth

    def _dead_utility(wealth: ContinuousState) -> FloatND:
        return jnp.log(wealth + 1)

    alive = Regime(
        functions={"utility": _alive_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={
            "wealth": {
                "alive": _next_wealth,
                "dead": _next_wealth,
            },
        },
        actions={"consumption": consumption_grid},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": _dead_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        active=lambda _age: True,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def test_simulate_with_runtime_action_grid_no_nan() -> None:
    """Simulate must substitute runtime-supplied action gridpoints into the
    state-action space; otherwise the action grid is filled with NaN
    placeholders, optimal_actions become NaN, next_states propagate NaN to
    the dead regime, and `validate_V` raises.
    """
    model = _make_action_grid_model_with_stateful_dead(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}},
    }
    initial_conditions = {
        "regime": jnp.array([RegimeId.alive, RegimeId.alive, RegimeId.alive]),
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([2.0, 5.0, 9.0]),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        check_initial_conditions=False,
    )
    df = result.to_dataframe()
    assert not df["value"].isna().any()
