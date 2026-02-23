"""Tests for initial conditions validation utilities."""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, IrregSpacedGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidInitialConditionsError
from lcm.input_processing.params_processing import process_params
from lcm.simulation.util import (
    convert_initial_states_to_nested,
    validate_initial_conditions,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    InternalParams,
    ScalarInt,
)


@pytest.fixture
def model() -> Model:
    """Minimal model with two states (wealth, health) for initial states tests."""

    @dataclass
    class HealthStatus:
        healthy: int = 0
        sick: int = 1

    @dataclass
    class RegimeId:
        active: int = 0
        terminal: int = 1

    def utility(wealth: ContinuousState, health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_regime(period: int) -> ScalarInt:
        return jnp.where(
            period + 1 >= 2,
            RegimeId.terminal,
            RegimeId.active,
        )

    n_periods = 2
    ages = AgeGrid(start=0, stop=n_periods, step="Y")

    alive = Regime(
        functions={"utility": utility},
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=lambda wealth: wealth
            ),
            "health": DiscreteGrid(
                category_class=HealthStatus, transition=lambda health: health
            ),
        },
        transition=next_regime,
        active=lambda age: age < n_periods - 1,
    )

    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= n_periods - 1,
    )

    return Model(
        regimes={"active": alive, "terminal": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


@pytest.fixture
def internal_params(model: Model) -> InternalParams:
    """Process params for the minimal model."""
    return process_params(
        params={"discount_factor": 0.95}, params_template=model.params_template
    )


# ==============================================================================
# Tests
# ==============================================================================
def test_convert_flat_to_nested_single_regime(model: Model) -> None:
    """Single regime gets its states from flat dict."""
    flat = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    nested = convert_initial_states_to_nested(
        initial_states=flat, internal_regimes=model.internal_regimes
    )

    assert set(nested) == {"active", "terminal"}
    assert "wealth" in nested["active"]
    assert "health" in nested["active"]


def test_validate_initial_conditions_valid_input(
    model: Model, internal_params: InternalParams
) -> None:
    """Valid input should not raise."""
    flat = {
        "wealth": jnp.array([10.0, 50.0]),
        "health": jnp.array([0, 1]),
    }
    validate_initial_conditions(
        initial_states=flat,
        initial_regimes=["active", "active"],
        internal_regimes=model.internal_regimes,
        internal_params=internal_params,
    )


def test_validate_initial_conditions_missing_state(
    model: Model, internal_params: InternalParams
) -> None:
    """Missing state should raise InvalidInitialConditionsError."""
    flat = {"wealth": jnp.array([10.0, 50.0])}

    with pytest.raises(
        InvalidInitialConditionsError, match=r"Missing initial states: \['health'\].*"
    ):
        validate_initial_conditions(
            initial_states=flat,
            initial_regimes=["active", "active"],
            internal_regimes=model.internal_regimes,
            internal_params=internal_params,
        )


def test_validate_initial_conditions_extra_state(
    model: Model, internal_params: InternalParams
) -> None:
    """Extra state should raise InvalidInitialConditionsError."""
    flat = {
        "wealth": jnp.array([10.0]),
        "health": jnp.array([0]),
        "unknown": jnp.array([1.0]),
    }

    with pytest.raises(InvalidInitialConditionsError, match="Unknown initial states"):
        validate_initial_conditions(
            initial_states=flat,
            initial_regimes=["active"],
            internal_regimes=model.internal_regimes,
            internal_params=internal_params,
        )


def test_validate_initial_conditions_inconsistent_lengths(
    model: Model, internal_params: InternalParams
) -> None:
    """Arrays with different lengths should raise InvalidInitialConditionsError."""
    flat = {
        "wealth": jnp.array([10.0, 20.0]),
        "health": jnp.array([0]),
    }

    with pytest.raises(InvalidInitialConditionsError, match="same length"):
        validate_initial_conditions(
            initial_states=flat,
            initial_regimes=["active", "active"],
            internal_regimes=model.internal_regimes,
            internal_params=internal_params,
        )


def test_validate_initial_conditions_invalid_discrete_value(
    model: Model, internal_params: InternalParams
) -> None:
    """Invalid discrete state code should raise InvalidInitialConditionsError."""
    flat = {
        "wealth": jnp.array([10.0]),
        "health": jnp.array([5]),
    }
    with pytest.raises(InvalidInitialConditionsError, match=r"Invalid values.*health"):
        validate_initial_conditions(
            initial_states=flat,
            initial_regimes=["active"],
            internal_regimes=model.internal_regimes,
            internal_params=internal_params,
        )


def test_validate_initial_conditions_invalid_regime_name(
    model: Model, internal_params: InternalParams
) -> None:
    """Invalid regime name should raise InvalidInitialConditionsError."""
    with pytest.raises(InvalidInitialConditionsError, match="Invalid regime names"):
        validate_initial_conditions(
            initial_states={
                "wealth": jnp.array([10.0]),
                "health": jnp.array([0]),
            },
            initial_regimes=["nonexistent"],
            internal_regimes=model.internal_regimes,
            internal_params=internal_params,
        )


# ==============================================================================
# Reproducer for GitHub issue #64
# ==============================================================================

# The key distinction is between constraint feasibility and grid bounds:
#
#   consumption grid start = 0.5  (= effective constraint threshold on wealth)
#   wealth grid start = 2.0       (= grid minimum, extrapolation below is fine)
#
#   v_0 = 0.25 < 0.5 = constraint < v_1 = 1.0 < 2.0 = grid_min < v_2 = 5.0


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2.0


def _make_constraint_model(wealth_grid) -> Model:
    """Create a constraint model with the given wealth grid."""
    final_age = 1

    @categorical
    class RegimeId:
        working: int
        dead: int

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    def next_regime(age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.working)

    working_regime = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.5, stop=10, n_points=20),
        },
        states={"wealth": wealth_grid},
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions={"utility": utility},
        active=lambda age: age <= final_age,
    )
    dead_regime = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > final_age,
    )
    return Model(
        regimes={"working": working_regime, "dead": dead_regime},
        ages=AgeGrid(start=0, stop=final_age + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_infeasible_initial_states_detected():
    """Issue #64: wealth below constraint threshold makes all actions infeasible.

    wealth=0.25 < min consumption (0.5), so consumption <= wealth is always False.
    """
    model = _make_constraint_model(
        wealth_grid=LinSpacedGrid(
            start=2.0, stop=10, n_points=15, transition=_next_wealth
        )
    )
    params = {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": 1}},
    }
    with pytest.raises(InvalidInitialConditionsError):
        model.solve_and_simulate(
            params=params,
            initial_states={"wealth": jnp.array([0.25])},
            initial_regimes=["working"],
        )


def test_on_grid_state_but_combination_infeasible():
    """State ON the grid but constraint fails for ALL action combinations.

    wealth=0.3 is the grid minimum, but min consumption (0.5) > 0.3,
    so consumption <= wealth is always False.
    """
    model = _make_constraint_model(
        wealth_grid=LinSpacedGrid(
            start=0.3, stop=10, n_points=15, transition=_next_wealth
        )
    )
    params = {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": 1}},
    }
    with pytest.raises(InvalidInitialConditionsError):
        model.solve_and_simulate(
            params=params,
            initial_states={"wealth": jnp.array([0.3])},
            initial_regimes=["working"],
        )


def test_extrapolated_initial_states_accepted():
    """wealth=1.0 is above constraint threshold but below grid min — feasible."""
    model = _make_constraint_model(
        wealth_grid=LinSpacedGrid(
            start=2.0, stop=10, n_points=15, transition=_next_wealth
        )
    )
    params = {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": 1}},
    }
    model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([1.0])},
        initial_regimes=["working"],
    )


def test_on_grid_initial_states_accepted():
    """wealth=5.0 is above grid min — fully on grid, feasible."""
    model = _make_constraint_model(
        wealth_grid=LinSpacedGrid(
            start=2.0, stop=10, n_points=15, transition=_next_wealth
        )
    )
    params = {
        "discount_factor": 0.95,
        "working": {"next_regime": {"final_age_alive": 1}},
    }
    model.solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([5.0])},
        initial_regimes=["working"],
    )


def test_irreg_spaced_grid_with_runtime_points():
    """Feasibility check works when grid points are supplied at runtime via params."""
    model = _make_constraint_model(
        wealth_grid=IrregSpacedGrid(n_points=15, transition=_next_wealth)
    )
    params = {
        "discount_factor": 0.95,
        "working": {
            "wealth": {"points": jnp.linspace(0.3, 10, 15)},
            "next_regime": {"final_age_alive": 1},
        },
    }
    with pytest.raises(InvalidInitialConditionsError):
        model.solve_and_simulate(
            params=params,
            initial_states={"wealth": jnp.array([0.3])},
            initial_regimes=["working"],
        )
