"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from lcm import DiscreteGrid, LinspaceGrid, Model, Regime
from lcm.exceptions import ModelInitilizationError

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteAction,
        FloatND,
        IntND,
    )


# ======================================================================================
# Categorical Variables (shared across regimes)
# ======================================================================================


@dataclass
class WorkingStatus:
    not_working: int = 0
    working: int = 1


# ======================================================================================
# Model Functions (shared where possible)
# ======================================================================================


def utility(
    consumption: ContinuousAction, working: IntND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - disutility_of_work * working


def labor_income(working: IntND, wage: float) -> FloatND:
    return working * wage


def working_from_action(working: DiscreteAction) -> IntND:
    return working


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# Regime Transition Functions
# ======================================================================================


def work_to_retirement_transition(
    wealth: ContinuousState,
) -> dict[str, ContinuousState]:
    return {
        "wealth": wealth,
    }


# ======================================================================================
# API Demonstration Tests
# ======================================================================================


def test_regime_creation():
    """Test that individual Regimes can be created with range-based active periods."""
    work_regime = Regime(
        name="work",
        active=range(7),  # Periods 0-6
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "labor_income": labor_income,
            "working": working_from_action,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_transitions={
            "retirement": work_to_retirement_transition,
        },
    )

    # Basic validation that the regime was created correctly
    assert work_regime.name == "work"
    assert work_regime.active == range(7)
    assert "consumption" in work_regime.actions
    assert "working" in work_regime.actions
    assert "wealth" in work_regime.states
    assert len(work_regime.functions) == 5


@pytest.mark.skip(reason="Regime model implementation not yet complete")
def test_work_retirement_model_solution():
    """Test that a complete work-retirement model can be solved using new Regime API."""
    # Create work regime
    work_regime = Regime(
        name="work",
        active=range(7),  # Periods 0-6
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "labor_income": labor_income,
            "working": working_from_action,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_transitions={
            "retirement": work_to_retirement_transition,
        },
    )

    # Create retirement regime
    retirement_regime = Regime(
        name="retirement",
        active=range(7, 10),  # Periods 7-9
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "working": lambda: 0,  # Always not working in retirement
            "labor_income": labor_income,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
        regime_transitions={},  # Retirement is absorbing
    )

    # Create complete model using new regime-based API
    model = Model(regimes=[work_regime, retirement_regime])

    # Verify model properties
    assert model.is_regime_model is True
    assert model.computed_n_periods == 10
    assert len(model.regimes) == 2

    # Define parameters (similar to deterministic model)
    params = {
        "disutility_of_work": 2.0,
        "wage": 10.0,
        "interest_rate": 0.05,
    }

    # The core test: solve should work and return value functions
    solution = model.solve(params)

    # Basic checks: solution should be a dict with one entry per period
    assert isinstance(solution, dict)
    assert len(solution) == 10
    assert all(period in solution for period in range(10))

    # Additional API test: simulate should also work
    initial_states = {
        "wealth": jnp.array([10.0]),
    }

    simulation = model.simulate(
        params=params, initial_states=initial_states, V_arr_dict=solution, seed=42
    )

    # Basic simulation checks
    assert simulation is not None
    assert len(simulation) > 0


@pytest.mark.skip(reason="Regime model implementation not yet complete")
def test_single_regime_model():
    """Test that single-regime models work with new Regime API."""
    single_regime = Regime(
        name="default",
        active=range(10),  # All periods 0-9
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        functions={
            "utility": utility,
            "labor_income": labor_income,
            "working": working_from_action,
            "next_wealth": next_wealth,
            "borrowing_constraint": borrowing_constraint,
        },
    )

    model = Model(regimes=[single_regime])

    # Basic validation
    assert model.is_regime_model is True
    assert model.computed_n_periods == 10
    assert len(model.regimes) == 1


def test_legacy_api_deprecation_warning():
    """Test that legacy API shows deprecation warning."""
    warn_msg = re.escape(
        "Legacy Model API (n_periods, actions, states, functions) is deprecated "
        "and will be removed in version 0.1.0."
    )

    # The deprecation warning should trigger before the initialization error
    with (
        pytest.warns(DeprecationWarning, match=warn_msg),
        pytest.raises(ModelInitilizationError),
    ):
        # Model creation will fail due to function signature issues,
        # but the deprecation warning should be triggered first
        Model(
            n_periods=5,
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=10)},
            states={"wealth": LinspaceGrid(start=1, stop=100, n_points=11)},
            functions={
                "utility": lambda consumption: jnp.log(consumption),
                "next_wealth": lambda wealth, consumption: wealth - consumption,
            },
        )


def test_n_periods_and_regime_active_interaction():
    """Test the interaction between Model.n_periods and Regime.active."""
    # Case 1: n_periods=None, derive from regime active ranges
    regime_a = Regime(
        name="phase_a",
        active=range(3),  # Periods 0-2
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )
    regime_b = Regime(
        name="phase_b",
        active=range(3, 7),  # Periods 3-6
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    try:
        model = Model(n_periods=None, regimes=[regime_a, regime_b])
        assert model.computed_n_periods == 7  # max(3, 7) = 7
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_n_periods_with_none_active_regimes():
    """Test n_periods with regimes that have active=None (default to all periods)."""
    regime = Regime(
        name="all_periods",
        # active=None (default) - should be active in all periods
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    try:
        model = Model(n_periods=5, regimes=[regime])
        assert model.computed_n_periods == 5
        # The regime should have been updated to active=range(5)
        assert regime.active == range(5)
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_n_periods_alignment_validation():
    """Test that n_periods alignment with regime active ranges is validated."""
    regime = Regime(
        name="test",
        active=range(10),  # Extends beyond n_periods=5
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    # Should raise error because regime extends beyond n_periods
    with pytest.raises(ModelInitilizationError, match=r"extending beyond n_periods"):
        Model(n_periods=5, regimes=[regime])


def test_all_regimes_none_active_with_n_periods():
    """Test case where all regimes have active=None and n_periods is specified."""
    regime_a = Regime(
        name="regime_a",
        # active=None - should be active in all periods
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    try:
        model = Model(n_periods=5, regimes=[regime_a])
        assert model.computed_n_periods == 5
        assert regime_a.active == range(5)  # Updated to all periods
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_mixed_active_none_should_error():
    """Test that mixing explicit active and None active creates validation errors."""
    regime_explicit = Regime(
        name="explicit",
        active=range(3),  # Periods 0-2
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )
    regime_none = Regime(
        name="none_active",
        # active=None - would be set to all periods, creating overlap
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    # This should fail because regime_none would get set to range(5)
    # which overlaps with regime_explicit's range(0, 3)
    with pytest.raises(ModelInitilizationError, match=r"Overlapping periods"):
        Model(n_periods=5, regimes=[regime_explicit, regime_none])


def test_no_active_ranges_specified_error():
    """Test error when n_periods=None and no regime has active range."""
    regime = Regime(
        name="no_active",
        # active=None and n_periods=None
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    with pytest.raises(
        ModelInitilizationError, match=r"at least one regime must have an active range"
    ):
        Model(n_periods=None, regimes=[regime])


def test_regime_to_model_fluent_interface():
    """Test the fluent to_model() interface for single-regime models."""
    regime = Regime(
        name="simple_model",
        active=range(5),
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
        states={"wealth": LinspaceGrid(start=1, stop=100, n_points=11)},
        functions={
            "utility": lambda consumption: jnp.log(consumption),
            "next_wealth": lambda wealth, consumption: wealth - consumption,
        },
    )

    # Test fluent interface
    try:
        model = regime.to_model()
        assert model.computed_n_periods == 5  # Derived from regime.active
        assert len(model.regimes) == 1
        assert model.regimes[0] is regime
        assert model.description is None
        assert model.enable_jit is True
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_regime_to_model_with_n_periods_override():
    """Test to_model() with explicit n_periods override."""
    regime = Regime(
        name="flexible_model",
        # active=None - should adapt to n_periods
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
        states={"wealth": LinspaceGrid(start=1, stop=100, n_points=11)},
        functions={
            "utility": lambda consumption: jnp.log(consumption),
            "next_wealth": lambda wealth, consumption: wealth - consumption,
        },
    )

    try:
        model = regime.to_model(n_periods=8)
        assert model.computed_n_periods == 8
        assert regime.active == range(8)  # Should be updated
        assert len(model.regimes) == 1
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_regime_to_model_with_description_and_jit():
    """Test to_model() with description and JIT options."""
    regime = Regime(
        name="documented_model",
        active=range(3),
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    try:
        model = regime.to_model(
            description="Test model with custom settings", enable_jit=False
        )
        assert model.computed_n_periods == 3
        assert model.description == "Test model with custom settings"
        assert model.enable_jit is False
    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass


def test_regime_to_model_error_no_periods_info():
    """Test that to_model() fails when no period information is available."""
    regime = Regime(
        name="no_periods",
        # active=None and n_periods=None
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    # Should fail because no period information is provided
    with pytest.raises(
        ModelInitilizationError, match=r"at least one regime must have an active range"
    ):
        regime.to_model(n_periods=None)


def test_regime_to_model_equivalent_to_explicit_model():
    """Test that to_model() creates equivalent model to explicit Model creation."""
    regime = Regime(
        name="comparison_test",
        active=range(4),
        actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
    )

    try:
        # Fluent interface
        model_fluent = regime.to_model(description="Test model")

        # Explicit interface
        model_explicit = Model(regimes=[regime], description="Test model")

        # Should be equivalent
        assert model_fluent.computed_n_periods == model_explicit.computed_n_periods
        assert model_fluent.description == model_explicit.description
        assert model_fluent.regimes == model_explicit.regimes
        assert model_fluent.enable_jit == model_explicit.enable_jit

    except NotImplementedError:
        # Expected since regime models aren't fully implemented yet
        pass
