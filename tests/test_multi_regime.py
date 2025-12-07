"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model, Regime

if TYPE_CHECKING:
    from lcm.typing import (
        BoolND,
        ContinuousAction,
        ContinuousState,
        DiscreteState,
        FloatND,
        IntND,
        ParamsDict,
        RegimeName,
    )


# ======================================================================================
# Categorical Variables (shared across regimes)
# ======================================================================================


@dataclass
class WorkingStatus:
    not_working: int = 0
    working: int = 1


@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class RegimeID:
    work: int = 0
    retirement: int = 1


# ======================================================================================
# Model Functions (shared where possible)
# ======================================================================================


def utility_work(
    consumption: ContinuousAction,
    working: IntND,
    disutility_of_work: float,
    health: DiscreteState,
) -> FloatND:
    return jnp.log(consumption) - (1 - health / 2) * disutility_of_work * working


def utility_retirement(
    consumption: ContinuousAction,
    working: IntND,
    disutility_of_work: float,
    health: DiscreteState,  # noqa: ARG001
) -> FloatND:
    return jnp.log(consumption) - disutility_of_work * working


def labor_income(working: IntND, wage: float) -> FloatND:
    return working * wage


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


@lcm.mark.stochastic
def next_health(health: DiscreteState, health_transition: FloatND) -> FloatND:
    return health_transition[health]


def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    return consumption <= wealth


# ======================================================================================
# API Demonstration Tests
# ======================================================================================


@lcm.mark.stochastic
def next_regime_from_working(period: int) -> FloatND:
    """Return probability array [P(work), P(retirement)] indexed by RegimeID."""
    return jnp.array(
        [
            jnp.where(period < 6, 1.0, 0.5),  # P(work)
            jnp.where(period < 6, 0.0, 0.5),  # P(retirement)
        ]
    )


def next_regime_from_retirement() -> int:
    """Return deterministic next regime (always stay in retirement)."""
    return RegimeID.retirement


def working_during_retirement() -> IntND:
    return jnp.array(0)


def test_work_retirement_model_solution():
    """Test that a complete work-retirement model can be solved using new Regime API."""
    # Create work regime
    work_regime = Regime(
        name="work",
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
            "working": DiscreteGrid(WorkingStatus),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
            "health": DiscreteGrid(HealthStatus),
        },
        utility=utility_work,
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={
            "labor_income": labor_income,
        },
        transitions={
            "next_wealth": next_wealth,
            "next_health": next_health,
            "next_regime": next_regime_from_working,
        },
    )

    # Create retirement regime
    retirement_regime = Regime(
        name="retirement",
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
            "health": DiscreteGrid(HealthStatus),
        },
        utility=utility_retirement,
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={
            "working": working_during_retirement,  # Always not working in retirement
            "labor_income": labor_income,
        },
        transitions={
            "next_wealth": next_wealth,
            "next_health": next_health,
            "next_regime": next_regime_from_retirement,
        },
    )

    # Create complete model using new regime-based API
    model = Model(
        regimes=[work_regime, retirement_regime],
        n_periods=10,
        regime_id_cls=RegimeID,
    )

    # Verify model properties
    assert model.n_periods == 10
    assert len(model.internal_regimes) == 2

    health_transition = jnp.array(
        [
            # From bad health today to (bad, good) tomorrow
            [0.9, 0.1],
            # From good health today to (bad, good) tomorrow
            [0.5, 0.5],
        ],
    )

    # Define parameters
    params_working = {
        "beta": 0.9,
        "utility": {"disutility_of_work": 2.0},
        "labor_income": {"wage": 25},
        "next_wealth": {"interest_rate": 0.1},
        "next_health": {"health_transition": health_transition},
        "borrowing_constraint": {},
    }

    params_retired = {
        "beta": 0.8,
        "utility": {"disutility_of_work": 2.0},
        "labor_income": {"wage": 20},
        "next_wealth": {"interest_rate": 0.1},
        "next_health": {"health_transition": health_transition},
        "working": {},
        "borrowing_constraint": {},
    }

    params: dict[RegimeName, ParamsDict] = {
        "work": params_working,
        "retirement": params_retired,
    }

    # The core test: solve should work and return value functions
    solution = model.solve(params)
    simulation = model.simulate(
        params=params,
        initial_states={
            "work": {
                "wealth": jnp.array([5.0, 20, 40, 70]),
                "health": jnp.array([1, 1, 1, 1]),
            },
            "retirement": {
                "wealth": jnp.array([5.0, 20, 40, 70]),
                "health": jnp.array([1, 1, 1, 1]),
            },
        },
        initial_regimes=["work"] * 4,
        V_arr_dict=solution,
    )

    # Basic checks: solution should be a dict with one entry per period
    assert isinstance(solution, dict)
    assert len(solution) == 10
    assert all(period in solution for period in range(10))

    assert isinstance(simulation, dict)
    assert len(simulation) == 2


# ======================================================================================
# Test for regimes with different states (like alive/dead in Mahler & Yum)
# ======================================================================================


@dataclass
class AliveDeadRegimeID:
    alive: int = 0
    dead: int = 1


@dataclass
class DeadStatus:
    dead: int = 0


@lcm.mark.stochastic
def next_regime_from_alive(health: DiscreteState) -> FloatND:
    """Return probability array [P(alive), P(dead)] based on health."""
    survival_prob = jnp.where(health == 1, 0.95, 0.8)
    return jnp.array([survival_prob, 1 - survival_prob])


def next_regime_from_dead() -> int:
    """Deterministic: once dead, always dead."""
    return AliveDeadRegimeID.dead


def alive_borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Borrowing constraint: consumption must not exceed wealth."""
    return consumption <= wealth


def alive_utility(
    consumption: ContinuousAction,
    health: DiscreteState,
) -> FloatND:
    """Utility function for alive regime."""
    return jnp.log(consumption) * (1 + health)


def dead_utility(dead: DiscreteState) -> FloatND:  # noqa: ARG001
    """Utility function for dead regime (always zero)."""
    return jnp.asarray([0.0])


def next_wealth_fn(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    """State transition for wealth."""
    return wealth - consumption


def next_dead_fn(dead: DiscreteState) -> int:  # noqa: ARG001
    """State transition for dead (always stay dead)."""
    return DeadStatus.dead


def next_dead_from_alive() -> int:
    """Transition to dead state when entering dead regime from alive."""
    return DeadStatus.dead


def test_multi_regime_with_different_states():
    """Test that regimes with different states work correctly.

    This tests the pattern used in Mahler & Yum (2024) where:
    - alive regime has states: wealth, health
    - dead regime has states: dead

    The key issue this tests is that flat transitions should only apply
    to the current regime, not create entries for all regimes.
    """
    # Create alive regime with wealth and health states
    alive_regime = Regime(
        name="alive",
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(HealthStatus),
        },
        utility=alive_utility,
        constraints={"borrowing_constraint": alive_borrowing_constraint},
        transitions={
            "next_wealth": next_wealth_fn,
            "next_health": next_health,
            "next_dead": next_dead_from_alive,  # Required for transition to dead regime
            "next_regime": next_regime_from_alive,
        },
    )

    # Create dead regime with only dead state (absorbing - never leaves)
    dead_regime = Regime(
        name="dead",
        absorbing=True,  # Absorbing regime - auto-generates next_regime
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": next_dead_fn,
            # No next_regime needed - auto-generated for absorbing regimes
        },
    )

    # Create model
    model = Model(
        regimes=[alive_regime, dead_regime],
        n_periods=3,
        regime_id_cls=AliveDeadRegimeID,
    )

    # Verify model properties
    assert model.n_periods == 3
    assert len(model.internal_regimes) == 2

    # Check params template has correct keys for each regime
    alive_params_keys = set(model.params_template["alive"].keys())
    dead_params_keys = set(model.params_template["dead"].keys())

    # alive should have next_wealth, next_health, next_regime
    assert "next_wealth" in alive_params_keys
    assert "next_health" in alive_params_keys
    assert "next_regime" in alive_params_keys

    # dead should have next_dead, next_regime (NOT next_wealth, next_health)
    assert "next_dead" in dead_params_keys
    assert "next_regime" in dead_params_keys
    assert "next_wealth" not in dead_params_keys
    assert "next_health" not in dead_params_keys

    # Define parameters
    health_transition = jnp.array([[0.9, 0.1], [0.5, 0.5]])

    params = {
        "alive": {
            "beta": 0.9,
            "utility": {},
            "borrowing_constraint": {},
            "next_wealth": {},
            "next_health": {"health_transition": health_transition},
            "next_regime": {},
        },
        "dead": {
            "beta": 0.9,
            "utility": {},
            "next_dead": {},
            "next_regime": {},
        },
    }

    # Solve model - this should not raise KeyError for dead__health
    solution = model.solve(params)

    # Basic checks
    assert isinstance(solution, dict)
    assert len(solution) == 3
    assert all(period in solution for period in range(3))


# ======================================================================================
# Test for regimes with OVERLAPPING state-spaces (neither is subset of the other)
# ======================================================================================


@dataclass
class EducationLevel:
    low: int = 0
    high: int = 1


@dataclass
class PensionType:
    basic: int = 0
    premium: int = 1


@dataclass
class YoungOldRegimeID:
    young: int = 0
    old: int = 1


@lcm.mark.stochastic
def next_regime_young_to_old(education: DiscreteState) -> FloatND:
    """Probability of transitioning from young to old based on education."""
    # Higher education -> lower probability of aging (stay young longer)
    stay_young_prob = jnp.where(education == 1, 0.8, 0.6)
    return jnp.array([stay_young_prob, 1 - stay_young_prob])


def next_regime_stay_old() -> int:
    """Once old, always old (absorbing state)."""
    return YoungOldRegimeID.old


def young_utility(
    consumption: ContinuousAction,
    education: DiscreteState,
) -> FloatND:
    """Utility for young regime."""
    return jnp.log(consumption) * (1 + 0.5 * education)


def old_utility(
    consumption: ContinuousAction,
    pension: DiscreteState,
) -> FloatND:
    """Utility for old regime."""
    return jnp.log(consumption) * (1 + 0.3 * pension)


def young_borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Borrowing constraint for young."""
    return consumption <= wealth


def old_borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Borrowing constraint for old."""
    return consumption <= wealth


def next_wealth_young(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    """Wealth transition for young (can save)."""
    return wealth - consumption + 10  # Young earn income


def next_wealth_old(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    """Wealth transition for old (pension income)."""
    return wealth - consumption + 5  # Old get pension


def next_education(education: DiscreteState) -> DiscreteState:
    """Education stays constant."""
    return education


def next_pension(education: DiscreteState) -> DiscreteState:
    """Pension type determined by education when transitioning to old."""
    # High education -> premium pension, low education -> basic pension
    return education  # Maps education level to pension type


def test_multi_regime_with_overlapping_states():
    """Test regimes with overlapping but non-subset state-spaces.

    This tests a critical case where:
    - young regime has states: {wealth, education}
    - old regime has states: {wealth, pension}
    - Shared state: wealth
    - Unique to young: education
    - Unique to old: pension

    When young transitions to old, it needs:
    - next_wealth (shared state)
    - next_pension (old's unique state, derived from education)

    The flat transitions interface should automatically map:
    - next_wealth -> both young and old regimes (shared)
    - next_education -> young regime only
    - next_pension -> old regime only
    """
    # Young regime: has wealth (shared) and education (unique)
    young_regime = Regime(
        name="young",
        actions={
            "consumption": LinspaceGrid(start=1, stop=50, n_points=10),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "education": DiscreteGrid(EducationLevel),
        },
        utility=young_utility,
        constraints={"borrowing_constraint": young_borrowing_constraint},
        transitions={
            "next_wealth": next_wealth_young,
            "next_education": next_education,
            "next_pension": next_pension,  # Needed for transition to old!
            "next_regime": next_regime_young_to_old,
        },
    )

    # Old regime: has wealth (shared) and pension (unique)
    # Marked as absorbing since agents can't leave old regime
    old_regime = Regime(
        name="old",
        absorbing=True,  # Absorbing regime - can't transition back to young
        actions={
            "consumption": LinspaceGrid(start=1, stop=50, n_points=10),
        },
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "pension": DiscreteGrid(PensionType),
        },
        utility=old_utility,
        constraints={"borrowing_constraint": old_borrowing_constraint},
        transitions={
            "next_wealth": next_wealth_old,
            "next_pension": lambda pension: pension,  # Pension stays constant
            # No next_regime needed - auto-generated for absorbing regimes
        },
    )

    # Create model - this should work with correct flat->nested conversion
    model = Model(
        regimes=[young_regime, old_regime],
        n_periods=3,
        regime_id_cls=YoungOldRegimeID,
    )

    # Verify model properties
    assert model.n_periods == 3
    assert len(model.internal_regimes) == 2

    # Check params template structure
    young_params_keys = set(model.params_template["young"].keys())
    old_params_keys = set(model.params_template["old"].keys())

    # Young should have: next_wealth, next_education, next_pension, next_regime
    assert "next_wealth" in young_params_keys
    assert "next_education" in young_params_keys
    assert "next_pension" in young_params_keys  # For transition to old!
    assert "next_regime" in young_params_keys

    # Old should have: next_wealth, next_pension, next_regime
    assert "next_wealth" in old_params_keys
    assert "next_pension" in old_params_keys
    assert "next_regime" in old_params_keys
    # Old should NOT have next_education (not in old's state space)
    assert "next_education" not in old_params_keys

    # Define parameters
    params = {
        "young": {
            "beta": 0.95,
            "utility": {},
            "borrowing_constraint": {},
            "next_wealth": {},
            "next_education": {},
            "next_pension": {},
            "next_regime": {},
        },
        "old": {
            "beta": 0.95,
            "utility": {},
            "borrowing_constraint": {},
            "next_wealth": {},
            "next_pension": {},
            "next_regime": {},
        },
    }

    # Solve model - this will fail if flat->nested conversion is broken
    # because young regime needs next_pension mapped to old regime's state space
    solution = model.solve(params)

    # Basic checks
    assert isinstance(solution, dict)
    assert len(solution) == 3
    assert all(period in solution for period in range(3))


# ======================================================================================
# Tests for absorbing regimes and transition completeness validation
# ======================================================================================

# Note: AliveDeadRegimeID and DeadStatus are already defined above (around line 252)


def test_non_absorbing_regime_missing_transitions_raises_error():
    """Non-absorbing regime must have transitions for ALL states across ALL regimes.

    If alive regime can transition to dead regime, alive must have next_dead.
    """

    def alive_utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def dead_utility(dead: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    @lcm.mark.stochastic
    def next_regime_from_alive() -> FloatND:
        return jnp.array([0.9, 0.1])  # 90% stay alive, 10% die

    # Alive regime is MISSING next_dead!
    alive_regime = Regime(
        name="alive",
        utility=alive_utility,
        states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        constraints={"budget": lambda wealth, consumption: consumption <= wealth},
        transitions={
            "next_wealth": next_wealth,
            # MISSING: "next_dead": ...,  <-- This should cause an error!
            "next_regime": next_regime_from_alive,
        },
    )

    dead_regime = Regime(
        name="dead",
        absorbing=True,
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
        },
    )

    # This should raise an error because alive is missing next_dead
    with pytest.raises(ValueError, match="missing transitions"):
        Model(
            regimes=[alive_regime, dead_regime],
            n_periods=3,
            regime_id_cls=AliveDeadRegimeID,
        )


def test_missing_transitions_error_message_is_descriptive():
    """Error message should clearly state which transitions are missing."""

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    @lcm.mark.stochastic
    def next_regime() -> FloatND:
        return jnp.array([0.5, 0.5])

    # Regime A has states: wealth, health
    # Regime B has states: wealth, pension
    # Each regime is missing transitions for the other's unique state

    regime_a = Regime(
        name="regime_a",
        utility=utility,
        states={
            "wealth": LinspaceGrid(start=1, stop=10, n_points=5),
            "health": DiscreteGrid(HealthStatus),
        },
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        transitions={
            "next_wealth": next_wealth,
            "next_health": lambda health: health,
            # Note: next_pension is intentionally missing
            "next_regime": next_regime,
        },
    )

    regime_b = Regime(
        name="regime_b",
        utility=utility,
        states={
            "wealth": LinspaceGrid(start=1, stop=10, n_points=5),
            "pension": DiscreteGrid(HealthStatus),  # Reuse HealthStatus for simplicity
        },
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        transitions={
            "next_wealth": next_wealth,
            "next_pension": lambda pension: pension,
            # Note: next_health is intentionally missing
            "next_regime": next_regime,
        },
    )

    @dataclass
    class ABRegimeID:
        regime_a: int = 0
        regime_b: int = 1

    with pytest.raises(ValueError, match="next_pension") as exc_info:
        Model(
            regimes=[regime_a, regime_b],
            n_periods=3,
            regime_id_cls=ABRegimeID,
        )

    # Error should mention both regimes and their missing transitions
    error_msg = str(exc_info.value)
    assert "regime_a" in error_msg
    assert "next_pension" in error_msg
    assert "regime_b" in error_msg
    assert "next_health" in error_msg


def test_absorbing_regime_only_needs_own_state_transitions():
    """Absorbing regime should only require transitions for its own states.

    Dead regime with absorbing=True should NOT need next_wealth.
    """

    def alive_utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def dead_utility(dead: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    @lcm.mark.stochastic
    def next_regime_from_alive() -> FloatND:
        return jnp.array([0.9, 0.1])

    def borrowing_constraint(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> BoolND:
        return consumption <= wealth

    alive_regime = Regime(
        name="alive",
        utility=alive_utility,
        states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
            "next_regime": next_regime_from_alive,
        },
    )

    # Dead regime with absorbing=True - only needs next_dead, NOT next_wealth
    dead_regime = Regime(
        name="dead",
        absorbing=True,
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
            # No next_wealth needed because absorbing=True!
        },
    )

    # This should work without error
    model = Model(
        regimes=[alive_regime, dead_regime],
        n_periods=3,
        regime_id_cls=AliveDeadRegimeID,
    )

    assert model.n_periods == 3
    assert len(model.internal_regimes) == 2


def test_absorbing_regime_auto_generates_next_regime():
    """Absorbing regime without next_regime should auto-generate it."""

    def dead_utility(dead: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    # Dead regime without explicit next_regime
    dead_regime = Regime(
        name="dead",
        absorbing=True,
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
            # No next_regime - should be auto-generated!
        },
    )

    @dataclass
    class SingleDeadRegimeID:
        dead: int = 0

    # Single absorbing regime model should work
    model = Model(
        regimes=[dead_regime],
        n_periods=3,
        regime_id_cls=SingleDeadRegimeID,
    )

    assert model.n_periods == 3

    # Verify next_regime was auto-generated (returns 100% for dead)
    internal_dead = model.internal_regimes["dead"]
    # The regime_transition_probs function should exist
    assert internal_dead.regime_transition_probs is not None


def test_absorbing_regime_with_explicit_next_regime_warns():
    """Absorbing regime with explicit next_regime should warn (redundant)."""

    def alive_utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def dead_utility(dead: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array(0.0)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    @lcm.mark.stochastic
    def next_regime_from_alive() -> FloatND:
        return jnp.array([0.9, 0.1])  # 90% stay alive, 10% die

    @lcm.mark.stochastic
    def explicit_next_regime_from_dead() -> FloatND:
        return jnp.array([0.0, 1.0])  # 100% stay dead - redundant for absorbing!

    def borrowing_constraint(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> BoolND:
        return consumption <= wealth

    alive_regime = Regime(
        name="alive",
        utility=alive_utility,
        states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
            "next_regime": next_regime_from_alive,
        },
    )

    dead_regime = Regime(
        name="dead",
        absorbing=True,
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": lambda dead: DeadStatus.dead,  # noqa: ARG005
            "next_regime": explicit_next_regime_from_dead,  # Redundant for absorbing!
        },
    )

    # Should warn about redundant next_regime on absorbing regime
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Model(
            regimes=[alive_regime, dead_regime],
            n_periods=3,
            regime_id_cls=AliveDeadRegimeID,
        )
        # Check that a warning was issued about absorbing regime
        assert any("absorbing" in str(warning.message).lower() for warning in w)


def test_single_regime_model_treated_as_absorbing():
    """Single-regime models should be treated as absorbing internally."""

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    # Single regime without absorbing=True or next_regime
    single_regime = Regime(
        name="single",
        utility=utility,
        states={"wealth": LinspaceGrid(start=1, stop=10, n_points=5)},
        actions={"consumption": LinspaceGrid(start=1, stop=5, n_points=5)},
        transitions={
            "next_wealth": next_wealth,
            # No next_regime needed for single-regime model
        },
    )

    @dataclass
    class SingleRegimeID:
        single: int = 0

    # Should work without error
    model = Model(
        regimes=[single_regime],
        n_periods=3,
        regime_id_cls=SingleRegimeID,
    )

    assert model.n_periods == 3
    assert len(model.internal_regimes) == 1
