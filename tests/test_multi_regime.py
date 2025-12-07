"""Test Model Regimes functionality using work-retirement example.

This test file demonstrates the new Regime-based API for PyLCM and serves as
the target specification for the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

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
            "next_regime": next_regime_from_alive,
        },
    )

    # Create dead regime with only dead state
    dead_regime = Regime(
        name="dead",
        utility=dead_utility,
        states={"dead": DiscreteGrid(DeadStatus)},
        actions={},
        transitions={
            "next_dead": next_dead_fn,
            "next_regime": next_regime_from_dead,
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
    old_regime = Regime(
        name="old",
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
            "next_regime": next_regime_stay_old,
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
