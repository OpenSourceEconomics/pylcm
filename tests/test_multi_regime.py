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
    )


# ======================================================================================
# Base Model Components
# ======================================================================================


@dataclass
class RegimeId:
    """Regime identifiers for work-retirement model."""

    work: int = 0
    retirement: int = 1


@dataclass
class HealthStatus:
    """Health states used in the model."""

    bad: int = 0
    good: int = 1


@dataclass
class WorkingStatus:
    """Working decision (action in work regime)."""

    not_working: int = 0
    working: int = 1


# --------------------------------------------------------------------------------------
# Shared Functions
# --------------------------------------------------------------------------------------


def utility(
    consumption: ContinuousAction,
    working: IntND,
    health: DiscreteState,
    disutility_of_work: float,
) -> FloatND:
    """Utility from consumption, reduced by disutility of work."""
    return jnp.log(consumption) - disutility_of_work * working * (1 - health / 2)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    working: IntND,
    wage: float,
    interest_rate: float,
) -> ContinuousState:
    """Wealth transition: savings plus labor income."""
    return (1 + interest_rate) * (wealth - consumption) + working * wage


def next_health() -> FloatND:
    """Stochastic health transition."""
    pass


def borrowing_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    """Cannot consume more than wealth."""
    return consumption <= wealth


@lcm.mark.stochastic
def next_regime_from_work(period: int) -> FloatND:
    """Stochastic transition: probability of retiring increases over time."""
    retire_prob = jnp.where(period < 3, 0.0, 0.5)
    return jnp.array([1 - retire_prob, retire_prob])


def retired_working() -> IntND:
    """Retired agents don't work."""
    return jnp.array(WorkingStatus.not_working)


# --------------------------------------------------------------------------------------
# Factory Function
# --------------------------------------------------------------------------------------


def create_base_regimes() -> tuple[Regime, Regime]:
    """Create work and retirement regimes for the base model."""
    work_regime = Regime(
        name="work",
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(HealthStatus),
        },
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
            "working": DiscreteGrid(WorkingStatus),
        },
        utility=utility,
        constraints={"borrowing_constraint": borrowing_constraint},
        transitions={
            "next_wealth": next_wealth,
            "next_health": lcm.mark.stochastic(next_health, type="uniform"),
            "next_regime": next_regime_from_work,
        },
    )

    retirement_regime = Regime(
        name="retirement",
        absorbing=True,
        states={
            "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
            "health": DiscreteGrid(HealthStatus),
        },
        actions={
            "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
        },
        utility=utility,
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={
            "working": retired_working,
        },
        transitions={
            "next_wealth": next_wealth,
            "next_health": lcm.mark.stochastic(next_health, type="uniform")
        },
    )

    return work_regime, retirement_regime


def create_base_params() -> ParamsDict:
    """Create parameters for the base work-retirement model."""
    health_transition = jnp.array(
        [
            [0.8, 0.2],  # From bad health: 80% stay bad, 20% become good
            [0.3, 0.7],  # From good health: 30% become bad, 70% stay good
        ]
    )

    return {
        "work": {
            "beta": 0.95,
            "utility": {"disutility_of_work": 0.5},
            "borrowing_constraint": {},
            "next_wealth": {"wage": 20.0, "interest_rate": 0.05},
            "next_health": {"n": 2,  "x_min": 0, "x_max": 1},
            "next_regime": {},
        },
        "retirement": {
            "beta": 0.95,
            "utility": {"disutility_of_work": 0.5},
            "borrowing_constraint": {},
            "working": {},
            "next_wealth": {"wage": 20.0, "interest_rate": 0.05},
            "next_health": {"n": 2,  "x_min": 0, "x_max": 1},
            "next_regime": {},
        },
    }


# ======================================================================================
# Test Cases
# ======================================================================================


class TestBasicModel:
    """Test the base work-retirement model."""

    def test_solve(self):
        """Base model can be solved."""
        work_regime, retirement_regime = create_base_regimes()
        model = Model(
            regimes=[work_regime, retirement_regime],
            n_periods=3,
            regime_id_cls=RegimeId,
        )
        params = create_base_params()

        solution = model.solve(params)

        assert isinstance(solution, dict)
        assert len(solution) == 3
        assert all(period in solution for period in range(3))

    def test_solve_and_simulate(self):
        """Base model can be solved and simulated."""
        work_regime, retirement_regime = create_base_regimes()
        model = Model(
            regimes=[work_regime, retirement_regime],
            n_periods=3,
            regime_id_cls=RegimeId,
        )
        params = create_base_params()

        solution = model.solve(params)

        # Simulate starting from work regime
        simulation = model.simulate(
            params=params,
            initial_states={
                "wealth": jnp.array([10.0, 50.0]),
                "health": jnp.array([0, 1]),
            },
            initial_regimes=["work", "work"],
            V_arr_dict=solution,
        )
        assert "work" in simulation or "retirement" in simulation


class TestAbsorbingRegimes:
    """Test absorbing regime behavior (retirement is absorbing)."""

    def test_absorbing_regime_auto_generates_next_regime(self):
        """Absorbing regime without explicit next_regime gets one auto-generated."""
        work_regime, retirement_regime = create_base_regimes()

        # Verify retirement regime doesn't have explicit next_regime in transitions
        # (it's absorbing=True, so it should be auto-generated)
        model = Model(
            regimes=[work_regime, retirement_regime],
            n_periods=3,
            regime_id_cls=RegimeId,
        )

        # Internal regime should have regime_transition_probs
        internal_retirement = model.internal_regimes["retirement"]
        assert internal_retirement.regime_transition_probs is not None

    def test_absorbing_regime_with_explicit_next_regime_warns(self):
        """Providing next_regime for absorbing regime issues a warning."""
        work_regime, _ = create_base_regimes()

        @lcm.mark.stochastic
        def explicit_next_regime() -> FloatND:
            return jnp.array([0.0, 1.0])  # 100% stay in retirement

        # Create retirement regime WITH explicit next_regime (redundant)
        retirement_with_explicit = Regime(
            name="retirement",
            absorbing=True,
            states={
                "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
                "health": DiscreteGrid(HealthStatus),
            },
            actions={
                "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
            },
            utility=utility,
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={"working": retired_working},
            transitions={
                "next_wealth": next_wealth,
                "next_health": lcm.mark.stochastic(next_health, type="uniform"),
                "next_regime": explicit_next_regime,  # Redundant!
            },
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Model(
                regimes=[work_regime, retirement_with_explicit],
                n_periods=3,
                regime_id_cls=RegimeId,
            )
            assert any("absorbing" in str(warning.message).lower() for warning in w)


class TestDifferentStateSpaces:
    """Test regimes with completely different state spaces.

    Extends base model by adding a 'dead' regime with only a 'funerary_wealth' state
    (no wealth or health). This tests that transitions are correctly scoped
    to each regime's state space.

    """

    def test_regime_with_disjoint_states(self):
        """Add dead regime with completely different states than work/retirement."""

        @dataclass
        class ExtendedRegimeId:
            work: int = 0
            retirement: int = 1
            dead: int = 2

        # Modify work regime to possibly die
        @lcm.mark.stochastic
        def next_regime_work_with_death(health: DiscreteState) -> FloatND:
            # Healthy: 80% work, 15% retire, 5% die
            # Unhealthy: 60% work, 20% retire, 20% die
            return jnp.where(
                health == HealthStatus.good,
                jnp.array([0.80, 0.15, 0.05]),
                jnp.array([0.60, 0.20, 0.20]),
            )

        @lcm.mark.stochastic
        def next_regime_retirement_with_death(health: DiscreteState) -> FloatND:
            # Healthy: 95% stay retired, 5% die
            # Unhealthy: 80% stay retired, 20% die
            return jnp.where(
                health == HealthStatus.good,
                jnp.array([0.0, 0.95, 0.05]),
                jnp.array([0.0, 0.80, 0.20]),
            )

        def next_funerary_wealth_from_alive(wealth: ContinuousState) -> ContinuousState:
            """Transition to funerary wealth state when entering dead regime."""
            return wealth / 2

        work_regime = Regime(
            name="work",
            states={
                "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
                "health": DiscreteGrid(HealthStatus),
            },
            actions={
                "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
                "working": DiscreteGrid(WorkingStatus),
            },
            utility=utility,
            constraints={"borrowing_constraint": borrowing_constraint},
            transitions={
                "next_wealth": next_wealth,
                "next_health": lcm.mark.stochastic(next_health, type="uniform"),
                # Required for transition to dead
                "next_funerary_wealth": next_funerary_wealth_from_alive,
                "next_regime": next_regime_work_with_death,
            },
        )

        retirement_regime = Regime(
            name="retirement",
            states={
                "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
                "health": DiscreteGrid(HealthStatus),
            },
            actions={
                "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
            },
            utility=utility,
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={"working": retired_working},
            transitions={
                "next_wealth": next_wealth,
                "next_health": lcm.mark.stochastic(next_health, type="uniform"),
                # Required for transition to dead
                "next_funerary_wealth": next_funerary_wealth_from_alive,
                "next_regime": next_regime_retirement_with_death,
            },
        )

        dead_regime = Regime(
            name="dead",
            absorbing=True,
            states={"funerary_wealth": LinspaceGrid(start=0, stop=50, n_points=5)},
            actions={},
            utility=lambda funerary_wealth: jnp.array(0.0),  # noqa: ARG005
            transitions={
                "next_funerary_wealth": lambda funerary_wealth: funerary_wealth
            },
        )

        model = Model(
            regimes=[work_regime, retirement_regime, dead_regime],
            n_periods=3,
            regime_id_cls=ExtendedRegimeId,
        )

        # Verify dead regime doesn't have wealth/health in params template
        dead_params_keys = set(model.params_template["dead"].keys())
        assert "next_wealth" not in dead_params_keys
        assert "next_health" not in dead_params_keys
        assert "next_funerary_wealth" in dead_params_keys

        # Verify model can be solved
        health_transition = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        params = {
            "work": {
                "beta": 0.95,
                "utility": {"disutility_of_work": 0.5},
                "borrowing_constraint": {},
                "next_wealth": {"wage": 20.0, "interest_rate": 0.05},
                "next_health": {"n": 2,  "x_min": 0, "x_max": 1},
                "next_funerary_wealth": {},
                "next_regime": {},
            },
            "retirement": {
                "beta": 0.95,
                "utility": {"disutility_of_work": 0.5},
                "borrowing_constraint": {},
                "working": {},
                "next_wealth": {"wage": 20.0, "interest_rate": 0.05},
                "next_health": {"n": 2,  "x_min": 0, "x_max": 1},
                "next_funerary_wealth": {},
                "next_regime": {},
            },
            "dead": {
                "beta": 0.95,
                "utility": {},
                "next_funerary_wealth": {},
                "next_regime": {},
            },
        }

        solution = model.solve(params)
        assert len(solution) == 3


class TestOverlappingStateSpaces:
    """Test regimes with partially overlapping state spaces.

    Modifies retirement to have 'pension' instead of 'health'.
    - Work regime: {wealth, health}
    - Retirement regime: {wealth, pension}
    - Shared: wealth
    - Work-only: health
    - Retirement-only: pension
    """

    def test_regime_with_overlapping_states(self):
        """Retirement has pension instead of health (partial overlap with work)."""

        @dataclass
        class PensionType:
            basic: int = 0
            premium: int = 1

        def work_utility(
            consumption: ContinuousAction,
            health: DiscreteState,
        ) -> FloatND:
            """Utility in work depends on health."""
            return jnp.log(consumption) * (1 + 0.1 * health)

        def retirement_utility(
            consumption: ContinuousAction,
            pension: DiscreteState,
        ) -> FloatND:
            """Utility in retirement depends on pension type."""
            return jnp.log(consumption) * (1 + 0.2 * pension)

        def next_pension(pension: DiscreteState) -> DiscreteState:
            """Pension type stays constant."""
            return pension

        def next_pension_from_work(health: DiscreteState) -> DiscreteState:
            """Pension type determined by health at retirement."""
            # Good health -> premium pension, bad health -> basic pension
            return health

        def simple_next_wealth(
            wealth: ContinuousState,
            consumption: ContinuousAction,
            interest_rate: float,
        ) -> ContinuousState:
            """Simple wealth transition: savings with interest."""
            return (1 + interest_rate) * (wealth - consumption)

        @lcm.mark.stochastic
        def next_regime_to_retirement(period: int) -> FloatND:
            """Transition to retirement after period 1."""
            retire_prob = jnp.where(period < 1, 0.0, 0.5)
            return jnp.array([1 - retire_prob, retire_prob])

        work_regime = Regime(
            name="work",
            states={
                "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
                "health": DiscreteGrid(HealthStatus),
            },
            actions={
                "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
            },
            utility=work_utility,
            constraints={"borrowing_constraint": borrowing_constraint},
            transitions={
                "next_wealth": simple_next_wealth,
                "next_health": lcm.mark.stochastic(next_health, type="uniform"),
                "next_pension": next_pension_from_work,  # For transition to retirement
                "next_regime": next_regime_to_retirement,
            },
        )

        def retirement_budget(
            consumption: ContinuousAction, wealth: ContinuousState
        ) -> BoolND:
            return consumption <= wealth

        retirement_regime = Regime(
            name="retirement",
            absorbing=True,
            states={
                "wealth": LinspaceGrid(start=1, stop=100, n_points=10),
                "pension": DiscreteGrid(PensionType),
            },
            actions={
                "consumption": LinspaceGrid(start=1, stop=100, n_points=10),
            },
            utility=retirement_utility,
            constraints={"borrowing_constraint": retirement_budget},
            transitions={
                "next_wealth": simple_next_wealth,
                "next_pension": next_pension,
            },
        )

        model = Model(
            regimes=[work_regime, retirement_regime],
            n_periods=3,
            regime_id_cls=RegimeId,
        )

        # Verify correct state transitions are in each regime
        work_keys = set(model.params_template["work"].keys())
        retirement_keys = set(model.params_template["retirement"].keys())

        # Work should have next_pension (for transition to retirement)
        assert "next_pension" in work_keys
        assert "next_health" in work_keys

        # Retirement should NOT have next_health
        assert "next_health" not in retirement_keys
        assert "next_pension" in retirement_keys

        # Solve model
        health_transition = jnp.array([[0.8, 0.2], [0.3, 0.7]])
        params = {
            "work": {
                "beta": 0.95,
                "utility": {},
                "borrowing_constraint": {},
                "next_wealth": {"interest_rate": 0.05},
                "next_health": {"n": 2,  "x_min": 0, "x_max": 1},
                "next_pension": {},
                "next_regime": {},
            },
            "retirement": {
                "beta": 0.95,
                "utility": {},
                "borrowing_constraint": {},
                "next_wealth": {"interest_rate": 0.05},
                "next_pension": {},
                "next_regime": {},
            },
        }

        solution = model.solve(params)
        assert len(solution) == 3


class TestValidation:
    """Test validation errors for missing transitions."""

    def test_missing_transition_raises_error(self):
        """Non-absorbing regime missing required transition raises error."""
        work_regime, _ = create_base_regimes()

        @dataclass
        class WorkDeadRegimeId:
            work: int = 0
            dead: int = 1

        dead_regime = Regime(
            name="dead",
            absorbing=True,
            states={"funerary_wealth": LinspaceGrid(start=0, stop=50, n_points=5)},
            actions={},
            utility=lambda: jnp.array(0.0),
            transitions={
                "next_funerary_wealth": lambda funerary_wealth: funerary_wealth
            },
        )

        with pytest.raises(ValueError, match="missing transitions"):
            Model(
                regimes=[work_regime, dead_regime],
                n_periods=3,
                regime_id_cls=WorkDeadRegimeId,
            )

    def test_error_message_identifies_missing_transitions(self):
        """Error message clearly identifies which transitions are missing."""

        @dataclass
        class ABRegimeId:
            a: int = 0
            b: int = 1

        @dataclass
        class StateX:
            x: int = 0

        @dataclass
        class StateY:
            y: int = 0

        def utility_fn(consumption: ContinuousAction) -> FloatND:
            return jnp.log(consumption)

        @lcm.mark.stochastic
        def next_regime_stochastic() -> FloatND:
            return jnp.array([0.5, 0.5])

        regime_a = Regime(
            name="a",
            states={"state_x": DiscreteGrid(StateX)},
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
            utility=utility_fn,
            transitions={
                "next_state_x": lambda state_x: state_x,
                "next_state_y": lambda: 0,
                "next_regime": next_regime_stochastic,
            },
        )

        # Regime B is missing 'next_state_x' transition
        regime_b = Regime(
            name="b",
            states={"state_y": DiscreteGrid(StateY)},
            actions={"consumption": LinspaceGrid(start=1, stop=10, n_points=5)},
            utility=utility_fn,
            transitions={
                "next_state_y": lambda state_y: state_y,
                "next_regime": next_regime_stochastic,
            },
        )

        with pytest.raises(ValueError, match="next_state_x"):
            Model(regimes=[regime_a, regime_b], n_periods=3, regime_id_cls=ABRegimeId)
