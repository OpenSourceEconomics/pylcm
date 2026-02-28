"""Tests for cross-regime state transition handling.

Covers:
- Discrete states with different categories across regimes (validation error)
- States only present in the target regime (mapping transition API)
- Continuous states with per-boundary mapping transitions
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    RegimeTransition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical
class HealthWorkingLife:
    disabled: int
    bad: int
    good: int


@categorical
class HealthRetirement:
    bad: int
    good: int


@categorical
class RegimeId:
    working: int
    retired: int
    dead: int


def hm_utility_working(consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + health * 0.1


def hm_utility_retired(consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + health * 0.05


def hm_next_health_working(health: DiscreteState) -> DiscreteState:
    """Identity transition within working regime (3 categories)."""
    return health


def hm_next_regime_working(age: float) -> ScalarInt:
    return jnp.where(
        age >= 3,
        RegimeId.dead,
        jnp.where(
            age >= 2,
            RegimeId.retired,
            RegimeId.working,
        ),
    )


def hm_next_regime_retired(age: float) -> ScalarInt:
    return jnp.where(age >= 3, RegimeId.dead, RegimeId.retired)


def test_discrete_state_different_categories_raises_without_mapping():
    """Category mismatch without per-boundary transition raises at model construction.

    Health has 3 categories in working (disabled, bad, good) but only 2 in retirement
    (bad, good). Without a per-boundary mapping transition, model construction raises
    ``ModelInitializationError`` to prevent silent index clipping.
    """
    working = Regime(
        states={
            "health": DiscreteGrid(
                HealthWorkingLife, transition=hm_next_health_working
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_working},
        transition=RegimeTransition(hm_next_regime_working),
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "health": DiscreteGrid(HealthRetirement, transition=None),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_retired},
        transition=RegimeTransition(hm_next_regime_retired),
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    with pytest.raises(ModelInitializationError, match="health"):
        Model(
            regimes={"working": working, "retired": retired, "dead": dead},
            ages=AgeGrid(start=0, stop=4, step="Y"),
            regime_id_class=RegimeId,
        )


def test_discrete_state_different_categories_with_mapping_transition():
    """Category mismatch resolved with a per-boundary mapping transition.

    The retired regime's health grid uses a mapping transition to define how
    3-category working health maps to 2-category retirement health.
    """

    def map_working_health_to_retired(health: DiscreteState) -> DiscreteState:
        """Map working health {disabled=0, bad=1, good=2} to retired {bad=0, good=1}.

        disabled or bad → bad (0), good → good (1).
        """
        return jnp.where(health >= 2, HealthRetirement.good, HealthRetirement.bad)

    working = Regime(
        states={
            "health": DiscreteGrid(
                HealthWorkingLife, transition=hm_next_health_working
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_working},
        transition=RegimeTransition(hm_next_regime_working),
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "health": DiscreteGrid(
                HealthRetirement,
                transition={
                    ("working", "retired"): map_working_health_to_retired,
                },
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_retired},
        transition=RegimeTransition(hm_next_regime_retired),
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    params = {"discount_factor": 0.95}

    model = Model(
        regimes={"working": working, "retired": retired, "dead": dead},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=RegimeId,
    )
    model.solve(params)


# ======================================================================================
# Cross-regime state transition gap
# ======================================================================================


def test_transition_to_state_only_in_target_regime() -> None:
    """Alive transitions to dead; dead has heir_present which alive doesn't.

    The dead regime's heir_present grid uses a mapping transition to define
    how heir_present is initialized when entering from alive.
    """

    @categorical
    class HeirPresent:
        no: int
        yes: int

    @categorical
    class RegimeId:
        alive: int
        dead: int

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, RegimeId.dead, RegimeId.alive)

    def determine_heir(wealth: ContinuousState) -> DiscreteState:
        """Set heir_present based on wealth: wealthy agents leave an heir."""
        return jnp.where(wealth > 50, HeirPresent.yes, HeirPresent.no)

    def next_wealth(wealth: ContinuousState) -> ContinuousState:
        return wealth

    def utility_alive(wealth: ContinuousState) -> FloatND:
        return wealth

    def utility_dead(wealth: ContinuousState, heir_present: DiscreteState) -> FloatND:
        return wealth * heir_present

    alive = Regime(
        functions={"utility": utility_alive},
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=next_wealth
            ),
        },
        transition=RegimeTransition(next_regime),
        active=lambda age: age < 2,
    )

    dead = Regime(
        transition=None,
        functions={"utility": utility_dead},
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
                transition=None,
            ),
            "heir_present": DiscreteGrid(
                category_class=HeirPresent,
                transition={("alive", "dead"): determine_heir},
            ),
        },
        active=lambda age: age >= 2,
    )

    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
    )

    params = {"discount_factor": 0.95}
    V_arr_dict = model.solve(params)
    result = model.simulate(
        params=params,
        initial_states={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([50.0]),
        },
        initial_regimes=["alive"],
        V_arr_dict=V_arr_dict,
    )
    df = result.to_dataframe()
    dead_rows = df[df["regime"] == "dead"]
    valid_labels = {"no", "yes"}
    assert dead_rows["heir_present"].isin(valid_labels).all()


# ======================================================================================
# Continuous state per-boundary mapping transition
# ======================================================================================


def test_continuous_state_per_boundary_mapping_transition() -> None:
    """LinSpacedGrid with per-boundary mapping transition across regime boundaries.

    Wealth uses a different transition when crossing from working to retired
    (taxed at 80%) vs within the working regime (5% growth). The retired
    regime's grid specifies a mapping transition keyed by ``("working", "retired")``;
    unlisted boundaries fall back to identity.
    """

    @categorical
    class _RegimeId:
        working: int
        retired: int
        dead: int

    def next_regime_working(age: float) -> ScalarInt:
        return jnp.where(age >= 2, _RegimeId.retired, _RegimeId.working)

    def next_regime_retired(age: float) -> ScalarInt:
        return jnp.where(age >= 3, _RegimeId.dead, _RegimeId.retired)

    def next_wealth_working(wealth: ContinuousState) -> ContinuousState:
        return wealth * 1.05

    def next_wealth_working_to_retired(wealth: ContinuousState) -> ContinuousState:
        return wealth * 0.8

    def utility(wealth: ContinuousState) -> FloatND:
        return jnp.log(wealth)

    working = Regime(
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=next_wealth_working
            ),
        },
        functions={"utility": utility},
        transition=RegimeTransition(next_regime_working),
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
                transition={
                    ("working", "retired"): next_wealth_working_to_retired,
                },
            ),
        },
        functions={"utility": utility},
        transition=RegimeTransition(next_regime_retired),
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    model = Model(
        regimes={"working": working, "retired": retired, "dead": dead},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_RegimeId,
    )

    params = {"discount_factor": 0.95}
    V_arr_dict = model.solve(params)
    result = model.simulate(
        params=params,
        initial_states={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([50.0]),
        },
        initial_regimes=["working"],
        V_arr_dict=V_arr_dict,
    )
    df = result.to_dataframe()
    assert len(df) > 0
    assert "wealth" in df.columns

    # Verify the per-boundary transition applied the 80% tax at the regime boundary.
    # The agent starts in "working" at age 0 with wealth ~50. Within working, wealth
    # grows by 5% each period. At the working→retired boundary (age 2→3), the
    # per-boundary mapping applies a 0.8 multiplier instead of the 1.05 growth.
    working_rows = df[df["regime"] == "working"]
    retired_rows = df[df["regime"] == "retired"]
    assert len(retired_rows) > 0, "Agent should transition to retired regime"
    last_working_wealth = working_rows["wealth"].iloc[-1]
    first_retired_wealth = retired_rows["wealth"].iloc[0]
    # Retired wealth should reflect the 80% tax on the transitioned wealth
    expected = last_working_wealth * 0.8
    assert abs(first_retired_wealth - expected) < 1.0, (
        f"Expected retired wealth ~{expected:.1f} (80% of {last_working_wealth:.1f}), "
        f"got {first_retired_wealth:.1f}"
    )
