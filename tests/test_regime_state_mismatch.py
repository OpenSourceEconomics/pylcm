"""Reproducer: discrete state with different categories across regimes."""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.input_processing.regime_processing import _merge_ordered_categories
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class HealthWorkingLife:
    disabled: int
    bad: int
    good: int


@categorical(ordered=False)
class HealthRetirement:
    bad: int
    good: int


@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int
    dead: int


def hm_utility_working(consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + health * 0.1


def hm_utility_retirement(
    consumption: ContinuousAction, health: DiscreteState
) -> FloatND:
    return jnp.log(consumption) + health * 0.05


def hm_next_health_working(health: DiscreteState) -> DiscreteState:
    """Identity transition within working regime (3 categories).

    When transitioning working -> retired, we would need a DIFFERENT function
    that maps {0, 1, 2} -> {0, 1} (e.g., disabled -> bad, bad -> bad,
    good -> good). But the API has no way to attach a separate boundary
    transition for different target regimes.
    """
    return health


def hm_next_regime_working(age: float) -> ScalarInt:
    return jnp.where(
        age >= 3,
        RegimeId.dead,
        jnp.where(
            age >= 2,
            RegimeId.retirement,
            RegimeId.working_life,
        ),
    )


def hm_next_regime_retired(age: float) -> ScalarInt:
    return jnp.where(age >= 3, RegimeId.dead, RegimeId.retirement)


def test_discrete_state_different_categories_across_regimes():
    """Single transition for a state with different categories across regimes.

    A 'health' state has 3 categories (disabled, bad, good) during working life
    but only 2 (bad, good) during retirement. The system needs to map 3 -> 2 at
    the working -> retired boundary. While a per-target dict transition can handle
    this (see `test_per_target_dict_transitions`), this test uses a single
    transition function for all targets.

    Model construction should raise a validation error for this category mismatch.
    """
    working = Regime(
        states={
            "health": DiscreteGrid(HealthWorkingLife),
        },
        state_transitions={
            "health": hm_next_health_working,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_working},
        transition=hm_next_regime_working,
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "health": DiscreteGrid(HealthRetirement),
        },
        state_transitions={
            "health": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_retirement},
        transition=hm_next_regime_retired,
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    with pytest.raises(ModelInitializationError, match="health"):
        Model(
            regimes={"working_life": working, "retirement": retired, "dead": dead},
            ages=AgeGrid(start=0, stop=4, step="Y"),
            regime_id_class=RegimeId,
        )


# ======================================================================================
# Cross-regime state transition gap
# ======================================================================================


def test_deterministic_target_only_state() -> None:
    """Target-only state with a deterministic per-target transition.

    alive transitions to dead, which has an heir_present state that alive
    does not. Whether the deceased has an heir is determined by their
    wealth (never mind the reverse causality).
    """

    @categorical(ordered=False)
    class HeirPresent:
        no: int
        yes: int

    @categorical(ordered=False)
    class _RegimeId:
        alive: int
        dead: int

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, _RegimeId.dead, _RegimeId.alive)

    alive = Regime(
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        state_transitions={
            "wealth": lambda wealth: wealth,
            "heir_present": {
                "dead": lambda wealth: jnp.where(
                    wealth >= 50, HeirPresent.yes, HeirPresent.no
                ),
            },
        },
        transition=next_regime,
        active=lambda age: age < 2,
    )

    dead = Regime(
        transition=None,
        functions={"utility": lambda wealth, heir_present: wealth * heir_present},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "heir_present": DiscreteGrid(HeirPresent),
        },
    )

    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )

    params = {"discount_factor": 0.95}
    V_arr_dict = model.solve(params=params)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([20.0, 80.0]),
            "regime": jnp.array([_RegimeId.alive] * 2),
        },
        V_arr_dict=V_arr_dict,
    )
    df = result.to_dataframe(use_labels=False)
    dead_rows = df[df["regime"] == "dead"]
    valid_codes = {float(HeirPresent.no), float(HeirPresent.yes)}
    assert not dead_rows.empty, "Expected some rows in the dead regime"
    assert dead_rows["heir_present"].isin(valid_codes).all()
    # Wealthy subject (wealth=80 >= 50) should have heir
    # Poor subject (wealth=20 < 50) should not
    wealthy_dead = dead_rows[dead_rows["wealth"] >= 50]
    poor_dead = dead_rows[dead_rows["wealth"] < 50]
    if not wealthy_dead.empty:
        assert (wealthy_dead["heir_present"] == HeirPresent.yes).all()
    if not poor_dead.empty:
        assert (poor_dead["heir_present"] == HeirPresent.no).all()


def test_stochastic_target_only_state() -> None:
    """Target-only state with a stochastic per-target transition.

    Same scenario as the deterministic test, but heir_present is assigned
    stochastically: wealthier individuals are more likely to have an heir.
    """

    @categorical(ordered=False)
    class HeirPresent:
        no: int
        yes: int

    @categorical(ordered=False)
    class _RegimeId:
        alive: int
        dead: int

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 1, _RegimeId.dead, _RegimeId.alive)

    def heir_present_probs(wealth: ContinuousState) -> FloatND:
        return jnp.where(
            wealth >= 50,
            jnp.array([0.2, 0.8]),  # mostly yes
            jnp.array([0.7, 0.3]),  # mostly no
        )

    def utility_alive(wealth: ContinuousState) -> FloatND:
        return wealth

    def next_wealth(wealth: ContinuousState) -> ContinuousState:
        return wealth

    def utility_dead(wealth: ContinuousState, heir_present: DiscreteState) -> FloatND:
        return wealth * heir_present

    alive = Regime(
        functions={"utility": utility_alive},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
        },
        state_transitions={
            "wealth": next_wealth,
            "heir_present": {
                "dead": MarkovTransition(heir_present_probs),
            },
        },
        transition=next_regime,
        active=lambda age: age < 2,
    )

    dead = Regime(
        transition=None,
        functions={"utility": utility_dead},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "heir_present": DiscreteGrid(HeirPresent),
        },
    )

    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )

    params = {"discount_factor": 0.95}
    V_arr_dict = model.solve(params=params)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([20.0, 80.0]),
            "regime": jnp.array([_RegimeId.alive] * 2),
        },
        V_arr_dict=V_arr_dict,
    )
    df = result.to_dataframe(use_labels=False)
    dead_rows = df[df["regime"] == "dead"]
    valid_codes = {float(HeirPresent.no), float(HeirPresent.yes)}
    assert not dead_rows.empty, "Expected some rows in the dead regime"
    assert dead_rows["heir_present"].isin(valid_codes).all()


# ======================================================================================
# Per-target dict transitions (passing test)
# ======================================================================================


def map_working_health_to_retired(health: DiscreteState) -> DiscreteState:
    """Map 3 working-life health categories to 2 retirement categories.

    Working: disabled=0, bad=1, good=2
    Retired: bad=0, good=1

    Mapping: disabled -> bad (0->0), bad -> bad (1->0), good -> good (2->1).
    """
    return jnp.where(health >= 2, HealthRetirement.good, HealthRetirement.bad)


def test_per_target_dict_transitions():
    """Per-target dict transitions correctly handle different categories across regimes.

    Uses the per-target dict syntax in `state_transitions` to provide a boundary
    transition that maps 3 working-life health states to 2 retirement health states
    when transitioning from working to retired.
    """
    working = Regime(
        states={
            "health": DiscreteGrid(HealthWorkingLife),
        },
        state_transitions={
            "health": {
                "working_life": hm_next_health_working,
                "retirement": map_working_health_to_retired,
                "dead": hm_next_health_working,
            },
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_working},
        transition=hm_next_regime_working,
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "health": DiscreteGrid(HealthRetirement),
        },
        state_transitions={
            "health": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_retirement},
        transition=hm_next_regime_retired,
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    model = Model(
        regimes={"working_life": working, "retirement": retired, "dead": dead},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=RegimeId,
    )

    params = {"discount_factor": 0.95}
    V_arr_dict = model.solve(params=params)

    n_subjects = 4
    # Use codes 0 (disabled) and 1 (bad) — valid in both regimes.
    # Code 2 (good) is only valid in working and fails cross-regime validation.
    initial_health = jnp.array(
        [
            HealthWorkingLife.disabled,
            HealthWorkingLife.disabled,
            HealthWorkingLife.bad,
            HealthWorkingLife.bad,
        ]
    )

    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(n_subjects),
            "health": initial_health,
            "regime": jnp.array([RegimeId.working_life] * n_subjects),
        },
        V_arr_dict=V_arr_dict,
    )
    df = result.to_dataframe(use_labels=False)

    retired_rows = df[df["regime"] == "retirement"]
    valid_retired_codes = {float(HealthRetirement.bad), float(HealthRetirement.good)}
    assert not retired_rows.empty, "Expected some rows in the retired regime"
    assert retired_rows["health"].isin(valid_retired_codes).all(), (
        f"Retired health codes should be in {valid_retired_codes}, "
        f"got {sorted(retired_rows['health'].unique())}"
    )


def test_discrete_state_same_count_different_names():
    """Same number of categories but different names should still raise."""

    @categorical(ordered=False)
    class StatusA:
        employed: int
        unemployed: int

    @categorical(ordered=False)
    class StatusB:
        married: int
        single: int

    @categorical(ordered=False)
    class _RegimeId:
        work: int
        retire: int
        dead: int

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(
            age >= 2,
            _RegimeId.dead,
            jnp.where(age >= 1, _RegimeId.retire, _RegimeId.work),
        )

    work = Regime(
        states={"status": DiscreteGrid(StatusA)},
        state_transitions={"status": lambda status: status},
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={
            "utility": lambda consumption, status: jnp.log(consumption) + status
        },
        transition=next_regime,
        active=lambda age: age < 2,
    )

    retire = Regime(
        states={"status": DiscreteGrid(StatusB)},
        state_transitions={"status": None},
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={
            "utility": lambda consumption, status: jnp.log(consumption) + status
        },
        transition=lambda age: jnp.where(age >= 2, _RegimeId.dead, _RegimeId.retire),
        active=lambda age: age < 3,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    with pytest.raises(ModelInitializationError, match="status"):
        Model(
            regimes={"work": work, "retire": retire, "dead": dead},
            ages=AgeGrid(start=0, stop=3, step="Y"),
            regime_id_class=_RegimeId,
        )


# ======================================================================================
# Ordered flag validation tests
# ======================================================================================


def test_mixed_ordered_flags_raises():
    """Mixed ordered flags (True in one regime, False in another) should raise."""

    @categorical(ordered=True)
    class HealthOrdered:
        bad: int
        good: int

    @categorical(ordered=False)
    class HealthUnordered:
        bad: int
        good: int

    @categorical(ordered=False)
    class _RegimeId:
        a: int
        b: int
        dead: int

    def next_regime() -> ScalarInt:
        return _RegimeId.dead

    a = Regime(
        states={"health": DiscreteGrid(HealthOrdered)},
        state_transitions={"health": None},
        functions={"utility": lambda health: health},
        transition=next_regime,
    )
    b = Regime(
        states={"health": DiscreteGrid(HealthUnordered)},
        state_transitions={"health": None},
        functions={"utility": lambda health: health},
        transition=next_regime,
    )
    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    with pytest.raises(ModelInitializationError, match="inconsistent ordered flags"):
        Model(
            regimes={"a": a, "b": b, "dead": dead},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_both_ordered_same_categories_passes():
    """Both ordered with same categories should pass without error."""

    @categorical(ordered=True)
    class HealthA:
        bad: int
        good: int

    @categorical(ordered=True)
    class HealthB:
        bad: int
        good: int

    @categorical(ordered=False)
    class _RegimeId:
        a: int
        b: int
        dead: int

    def next_regime() -> ScalarInt:
        return _RegimeId.dead

    a = Regime(
        states={"health": DiscreteGrid(HealthA)},
        state_transitions={"health": None},
        functions={"utility": lambda health: health},
        transition=next_regime,
    )
    b = Regime(
        states={"health": DiscreteGrid(HealthB)},
        state_transitions={"health": None},
        functions={"utility": lambda health: health},
        transition=next_regime,
    )
    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    # Should not raise
    Model(
        regimes={"a": a, "b": b, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_both_ordered_different_categories_ambiguous_raises():
    """Both ordered with ambiguous merge should raise."""
    # p < r and q < r — p vs q ordering is undetermined
    result = _merge_ordered_categories(
        [
            ("regime_a", ("p", "r")),
            ("regime_b", ("q", "r")),
        ]
    )
    assert result is None


def test_both_ordered_different_categories_unique_merge():
    """Both ordered with unique topological merge should succeed."""
    # p < q from regime_a, q < r from regime_b → p < q < r
    result = _merge_ordered_categories(
        [
            ("regime_a", ("p", "q")),
            ("regime_b", ("q", "r")),
        ]
    )
    assert result == ("p", "q", "r")


def test_both_ordered_contradictory_raises():
    """Contradictory orderings (cycle) should fail merge."""
    # a < b from regime_a, b < a from regime_b → cycle
    result = _merge_ordered_categories(
        [
            ("regime_a", ("a", "b")),
            ("regime_b", ("b", "a")),
        ]
    )
    assert result is None
