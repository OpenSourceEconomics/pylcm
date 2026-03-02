"""Reproducer: discrete state with different categories across regimes."""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousAction, DiscreteState, FloatND, ScalarInt


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
            RegimeId.retired,
            RegimeId.working,
        ),
    )


def hm_next_regime_retired(age: float) -> ScalarInt:
    return jnp.where(age >= 3, RegimeId.dead, RegimeId.retired)


@pytest.mark.xfail(
    reason=(
        "Unsupported: discrete state with different categories across regimes. "
        "The transition parameter on a grid is a single function reused for all "
        "target regimes, so there is no way to express a boundary transition "
        "that maps 3 working-life health states to 2 retirement health states. "
        "Currently model construction and solve silently succeed, producing "
        "incorrect results due to JAX's out-of-bounds index clipping."
    ),
    strict=True,
)
def test_discrete_state_different_categories_across_regimes():
    """Discrete state with different category sets across regimes is unsupported.

    A 'health' state has 3 categories (disabled, bad, good) during working life
    but only 2 (bad, good) during retirement. The system needs to map 3 -> 2 at
    the working -> retired boundary, but the transition on the grid is reused for
    all target regimes (_extract_transitions_from_regime in regime_processing.py).

    Model construction or solve should raise a validation error for this
    category mismatch. Currently it silently succeeds, producing incorrect
    continuation values because JAX clips out-of-bounds indices.
    """
    working = Regime(
        states={
            "health": DiscreteGrid(
                HealthWorkingLife, transition=hm_next_health_working
            ),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_working},
        transition=hm_next_regime_working,
        active=lambda age: age < 3,
    )

    retired = Regime(
        states={
            "health": DiscreteGrid(HealthRetirement, transition=None),
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=10, n_points=5)},
        functions={"utility": hm_utility_retired},
        transition=hm_next_regime_retired,
        active=lambda age: age < 4,
    )

    dead = Regime(transition=None, functions={"utility": lambda: 0.0})

    params = {"discount_factor": 0.95}

    # Should raise a validation error because 'health' has incompatible
    # categories across regimes (3 in working vs 2 in retired).
    with pytest.raises(ValueError, match="health"):  # noqa: PT012
        model = Model(
            regimes={"working": working, "retired": retired, "dead": dead},
            ages=AgeGrid(start=0, stop=4, step="Y"),
            regime_id_class=RegimeId,
        )
        model.solve(params)


# ======================================================================================
# Cross-regime state transition gap
# ======================================================================================


@pytest.mark.xfail(
    reason=(
        "Unsupported: source regime transitions to a target regime that has a state "
        "not present in the source. The missing state is filled with MISSING_CAT_CODE "
        "so simulation completes, but the results are incorrect because the framework "
        "cannot determine the correct initial value for the missing state."
    ),
    strict=True,
)
def test_transition_to_state_only_in_target_regime() -> None:
    """alive transitions to dead, but dead has heir_present which alive doesn't."""

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

    alive = Regime(
        functions={"utility": lambda wealth: wealth},
        states={
            "wealth": LinSpacedGrid(
                start=1, stop=100, n_points=10, transition=lambda wealth: wealth
            ),
        },
        transition=next_regime,
        active=lambda age: age < 2,
    )

    dead = Regime(
        transition=None,
        functions={"utility": lambda wealth, heir_present: wealth * heir_present},
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
                transition=None,
            ),
            "heir_present": DiscreteGrid(
                category_class=HeirPresent,
                transition=None,
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
    dead_rows = df[df["regime_id"] == "dead"]
    valid_codes = {HeirPresent.no, HeirPresent.yes}
    assert dead_rows["heir_present"].isin(valid_codes).all()
