"""DC-EGM splaying: `batch_size` on a discrete combo axis is a memory knob only.

In asset-row mode the kernel maps the per-combo solve over the Cartesian product
of the regime's discrete states, passive states, and discrete actions. Splaying
a combo axis — setting `batch_size` on a discrete state grid (or a process /
passive grid) — processes that axis's slices in blocks rather than one fused
vmap, shedding peak working-set memory. It is a pure scheduling choice: the
solved value function is identical to the unsplayed (`batch_size=0`) solve at
any block size, including sizes that do not divide the axis.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

N_PERIODS = 4
N_WEALTH = 40
BAND_START = 5.0
BAND_WIDTH = 40.0

CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=100.0, n_points=2000)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(110.0 * (i / 149) ** 3 for i in range(150)))


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    fair: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def smoothstep(value: FloatND) -> FloatND:
    t = jnp.clip((value - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    # Reading the Euler state in the regime-transition probability switches the
    # kernel into the per-exogenous-asset-node (asset-row) solve.
    return 0.5 + 0.45 * smoothstep(wealth)


def stay_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth, age, final_age_alive)


def health_transition(health: DiscreteState) -> FloatND:
    # A genuine Markov health combo axis carried into the child.
    stay = jnp.where(health == Health.good, 0.7, 0.5)
    others = (1.0 - stay) / 2.0
    return jnp.stack([others, others, stay])


def utility(consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    penalty = jnp.where(
        health == Health.bad, 0.2, jnp.where(health == Health.fair, 0.1, 0.0)
    )
    return jnp.log(consumption) - penalty


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


@functools.cache
def _model(health_batch_size: int) -> Model:
    """Asset-row DC-EGM with a Markov health combo axis splayed by `batch_size`."""
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = UserRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(Health, batch_size=health_batch_size),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
        },
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=SAVINGS_GRID,
            n_constrained_points=32,
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=200)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def _params() -> dict:
    return {"discount_factor": 0.95, "final_age_alive": 40 + (N_PERIODS - 2) * 10}


def _solve(health_batch_size: int) -> PeriodToRegimeToVArr:
    return _model(health_batch_size).solve(params=_params(), log_level="debug")


@pytest.mark.parametrize("health_batch_size", [1, 2, 3])
def test_discrete_combo_batch_size_leaves_value_function_unchanged(
    health_batch_size: int,
):
    """Splaying a discrete combo axis does not change the solved V.

    `batch_size` on the `health` grid only changes how the discrete-state combo
    product is scheduled (blocks via `lax.map` instead of one fused vmap over
    the flattened product), so the value function at every period matches the
    unsplayed `batch_size=0` solve exactly — including block sizes that do not
    divide the axis.
    """
    reference = _solve(0)
    splayed = _solve(health_batch_size)
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"period={period}, regime={regime_name}",
            )


@categorical(ordered=False)
class Marital:
    single: ScalarInt
    married: ScalarInt


def utility_two_combos(
    consumption: ContinuousAction, health: DiscreteState, married: DiscreteState
) -> FloatND:
    penalty = jnp.where(
        health == Health.bad, 0.2, jnp.where(health == Health.fair, 0.1, 0.0)
    )
    bonus = jnp.where(married == Marital.married, 0.05, 0.0)
    return jnp.log(consumption) - penalty + bonus


@functools.cache
def _two_combo_model(batch_size: int) -> Model:
    """Asset-row DC-EGM with TWO discrete combo axes (health + married)."""
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = UserRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(Health, batch_size=batch_size),
            "married": DiscreteGrid(Marital, batch_size=batch_size),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
            "married": fixed_transition("married"),
        },
        functions={
            "utility": utility_two_combos,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=SAVINGS_GRID,
            n_constrained_points=32,
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=200)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_two_discrete_combo_axes_splayed_together_match_unsplayed(batch_size: int):
    """Splaying TWO combo axes at once does not change the solved V.

    With more than one splayed axis the kernel runs a single `lax.map` over the
    flattened (health-by-married) product (one scan carry) rather than nesting a
    `lax.map` per axis. The value function at every period must match the
    unsplayed `batch_size=0` solve exactly, with the combo axes in the same
    canonical order — guarding the flatten-and-transpose path.
    """
    reference = _two_combo_model(0).solve(params=_params(), log_level="debug")
    splayed = _two_combo_model(batch_size).solve(params=_params(), log_level="debug")
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V, ref_V, rtol=1e-12, atol=1e-12, err_msg=f"period={period}"
            )
