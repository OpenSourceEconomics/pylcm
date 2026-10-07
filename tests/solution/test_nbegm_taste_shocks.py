"""NB-EGM rejects declared taste shocks instead of ignoring them.

The NB-EGM envelopes take a hard maximum over branches and every published
carry pins the taste-shock scale to zero, while the simulate phase applies the
declared EV1 shocks. Solving one problem and simulating another is silently
wrong, so a regime that declares taste shocks alongside an NB-EGM solver — bare
or nested inside `NNBEGM` — is rejected at model build.
"""

import jax.numpy as jnp
import pytest

from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExtremeValueTasteShocks,
    LinSpacedGrid,
    Regime,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models import n_nbegm_toy
from tests.test_models.graph import with_fixture_graph
from tests.test_models.schedules import until_exit

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=20.0, n_points=8)
CONSUMPTION_GRID = LinSpacedGrid(start=0.05, stop=19.0, n_points=8)
SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=19.0, n_points=8)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class Work:
    work: ScalarInt
    rest: ScalarInt


def labor_income(labor_supply: DiscreteAction) -> FloatND:
    return jnp.where(labor_supply == Work.work, 5.0, 1.0)


def resources(*, wealth: ContinuousState, labor_income: FloatND) -> FloatND:
    return wealth + labor_income


def liquid_savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    return 1.02 * liquid_savings


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def utility_with_labor_disutility(
    *, consumption: ContinuousAction, labor_income: FloatND
) -> FloatND:
    """Additively separable in the discrete labor choice and in consumption."""
    return jnp.log(consumption) - 0.1 * labor_income


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_regime(age: int) -> ScalarInt:
    return jnp.where(age < 25, RegimeId.alive, RegimeId.dead)


def test_nbegm_regime_declaring_taste_shocks_is_rejected():
    """A bare NB-EGM regime with EV1 taste shocks fails at model build."""
    alive = ConsumptionSavingsRegime(
        states={"wealth": WEALTH_GRID},
        state_transitions={"wealth": next_wealth},
        actions={
            "consumption": CONSUMPTION_GRID,
            "labor_supply": DiscreteGrid(category_class=Work),
        },
        regime_transitions=until_exit(
            25,
            law=_SupportedDeterministicTransition(
                func=next_regime, targets=("alive", "dead")
            ),
            exits=("dead",),
        ),
        taste_shocks=ExtremeValueTasteShocks(),
        functions={
            "utility": utility,
            "resources": resources,
            "liquid_savings": liquid_savings,
            "labor_income": labor_income,
        },
        solver=NBEGM(savings_grid=SAVINGS_GRID),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="liquid_savings",
        ),
    )
    dead = Regime(
        regime_transitions=None,
        states={"wealth": WEALTH_GRID},
        functions={"utility": terminal_utility},
    )
    with pytest.raises(RegimeInitializationError, match="does not implement taste"):
        with_fixture_graph(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=RegimeId,
            ages=AgeGrid(start=20, inclusive_stop=25, step="5Y"),
            initial_nodes={20: "alive"},
        )


def test_nnbegm_regime_declaring_taste_shocks_is_rejected():
    """The nested solver inherits the inner NB-EGM's taste-shock rejection."""
    alive = NestedConsumptionSavingsRegime(
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": n_nbegm_toy.durable_transition,
        },
        actions={
            "consumption": n_nbegm_toy.CONSUMPTION_GRID,
            "illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID,
            "labor_supply": DiscreteGrid(category_class=Work),
        },
        regime_transitions=until_exit(
            25,
            law=_SupportedDeterministicTransition(
                func=n_nbegm_toy.next_regime, targets=("alive", "dead")
            ),
            exits=("dead",),
        ),
        taste_shocks=ExtremeValueTasteShocks(),
        functions={
            "utility": utility_with_labor_disutility,
            "new_illiquid": n_nbegm_toy.new_illiquid,
            "resources": n_nbegm_toy.resources,
            "liquid_savings": n_nbegm_toy.liquid_savings,
            "credited": n_nbegm_toy.credited,
            "labor_income": labor_income,
        },
        solver=n_nbegm_toy.build_solver(variant="n_nbegm"),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
    )
    dead = Regime(
        regime_transitions=None,
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        functions={"utility": n_nbegm_toy.terminal_utility},
    )
    with pytest.raises(RegimeInitializationError, match="does not implement taste"):
        with_fixture_graph(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=RegimeId,
            ages=AgeGrid(start=20, inclusive_stop=25, step="5Y"),
            fixed_params={"final_age_alive": 20},
            initial_nodes={20: "alive"},
        )
