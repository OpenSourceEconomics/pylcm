"""A broadcast state read only by an adjustment cost survives pruning.

`NetOfAdjustmentCost` composes the resources node at model finalization, which
runs after broadcast pruning. Pruning therefore has to know the composition
exists, or a state whose only reader is the cost looks unused and is dropped —
after which finalization has nothing to compose.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    DCEGM,
    AgeGrid,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    FUESEnvelope,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NetOfAdjustmentCost,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Tier:
    low: ScalarInt
    high: ScalarInt


_WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=4)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=4.5, n_points=4)


def utility(consumption: FloatND) -> FloatND:
    return jnp.log(consumption)


def gross_resources(wealth: ContinuousState, interest_rate: float = 0.05) -> FloatND:
    return (1 + interest_rate) * wealth


def adjustment_cost(tier: DiscreteState, fee: float = 0.1) -> FloatND:
    """The only reader of `tier` anywhere in the regime."""
    return fee * tier


def savings(resources: FloatND, consumption: FloatND) -> FloatND:
    return resources - consumption


def next_wealth(savings: FloatND) -> FloatND:
    return savings


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def next_regime(age: int) -> DiscreteState:
    return jnp.where(age < 2, RegimeId.working, RegimeId.dead)


def _build_model(*, broadcast_tier: bool, cost=adjustment_cost) -> Model:
    working = ConsumptionSavingsRegime(
        transition=next_regime,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={
            "utility": utility,
            "gross_resources": gross_resources,
            "adjustment_cost": cost,
            "savings": savings,
        },
        state_transitions={"wealth": next_wealth},
        solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.1, stop=5.0, n_points=5),
            envelope=FUESEnvelope(),
        ),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                name_in_dag="resources",
                before_cost="gross_resources",
                cost="adjustment_cost",
            ),
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        transition=None,
        functions={"utility": terminal_utility},
        states={"wealth": _WEALTH},
    )
    model_states = {"tier": DiscreteGrid(Tier)} if broadcast_tier else {}
    model_laws = {"tier": fixed_transition("tier")} if broadcast_tier else {}
    if not broadcast_tier:
        working = working.replace(
            states={**dict(working.states), "tier": DiscreteGrid(Tier)},
            state_transitions={
                **dict(working.state_transitions),
                "tier": fixed_transition("tier"),
            },
        )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
        states=model_states,
        state_transitions=model_laws,
    )


def test_a_state_read_only_by_the_adjustment_cost_is_not_pruned():
    """`tier` reaches utility through the composed resources, so it is kept."""
    model = _build_model(broadcast_tier=True)

    assert model.pruned_variables["working"] == frozenset()


def test_the_pruner_still_drops_a_state_nothing_reads():
    """A broadcast state with no reader at all is still pruned."""

    def unread_cost(fee: float = 0.1) -> FloatND:
        return jnp.asarray(fee)

    model = _build_model(broadcast_tier=True, cost=unread_cost)

    assert model.pruned_variables["working"] == frozenset({"tier"})


@pytest.mark.parametrize("broadcast", [True, False])
def test_the_model_builds_either_way(broadcast):
    """Declaring `tier` on the regime and broadcasting it agree."""
    model = _build_model(broadcast_tier=broadcast)

    assert "tier" in model.user_regimes["working"].states
