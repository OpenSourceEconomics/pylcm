"""The FLOW half of a COLLECTIVE regime's simulate-phase Q is closed under simulate.

Collective analogue of `test_flow_transitions_are_phase_closed.py`.
`get_Q_and_F_collective` must thread the SAME flow/continuation phase split as
`get_Q_and_F`:

    flow         = simulate function pool
    continuation = solve function pool (`continuation_functions`)

A collective branch of `_build_Q_and_F_per_period` that dropped
`continuation_functions` would leave the collective simulator building the current
per-stakeholder flow from the SOLVE pool — a hybrid sub-DAG that is neither phase, and
that reverses the household argmax.

Construction mirrors the singleton test: a two-period model whose period-1 continuation
is FLAT, so `E[V]` is identical across period-0 actions and the household argmax is
decided purely by whichever variant the FLOW resolves. Each stakeholder's within-period
utility reads the stock chosen this period — `new_stock`, an ordinary function of this
period's states and actions, which the state law then carries forward (the NEGM
service-flow pattern). That is the only observable exposing which pool the flow sub-DAG
resolved. `new_stock` is `Phased`: the agent BELIEVES `stay` leads to good but LIVES
the opposite. A flow closed under simulate therefore chooses `switch`; a collective
flow built from the solve pool would choose `stay`.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Move:
    stay: ScalarInt
    switch: ScalarInt


@categorical(ordered=True)
class Stock:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    live: ScalarInt
    last: ScalarInt


def _service_flow(
    *, new_stock: FloatND, move: DiscreteAction, stock: FloatND
) -> FloatND:
    """Within-period utility reading the CHOSEN stock (NEGM service flow).

    `new_stock` is an ordinary function of this period's states and actions, not a
    `next_<state>`: the stock chosen NOW is a within-period quantity, and the law merely
    carries it forward. Flat in `move` and in the CURRENT `stock` (both enter with a
    zero weight, only so that every state and action is used), so the action is decided
    purely by `new_stock` -- i.e. by whichever pool the flow sub-DAG resolves. Shared
    verbatim by both stakeholders so the household argmax (the `H_linear` sum of the two
    `Q^s`) is unambiguous.
    """
    return 1.0 * new_stock + 0.0 * move + 0.0 * stock


def _carry_new_stock(new_stock: FloatND) -> FloatND:
    """The law: next period's stock is the one chosen this period."""
    return new_stock


def _flat_utility(*, stock: FloatND, move: DiscreteAction) -> FloatND:
    """Terminal utility FLAT in the state, so continuation cannot drive the argmax."""
    return 0.0 * stock + 0.0 * move


def _new_stock_belief(move: DiscreteAction) -> FloatND:
    """BELIEF (solve): `stay` leads to good."""
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _new_stock_actual(move: DiscreteAction) -> FloatND:
    """TRUTH (simulate): exactly the opposite -- `switch` leads to good."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _next_regime() -> ScalarInt:
    return RegimeId.last


PARAMS = {"discount_factor": 0.95}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulate(*, live_functions, state_transitions) -> pd.DataFrame:
    live = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        state_transitions=state_transitions,
        states={"stock": DiscreteGrid(category_class=Stock)},
        actions={"move": DiscreteGrid(category_class=Move)},
        functions=live_functions,
    )
    last = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"stock": DiscreteGrid(category_class=Stock)},
        actions={"move": DiscreteGrid(category_class=Move)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _flat_utility, "m": _flat_utility}
            )
        },
    )
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1)),
        regime_id_class=RegimeId,
        description="phase closure of a collective regime's flow sub-DAG",
    )
    V = model.solve(params=PARAMS, log_level="off")
    return (
        model.simulate(
            params=PARAMS,
            initial_conditions=IC,
            solution=V,
            log_level="off",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )


def test_collective_flow_reads_the_simulate_variant_of_a_phased_chosen_stock():
    """A `Phased` `new_stock` read by a stakeholder utility uses the SIMULATE variant.

    A collective branch that built the flow from the SOLVE pool while its helpers came
    from the simulate pool -- a sub-DAG that is neither phase -- would have every
    household choose `stay`, valuing its current service flow under a rule it only
    believes rather than the one it lives in.
    """
    df = _simulate(
        live_functions={
            "utility": CollectiveUtility(
                utilities={"f": _service_flow, "m": _service_flow}
            ),
            "new_stock": Phased(solve=_new_stock_belief, simulate=_new_stock_actual),
        },
        state_transitions={"stock": _carry_new_stock},
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the collective FLOW must value the chosen stock under the SIMULATE variant"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must follow the SIMULATE variant"
    )


def test_collective_flow_reads_the_simulate_variant_of_a_phased_helper():
    """The collective flow is closed: a `Phased` HELPER under `new_stock` is simulate.

    Complements the chosen-stock case: here `new_stock` is a single bare function and
    the belief/truth wedge lives entirely in the helper it reads. A flow that mixed the
    solve chosen stock with simulate helpers -- the exact hybrid the dropped
    `continuation_functions` produced -- is not phase-closed.
    """

    def stay_target_belief() -> FloatND:
        return jnp.array(Stock.good)

    def stay_target_actual() -> FloatND:
        return jnp.array(Stock.bad)

    def new_stock(*, move: DiscreteAction, stay_target: FloatND) -> FloatND:
        # Stays integral: the stock indexes a DiscreteGrid.
        return jnp.where(move == Move.stay, stay_target, 1 - stay_target)

    df = _simulate(
        live_functions={
            "utility": CollectiveUtility(
                utilities={"f": _service_flow, "m": _service_flow}
            ),
            "new_stock": new_stock,
            "stay_target": Phased(
                solve=stay_target_belief, simulate=stay_target_actual
            ),
        },
        state_transitions={"stock": _carry_new_stock},
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the collective FLOW must read the SIMULATE variant of a Phased helper"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must read the SIMULATE variant of a Phased helper"
    )
