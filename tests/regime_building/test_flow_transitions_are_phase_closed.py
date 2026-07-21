"""The FLOW half of the simulate-phase Q is closed under the simulate phase.

The simulate-phase Q is built from two phase-closed halves:

    flow         = simulate transitions + simulate function pool
    continuation = solve transitions    + solve function pool

The agent acts on its beliefs about the FUTURE and lives in the TRUTH NOW, so the
belief prices only the continuation. `test_perceived_stochastic_transitions.py` pins the
continuation half; this module pins the flow half.

The flow half is observable because pylcm lets a within-period utility read a chosen
`next_<state>` (the NEGM service-flow pattern). Here the continuation is made FLAT in
the state, so E[V] is identical across actions and the argmax is decided purely by the
law the FLOW resolves. That isolates the flow: a build that took the flow's outer
`next_<state>` from the solve bundle chooses the opposite action.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import AgeGrid, DiscreteGrid, Model, Phased, Regime, categorical
from lcm.typing import DiscreteAction, FloatND, Period, ScalarInt


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


def service_flow(next_stock: FloatND, move: DiscreteAction, stock: FloatND) -> FloatND:
    """Within-period utility reading the CHOSEN next state (NEGM service flow).

    Flat in `move` and in the CURRENT `stock` (both enter with a zero weight, only so
    that every state and action is used), so the action is decided purely by
    `next_stock` -- i.e. by whichever law the flow sub-DAG resolves.
    """
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock


def flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    """Terminal utility FLAT in the state, so continuation cannot drive the argmax."""
    return 0.0 * stock + 0.0 * move


def next_stock_belief(move: DiscreteAction) -> FloatND:
    """BELIEF: `stay` leads to good."""
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def next_stock_actual(move: DiscreteAction) -> FloatND:
    """TRUTH: exactly the opposite -- `switch` leads to good."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulate(*, live_functions, state_transitions) -> pd.DataFrame:
    live = Regime(
        transition=_next_regime,
        state_transitions=state_transitions,
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=live_functions,
    ).replace(active=lambda age: age < 2)
    last = Regime(
        transition=None,
        state_transitions={},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": flat_utility},
    ).replace(active=lambda age: age >= 2)
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="phase closure of the flow sub-DAG",
    )
    V = model.solve(params=PARAMS, log_level="off")
    return (
        model.simulate(
            params=PARAMS,
            initial_conditions=IC,
            period_to_regime_to_V_arr=V,
            log_level="off",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )


def test_flow_reads_the_simulate_variant_of_a_phased_outer_law():
    """A `Phased` `next_<state>` read by utility resolves to the SIMULATE law in flow.

    Before the fix the flow's outer `next_<state>` was taken from the SOLVE bundle
    (while its helpers still came from the simulate pool -- a sub-DAG that was neither
    phase), so the agent chose `stay`: it valued its current service flow under a law it
    only believes, rather than the one it lives in.
    """
    df = _simulate(
        live_functions={"utility": service_flow},
        state_transitions={
            "stock": Phased(solve=next_stock_belief, simulate=next_stock_actual)
        },
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the FLOW must value the chosen next state under the SIMULATE law"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must follow the SIMULATE law"
    )


def test_flow_reads_the_simulate_variant_of_a_phased_helper():
    """The flow sub-DAG is closed: a `Phased` HELPER under the law is simulate too.

    Complements the outer-law case: here the law is a single bare function and the
    belief/truth wedge lives entirely in the helper it reads. A flow that mixed the
    solve outer law with simulate helpers -- or vice versa -- is not phase-closed.
    """

    def stay_target_belief() -> FloatND:
        return jnp.array(Stock.good)

    def stay_target_actual() -> FloatND:
        return jnp.array(Stock.bad)

    def next_stock(move: DiscreteAction, stay_target: FloatND) -> FloatND:
        # Stays integral: `next_stock` indexes a DiscreteGrid.
        return jnp.where(move == Move.stay, stay_target, 1 - stay_target)

    df = _simulate(
        live_functions={
            "utility": service_flow,
            "stay_target": Phased(
                solve=stay_target_belief, simulate=stay_target_actual
            ),
        },
        state_transitions={"stock": next_stock},
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the FLOW must read the SIMULATE variant of a Phased helper"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must read the SIMULATE variant of a Phased helper"
    )
