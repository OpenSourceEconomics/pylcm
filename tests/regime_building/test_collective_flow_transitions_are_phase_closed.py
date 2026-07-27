"""The FLOW half of a COLLECTIVE regime's simulate-phase Q is closed under simulate.

Collective analogue of `test_flow_transitions_are_phase_closed.py`.
`get_Q_and_F_collective` must thread the SAME flow/continuation phase split as
`get_Q_and_F`:

    flow         = simulate transitions + simulate function pool
    continuation = solve transitions    + solve function pool

Before the collective-regimes-branch-2 audit F1 fix, the *collective* branch of
`_build_Q_and_F_per_period` dropped `flow_transitions`, `continuation_functions`,
`flow_stochastic_transition_names`, and `next_state_names` when calling
`get_Q_and_F_collective`, so the collective simulator built the current per-stakeholder
flow from the SOLVE transitions — a hybrid sub-DAG that is neither phase and reverses
the household argmax.

Construction mirrors the singleton test: a two-period model whose period-1 continuation
is FLAT, so `E[V]` is identical across period-0 actions and the household argmax is
decided purely by the law the FLOW resolves. Each stakeholder's within-period utility
reads the CHOSEN `next_stock` (the NEGM service-flow pattern), the only observable that
exposes which phase the flow sub-DAG resolved. The `next_stock` law is `Phased`: the
agent BELIEVES `stay` leads to good but LIVES the opposite. A flow closed under the
simulate law therefore chooses `switch`; the pre-fix collective flow (solve law) chose
`stay`.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import AgeGrid, DiscreteGrid, Model, Phased, Regime, categorical
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


def _service_flow(next_stock: FloatND, move: DiscreteAction, stock: FloatND) -> FloatND:
    """Within-period utility reading the CHOSEN next state (NEGM service flow).

    Flat in `move` and in the CURRENT `stock` (both enter with a zero weight, only so
    that every state and action is used), so the action is decided purely by
    `next_stock` -- i.e. by whichever law the flow sub-DAG resolves. Shared verbatim by
    both stakeholders so the household argmax (the `H_linear` sum of the two `Q^s`) is
    unambiguous.
    """
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock


def _flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    """Terminal utility FLAT in the state, so continuation cannot drive the argmax."""
    return 0.0 * stock + 0.0 * move


def _next_stock_belief(move: DiscreteAction) -> FloatND:
    """BELIEF (solve law): `stay` leads to good."""
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_stock_actual(move: DiscreteAction) -> FloatND:
    """TRUTH (simulate law): exactly the opposite -- `switch` leads to good."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _next_regime() -> ScalarInt:
    return RegimeId.last


PARAMS = {"discount_factor": 0.95}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulate(*, live_functions, state_transitions) -> pd.DataFrame:
    live = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        state_transitions=state_transitions,
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions=live_functions,
    )
    last = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility_f": _flat_utility, "utility_m": _flat_utility},
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
            period_to_regime_to_V_arr=V,
            log_level="off",
            seed=1,
        )
        .to_dataframe()
        .reset_index()
    )


def test_collective_flow_reads_the_simulate_variant_of_a_phased_outer_law():
    """A `Phased` `next_<state>` read by a stakeholder utility uses the SIMULATE law.

    Pre-fix, the collective branch dropped `flow_transitions`, so the flow's outer
    `next_stock` was taken from the SOLVE bundle (while its helpers still came from the
    simulate pool -- a sub-DAG that was neither phase), and every household chose
    `stay`: it valued its current service flow under a law it only believes, rather
    than the one it lives in.
    """
    df = _simulate(
        live_functions={"utility_f": _service_flow, "utility_m": _service_flow},
        state_transitions={
            "stock": Phased(solve=_next_stock_belief, simulate=_next_stock_actual)
        },
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the collective FLOW must value the chosen next state under the SIMULATE law"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must follow the SIMULATE law"
    )


def test_collective_flow_reads_the_simulate_variant_of_a_phased_helper():
    """The collective flow is closed: a `Phased` HELPER under the law is simulate too.

    Complements the outer-law case: the law is a single bare function and the
    belief/truth wedge lives entirely in the helper it reads. A flow that mixed the
    solve outer law with simulate helpers -- the exact hybrid the dropped
    `continuation_functions` produced -- is not phase-closed.
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
            "utility_f": _service_flow,
            "utility_m": _service_flow,
            "stay_target": Phased(
                solve=stay_target_belief, simulate=stay_target_actual
            ),
        },
        state_transitions={"stock": next_stock},
    )
    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "the collective FLOW must read the SIMULATE variant of a Phased helper"
    )
    assert (df[df["period"] == 1]["stock"] == "good").all(), (
        "the realized draw must read the SIMULATE variant of a Phased helper"
    )
