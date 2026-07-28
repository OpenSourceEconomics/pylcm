"""Phase resolution is SLOT-LOCAL: it does not reach a nested plain-Python call.

`test_flow_transitions_are_phase_closed.py` pins that a `Phased` outer law, and a
`Phased` DAG HELPER under it, both resolve to the simulate variant in the flow. This
module pins the boundary of that guarantee, which is where the guarantee is easiest to
over-read:

    declaring `state_transitions["a"] = Phased(solve=f_solve, simulate=f_sim)`
    phases the slot "a" -- and NOTHING else.

If some other slot's law calls `f_solve` as an ordinary Python function, that call is
invisible to phase resolution: it is not a DAG node, so there is no slot to rewrite. The
second slot silently keeps the SOLVE behaviour in both phases.

This is not hypothetical. It is the defect class `phase-incomplete-state-consumer-closure`
found in the EKL (2019) replication: phasing the `experience` transition left
`_job_offer_probs` -- a *different* transition that called the capped `_next_experience`
helper directly -- computing its offer/retention logits on capped experience in the
simulate phase, so only part of the intended belief/truth wedge took effect. The repair is
never "phase the state harder"; it is to enumerate every consumer and phase each slot, or
route them all through one phase-resolved DAG helper.

Both cases below are asserted, so the test documents the defect AND its repair.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import AgeGrid, DiscreteGrid, Model, Phased, Regime, categorical
from lcm.typing import DiscreteAction, FloatND, Period, ScalarInt, UserFunction


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


def _next_stock_belief(move: DiscreteAction) -> FloatND:
    """BELIEF (solve): `stay` leads to good."""
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_stock_actual(move: DiscreteAction) -> FloatND:
    """TRUTH (simulate): exactly the opposite -- `switch` leads to good."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def next_tag_calling_the_solve_helper(move: DiscreteAction) -> FloatND:
    """A SECOND slot's law that calls the belief helper by name.

    The nested call is plain Python, not a DAG edge, so nothing in the model can tell
    that this slot also consumes the phased quantity.
    """
    return _next_stock_belief(move)


def next_tag_calling_the_simulate_helper(move: DiscreteAction) -> FloatND:
    """The simulate twin of `next_tag_calling_the_solve_helper` (the repair)."""
    return _next_stock_actual(move)


def service_flow(
    next_stock: FloatND, move: DiscreteAction, stock: FloatND, tag: FloatND
) -> FloatND:
    """Within-period utility reading the CHOSEN next state (NEGM service flow).

    Flat in everything except `next_stock`, so the argmax is decided purely by the law
    the FLOW resolves -- which `test_flow_transitions_are_phase_closed.py` already pins
    as the simulate one. That makes `move == switch` a fixed, known input here, and the
    realized `stock` / `tag` the only things under test.
    """
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock + 0.0 * tag


def flat_utility(stock: FloatND, move: DiscreteAction, tag: FloatND) -> FloatND:
    """Terminal utility FLAT in the states, so continuation cannot drive the argmax."""
    return 0.0 * stock + 0.0 * move + 0.0 * tag


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame(
    {"regime_name": "live", "age": 0, "stock": ["bad"] * 8, "tag": ["bad"] * 8}
)
_STATES = {"stock": DiscreteGrid(Stock), "tag": DiscreteGrid(Stock)}


def _simulate(tag_law: UserFunction | Phased) -> pd.DataFrame:
    live = Regime(
        transition=_next_regime,
        state_transitions={
            "stock": Phased(solve=_next_stock_belief, simulate=_next_stock_actual),
            "tag": tag_law,
        },
        states=_STATES,
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": service_flow},
    ).replace(active=lambda age: age < 2)
    last = Regime(
        transition=None,
        state_transitions={},
        states=_STATES,
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": flat_utility},
    ).replace(active=lambda age: age >= 2)
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="slot-locality of phase resolution",
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


def test_an_unphased_slot_keeps_the_solve_behaviour_even_next_to_a_phased_one():
    """Phasing `stock` does NOT phase `tag`, which calls the solve helper directly.

    This asserts the trap, not a wish: the two slots consume the SAME quantity and
    DIVERGE, because only one of them was declared `Phased`. A future engine change
    that started rewriting nested Python calls would flip this -- deliberately, and
    with this test as the place to say so.
    """
    df = _simulate(next_tag_calling_the_solve_helper)
    landed = df[df["period"] == 1]

    assert (df[df["period"] == 0]["move"] == "switch").all(), (
        "precondition: the flow must resolve the SIMULATE law (see "
        "test_flow_transitions_are_phase_closed.py)"
    )
    assert (landed["stock"] == "good").all(), (
        "the phased slot must realize the SIMULATE law"
    )
    assert (landed["tag"] == "bad").all(), (
        "the UNPHASED slot must keep the SOLVE law -- phase resolution is slot-local "
        "and cannot see a nested plain-Python call to the solve helper"
    )


def test_phasing_every_consumer_closes_the_gap():
    """The repair: phase the second slot too, and both follow the simulate law.

    Note what the fix is NOT -- nothing about the `stock` declaration changed. Closure
    is achieved by enumerating consumers, which is why the EKL repair had to split its
    job-offer transition into capped/raw twins rather than adjust the experience state.
    """
    df = _simulate(
        Phased(
            solve=next_tag_calling_the_solve_helper,
            simulate=next_tag_calling_the_simulate_helper,
        )
    )
    landed = df[df["period"] == 1]

    assert (landed["stock"] == "good").all()
    assert (landed["tag"] == "good").all(), (
        "with every consumer phased, the second slot follows the SIMULATE law too"
    )
