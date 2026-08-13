"""Whether a target reading the chosen stock is realized or decision-only.

The stock chosen this period is `new_stock`, an ordinary function of this period's
states and actions; the state law merely carries it forward. Utility reads it because
the service flow is enjoyed now.

Reading it does not by itself make a function decision-only. A phase-invariant
`new_stock` is a function of the row's own state and action, so recomputing it from a
realized row reproduces exactly the value that entered the argmax — such a target is
realized and is published.

A `Phased` `new_stock` is still realized. The flow is *now*, so it is priced under the
simulate variant while only the continuation is priced under the perceived one; the
target recomputes from that same simulate pool and so reproduces the quantity that
entered the argmax.

`_decision_only_target_names` excludes a target whose ancestry reaches a phase-split
or stochastic transition. A regime function cannot read a `next_<state>` at all, and
a transition is not itself an available target, so that exclusion has nothing to fire
on and is not covered here.

No case may raise a confusing missing-argument failure from deep in the target DAG.
"""

import jax.numpy as jnp
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from lcm import AgeGrid, DiscreteGrid, Model, Phased, Regime, categorical
from lcm.typing import DiscreteAction, FloatND, Period, ScalarInt
from tests.conftest import DECIMAL_PRECISION


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


def _service_flow(new_stock: FloatND, move: DiscreteAction, stock: FloatND) -> FloatND:
    """NEGM service-flow utility: reads the stock CHOSEN this period."""
    return 1.0 * new_stock + 0.0 * move + 0.0 * stock


def _flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _new_stock(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _new_stock_believed(move: DiscreteAction) -> FloatND:
    """A perceived rule that inverts the true one, so the two cannot coincide."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _carry_new_stock(new_stock: FloatND) -> FloatND:
    """The law: next period's stock is the one chosen this period."""
    return new_stock


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulated(new_stock=_new_stock):
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": _carry_new_stock},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _service_flow, "new_stock": new_stock},
    ).replace(active=lambda age: age < 2)
    last = Regime(
        transition=None,
        state_transitions={},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _flat_utility},
    ).replace(active=lambda age: age >= 2)
    model = Model(
        regimes={"live": live, "last": last},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        description="transition-dependent target",
    )
    V = model.solve(params=PARAMS, log_level="debug")
    return model.simulate(
        params=PARAMS,
        initial_conditions=IC,
        period_to_regime_to_V_arr=V,
        log_level="debug",
        seed=1,
    )


def test_service_flow_utility_under_a_bare_rule_is_a_realized_target():
    """A phase-invariant `new_stock` makes the service flow realized, so it computes.

    `utility = 1.0 * new_stock` and `new_stock` is `good` (code 1) after `stay`
    and `bad` (code 0) otherwise, so each published row must equal the service flow
    implied by that row's own action — which is what the argmax priced.
    """
    df = _simulated().to_dataframe(additional_targets=["utility"]).reset_index()
    live_rows = df[df["period"] < 2]
    expected = (live_rows["move"] == "stay").astype(float)
    aaae(
        live_rows["utility"].to_numpy(), expected.to_numpy(), decimal=DECIMAL_PRECISION
    )


def test_ordinary_utility_still_computes():
    """A regime whose utility reads no chosen stock is unaffected."""
    df = _simulated().to_dataframe(additional_targets=["utility"]).reset_index()
    assert df[df["period"] == 2]["utility"].notna().all()


def test_service_flow_under_a_phased_rule_is_realized_at_the_true_variant():
    """A `Phased` `new_stock` stays realized, and reports the TRUE service flow.

    The flow is *now*, so it is priced under the simulate variant even though the
    continuation is priced under the perceived one. Recomputing the target resolves
    `new_stock` from the same simulate pool, so it reproduces exactly the quantity
    that entered the simulate argmax — the target is realized, not decision-only.

    Belief and truth are exact opposites here, so the two are distinguishable: under
    the truth `stay` yields `good`, and that is what the agent takes and what is
    published. A build that priced the flow under the belief would choose `switch`
    and report `0.0`.
    """
    result = _simulated(Phased(solve=_new_stock_believed, simulate=_new_stock))
    df = result.to_dataframe(additional_targets=["utility"]).reset_index()
    live_rows = df[df["period"] < 2]
    assert live_rows["utility"].notna().all()
    assert (live_rows["move"] == "stay").all(), (
        "the flow must be priced under the SIMULATE variant of `new_stock`"
    )
    expected = (live_rows["move"] == "stay").astype(float)
    aaae(
        live_rows["utility"].to_numpy(), expected.to_numpy(), decimal=DECIMAL_PRECISION
    )
