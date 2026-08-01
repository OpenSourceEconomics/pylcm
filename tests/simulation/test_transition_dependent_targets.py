"""Whether a target reading a chosen `next_<state>` is realized or decision-only.

Reading a chosen next state does not by itself make a function decision-only. A
deterministic, phase-invariant transition is a function of the row's own state and
action, so recomputing it from a realized row reproduces exactly the value that
entered the argmax — such a target is realized and is published.

A `Phased` transition is different: the perceived (solve) law prices the decision
while the true (simulate) law governs the realized draw. Recomputing under the
objective law yields a well-defined number that is *not* the quantity the agent
decided on, so such a target is decision-only and its rows are left unfilled.

Neither case may raise the confusing "missing `next_<state>` argument" failure from
deep in the target DAG.
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


def _service_flow(next_stock: FloatND, move: DiscreteAction, stock: FloatND) -> FloatND:
    """NEGM service-flow utility: reads the CHOSEN next state."""
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock


def _flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _next_stock(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_stock_believed(move: DiscreteAction) -> FloatND:
    """A perceived law that inverts the true one, so the two cannot coincide."""
    return jnp.where(move == Move.stay, Stock.bad, Stock.good)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulated(stock_law=_next_stock):
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": stock_law},
        states={"stock": DiscreteGrid(Stock)},
        actions={"move": DiscreteGrid(Move)},
        functions={"utility": _service_flow},
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
    V = model.solve(params=PARAMS, log_level="off")
    return model.simulate(
        params=PARAMS,
        initial_conditions=IC,
        period_to_regime_to_V_arr=V,
        log_level="off",
        seed=1,
    )


def test_service_flow_utility_under_a_bare_law_is_a_realized_target():
    """A phase-invariant `next_stock` makes the service flow realized, so it computes.

    `utility = 1.0 * next_stock` and `next_stock` is `good` (code 1) after `stay`
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
    """A regime whose utility reads no transition is unaffected."""
    df = _simulated().to_dataframe(additional_targets=["utility"]).reset_index()
    assert df[df["period"] == 2]["utility"].notna().all()


def test_service_flow_utility_under_a_phased_law_is_decision_only():
    """A `Phased` law makes the service flow decision-only, so its rows stay unfilled.

    The agent decides under the perceived law and the realized draw follows the true
    one, so recomputing the service flow from a realized row would report a
    well-defined number that is not the quantity that entered the argmax.
    """
    result = _simulated(Phased(solve=_next_stock_believed, simulate=_next_stock))
    df = result.to_dataframe(additional_targets=["utility"]).reset_index()
    assert df[df["period"] < 2]["utility"].isna().all()
