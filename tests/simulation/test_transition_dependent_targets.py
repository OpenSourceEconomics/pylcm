"""A utility/constraint reading a chosen `next_<state>` is a decision-only target.

The chosen next state exists only inside the argmax, not in realized simulation
data, and the realized-target pool omits state transitions. Such a function must
therefore NOT be computed as a realized additional target: exposing it produced a
confusing "missing next_<state> argument" failure deep in the target DAG. It is now
excluded per regime, so a service-flow regime's rows are left unfilled rather than
crashing, while a regime whose utility is ordinary still computes.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import AgeGrid, DiscreteGrid, Model, Regime, categorical
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


def _service_flow(next_stock: FloatND, move: DiscreteAction, stock: FloatND) -> FloatND:
    """NEGM service-flow utility: reads the CHOSEN next state."""
    return 1.0 * next_stock + 0.0 * move + 0.0 * stock


def _flat_utility(stock: FloatND, move: DiscreteAction) -> FloatND:
    return 0.0 * stock + 0.0 * move


def _next_stock(move: DiscreteAction) -> FloatND:
    return jnp.where(move == Move.stay, Stock.good, Stock.bad)


def _next_regime(period: Period) -> ScalarInt:
    return jnp.where(period >= 1, RegimeId.last, RegimeId.live)


PARAMS = {"discount_factor": 0.95, "live": {}, "last": {}}
IC = pd.DataFrame({"regime_name": "live", "age": 0, "stock": ["bad"] * 8})


def _simulated():
    live = Regime(
        transition=_next_regime,
        state_transitions={"stock": _next_stock},
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


def test_transition_dependent_utility_does_not_crash_as_a_target():
    """Requesting a service-flow `utility` as a realized target used to raise a
    confusing missing-`next_stock` ValueError. It must now succeed: the service-flow
    (`live`) regime rows are left unfilled, the flat (`last`) regime rows compute.
    """
    result = _simulated()
    df = result.to_dataframe(additional_targets=["utility"]).reset_index()
    live_rows = df[df["period"] < 2]
    last_rows = df[df["period"] == 2]
    assert live_rows["utility"].isna().all(), (
        "a service-flow regime's utility is decision-only, not a realized target"
    )
    assert last_rows["utility"].notna().all(), (
        "a regime with an ordinary utility must still compute it"
    )
