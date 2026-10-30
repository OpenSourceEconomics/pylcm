"""A nested housing-and-savings regime whose utility reads a current-period shock.

The `child` regime chooses consumption and a housing investment after drawing
`eps`, then moves to a terminal `done` regime. `_oracle` solves its decision in
closed form, so tests can check a simulated decision against an exact one.
"""

import math

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Regime,
    categorical,
)
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.model import _ModelConstructorInputs
from lcm.shocks import ShockGroup, Uniform, UserNodes
from lcm.solvers import DCEGM, NEGM, ExactEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)


@categorical(ordered=False)
class ReplayRegime:
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return (1 + eps) * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, eps: FloatND) -> FloatND:
    return (1 + eps) / marginal_continuation


def _base_resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _new_housing(*, housing: ContinuousState, investment: ContinuousAction) -> FloatND:
    return housing + investment


def _cost(*, housing: ContinuousState, new_housing: FloatND) -> FloatND:
    return new_housing - housing


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(*, savings: FloatND) -> FloatND:
    return savings


def _next_housing(*, new_housing: FloatND) -> FloatND:
    return new_housing


def _terminal(*, wealth: ContinuousState, housing: ContinuousState) -> FloatND:
    return jnp.log(wealth) + 2 * jnp.log1p(housing)


def _params() -> UserParams:
    return {"discount_factor": 1.0, "child": {}}


def _declarations() -> _ModelConstructorInputs:
    """Declare the NEGM `child` at age 1 and its terminal `done` at age 2."""
    savings = IrregSpacedGrid(
        points=np.asarray([0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0])
    )
    housing = IrregSpacedGrid(points=np.asarray([0.0, 1.0, 2.0]))
    child = NestedConsumptionSavingsRegime(
        states={
            "wealth": IrregSpacedGrid(points=np.asarray([1.5, 2.0, 3.0])),
            "housing": housing,
        },
        actions={
            "consumption": IrregSpacedGrid(points=np.asarray([0.1, 5.0])),
            "investment": IrregSpacedGrid(points=np.asarray([-2.0, 0.0, 2.0])),
        },
        functions={
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "base_resources": _base_resources,
            "new_housing": _new_housing,
            "cost": _cost,
            "savings": _savings,
        },
        state_transitions={
            "wealth": {"done": _next_wealth},
            "housing": {"done": _next_housing},
        },
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=NetOfAdjustmentCost(
                output="resources", before_cost="base_resources", cost="cost"
            ),
            post_decision_state="savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="housing",
            action="investment",
            post_decision_state="new_housing",
            no_adjustment=outer_unchanged,
        ),
        solver=NEGM(
            inner=DCEGM(savings_grid=savings, envelope=ExactEnvelope()),
            outer_grid=IrregSpacedGrid(points=np.asarray([0.0, 2.0])),
        ),
        shocks=("income",),
    )
    return _ModelConstructorInputs(
        regimes={
            "child": child,
            "done": Regime(
                states={"wealth": savings, "housing": housing},
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=ReplayRegime,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=3.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
    )


def _oracle(*, wealth: float, housing: float, shock: float) -> tuple[float, ...]:
    """Solve the liquid Euler equation and enumerate keeper then declared nodes."""
    candidates = []
    for index, target in enumerate((housing, 0.0, 2.0)):
        resources = wealth + housing - target
        saving = max(0.25, resources / (2 + shock))
        consumption = resources - saving
        value = (
            (
                (1 + shock) * math.log(consumption)
                + math.log(saving)
                + 2 * math.log1p(target)
            )
            if consumption > 0
            else -math.inf
        )
        candidates.append(
            (
                value,
                (1 + shock) / consumption if consumption > 0 else 0.0,
                consumption,
                saving,
                target - housing,
                target,
                float(index),
            )
        )
    return max(candidates, key=lambda item: item[0])
