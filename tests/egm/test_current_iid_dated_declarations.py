"""Dated preparation source contracts and genuine ordinary fixed-param controls.

These definitions are held for the isolated native-qualified CPU fixture gate.
The metadata observer returns every original result unchanged; it supplies no
solver output, parameter shape, economic continuation, or successful native row.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction, scale: float) -> FloatND:
    return scale * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, scale: float) -> FloatND:
    return scale / marginal_continuation


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _landing(*, savings: FloatND) -> ContinuousState:
    return savings + 1.0


def _terminal(*, wealth: ContinuousState, scale: float) -> FloatND:
    return scale * jnp.log(wealth)


def _ordinary_model(*, fixed_params: UserParams, terminal_start: bool = False) -> Model:
    grid = IrregSpacedGrid(points=np.array([1.0, 2.0, 4.0]))
    live = ConsumptionSavingsRegime(
        states={"wealth": grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.05, 0.5, 2.0]))},
        functions={
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "resources": _resources,
            "savings": _savings,
        },
        state_transitions={"wealth": {"live": _landing, "done": _landing}},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.25, 0.5, 4.0])),
            envelope=ExactEnvelope(),
        ),
    )
    return Model(
        regimes={
            "live": live,
            "done": Regime(states={"wealth": grid}, functions={"utility": _terminal}),
        },
        n_periods=3,
        initial_nodes=InitialNodes(
            by_period={2: "done"} if terminal_start else {0: "live"}
        ),
        edges={
            "live": {
                "live": Periods(values=(0,)),
                "done": Periods(values=(1,)),
            }
        },
        regime_id_class=_RegimeId,
        fixed_params=fixed_params,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_ordinary_fixed_and_free_utility_operands_keep_the_same_values() -> None:
    """Actual ordinary solves retain fixed/free algebra and terminal carry binding."""
    fixed = _ordinary_model(fixed_params={"scale": 2.0}).solve(
        params={"discount_factor": 0.95}, log_level="off"
    )
    free = _ordinary_model(fixed_params={}).solve(
        params={"scale": 2.0, "discount_factor": 0.95}, log_level="off"
    )
    assert fixed.values.keys() == free.values.keys()
    for period, regimes in fixed.values.items():
        assert regimes.keys() == free.values[period].keys()
        for name, value in regimes.items():
            np.testing.assert_array_almost_equal(
                value, free.values[period][name], decimal=DECIMAL_PRECISION
            )
