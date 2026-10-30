"""DC-EGM publishes the closed-form value wherever a literal savings floor binds.

One decision period before a terminal regime, deterministic resources `R = w + 1`,
log felicity, terminal utility `log(wealth)`, no discounting, and a savings floor of
1.5 at the savings-grid origin. Saving `R / 2` is optimal while that clears the floor,
so the value is `2 log(R / 2)` for `R >= 3` and `log(R - 1.5) + log(1.5)` below it,
where the floor binds and the agent consumes everything above it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from lcm import (
    ConsumptionSavingsRegime,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    Periods,
    Regime,
    categorical,
    post_decision_lower_bound,
)
from lcm.solvers import DCEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_FLOOR = 1.5
# Resources 1.6, 1.7, ..., 6.0: the cells below the kink at R = 3 run from just above
# the floor to just below the first Euler point, and the cells above it stretch the
# constrained candidates all the way up to that point.
_WEALTH = LinSpacedGrid(start=0.6, stop=5.0, n_points=45)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _resources(*, wealth: ContinuousState) -> FloatND:
    return wealth + 1.0


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _terminal_utility(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


_MARGIN = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def _model(*, n_savings: int) -> Model:
    savings_grid = IrregSpacedGrid(points=np.linspace(_FLOOR, 6.0, n_savings))
    alive = ConsumptionSavingsRegime(
        states={"wealth": _WEALTH},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=6.0, n_points=12)},
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse_marginal_utility,
        },
        state_transitions={"wealth": {"done": _next_wealth}},
        constraints={
            "borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=_FLOOR)
        },
        liquid=_MARGIN,
        solver=DCEGM(savings_grid=savings_grid),
    )
    done = Regime(
        states={"wealth": savings_grid}, functions={"utility": _terminal_utility}
    )
    return Model(
        regimes={"alive": alive, "done": done},
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_RegimeId,
        fixed_params={"discount_factor": 1.0},
    )


@pytest.mark.parametrize("n_savings", [12, 96])
def test_binding_floor_cells_publish_the_closed_form_value(n_savings: int) -> None:
    """Below the kink the published value is `log(R - 1.5) + log(1.5)`."""
    model = _model(n_savings=n_savings)
    published = np.asarray(model.solve(params={}, log_level="off").values[0]["alive"])
    resources = np.asarray(_WEALTH.to_jax()) + 1.0
    binding = resources < 2 * _FLOOR
    expected = np.log(resources[binding] - _FLOOR) + np.log(_FLOOR)

    assert_array_almost_equal(published[binding], expected, decimal=DECIMAL_PRECISION)
