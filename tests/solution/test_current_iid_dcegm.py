"""Current resources draws are selected before their values are integrated."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

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
)
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class ResourcesRegimeId:
    saving: ScalarInt
    done: ScalarInt


def _resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _terminal_utility(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _resources_only_model() -> Model:
    """A declared pre-state domain and an exact-node log Euler problem."""
    savings_grid = IrregSpacedGrid(points=np.array([0.5, 1.0, 2.0, 3.0, 4.0]))
    saving = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=np.array([1.0, 3.0]))},
        # Native Euler candidates 1, 2, 3 are deliberately not this grid's nodes.
        actions={"consumption": LinSpacedGrid(start=0.25, stop=4.0, n_points=2)},
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse_marginal_utility,
        },
        state_transitions={"wealth": {"done": _next_wealth}},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(savings_grid=savings_grid, envelope=ExactEnvelope()),
        shocks=("income",),
    )
    done = Regime(
        states={"wealth": savings_grid},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"saving": saving, "done": done},
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "saving"}),
        edges={"saving": {"done": Periods(values=(0,))}},
        regime_id_class=ResourcesRegimeId,
        fixed_params={"discount_factor": 1.0},
        shocks={
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[1.0], [3.0]]),
                    weights=np.array([0.5, 0.5]),
                ),
            )
        },
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_resources_only_current_iid_dcegm_integrates_nodewise_log_optima() -> None:
    """Each positive resources draw uses its continuous log Euler optimum."""
    model = _resources_only_model()
    solution = model.solve(params={}, log_level="off")

    # beta=1 gives c=s=m/2, V=log(c)+log(s)=2log(m/2).
    # Every optimal saving lies exactly on the declared native savings grid.
    expected = np.array(
        [
            0.5 * (2.0 * math.log(1.0)) + 0.5 * (2.0 * math.log(2.0)),
            0.5 * (2.0 * math.log(2.0)) + 0.5 * (2.0 * math.log(3.0)),
        ]
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["saving"], expected, decimal=DECIMAL_PRECISION
    )
