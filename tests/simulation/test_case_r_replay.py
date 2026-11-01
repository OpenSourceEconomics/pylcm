"""Original resources rows produce actions under the same addressed physical frame."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import InitialNodes, LinSpacedGrid, Model, Periods
from lcm import Regime as UserRegime
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId


def _flow(*, consumption: ContinuousAction) -> FloatND:
    return consumption


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return wealth


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_grid_policy_lookup_retains_the_resolved_solution_schema() -> None:
    """An authenticated ordinary solution still selects its literal grid optimum."""
    model = Model(
        regimes={
            "alive": UserRegime(
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _flow},
            ),
            "done": UserRegime(functions={"utility": _terminal}),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
        state_transitions={"wealth": _terminal},
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_LifecycleRegimeId,
        enable_jit=True,
    )
    params = {"discount_factor": 0.0}
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={"wealth": jnp.asarray([1.0, 2.0])},
    )
    assert lookup.actions is not None
    np.testing.assert_array_equal(lookup.actions["consumption"], [1.0, 1.0])
