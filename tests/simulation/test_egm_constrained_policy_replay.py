"""Replay the canonical value of borrowing-constrained consumption policies."""

import math
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lcm import (
    ConsumptionSavingsRegime,
    DiscreteGrid,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.solvers import DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

pytestmark = pytest.mark.slow


@categorical(ordered=False)
class _Job:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _resources(
    *, wealth: ContinuousState, job: DiscreteAction, income: FloatND
) -> FloatND:
    return wealth + income * job


def _utility(
    *, consumption: ContinuousAction, job: DiscreteAction, job_cost: FloatND
) -> FloatND:
    return jnp.log(consumption) - job_cost * job


def _inverse_marginal_utility(*, marginal_continuation: FloatND) -> FloatND:
    return 1 / marginal_continuation


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _bequest(*, wealth: ContinuousState, bequest_slope: FloatND) -> FloatND:
    return bequest_slope * wealth


def _model(
    *,
    wealth: np.ndarray,
    scale: float,
    floor: float,
    cost: float,
    beta: float,
    alpha: float,
    income_factor: float = 1.0,
    lower_wealth: float = 1.0,
    replay: Literal["candidate", "grid"] = "candidate",
    budget: int | None = None,
    enable_jit: bool = True,
) -> Model:
    """Build log consumption with a linear terminal payoff and two income choices."""
    savings_grid = IrregSpacedGrid(points=floor + scale * np.arange(6.0))
    # Include each analytical action so grid replay has no candidate error.
    actions = np.minimum(
        wealth[:, None] + income_factor * np.arange(2), 1 / alpha
    ).ravel()
    action_grid = IrregSpacedGrid(
        points=scale * np.unique(np.r_[0.1, 0.5, 1.0, 2.0, 3.0, actions])
    )
    alive = ConsumptionSavingsRegime(
        states={
            "wealth": IrregSpacedGrid(
                points=floor + scale * np.linspace(lower_wealth, 5.0, 5)
            )
        },
        actions={"consumption": action_grid, "job": DiscreteGrid(category_class=_Job)},
        functions={
            "utility": _utility,
            "resources": _resources,
            "inverse_marginal_utility": _inverse_marginal_utility,
            "savings": _savings,
        },
        state_transitions={"wealth": {"done": _next_wealth}},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        solver=DCEGM(savings_grid=savings_grid, replay=replay),
    )
    return Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={"wealth": savings_grid}, functions={"utility": _bequest}
            ),
        },
        n_periods=2,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"done": Periods(values=(0,))}},
        regime_id_class=_RegimeId,
        fixed_params={
            "discount_factor": beta,
            "income": scale * income_factor,
            "job_cost": cost,
            "bequest_slope": alpha / (beta * scale),
        },
        execution_config=ExecutionConfig(device_memory_bytes=budget),
        enable_jit=enable_jit,
    )


def _assert_panel(
    *,
    wealth: np.ndarray,
    scale: float = 1.0,
    floor: float = 0.0,
    beta: float = 1.0,
    alpha: float = 0.01,
    cost: float,
    income_factor: float = 1.0,
    lower_wealth: float = 1.0,
    replay: Literal["candidate", "grid"] = "candidate",
    budget: int | None = None,
    enable_jit: bool = True,
) -> None:
    model = _model(
        wealth=wealth,
        scale=scale,
        floor=floor,
        cost=cost,
        beta=beta,
        alpha=alpha,
        income_factor=income_factor,
        lower_wealth=lower_wealth,
        replay=replay,
        budget=budget,
        enable_jit=enable_jit,
    )
    result = model.simulate(
        params={},
        seed=0,
        log_level="off",
        initial_conditions={
            "wealth": jnp.asarray(floor + scale * wealth),
            "period": jnp.zeros(wealth.shape, dtype=jnp.int32),
            "regime_id": jnp.zeros(wealth.shape, dtype=jnp.int32),
        },
    ).to_dataframe(use_categorical_codes=True)
    first = result[result["period"] == 0].sort_values("subject_id")
    # Log utility and a linear bequest imply c=min(resources-floor, scale/alpha).
    resources = scale * (wealth[:, None] + income_factor * np.arange(2))
    consumption = np.minimum(resources, scale / alpha)
    savings = floor + resources - consumption
    values = np.log(consumption) - cost * np.arange(2) + alpha * savings / scale
    owner = np.argmax(values, axis=1)
    rows = np.arange(wealth.size)
    assert_array_equal(first["job"], owner)
    assert_allclose(
        first["consumption"],
        consumption[rows, owner],
        rtol=10.0**-DECIMAL_PRECISION,
        atol=10.0**-DECIMAL_PRECISION,
    )
    assert_allclose(
        first["value"],
        values[rows, owner],
        rtol=10.0**-DECIMAL_PRECISION,
        atol=10.0**-DECIMAL_PRECISION,
    )


@pytest.mark.parametrize(
    ("scale", "floor", "beta"), [(1.0, 0.0, 1.0), (3.0, 0.75, 0.5)]
)
@pytest.mark.parametrize(("wealth", "offset"), [(2.0, 0.0003), (2.5, -0.0003)])
@pytest.mark.parametrize("replay", ["candidate", "grid"])
def test_constrained_branch_ranking_uses_attained_values(
    *,
    scale: float,
    floor: float,
    beta: float,
    wealth: float,
    offset: float,
    replay: Literal["candidate", "grid"],
) -> None:
    """Binding branches select the unique analytical optimum between row knots."""
    _assert_panel(
        wealth=np.asarray([wealth]),
        scale=scale,
        floor=floor,
        beta=beta,
        cost=math.log((wealth + 1) / wealth) + offset,
        replay=replay,
    )


@pytest.mark.parametrize(
    ("scale", "floor", "beta", "income"), [(1.0, 0.0, 1.0, 1.0), (3.0, 0.75, 0.5, 0.5)]
)
def test_binding_and_interior_policies_share_the_canonical_objective(
    *, scale: float, floor: float, beta: float, income: float
) -> None:
    """A mixed panel preserves value and branch choice around the Euler boundary."""
    pivot, alpha = 1.4, 0.5
    left, right = min(pivot, 1 / alpha), min(pivot + income, 1 / alpha)
    cost = math.log(right / left) + alpha * (income - right + left)
    _assert_panel(
        wealth=np.asarray(
            [1.1, pivot - 0.02, pivot + 0.02, 1.999, 2.0, 2.001, 2.5, 4.0]
        ),
        scale=scale,
        floor=floor,
        beta=beta,
        alpha=alpha,
        cost=cost,
        income_factor=income,
    )


def test_constrained_row_head_and_first_euler_point_preserve_values() -> None:
    """Values remain canonical below the row head and around the first Euler point."""
    alpha = 0.5
    head = 1e-4 / alpha
    ratio = 10000 ** (1 / 19)
    _assert_panel(
        wealth=np.asarray(
            [
                head * 0.999,
                head,
                head * 1.001,
                head * math.sqrt(ratio),
                1.1,
                1 / alpha - 0.001,
                1 / alpha,
                1 / alpha + 0.001,
                2.5,
                4.0,
            ]
        ),
        alpha=alpha,
        cost=20.0,
        lower_wealth=head * 0.99,
    )


@pytest.mark.parametrize(("budget", "enable_jit"), [(2**28, True), (None, False)])
def test_constrained_policy_replay_preserves_execution_variants(
    *, budget: int | None, enable_jit: bool
) -> None:
    """Borrowing-constrained choices agree in budgeted and eager simulation."""
    _assert_panel(
        wealth=np.asarray([2.0]),
        cost=math.log(1.5) + 0.0003,
        budget=budget,
        enable_jit=enable_jit,
    )
