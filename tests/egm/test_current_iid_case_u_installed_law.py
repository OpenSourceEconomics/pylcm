"""Installed U publication retains current-law support and probability derivatives."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _ChildEulerState
from _lcm.egm.projected_child import ProjectedChildReader
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.execution.core_program import CoreBuildContext
from _lcm.solution.current_iid_case_u_native import acquire_installed_case_u_date
from _lcm.typing import EconFunctionKwargs
from lcm import ConsumptionSavingsRegime, Model, categorical
from lcm.shocks import Discrete, GaussHermite, Normal, Product, ShockGroup
from lcm.solver_api import EGM_CONTINUATION, ArtifactRef, ResultRetention
from lcm.solvers import DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarFloat,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations, _params
from tests.egm.test_current_iid_parent_demands import _resources

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _CurrentOutcome:
    low: ScalarInt
    high: ScalarInt


def _current_probabilities(*, wealth: ContinuousState) -> FloatND:
    high = (wealth - 1.0) / 16.0
    return jnp.stack((1.0 - high, high))


def _current_location(*, wealth: ContinuousState) -> ScalarFloat:
    return wealth / 4.0


def _utility_with_current_law(
    *,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    eps: ScalarInt,
    eta: ScalarFloat,
) -> FloatND:
    return jnp.log(consumption) + (eps + eta) * wealth


def _model_with_current_law() -> Model:
    """Keep the original grid/candidates and replace the declared current law."""
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    assert isinstance(child.solver, DCEGM)
    savings = child.solver.savings_grid.to_jax()
    child = replace(
        child,
        solver=replace(child.solver, current_iid_projection_capacity=8),
        functions={
            **child.functions,
            "utility": _utility_with_current_law,
            "resources": _resources,
        },
    )
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": child},
        shocks={
            "income": ShockGroup(
                shocks={
                    "eps": Discrete(
                        categories=_CurrentOutcome, probs=_current_probabilities
                    ),
                    "eta": Normal(mu=_current_location, sigma=1.0),
                },
                nodes=Product(eta=GaussHermite(n=1)),
            )
        },
    )
    # Both current events have the same optimum c=s=m/2: their additive
    # current utility is independent of savings. These are original grid nodes.
    for optimum in (0.5, 0.75, 6.25):
        assert bool(jnp.any(savings == optimum))
    return Model._from_call_owned_params(declarations=declarations, params=_params())


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_installed_u_current_law_marginal_reaches_original_parent() -> None:
    """Original candidate selection and dated parent consume the full law tangent."""
    params = _params()
    model = _model_with_current_law()
    flat, generation = model._prepare_call(params=params)
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    owner = generation.solver_preparation(regime_name="child", period=1)
    assert prepared.context is owner.context
    assert prepared.context.current_iid_plans[1] is owner.plan
    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    carry = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, ProjectedEGMCarry)
    assert bool(carry.valid)
    live_liquid = carry.grid[: int(carry.valid_length)]
    assert bool(jnp.all((live_liquid >= 1.0) & (live_liquid <= 12.5)))
    probabilities = _current_probabilities(wealth=live_liquid)
    assert bool(jnp.all(jnp.isfinite(probabilities)))
    assert bool(jnp.all((probabilities >= 0.0) & (probabilities <= 1.0)))
    np.testing.assert_array_equal(jnp.sum(probabilities, axis=0), 1.0)
    np.testing.assert_array_equal(
        _current_probabilities(wealth=jnp.asarray(1.0)), [1.0, 0.0]
    )
    # E[eps+eta]=(5m-1)/16; dE/dm=1/16+1/4. At m=1,
    # P(eps=high)=0 but its derivative still contributes m/16=1/16.
    for liquid in (1.0, 1.5, 12.5):
        index = jnp.nonzero(carry.grid == liquid, size=1, fill_value=-1)[0][0]
        assert int(index) >= 0
        assert int(index) < int(carry.valid_length)
        expected_value = 2 * np.log(liquid / 2) + liquid * (5 * liquid - 1) / 16
        expected_marginal = 2 / liquid + (10 * liquid - 1) / 16
        np.testing.assert_almost_equal(
            (carry.value[index], carry.economic_marginal[index]),
            (expected_value, expected_marginal),
            decimal=DECIMAL_PRECISION,
        )
    done = solution.retained_continuations[
        ArtifactRef(period=2, regime="done", key=EGM_CONTINUATION)
    ]
    assert isinstance(done, EGMCarry)
    arguments = prepared.program.argument_builder(
        CoreBuildContext(
            state_action_space=prepared.context.state_action_space,
            next_regime_to_V_arr={},
            next_regime_to_continuation={"done": done},
            flat_params=flat,
            period=1,
            ages=prepared.context.ages,
        )
    )
    pools = arguments["_lcm_parent_demand_pools"]
    assert isinstance(pools, tuple)
    (raw_pool,) = pools
    assert isinstance(raw_pool, Mapping)
    exact_pool = {}
    for name, value in raw_pool.items():
        assert isinstance(name, str)
        assert isinstance(value, jax.Array)
        exact_pool[name] = value
    pool: EconFunctionKwargs = MappingProxyType(exact_pool)
    parent = generation.solver_preparation(regime_name="parent", period=0)
    read = parent.step.pieces.continuation_plan.child_reads["child"]
    assert read.projected_date is prepared.date
    reader = ProjectedChildReader.bind_date(
        date=prepared.date,
        target="child",
        carry=carry,
        next_liquid=_ChildEulerState(
            next_state_func=read.euler_state_func,
            combo_pool=pool,
            post_decision_name=parent.step.pieces.continuation_plan.post_decision_name,
            next_state_key=read.next_state_key,
        ),
    )
    # Actual parent law m=2s+.5, beta=1, so dV/ds=2*dV/dm.
    actual = reader(jnp.asarray(0.5))
    assert actual.query_status is not None
    assert not bool(actual.query_status[0].coverage_missing)
    np.testing.assert_almost_equal(
        actual.output,
        (2 * np.log(0.75) + 1.5 * 6.5 / 16, 2 * (2 / 1.5 + 14 / 16)),
        decimal=DECIMAL_PRECISION,
    )
    parent_values = solution.values[0]["parent"]
    assert isinstance(parent_values, jax.Array)
    assert bool(jnp.all(jnp.isfinite(parent_values)))
