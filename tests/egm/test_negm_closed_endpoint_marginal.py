"""Retain selected NEGM economic marginals at valid closed resource nodes."""

import math
import operator
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.current_iid_negm import NEGMCaseUPoint
from _lcm.egm.current_iid_negm_program import NEGMCaseUArguments
from _lcm.egm.current_iid_resource_query import read_resources_candidate
from _lcm.egm.preferences import BoundUtilityOfAction
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.step_core import OwnedResourcesRow, _EgmKernelPieces
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.current_iid_case_u_native import CaseUExactArguments
from _lcm.solution.current_iid_negm import NativeNEGMCaseUPeriodKernel
from _lcm.typing import EconFunctionKwargs
from lcm import IrregSpacedGrid, Model
from lcm.exceptions import UnsupportedOperationError
from lcm.model import _ModelConstructorInputs
from lcm.shocks import ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ResultRetention, SolutionResult
from lcm.typing import ContinuousState, FloatND, RegimeName, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_negm_model import _declarations, _params

pytestmark = [
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
]


type _EndpointOracle = tuple[float, float, float, float, float, int]


class _EndpointConfig(NamedTuple):
    slope: float
    housing: float


class _ClosedCase(NamedTuple):
    model: Model
    solution: SolutionResult
    point: NEGMCaseUPoint
    pool: EconFunctionKwargs
    physical: EconFunctionKwargs
    children: Mapping[RegimeName, EGMCarry]
    keeper: OwnedResourcesRow
    adjusters: OwnedResourcesRow
    native: OwnedResourcesRow
    pieces: _EgmKernelPieces
    frame: EconFunctionKwargs
    resources: ScalarFloat
    oracle: _EndpointOracle
    liquid: float
    housing: float
    slope: float


def _resources_twice(*, wealth: ContinuousState) -> FloatND:
    return 2.0 * wealth


def _resources_decreasing(*, wealth: ContinuousState) -> FloatND:
    return 20.0 - wealth


def _closed_declarations(
    *, slope: float, housing: float
) -> tuple[_ModelConstructorInputs, float]:
    """Keep the real family fixture; move one declared source point to its top.

    The terminal function is log(s) + 2 log(1 + h_next), beta=1, eps=0.
    At a selected target h_next=2 and resources=6, c=s=3 is an original
    savings node and the original native upper endpoint. The target is either
    the keeper (housing=2) or an adjuster (housing=1).
    """
    declarations = _declarations()
    original = declarations.regimes["child"]
    liquid = (8.0 - housing) / slope
    functions = dict(original.functions)
    if slope == 2.0:
        functions["base_resources"] = _resources_twice
    child = replace(
        original,
        states={
            **original.states,
            "wealth": IrregSpacedGrid(
                points=np.asarray([liquid / 4, liquid / 2, liquid])
            ),
        },
        functions=functions,
    )
    # Current shock remains a genuine U input to utility and inverse utility.
    # Its zero-mass second event isolates the exact expected marginal without
    # relying on another off-grid numerical oracle.
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": child},
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=3.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]),
                    weights=np.asarray([1.0, 0.0]),
                ),
            )
        },
    )
    return declarations, liquid


def _independent_oracle(
    *, liquid: float, housing: float, slope: float
) -> _EndpointOracle:
    """Analytic log/log Euler solution and explicit finite outer enumeration.

    This imports no production selector/interpolator. Although losing choices
    can optimize outside the finite savings bank, the winning c=s=3 lies
    exactly in it, so the claimed winner and tuple need no grid-error premise.
    Stable first ownership of an exact tie is part of this fixture.
    """
    rows: list[_EndpointOracle] = []
    for index, target in enumerate((housing, 0.0, 2.0)):
        cash = slope * liquid + housing - target
        saving = max(0.25, cash / 2.0)
        consumption = cash - saving
        rows.append(
            (
                math.log(consumption) + math.log(saving) + 2.0 * math.log1p(target),
                slope / consumption,
                consumption,
                saving,
                target - housing,
                index,
            )
        )
    return max(rows, key=lambda row: row[0])


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(_EndpointConfig(1.0, 2.0), id="keeper_unit"),
        pytest.param(_EndpointConfig(1.0, 1.0), id="adjuster_unit"),
        pytest.param(_EndpointConfig(2.0, 2.0), id="keeper_scaled"),
        pytest.param(_EndpointConfig(2.0, 1.0), id="adjuster_scaled"),
    ],
)
def closed_case(request: pytest.FixtureRequest) -> _ClosedCase:
    config = request.param
    assert isinstance(config, _EndpointConfig)
    slope, housing = config
    declarations, liquid = _closed_declarations(slope=slope, housing=housing)
    model = Model._from_call_owned_params(declarations=declarations, params=_params())
    solution = model.solve(
        params=_params(),
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NEGMCaseUPoint)
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    flat, _generation = model._prepare_call(params=_params())
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NEGMCaseUArguments)
    pool = {
        **union_fixed_params(
            fixed_flat_params=arguments.point.fixed_flat_params,
            regime_name="child",
            transition_target_names=("done",),
        ),
        **union_free_params(
            flat_params=flat,
            regime_name="child",
            transition_target_names=("done",),
        ),
        "period": jnp.int32(1),
        "age": kernel.context.ages.values[1],
        "housing": jnp.asarray(housing),
    }
    physical = {"eps": jnp.asarray(0.0)}
    children = MappingProxyType(dict(future.children))
    keeper, adjusters = point.construct(physical=physical, pool=pool, children=children)
    oracle = _independent_oracle(liquid=liquid, housing=housing, slope=slope)
    index = oracle[-1]
    native = (
        keeper
        if index == 0
        else jax.tree.map(operator.itemgetter(index - 1), adjusters)
    )
    pieces = point.keeper if index == 0 else point.adjuster
    frame = {**pool, **physical, point.euler_state_name: jnp.asarray(liquid)}
    if index:
        frame[point.owner.outer_post_decision] = point.outer_nodes[index - 1]
    resources = pieces.own_resources_func(**frame)
    return _ClosedCase(
        model=model,
        solution=solution,
        point=point,
        pool=pool,
        physical=physical,
        children=children,
        keeper=keeper,
        adjusters=adjusters,
        native=native,
        pieces=pieces,
        frame=frame,
        resources=resources,
        oracle=oracle,
        liquid=liquid,
        housing=housing,
        slope=slope,
    )


def test_closed_native_tuple_keeps_economic_marginal(closed_case: _ClosedCase) -> None:
    case = closed_case
    point, native = case.point, case.native
    finite = np.asarray(native.envelope.grid)
    finite = finite[np.isfinite(finite)]
    # These preconditions make failure discriminating. They must pass before
    # interpreting the final comparison as evidence for the proposed defect.
    assert float(finite[-1]) == float(case.resources) == 6.0
    original = read_resources_candidate(
        row=native,
        resources=case.resources,
        owner_capacity=point.owner_capacity,
        utility_of_action=BoundUtilityOfAction(
            utility_func=case.pieces.utility_func,
            action_name=point.action_name,
            bound=case.frame,
        ),
    )
    assert bool(original.valid)
    assert bool(original.feasible)
    assert not bool(original.corner)
    expected = case.oracle
    np.testing.assert_almost_equal(
        [
            original.value,
            original.marginal * case.slope,
            original.policy,
            original.savings,
        ],
        expected[:4],
        decimal=DECIMAL_PRECISION,
    )
    profile, valid = point._profile(row=native, pieces=case.pieces, actual=case.frame)
    assert bool(valid)
    assert bool(profile.right.dead)  # A dead right extension is valid geometry.
    np.testing.assert_almost_equal(
        profile.right.marginal, expected[1], decimal=DECIMAL_PRECISION
    )
    result = point.read_constructed(
        liquid=jnp.asarray(case.liquid),
        physical=case.physical,
        pool=case.pool,
        keeper=case.keeper,
        adjusters=case.adjusters,
    )
    selected, investment, index = result.output
    assert bool(selected.valid)
    assert bool(selected.feasible)
    assert not bool(selected.corner)
    assert int(index) == expected[-1]
    assert int(selected.native_owner) == int(original.native_owner)
    assert int(selected.side_id) == int(original.side_id)
    np.testing.assert_almost_equal(
        [selected.value, selected.policy, selected.savings, investment],
        [expected[0], expected[2], expected[3], expected[4]],
        decimal=DECIMAL_PRECISION,
    )
    # Economic M belongs to the selected native tuple, separately from its germ.
    np.testing.assert_almost_equal(
        selected.marginal, expected[1], decimal=DECIMAL_PRECISION
    )


def test_closed_marginal_reaches_actual_projected_publication(
    closed_case: _ClosedCase,
) -> None:
    case = closed_case
    stored = case.solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    passive = int(case.housing)
    length = int(stored.valid_length[passive])
    slots = np.flatnonzero(np.asarray(stored.grid[passive, :length]) == case.liquid)
    assert slots.size == 1
    slot = int(slots[0])
    np.testing.assert_almost_equal(
        stored.value[passive, slot], case.oracle[0], decimal=DECIMAL_PRECISION
    )
    # This is a consumed publication channel, not a demand for particular
    # topology metadata or a fabricated projected carry.
    np.testing.assert_almost_equal(
        stored.economic_marginal[passive, slot],
        case.oracle[1],
        decimal=DECIMAL_PRECISION,
    )


def test_neighbor_and_extrapolation_controls(closed_case: _ClosedCase) -> None:
    case = closed_case
    point = case.point
    dtype = np.asarray(point.source_grid).dtype
    q = np.asarray(case.liquid, dtype=dtype)
    below = np.nextafter(q, np.asarray(-np.inf, dtype=dtype))
    above = np.nextafter(q, np.asarray(np.inf, dtype=dtype))
    for coordinate, expected_positive in ((below, True), (above, False)):
        frame = {**case.frame, point.euler_state_name: jnp.asarray(coordinate)}
        profile, valid = point._profile(
            row=case.native, pieces=case.pieces, actual=frame
        )
        assert bool(valid)
        if expected_positive:
            assert float(profile.right.marginal) > 0.0
            assert not bool(profile.right.dead)
        else:
            # Existing strict-above clamp convention is separate from the
            # economic marginal at the exact closed endpoint.
            assert bool(profile.right.dead)
            assert float(profile.right.marginal) == 0.0


def test_negative_resource_jacobian_is_a_declared_scope_refusal() -> None:
    declarations, _liquid = _closed_declarations(slope=1.0, housing=2.0)
    original = declarations.regimes["child"]
    child = replace(
        original,
        functions={**original.functions, "base_resources": _resources_decreasing},
    )
    with pytest.raises(
        UnsupportedOperationError, match=r"positive.*affine|resources proof"
    ):
        Model._from_call_owned_params(
            declarations=replace(
                declarations, regimes={**declarations.regimes, "child": child}
            ),
            params=_params(),
        )
