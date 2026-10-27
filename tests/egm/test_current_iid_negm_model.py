"""Finite original NEGM choices are selected before current-shock expectation."""

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from lcm import ExecutionConfig, InitialNodes, IrregSpacedGrid, Model, Regime
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.model import _ModelConstructorInputs
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.solvers import DCEGM, NEGM, ExactEnvelope
from lcm.typing import ContinuousAction, ContinuousState, FloatND, UserParams
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.simulation.test_case_u_reconstruction import _ReplayRegime

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


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
    original = _actual_declarations()
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
    return replace(
        original,
        regimes={
            "child": child,
            "done": Regime(
                states={"wealth": savings, "housing": housing},
                functions={"utility": _terminal},
            ),
        },
        edges={"child": {"done": (1,)}},
        initial_nodes=InitialNodes(by_age={1: "child"}),
        regime_id_class=_ReplayRegime,
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


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_negm_u_selects_complete_outer_choice_before_expectation() -> None:
    """Different shocks choose keeper and adjuster before values are averaged."""
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params()
    )
    solution = model.solve(
        params=_params(),
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    reference = np.mean(
        [_oracle(wealth=2.0, housing=1.0, shock=z)[:2] for z in (0.0, 3.0)], axis=0
    )
    assert _oracle(wealth=2.0, housing=1.0, shock=0.0)[6] == 0
    assert _oracle(wealth=2.0, housing=1.0, shock=3.0)[6] == 1
    np.testing.assert_almost_equal(
        solution.values[1]["child"][1, 1], reference[0], decimal=DECIMAL_PRECISION
    )
    stored = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    slots = np.flatnonzero(np.asarray(stored.grid[1]) == 2.0)
    assert slots.size == 1
    np.testing.assert_almost_equal(
        stored.economic_marginal[1, slots[0]], reference[1], decimal=DECIMAL_PRECISION
    )
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    assert future.child_names == ("done",)
    physical = np.asarray([0.0, 2.0, 3.0])
    result = model.simulate(
        params=_params(),
        solution=solution,
        log_level="off",
        seed=31,
        initial_conditions={
            "wealth": jnp.full((3,), 2.0),
            "housing": jnp.ones((3,)),
            "age": jnp.ones((3,)),
            "regime_id": jnp.full((3,), model.regime_names_to_ids["child"], jnp.int32),
            "eps": PhysicalValues(values=physical),
        },
    )
    actual = result.raw_results["child"][1]
    expected = np.asarray([_oracle(wealth=2.0, housing=1.0, shock=z) for z in physical])
    np.testing.assert_array_almost_equal(
        actual.actions["consumption"], expected[:, 2], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        actual.actions["investment"], expected[:, 4], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual.shocks["eps"], physical)
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        expected[:, 3],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["housing"],
        expected[:, 5],
        decimal=DECIMAL_PRECISION,
    )
    assert float(actual.actions["investment"][2]) not in (-2.0, 0.0, 2.0)
    assert float(actual.actions["consumption"][1]) not in (0.1, 5.0)
    _assert_original_nested_tuple(model=model, future=future)


def _assert_original_nested_tuple(  # noqa: PLR0915 - one genuine tuple and owner boundary
    *, model: Model, future: CaseUReplayChildren
) -> None:
    """The original winning outer node owns its inner saving and native record."""
    import operator  # noqa: PLC0415 - assertions require successful original Model solve
    from types import MappingProxyType  # noqa: PLC0415

    import jax  # noqa: PLC0415

    from _lcm.egm.current_iid_negm import NEGMCaseUPoint  # noqa: PLC0415
    from _lcm.egm.current_iid_negm_program import NEGMCaseUArguments  # noqa: PLC0415
    from _lcm.egm.current_iid_resource_query import (  # noqa: PLC0415
        read_resources_candidate,
    )
    from _lcm.egm.preferences import BoundUtilityOfAction  # noqa: PLC0415
    from _lcm.egm.step import _EGMStep  # noqa: PLC0415
    from _lcm.solution.continuation_target import (  # noqa: PLC0415
        union_fixed_params,
        union_free_params,
    )
    from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
        CaseUExactArguments,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from _lcm.solution.dcegm import _DCEGMPeriodKernel  # noqa: PLC0415
    from _lcm.solution.model_authority import (  # noqa: PLC0415
        _is_installed_negm_u_continuation,
    )

    regime = model._regimes["child"]
    kernel = regime.solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    assert isinstance(point, NEGMCaseUPoint)
    assert point.context is kernel.context
    assert point.plan is kernel.context.current_iid_plans[1]
    assert isinstance(point.owner.keeper_kernel, _DCEGMPeriodKernel)
    assert isinstance(point.owner.adjuster_kernel, _DCEGMPeriodKernel)
    assert isinstance(point.owner.keeper_kernel.demand_step, _EGMStep)
    assert isinstance(point.owner.adjuster_kernel.demand_step, _EGMStep)
    assert point.keeper is point.owner.keeper_kernel.demand_step.pieces
    assert point.adjuster is point.owner.adjuster_kernel.demand_step.pieces
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NEGMCaseUArguments)
    flat, _generation = model._prepare_call(params=_params())
    pool = {
        **union_fixed_params(
            fixed_flat_params=arguments.point.fixed_flat_params,
            regime_name="child",
            transition_target_names=("done",),
        ),
        **union_free_params(
            flat_params=flat, regime_name="child", transition_target_names=("done",)
        ),
        "period": jnp.int32(1),
        "age": kernel.context.ages.values[1],
        "housing": jnp.asarray(1.0),
    }
    physical = {"eps": jnp.asarray(3.0)}
    children = MappingProxyType(dict(future.children))
    actual = point.select(
        liquid=jnp.asarray(2.0), physical=physical, pool=pool, children=children
    )
    selected, investment, outer_index = actual.output
    reference = _oracle(wealth=2.0, housing=1.0, shock=3.0)
    np.testing.assert_array_almost_equal(
        [
            selected.value,
            selected.marginal,
            selected.policy,
            selected.savings,
            investment,
        ],
        reference[:5],
        decimal=DECIMAL_PRECISION,
    )
    assert int(outer_index) == 1
    keeper, adjusters = point.construct(physical=physical, pool=pool, children=children)
    original_row = jax.tree.map(operator.itemgetter(0), adjusters)
    original = read_resources_candidate(
        row=original_row,
        resources=jnp.asarray(3.0),
        owner_capacity=point.owner_capacity,
        utility_of_action=BoundUtilityOfAction(
            utility_func=point.adjuster.utility_func,
            action_name=point.action_name,
            bound={
                **pool,
                **physical,
                point.owner.outer_post_decision: jnp.asarray(0.0),
            },
        ),
    )
    np.testing.assert_array_equal(
        [selected.native_owner, selected.side_id, selected.corner],
        [original.native_owner, original.side_id, original.corner],
    )
    np.testing.assert_array_almost_equal(
        [selected.value, selected.marginal, selected.policy, selected.savings],
        [original.value, original.marginal, original.policy, original.savings],
        decimal=DECIMAL_PRECISION,
    )
    bad = adjusters._replace(
        envelope=adjusters.envelope._replace(
            owner_id=adjusters.envelope.owner_id.at[0].set(point.owner_capacity + 1),
        )
    )
    corrupted = point.read_constructed(
        liquid=jnp.asarray(2.0),
        physical=physical,
        pool=pool,
        keeper=keeper,
        adjusters=bad,
    )
    assert not bool(corrupted.output[0].valid)
    with pytest.raises(TypeError, match="original nested owner"):
        replace(point, plan=replace(point.plan))
    with pytest.raises(TypeError, match="genuine inner pieces"):
        replace(point, keeper=replace(point.keeper))
    template = kernel.template.output.projected
    assert _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=kernel.spec,
        template=template,
    )
    assert not _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=replace(kernel.spec),
        template=template,
    )
    assert not _is_installed_negm_u_continuation(
        regime=regime,
        regime_name="child",
        period=1,
        spec=kernel.spec,
        template=replace(template),
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_negm_u_inner_preparation_keeps_dated_branch_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuine keeper/adjuster source custody cannot grant ordinary publication."""
    from types import MappingProxyType  # noqa: PLC0415

    from _lcm.egm.current_iid_envelope import (  # noqa: PLC0415
        DatedCaseUStepAuthority,
    )
    from _lcm.solution.contract import (  # noqa: PLC0415
        SolutionKernels,
        SolverBuildContext,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from _lcm.solution.dcegm import EGMStepBuild  # noqa: PLC0415
    from lcm.exceptions import UnsupportedOperationError  # noqa: PLC0415

    captured: list[
        tuple[DCEGM, SolverBuildContext, EGMStepBuild, DatedCaseUStepAuthority]
    ] = []
    original = DCEGM.finish_nested_current_u_period_kernels

    # keyword-only-exempt: library-callback=python.method
    def observe(
        self: DCEGM,
        *,
        context: SolverBuildContext,
        build: EGMStepBuild,
        authority: DatedCaseUStepAuthority,
    ) -> SolutionKernels:
        captured.append((self, context, build, authority))
        return original(self, context=context, build=build, authority=authority)

    monkeypatch.setattr(DCEGM, "finish_nested_current_u_period_kernels", observe)
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params()
    )
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    assert len(captured) == 2
    assert {
        authority.nested_origin.keeper
        for _solver, _context, _build, authority in captured
        if authority.nested_origin is not None
    } == {False, True}
    for solver, context, build, authority in captured:
        branch = authority.nested_origin
        assert branch is not None
        source = branch.source
        assert source.solver is context.user_regimes["child"].solver
        assert source.plan is kernel.point.plan
        assert source.context.current_iid_plans[1] is source.plan
        assert context.current_iid_plans is source.context.current_iid_plans
        assert context.laws is source.context.laws
        assert context.ages is source.context.ages
        assert build.native_case_u_sources[1] is authority
        authority.require(context=context, solver=branch.solver, period=1)
        with pytest.raises(UnsupportedOperationError, match="cannot publish ordinary"):
            solver.finish_period_kernels(context=context, build=build)
        with pytest.raises(UnsupportedOperationError, match="prepared build identity"):
            original(solver, context=context, build=replace(build), authority=authority)
        with pytest.raises(UnsupportedOperationError, match="source/branch bindings"):
            replace(authority, plan=replace(source.plan)).require(
                context=context, solver=branch.solver, period=1
            )
        with pytest.raises(UnsupportedOperationError, match="source/branch bindings"):
            replace(authority, date=replace(source.date)).require(
                context=context, solver=branch.solver, period=1
            )
        with pytest.raises(UnsupportedOperationError, match="dated outer source"):
            replace(source, solver=replace(source.solver)).require(
                context=source.context, solver=source.solver
            )
        changed = replace(
            context,
            functions=MappingProxyType({**context.functions, "utility": _terminal}),
        )
        changed_branch = replace(branch, context=changed)
        with pytest.raises(UnsupportedOperationError, match="branch function pool"):
            replace(authority, context=changed, nested_origin=changed_branch).require(
                context=changed, solver=branch.solver, period=1
            )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_negm_u_keeper_authority_refuses_moved_raw_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An original wrapped marker cannot authorize a foreign numerical callee."""
    from types import CellType, FunctionType, MappingProxyType  # noqa: PLC0415

    from _lcm.egm.current_iid_negm_origin import (  # noqa: PLC0415
        DatedNEGMCaseUSource,
    )
    from _lcm.solution.contract import (  # noqa: PLC0415
        SolutionKernels,
        SolverBuildContext,
    )
    from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415
    from lcm.exceptions import UnsupportedOperationError  # noqa: PLC0415

    original = DatedNEGMCaseUSource.build_inner
    checked: list[bool] = []

    # keyword-only-exempt: library-callback=python.method
    def replace_keeper(
        self: DatedNEGMCaseUSource,
        *,
        context: SolverBuildContext,
        solver: _BoundDCEGM,
        keeper: bool,
    ) -> SolutionKernels:
        if keeper:
            function = context.functions[self.solver.outer_post_decision]
            assert isinstance(function, FunctionType)
            assert function.__closure__ is not None
            cells = dict(
                zip(function.__code__.co_freevars, function.__closure__, strict=True)
            )
            raw = cells["func"].cell_contents
            assert vars(function)["__wrapped__"] is raw
            # The original wrapper still advertises its canonical raw callback,
            # but calls a foreign function through the actual named func cell.
            changed = {
                **cells,
                "func": CellType(_terminal),
                "enforce": CellType(raw),
            }
            forged = FunctionType(
                function.__code__,
                function.__globals__,
                function.__name__,
                function.__defaults__,
                tuple(changed[name] for name in function.__code__.co_freevars),
            )
            forged.__dict__.update(function.__dict__)
            assert forged.__code__ is function.__code__
            assert vars(forged)["__wrapped__"] is raw
            assert any(cell.cell_contents is raw for cell in forged.__closure__ or ())
            changed_context = replace(
                context,
                functions=MappingProxyType(
                    {**context.functions, self.solver.outer_post_decision: forged}
                ),
            )
            checked.append(True)
            return original(self, context=changed_context, solver=solver, keeper=keeper)
        return original(self, context=context, solver=solver, keeper=keeper)

    monkeypatch.setattr(DatedNEGMCaseUSource, "build_inner", replace_keeper)
    with pytest.raises(UnsupportedOperationError, match="branch function pool"):
        Model._from_call_owned_params(declarations=_declarations(), params=_params())
    assert checked == [True]


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_negm_u_ordinary_passive_child_rows_keep_generation_custody() -> None:
    """Actual ordinary housing rows remain original numeric generation roots."""
    from _lcm.egm.current_iid_case_u_replay import (  # noqa: PLC0415
        _require_ordinary_passive_child_rows,
    )
    from _lcm.solution.current_iid_negm import (  # noqa: PLC0415
        NativeNEGMCaseUPeriodKernel,
    )
    from lcm.exceptions import UnsupportedOperationError  # noqa: PLC0415

    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params()
    )
    _flat, generation = model._prepare_call(params=_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNEGMCaseUPeriodKernel)
    point = kernel.point
    for pieces in (point.keeper, point.adjuster):
        read = pieces.continuation_plan.child_reads["done"]
        spec = kernel.future.specs["done"]
        assert read.passive_state_names == ("housing",)
        assert read.row_arg_names == ("housing",)
        assert read.row_block_shape == (3,)
        assert read.row_values
        assert read.projected_date is None
        _require_ordinary_passive_child_rows(read=read, spec=spec, period=2)
        for array in (*read.passive_grids, *read.row_values, read.age_values):
            assert any(root is array for root in generation.roots)
        with pytest.raises(UnsupportedOperationError, match="passive-row schema"):
            _require_ordinary_passive_child_rows(
                read=replace(read, row_arg_names=()), spec=spec, period=2
            )
        with pytest.raises(UnsupportedOperationError, match="passive-row schema"):
            _require_ordinary_passive_child_rows(
                read=replace(read, row_values=(read.row_values[0][:-1],)),
                spec=spec,
                period=2,
            )
        with pytest.raises(UnsupportedOperationError, match="passive-row schema"):
            _require_ordinary_passive_child_rows(
                read=replace(read, projected_date=kernel.spec.date_at(period=1)),
                spec=spec,
                period=2,
            )
