"""Original scalar NB candidates survive private Model solve and physical replay."""

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType, MethodType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.current_iid_nbegm import NBCaseUArguments, NBCaseUPoint, NBCaseUProgram
from _lcm.egm.current_iid_nbegm_r import NBCaseRProgram
from _lcm.egm.current_iid_pullback import certify_resources_map
from _lcm.egm.preferences import _PreferencesBuilder
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.upper_envelope.certified_sign import backend_flushes_subnormals
from _lcm.execution.core_program import CoreBuildContext
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.current_iid_case_r_build import _PositionalResources
from _lcm.solution.current_iid_case_u_build import guard_case_u_shape
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactFunction,
    NativeCaseUPeriodKernel,
)
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
from _lcm.solution.current_iid_nbegm_r import NativeNBCaseRPeriodKernel
from _lcm.solution.feasibility import BoundedResult
from _lcm.solution.nbegm import (
    _DeferredProbe,
    _evaluate_liquid_probe_program,
    _fail_if_budget_nonaffine_in_liquid,
    _liquid_affinity_grid_samples,
    _NBEGMContinuousCore,
    _NBEGMSourcePoint,
)
from _lcm.solution.preconditions import check_solver_params
from _lcm.typing import PytreeValue
from lcm import (
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    Model,
    affine_breakpoint,
    categorical,
    piecewise_affine,
    post_decision_lower_bound,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    ModelInitializationError,
    RegimeInitializationError,
    UnsupportedOperationError,
)
from lcm.model import _ModelConstructorInputs
from lcm.shocks import Discrete, PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import EGM_CONTINUATION, ArtifactRef, ArtifactStore, ResultRetention
from lcm.solvers import DCEGM, NBEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    FloatND,
    IntND,
    ReferenceName,
    ScalarInt,
    UserParams,
    ValueND,
)
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_balanced import _model as balanced_model
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.egm.test_finite_nnbegm_query import _query
from tests.simulation.test_case_u_reconstruction import _ReplayRegime

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _mixed_dtype_resources(
    *, wealth: FloatND, category: IntND, period: IntND, active: BoolND
) -> FloatND:
    return wealth + jnp.where(active, category + period, 0)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_resources_certificate_accepts_actual_category_and_clock_dtypes() -> (
    None
):
    """Both original resource builders supply dtype-polymorphic array operands."""
    positional = _PositionalResources(
        function=_mixed_dtype_resources,
        names=("wealth", "category", "period", "active"),
    )
    dtype = jnp.asarray(0.0).dtype
    certificate = certify_resources_map(
        func=positional,
        state_name="wealth",
        abstract_args=(
            jax.ShapeDtypeStruct((), dtype),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ),
        arg_names=positional.names,
    )
    assert certificate.violation is None
    assert certificate.coefficient == 1
    actual = jax.jit(positional)(
        jnp.asarray(2.0), jnp.int32(3), jnp.int32(4), jnp.asarray(a=True)
    )
    np.testing.assert_array_equal(actual, 9.0)
    assert actual.dtype == dtype
    with pytest.raises(TypeError):
        jax.make_jaxpr(positional)("not-an-array")


def _utility(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
    return (1 + eps) * jnp.log(consumption)


def _inverse(*, marginal_continuation: FloatND, eps: FloatND) -> FloatND:
    return (1 + eps) / marginal_continuation


def _resources(*, wealth: FloatND) -> FloatND:
    return wealth


def _utility_with_backend_probe(*, consumption: FloatND, eps: FloatND) -> FloatND:
    if backend_flushes_subnormals(consumption.dtype):
        return (1 + eps) * jnp.log(consumption)
    return (1 + eps) * jnp.log(consumption)


def _curved_resources(*, wealth: FloatND, curvature: FloatND) -> FloatND:
    return wealth + curvature * wealth**2


@piecewise_affine(
    output="rebate",
    variable="wealth",
    breakpoints=(affine_breakpoint(threshold="cutoff", kind="continuous_kink"),),
)
def _rebate(*, wealth: FloatND, cutoff: FloatND) -> FloatND:
    return -0.5 * jnp.maximum(wealth - cutoff, 0.0)


def _kink_resources(*, wealth: FloatND, rebate: FloatND) -> FloatND:
    return wealth + rebate


def _nb_declarations(
    *, kink: bool = False, device_memory_bytes: int | None = None
) -> _ModelConstructorInputs:
    declarations = _actual_declarations()
    old = declarations.regimes["child"]
    assert isinstance(old, ConsumptionSavingsRegime)
    assert isinstance(old.solver, DCEGM)
    wealth_nodes = [0.5, 1.0, 2.0, 4.0] if kink else [0.5, 1.5, 3.0]
    savings = (
        IrregSpacedGrid(
            points=np.asarray([0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5, 2.0, 4.0])
        )
        if kink
        else old.solver.savings_grid
    )
    child = replace(
        old,
        states={"wealth": IrregSpacedGrid(points=np.asarray(wealth_nodes))},
        actions={"consumption": IrregSpacedGrid(points=np.asarray([0.1, 4.0]))},
        functions={
            **old.functions,
            "utility": _utility,
            "inverse_marginal_utility": _inverse,
            "resources": _kink_resources if kink else _resources,
            **({"rebate": _rebate} if kink else {}),
        },
        solver=NBEGM(savings_grid=savings),
    )
    return replace(
        declarations,
        regimes={
            "child": child,
            "done": replace(declarations.regimes["done"], states={"wealth": savings}),
        },
        edges={"child": {"done": (1,)}},
        initial_nodes=InitialNodes(by_age={1: "child"}),
        regime_id_class=_ReplayRegime,
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
    )


def _model(*, kink: bool = False, device_memory_bytes: int | None = None) -> Model:
    return Model._from_call_owned_params(
        params=_nb_params(kink=kink),
        declarations=_nb_declarations(
            kink=kink, device_memory_bytes=device_memory_bytes
        ),
    )


def _nb_params(*, kink: bool = False) -> UserParams:
    return {
        "discount_factor": 1.0,
        "child": {
            "next_wealth": {"slope": 1.0},
            **({"rebate": {"cutoff": 2.0}} if kink else {}),
        },
    }


def _oracle(*, wealth: float, shock: float) -> tuple[float, float, float, float]:
    alpha = 1 + shock
    saving = max(0.25, wealth / (1 + alpha))
    consumption = wealth - saving
    return (
        alpha * math.log(consumption) + math.log(saving),
        alpha / consumption,
        consumption,
        saving,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_budgeted_nbegm_preserves_original_literal_floor_validation() -> None:
    declarations = _nb_declarations(device_memory_bytes=1 << 28)
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=child.liquid, lower=0.0
                    )
                },
            ),
        },
    )
    with pytest.raises(ModelInitializationError, match=r"savings.grid"):
        Model._from_call_owned_params(declarations=declarations, params=_nb_params())


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_argument_clock_refuses_before_retaining_unbudgeted_outputs() -> None:
    model = _model()
    flat, generation = model._prepare_call(params=_nb_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NBCaseUArguments)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.roots)
    owner.budget_bytes = 0
    try:
        with pytest.raises(ExecutionPlanningError):
            arguments.point.prepare(context=context, materializer=owner)
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_shape_capture_keeps_original_plan_receiver_and_numeric_roots() -> None:
    model = _model()
    flat, generation = model._prepare_call(params=_nb_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    function = kernel.program.function
    assert isinstance(function, CaseUExactFunction)
    assert isinstance(function.point, NBCaseUProgram)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.roots)
    try:
        held = guard_case_u_shape(
            function=function,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
        assert any(array is function.point.plan.indices for array in held)
        for leaf in jax.tree.leaves(function.point.plan.node_operands):
            assert any(array is leaf for array in held)
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_shape_capture_refuses_hidden_original_utility_array() -> None:
    model = _model()
    flat, generation = model._prepare_call(params=_nb_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    core = kernel.point.core
    assert isinstance(core, _NBEGMContinuousCore)
    preferences = core.build_preferences
    assert isinstance(preferences, _PreferencesBuilder)
    hidden = np.asarray([2.0])

    def captured(*, consumption: FloatND, eps: FloatND) -> FloatND:
        return (1 + eps) * jnp.log(consumption) + hidden[0]

    point = replace(
        kernel.point,
        core=replace(
            core, build_preferences=replace(preferences, utility_func=captured)
        ),
    )
    function = kernel.program.function
    assert isinstance(function, CaseUExactFunction)
    assert isinstance(function.point, NBCaseUProgram)
    function = replace(function, point=replace(function.point, pieces=point))
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.roots)
    try:
        with pytest.raises(UnsupportedOperationError, match="host-array capture"):
            guard_case_u_shape(
                function=function,
                context=context,
                fixed_flat_params=MappingProxyType({}),
                materializer=owner,
            )
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_shape_capture_still_refuses_callback_backend_probe() -> None:
    """Selecting exact engine arithmetic does not admit a callback's cached probe."""
    model = _model()
    flat, generation = model._prepare_call(params=_nb_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    owner = _owner(roots=generation.roots)
    try:
        with pytest.raises(TypeError, match=r"functools\._lru_cache_wrapper"):
            guard_case_u_shape(
                function=_utility_with_backend_probe,
                context=context,
                fixed_flat_params=MappingProxyType({}),
                materializer=owner,
            )
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_arguments_preserve_original_numeric_inputs_under_owner() -> None:
    model = _model()
    flat, generation = model._prepare_call(params=_nb_params())
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NBCaseUArguments)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=kernel.future.template().children,
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    expected = arguments.point(context)
    owner = _owner(roots=generation.roots)
    try:
        actual = arguments.point.prepare(context=context, materializer=owner)
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        for observed, original in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        ):
            assert isinstance(observed, Array)
            assert isinstance(original, Array)
            assert observed.dtype == original.dtype
            assert jax.typeof(observed).weak_type == jax.typeof(original).weak_type
            np.testing.assert_array_equal(observed, original)
        for name in (
            "wealth",
            "_lcm_nbegm_source_grid",
            "_lcm_case_u_savings_nodes",
            "_lcm_case_u_borrowing_limit",
            "next_regime_to_continuation",
        ):
            for observed, original in zip(
                jax.tree.leaves(actual[name]),
                jax.tree.leaves(expected[name]),
                strict=True,
            ):
                assert observed is original
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_cold_nbegm_constructor_uses_original_canonical_source_precision() -> None:
    """Admitted source support determines the numerical empty-partition precision."""
    model = _model(device_memory_bytes=1 << 28)
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    source = kernel.point.source_grid
    assert source.dtype == jnp.asarray(0.0).dtype
    assert not jax.typeof(source).weak_type
    for leaf in jax.tree.leaves(kernel.template.output.source):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == source.dtype
            assert not jax.typeof(leaf).weak_type


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_cold_nbegm_preserves_original_checks_and_owned_sample_roots() -> None:
    model = _model(device_memory_bytes=1 << 28)
    _flat, generation = model._prepare_call(params=_nb_params())
    source = model._regimes["child"].solution
    kernel = source.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    assert source.param_checks is kernel.param_checks
    assert kernel.point.core.schedule_spec.param_checks == ()
    check = kernel.param_checks[0]
    assert isinstance(check, _DeferredProbe)
    assert check.probe is _fail_if_budget_nonaffine_in_liquid
    roots = jax.tree.leaves(generation.roots)
    for name in ("liquid_grid", "liquid_samples"):
        assert any(check.bound[name] is root for root in roots)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_runtime_probe_refuses_before_unadmitted_numeric_output() -> None:
    model = _model(device_memory_bytes=1 << 28)
    flat, generation = model._prepare_call(params=_nb_params())
    check = model._regimes["child"].solution.param_checks[0]
    assert isinstance(check, _DeferredProbe)
    assert not check.checked
    owner = _owner(roots=generation.roots)
    owner.budget_bytes = 0
    try:
        with pytest.raises(ExecutionPlanningError):
            check_solver_params(
                regimes=model._regimes, flat_params=flat, materializer=owner
            )
        assert not check.checked
        assert not owner._pending
    finally:
        owner.close()


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_each_draw_uses_live_owner_and_revalidates_changed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declarations = _nb_declarations(device_memory_bytes=1 << 28)
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child, functions={**child.functions, "resources": _curved_resources}
            ),
        },
    )
    params: UserParams = {
        "discount_factor": 1.0,
        "child": {"next_wealth": {"slope": 1.0}, "resources": {"curvature": 0.0}},
    }
    phase = "constructor"
    stages: list[tuple[str, int, Callable[..., PytreeValue]]] = []
    original = SimulationEntryAllocations.evaluate_validation

    class ObserveAllocations(SimulationEntryAllocations):
        """Observe original stages without retaining or substituting their owner."""

        def evaluate_validation[T: PytreeValue](
            self,
            *,
            function: Callable[..., T],
            arguments: Mapping[ReferenceName, PytreeValue],
        ) -> T:
            source = function.func if isinstance(function, partial) else function
            stages.append((phase, id(self), source))
            return original(self, function=function, arguments=arguments)

    monkeypatch.setattr(
        SimulationEntryAllocations,
        "evaluate_validation",
        ObserveAllocations.evaluate_validation,
    )
    model = Model._from_call_owned_params(declarations=declarations, params=params)
    assert any(
        label == "constructor" and function is _liquid_affinity_grid_samples
        for label, _owner_id, function in stages
    )
    phase = "first_draw"
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    assert np.isfinite(np.asarray(solution.values[1]["child"])).all()
    check = model._regimes["child"].solution.param_checks[0]
    assert isinstance(check, _DeferredProbe)
    assert check.checked
    assert check.probe_schedule == "every_solve"
    phase = "changed_draw"
    with pytest.raises(RegimeInitializationError, match="nonzero second derivative"):
        model.solve(
            params={
                "discount_factor": 1.0,
                "child": {
                    "next_wealth": {"slope": 1.0},
                    "resources": {"curvature": 0.25},
                },
            },
            log_level="off",
            max_compilation_workers=1,
        )
    for label in ("constructor", "first_draw", "changed_draw"):
        owners = {owner_id for actual, owner_id, _function in stages if actual == label}
        assert len(owners) == 1
    for label in ("first_draw", "changed_draw"):
        assert any(
            actual == label and function is _evaluate_liquid_probe_program
            for actual, _owner_id, function in stages
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("device_memory_bytes", [None, 1 << 28])
def test_nbegm_u_model_integrates_and_reconstructs_original_candidates(
    *, device_memory_bytes: int | None
) -> None:
    model = _model(device_memory_bytes=device_memory_bytes)
    solution = model.solve(
        params=_nb_params(),
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        log_level="off",
    )
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    stored = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    states = [0.5, 1.5, 3.0]
    reference = np.asarray(
        [
            np.mean(
                [_oracle(wealth=wealth, shock=shock)[:2] for shock in (0.0, 1.0)],
                axis=0,
            )
            for wealth in states
        ]
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"], reference[:, 0], decimal=DECIMAL_PRECISION
    )
    count = int(stored.valid_length)
    for index, wealth in enumerate(states):
        slot = np.flatnonzero(np.asarray(stored.grid[:count]) == wealth)
        assert slot.size == 1
        np.testing.assert_almost_equal(
            stored.economic_marginal[slot[0]],
            reference[index, 1],
            decimal=DECIMAL_PRECISION,
        )
    assert not np.any(np.asarray(stored.link_valid))
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    wealth = np.asarray([0.5, 1.5, 3.0, 1.5])
    physical = np.asarray([1.0, 0.0, 1.0, 0.5])
    result = model.simulate(
        params=_nb_params(),
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.ones((4,)),
            "regime_id": jnp.full(
                (4,), model.regime_names_to_ids["child"], dtype=jnp.int32
            ),
            "eps": PhysicalValues(values=physical),
        },
        log_level="off",
        seed=17,
    )
    row = result.raw_results["child"][1]
    expected = np.asarray(
        [_oracle(wealth=w, shock=z)[2] for w, z in zip(wealth, physical, strict=True)]
    )
    np.testing.assert_array_almost_equal(
        row.actions["consumption"], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        wealth - expected,
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(row.shocks["eps"], physical)
    flat, generation = model._prepare_call(params=_nb_params())
    actual = generation.execution._regimes["child"].solution.period_kernels[1]
    assert isinstance(actual, NativeNBCaseUPeriodKernel)
    arguments = actual.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NBCaseUArguments)
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
        "age": jnp.asarray(1.0),
    }
    for index in range(4):
        original = actual.point.node(
            liquid=jnp.asarray(wealth[index]),
            physical={"eps": jnp.asarray(physical[index])},
            pool=pool,
            children=future.children,
        )
        assert bool(original.output.valid & original.output.feasible)
        assert int(original.output.native_owner) >= 0
        np.testing.assert_array_almost_equal(
            original.output.policy,
            row.actions["consumption"][index],
            decimal=DECIMAL_PRECISION,
        )
    assert float(row.actions["consumption"][1]) not in (0.1, 4.0)
    corrupted_child = replace(
        future.children["done"],
        marginal_utility=jnp.full_like(
            future.children["done"].marginal_utility, jnp.nan
        ),
    )
    ref = ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    corrupted = replace(
        solution,
        replay_artifacts=ArtifactStore(
            {
                **dict(solution.replay_artifacts),
                ref: replace(future, child_values=(corrupted_child,)),
            }
        ),
    )
    with pytest.raises(
        ExecutionPlanningError,
        match=r"^Case-U original candidate replay has unresolved row integrity\.$",
    ):
        model.simulate(
            params=_nb_params(),
            solution=corrupted,
            initial_conditions={
                "wealth": jnp.asarray(wealth),
                "age": jnp.ones((4,)),
                "regime_id": jnp.full(
                    (4,), model.regime_names_to_ids["child"], dtype=jnp.int32
                ),
                "eps": PhysicalValues(values=physical),
            },
            log_level="off",
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("device_memory_bytes", [None, 1 << 28])
def test_original_nbegm_u_kink_and_corner_keep_original_candidates(
    *, device_memory_bytes: int | None
) -> None:
    model = _model(kink=True, device_memory_bytes=device_memory_bytes)
    params = _nb_params(kink=True)
    solution = model.solve(params=params, log_level="off")
    wealth = np.asarray([0.5, 1.0, 2.0, 4.0])
    shock = np.asarray([1.0, 0.0, 1.0, 0.0])
    # Declared budget is m=b below2 and m=1+b/2 above; the physical
    # preimage changes, while the same log/log EGM candidate remains optimal.
    cash = wealth - 0.5 * np.maximum(wealth - 2.0, 0.0)
    expected = np.asarray(
        [_oracle(wealth=m, shock=z)[2] for m, z in zip(cash, shock, strict=True)]
    )
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.ones((4,)),
            "regime_id": jnp.full(
                (4,), model.regime_names_to_ids["child"], dtype=jnp.int32
            ),
            "eps": PhysicalValues(values=shock),
        },
        log_level="off",
        seed=17,
    )
    row = result.raw_results["child"][1]
    np.testing.assert_array_almost_equal(
        row.actions["consumption"], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        result.raw_results["done"][2].states["wealth"],
        cash - expected,
        decimal=DECIMAL_PRECISION,
    )
    reference = np.asarray(
        [np.mean([_oracle(wealth=m, shock=z)[0] for z in (0.0, 1.0)]) for m in cash]
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"], reference, decimal=DECIMAL_PRECISION
    )
    # Preserve the default solve/replay witness above. Request the source
    # continuation explicitly for the independent economic-derivative check.
    marginal_solution = model.solve(
        params=params,
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        log_level="off",
    )
    stored = marginal_solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    for index in (0, 1, 3):
        slope = 1.0 if wealth[index] < 2.0 else 0.5
        reference_marginal = slope * np.mean(
            [_oracle(wealth=cash[index], shock=z)[1] for z in (0.0, 1.0)]
        )
        slot = np.flatnonzero(
            np.asarray(stored.grid[: int(stored.valid_length)]) == wealth[index]
        )
        assert slot.size == 1
        np.testing.assert_almost_equal(
            stored.economic_marginal[slot[0]],
            reference_marginal,
            decimal=DECIMAL_PRECISION,
        )
    # The exact kink retains the original selected point's marginal below;
    # this test does not assign an unproved universal side derivative to it.
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    future = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(future, CaseUReplayChildren)
    flat, _generation = model._prepare_call(params=params)
    arguments = kernel.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert isinstance(arguments.point, NBCaseUArguments)
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
        "age": jnp.asarray(1.0),
    }
    point = kernel.point
    assert isinstance(point, NBCaseUPoint)
    assert isinstance(point.core, _NBEGMContinuousCore)
    child = future.children["done"]
    for index, z in enumerate(shock):
        physical = {"eps": jnp.asarray(z)}
        landing, jacobian = point.declared_law(
            savings_grid=point.savings_nodes, **{**pool, **physical}
        )
        native_params: dict[ReferenceName, ValueND] = {}
        for name, operand in {**pool, **physical}.items():
            assert isinstance(operand, Array)
            native_params[name] = operand
        full = point.core.construct_point(
            liquid=point.source_grid,
            next_liquid_grid=child.endog_grid,
            next_value=child.value,
            next_marginal=child.marginal_utility,
            next_liquid=landing,
            marginal_return=jacobian,
            effective_savings_grid=point.savings_nodes,
            boundary_savings_targets=jnp.zeros((0,)),
            boundary_next_liquid=jnp.zeros((0,)),
            source_bound_status=None,
            continuation_read=None,
            **native_params,
        )
        selected = point.node(
            liquid=jnp.asarray(wealth[index]),
            physical=physical,
            pool=pool,
            children=future.children,
        )
        assert isinstance(full, _NBEGMSourcePoint)
        assert full.native_owner is not None
        assert full.carry.policy is not None
        data = full.carry.query_data_for_transport()
        assert data is not None
        # Querying a point must not rebuild a smaller bank or rename its winner.
        assert int(selected.output.native_owner) == int(full.native_owner[index])
        np.testing.assert_array_equal(selected.output.policy, full.carry.policy[index])
        np.testing.assert_array_equal(selected.output.savings, data.savings[index])
        np.testing.assert_array_equal(
            selected.output.marginal, full.carry.marginal_utility[index]
        )


def _r_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _r_inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1 / marginal_continuation


def _r_resources(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return wealth + eps


def _r_resources_doubled(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return 2 * wealth + eps


def _r_declarations(*, resource_scale: int = 1) -> _ModelConstructorInputs:
    declarations = _nb_declarations()
    old = declarations.regimes["child"]
    assert isinstance(old, ConsumptionSavingsRegime)
    savings = IrregSpacedGrid(
        points=np.asarray(
            [0.25, 0.375, 0.5, 0.6875, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        )
    )
    child = replace(
        old,
        states={"wealth": IrregSpacedGrid(points=np.asarray([0.375, 1.0, 2.0]))},
        functions={
            **old.functions,
            "utility": _r_utility,
            "inverse_marginal_utility": _r_inverse,
            "resources": _r_resources if resource_scale == 1 else _r_resources_doubled,
        },
        solver=NBEGM(savings_grid=savings),
    )
    return replace(
        declarations,
        regimes={
            "child": child,
            "done": replace(declarations.regimes["done"], states={"wealth": savings}),
        },
    )


def _r_oracle(
    *, wealth: float, shock: float, resource_scale: int = 1
) -> tuple[float, float, float, float]:
    cash = resource_scale * wealth + shock
    saving = max(0.25, cash / 2)
    consumption = cash - saving
    return (
        math.log(consumption) + math.log(saving),
        resource_scale / consumption,
        consumption,
        saving,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_r_durable_projection_requires_original_installed_authority() -> None:
    """A structurally similar date or program cannot authorize a retained carry."""
    from _lcm.execution.core_program import ProgramScope  # noqa: PLC0415
    from _lcm.solution.current_iid_nbegm_r import NBCaseRArguments  # noqa: PLC0415
    from _lcm.solution.model_authority import (  # noqa: PLC0415
        _is_installed_nbegm_r_continuation,
    )

    model = Model._from_call_owned_params(
        declarations=_r_declarations(), params=_nb_params()
    )
    regime = model._regimes["child"]
    spec = regime.solution.continuation_spec
    assert spec is not None
    kernel = regime.solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseRPeriodKernel)
    template = spec.template_at(period=1)
    assert isinstance(template, ProjectedEGMCarry)
    check = partial(
        _is_installed_nbegm_r_continuation,
        regime_name="child",
        period=1,
    )
    assert check(regime=regime, spec=spec, template=template)
    assert not check(regime=regime, spec=replace(spec), template=template)
    assert not check(regime=regime, spec=spec, template=replace(template))
    program = kernel.programs["main"]
    assert isinstance(program.function, NBCaseRProgram)
    replay = kernel.programs["replay"]
    assert isinstance(replay.function, NBCaseRProgram)
    assert isinstance(replay.argument_builder, NBCaseRArguments)
    for key, changed_program in (
        ("main", replace(program, function=replace(program.function, period=2))),
        ("replay", replace(replay, scope=ProgramScope.VALUES_ONLY)),
        (
            "replay",
            replace(
                replay,
                function=replace(replay.function, target="other"),
                argument_builder=replace(replay.argument_builder, target="other"),
            ),
        ),
        (
            "replay",
            replace(
                replay,
                function=replace(
                    replay.function, coefficient=2 * replay.function.coefficient
                ),
            ),
        ),
    ):
        changed = replace(
            kernel,
            programs=MappingProxyType({**kernel.programs, key: changed_program}),
        )
        changed_regime = replace(
            regime,
            solution=replace(regime.solution, period_kernels={1: changed}),
        )
        assert not check(regime=changed_regime, spec=spec, template=template)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("resource_scale", [1, 2])
def test_original_nbegm_r_model_retains_one_bank_and_replays_offnode(  # noqa: PLR0915 - complete Model integration, original-owner, replay, and status witness
    *, resource_scale: int
) -> None:
    """Select the same resource bank at each draw, including its borrowing corner."""
    declarations = _r_declarations(resource_scale=resource_scale)
    params = _nb_params()
    model = Model._from_call_owned_params(declarations=declarations, params=params)
    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    from _lcm.egm.current_iid_resources_artifact import (  # noqa: PLC0415
        restore_resources_row,
    )
    from _lcm.egm.nbegm_resources import (  # noqa: PLC0415
        NBEGMResourcesBank,
    )
    from _lcm.egm.nbegm_resources_read import read_nbegm_resources_bank  # noqa: PLC0415
    from _lcm.egm.upper_envelope.query import (  # noqa: PLC0415
        envelope_at_query,
        published_owner,
    )
    from _lcm.solution.current_iid_case_r import (  # noqa: PLC0415
        CASE_R_RESOURCES_REPLAY,
    )

    regime = model._regimes["child"]
    spec = regime.solution.case_r_replay_spec
    assert spec is not None
    assert regime.solution.case_u_replay_spec is None
    period = spec.periods[1]
    payload = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_R_RESOURCES_REPLAY)
    ]
    bank = restore_resources_row(
        payload=payload, tree=period.row_tree, leaf_specs=period.row_leaf_specs
    )
    assert isinstance(bank, NBEGMResourcesBank)
    np.testing.assert_array_equal(
        bank.resources_support,
        sorted(resource_scale * w + z for w in (0.375, 1, 2) for z in (0, 1)),
    )
    stored = solution.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(stored, ProjectedEGMCarry)
    for index, wealth in enumerate((0.375, 1.0, 2.0)):
        reference = np.mean(
            [
                _r_oracle(wealth=wealth, shock=z, resource_scale=resource_scale)[:2]
                for z in (0.0, 1.0)
            ],
            axis=0,
        )
        np.testing.assert_almost_equal(
            solution.values[1]["child"][index], reference[0], decimal=DECIMAL_PRECISION
        )
        slots = np.flatnonzero(np.asarray(stored.grid) == wealth)
        assert slots.size == 1
        np.testing.assert_almost_equal(
            stored.economic_marginal[slots[0]],
            reference[1],
            decimal=DECIMAL_PRECISION,
        )
        for shock in (0.0, 1.0):
            original = read_nbegm_resources_bank(
                bank=bank,
                resources=jnp.asarray(resource_scale * wealth + shock),
                arithmetic="certified",
            )
            assert bool(original.valid & original.feasible)
            assert int(original.native_owner) >= 0
            native = envelope_at_query(
                endog_grid=bank.endog_grid,
                policy=bank.policy,
                value=bank.value,
                marginal=bank.marginal,
                segment_id=bank.segment_id,
                x_query=jnp.asarray([resource_scale * wealth + shock]),
                arithmetic="certified",
                candidate_savings=bank.savings,
                return_owner=True,
            )
            np.testing.assert_array_equal(
                [original.value, original.policy, original.marginal, original.savings],
                [channel[0] for channel in native[:4]],
            )
            np.testing.assert_array_equal(
                original.native_owner,
                published_owner(value=native[0], stable_index=native[4])[0],
            )
            expected = _r_oracle(
                wealth=wealth, shock=shock, resource_scale=resource_scale
            )
            np.testing.assert_array_almost_equal(
                [
                    original.value,
                    resource_scale * original.marginal,
                    original.policy,
                    original.savings,
                ],
                expected,
                decimal=DECIMAL_PRECISION,
            )
    if resource_scale == 1:
        corner = read_nbegm_resources_bank(
            bank=bank, resources=jnp.asarray(0.375), arithmetic="certified"
        )
        assert bool(corner.corner)
    invalid = read_nbegm_resources_bank(
        bank=bank, resources=jnp.asarray(6.0), arithmetic="certified"
    )
    assert not bool(invalid.valid)

    wealth = np.asarray([0.375, 1.0, 1.0, 2.0])
    physical = np.asarray([0.0, 0.0, 0.5, 1.0])
    simulated = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.ones((4,)),
            "regime_id": jnp.full((4,), model.regime_names_to_ids["child"], jnp.int32),
            "eps": PhysicalValues(values=physical),
        },
        log_level="off",
        seed=29,
    )
    actual = simulated.raw_results["child"][1]
    expected = np.asarray(
        [
            _r_oracle(wealth=w, shock=z, resource_scale=resource_scale)[2]
            for w, z in zip(wealth, physical, strict=True)
        ]
    )
    np.testing.assert_array_almost_equal(
        actual.actions["consumption"], expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual.shocks["eps"], physical)
    np.testing.assert_array_almost_equal(
        simulated.raw_results["done"][2].states["wealth"],
        resource_scale * wealth + physical - expected,
        decimal=DECIMAL_PRECISION,
    )
    assert float(actual.actions["consumption"][2]) not in (0.1, 4.0)
    corrupted = bank._replace(marginal=jnp.full_like(bank.marginal, jnp.nan))
    corrupted_solution = replace(
        solution,
        replay_artifacts=ArtifactStore(
            {
                **dict(solution.replay_artifacts),
                ArtifactRef(
                    period=1, regime="child", key=CASE_R_RESOURCES_REPLAY
                ): tuple(jax.tree.leaves(corrupted)),
            }
        ),
    )
    with pytest.raises(
        ExecutionPlanningError,
        match=r"^Case-R original candidate replay has unresolved row integrity\.$",
    ):
        model.simulate(
            params=params,
            solution=corrupted_solution,
            initial_conditions={
                "wealth": jnp.asarray(wealth),
                "age": jnp.ones((4,)),
                "regime_id": jnp.full(
                    (4,), model.regime_names_to_ids["child"], jnp.int32
                ),
                "eps": PhysicalValues(values=physical),
            },
            log_level="off",
        )


def _r_shocked_landing(*, savings: FloatND, eps: FloatND, slope: FloatND) -> FloatND:
    return slope * savings + 0.125 * eps


def _r_state_law_start(*, wealth: FloatND) -> FloatND:
    return wealth / 8


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_r_direct_transition_shock_does_not_cross_resources_cut() -> None:
    """An ordinary savings law reading the draw cannot use a shared resource bank."""
    declarations = _r_declarations()
    old = declarations.regimes["child"]
    assert isinstance(old, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                old, state_transitions={"wealth": {"done": _r_shocked_landing}}
            ),
        },
    )
    with pytest.raises(UnsupportedOperationError, match="not resources-only"):
        Model._from_call_owned_params(declarations=declarations, params=_nb_params())


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_nbegm_r_law_derivatives_and_cold_bank_remain_explicit_refusals() -> None:
    """Missing current-law tangents and bank accounting fail before R construction."""
    declarations = _r_declarations()
    state_law = replace(
        declarations,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=_r_state_law_start, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            )
        },
    )
    with pytest.raises(UnsupportedOperationError, match="state-dependent current law"):
        Model._from_call_owned_params(declarations=state_law, params=_nb_params())
    with pytest.raises(UnsupportedOperationError, match="Cold original NB Case R"):
        Model._from_call_owned_params(
            declarations=replace(
                declarations,
                execution_config=ExecutionConfig(device_memory_bytes=1 << 28),
            ),
            params=_nb_params(),
        )


def _r_null_declarations(*, invalid_null: bool = False) -> _ModelConstructorInputs:
    declarations = _r_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    return replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                functions={
                    **child.functions,
                    "resources": _r_invalid_null_resources
                    if invalid_null
                    else _r_resources,
                },
            ),
        },
        shocks={
            "income": ShockGroup(
                shocks={"eps": Uniform(start=0.0, stop=2.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]),
                    weights=np.asarray([1.0, 0.0]),
                ),
            )
        },
    )


def _r_invalid_null_resources(*, wealth: FloatND, eps: FloatND) -> FloatND:
    return wealth + jnp.sqrt(1.0 - eps)


def _r_produced_program_arguments(
    *, declarations: _ModelConstructorInputs, params: UserParams | None = None
) -> tuple[NBCaseRProgram, Mapping[ReferenceName, PytreeValue]]:
    """Use the installed original R program with an actually produced terminal row."""
    params = _nb_params() if params is None else params
    model = Model._from_call_owned_params(declarations=declarations, params=params)
    solution = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseRPeriodKernel)
    program = kernel.programs["replay"]
    assert isinstance(program.function, NBCaseRProgram)
    flat, _generation = model._prepare_call(params=params)
    child = solution.retained_continuations[
        ArtifactRef(period=2, regime="done", key=EGM_CONTINUATION)
    ]
    assert isinstance(child, EGMCarry)
    context = CoreBuildContext(
        state_action_space=kernel.context.state_action_space,
        next_regime_to_V_arr={"done": solution.values[2]["done"]},
        next_regime_to_continuation={"done": child},
        flat_params=flat,
        period=1,
        ages=kernel.context.ages,
    )
    actual = program.argument_builder(context)
    assert not any(
        isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(actual)
    )
    return program.function, cast("Mapping[ReferenceName, PytreeValue]", actual)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_null_probability_direction_keeps_counterfactual_value() -> (
    None
):
    """A null resource outside live support contributes its finite weight tangent."""
    original, arguments = _r_produced_program_arguments(
        declarations=_r_null_declarations()
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    nodes = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["node_operands"])

    def value_and_economic(probability: FloatND) -> FloatND:
        actual = {
            **arguments,
            "node_operands": {
                **nodes,
                "income": {
                    **nodes["income"],
                    "weights": jnp.stack((1.0 - probability, probability)),
                },
            },
        }
        result = call(**actual)
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return jnp.stack((projected.value[1], projected.economic_marginal[1]))

    primal, direction = jax.jvp(
        value_and_economic, (jnp.asarray(0.0),), (jnp.asarray(1.0),)
    )
    np.testing.assert_array_almost_equal(
        primal, [2 * math.log(0.5), 2.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        direction, [2 * math.log(3.0), -4 / 3], decimal=DECIMAL_PRECISION
    )
    jitted = jax.jit(
        lambda probability: jax.jvp(
            value_and_economic, (probability,), (jnp.asarray(1.0),)
        )
    )(jnp.asarray(0.0))
    np.testing.assert_array_almost_equal(
        jitted[1], direction, decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_invalid_null_primal_survives_requested_tangent_refuses() -> (
    None
):
    """Undefined null resources are evaluated only for a requested weight direction."""
    original, arguments = _r_produced_program_arguments(
        declarations=_r_null_declarations(invalid_null=True)
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    nodes = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["node_operands"])

    def values(probability: FloatND) -> FloatND:
        result = call(
            **{
                **arguments,
                "node_operands": {
                    **nodes,
                    "income": {
                        **nodes["income"],
                        "weights": jnp.stack((1.0 - probability, probability)),
                    },
                },
            }
        )
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return projected.value[:3]

    primal = values(jnp.asarray(0.0))
    np.testing.assert_array_almost_equal(
        primal,
        [2 * math.log((wealth + 1) / 2) for wealth in (0.375, 1.0, 2.0)],
        decimal=DECIMAL_PRECISION,
    )
    _, zero_direction = jax.jvp(values, (jnp.asarray(0.0),), (jnp.asarray(0.0),))
    np.testing.assert_array_equal(zero_direction, np.zeros((3,)))
    _, required_direction = jax.jvp(values, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    assert bool(jnp.all(jnp.isnan(required_direction)))


def _r_joint_null_resources(*, wealth: FloatND, eps: FloatND, eta: FloatND) -> FloatND:
    intercept = jnp.sqrt(jnp.where((eps > 0) & (eta > 0), -1.0, 1.0)) - 1
    return wealth + eps + eta + intercept


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_double_zero_event_skips_first_order_counterfactual() -> None:
    """A joint event with two zero factors has no first-order weight coefficient."""
    declarations = _r_null_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                functions={**child.functions, "resources": _r_joint_null_resources},
                shocks=("income", "extra"),
            ),
        },
        shocks={
            **declarations.shocks,
            "extra": ShockGroup(
                shocks={"eta": Uniform(start=0.0, stop=2.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]),
                    weights=np.asarray([1.0, 0.0]),
                ),
            ),
        },
    )
    original, arguments = _r_produced_program_arguments(declarations=declarations)
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    nodes = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["node_operands"])

    def value(probability: FloatND) -> FloatND:
        result = call(
            **{
                **arguments,
                "node_operands": {
                    name: {
                        **node,
                        "weights": jnp.stack((1.0 - probability, probability)),
                    }
                    for name, node in nodes.items()
                },
            }
        )
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return jnp.stack((projected.value[1], projected.economic_marginal[1]))

    primal, direction = jax.jvp(value, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_array_almost_equal(
        primal, [2 * math.log(0.5), 2.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        direction, [4 * math.log(3.0), -8 / 3], decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_nonweight_direction_keeps_original_child_differential() -> (
    None
):
    """A terminal value shift retains its original unit Bellman derivative."""
    original, arguments = _r_produced_program_arguments(
        declarations=_r_null_declarations()
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    children = cast("Mapping[str, EGMCarry]", arguments["next_regime_to_continuation"])
    child = children["done"]

    def value(shift: FloatND) -> FloatND:
        result = call(
            **{
                **arguments,
                "next_regime_to_continuation": {
                    "done": replace(child, value=child.value + shift)
                },
            }
        )
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return jnp.stack((projected.value[1], projected.economic_marginal[1]))

    _, direction = jax.jvp(value, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_array_almost_equal(
        direction, [1.0, 0.0], decimal=DECIMAL_PRECISION
    )


def _r_economic_minus_inf_terminal(*, wealth: FloatND) -> FloatND:
    return jnp.where(wealth < 1.0, -jnp.inf, jnp.log(wealth))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_feasible_economic_minus_inf_has_precise_refusal() -> None:
    """Unrepresented economic values are distinct from an empty consumption set."""
    declarations = _r_declarations()
    done = declarations.regimes["done"]
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "done": replace(
                done, functions={"utility": _r_economic_minus_inf_terminal}
            ),
        },
    )
    model = Model._from_call_owned_params(
        declarations=declarations, params=_nb_params()
    )
    with pytest.raises(
        UnsupportedOperationError,
        match="cannot represent feasible economic minus-infinity candidates",
    ):
        model.solve(params=_nb_params(), log_level="off")


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_no_action_keeps_minus_inf_and_zero_infeasible_marginal() -> (
    None
):
    """A budget below the original floor remains an ordinary no-action state."""
    declarations = _r_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                states={
                    "wealth": IrregSpacedGrid(points=np.asarray([0.125, 1.0, 2.0]))
                },
            ),
        },
    )
    model = Model._from_call_owned_params(
        declarations=declarations, params=_nb_params()
    )
    solution = model.solve(
        params=_nb_params(),
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )
    from _lcm.egm.current_iid_resources_artifact import (  # noqa: PLC0415
        restore_resources_row,
    )
    from _lcm.egm.nbegm_resources import NBEGMResourcesBank  # noqa: PLC0415
    from _lcm.egm.nbegm_resources_read import read_nbegm_resources_bank  # noqa: PLC0415
    from _lcm.solution.current_iid_case_r import (  # noqa: PLC0415
        CASE_R_RESOURCES_REPLAY,
    )

    spec = model._regimes["child"].solution.case_r_replay_spec
    assert spec is not None
    period = spec.periods[1]
    bank = restore_resources_row(
        payload=solution.replay_artifacts[
            ArtifactRef(period=1, regime="child", key=CASE_R_RESOURCES_REPLAY)
        ],
        tree=period.row_tree,
        leaf_specs=period.row_leaf_specs,
    )
    assert isinstance(bank, NBEGMResourcesBank)
    selected = read_nbegm_resources_bank(
        bank=bank, resources=jnp.asarray(0.125), arithmetic="certified"
    )
    assert bool(selected.valid & ~selected.feasible)
    assert bool(jnp.isneginf(selected.value))
    np.testing.assert_array_equal(selected.marginal, 0.0)
    assert not bool(bank.unsupported_economic_minus_inf)
    assert bool(jnp.isneginf(solution.values[1]["child"][0]))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_nested_forward_request_refuses_explicitly() -> None:
    """First-order null support does not claim a mixed higher-order contract."""
    original, arguments = _r_produced_program_arguments(
        declarations=_r_null_declarations()
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    nodes = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["node_operands"])

    def value(probability: FloatND) -> FloatND:
        result = call(
            **{
                **arguments,
                "node_operands": {
                    **nodes,
                    "income": {
                        **nodes["income"],
                        "weights": jnp.stack((1.0 - probability, probability)),
                    },
                },
            }
        )
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return projected.value[1]

    def first(probability: FloatND) -> FloatND:
        return jax.jvp(value, (probability,), (jnp.asarray(1.0),))[1]

    with pytest.raises(
        UnsupportedOperationError, match="first-order forward sensitivities only"
    ):
        jax.jvp(first, (jnp.asarray(0.0),), (jnp.asarray(1.0),))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("probability", [0.0, 0.25])
def test_original_nbegm_r_bank_direction_keeps_original_live_geometry(
    *, probability: float
) -> None:
    """Transient null support changes no exported original bank tangent or owner."""
    from _lcm.egm.current_iid_nbegm_r import NBCaseROperands  # noqa: PLC0415
    from _lcm.egm.nbegm_resources import NBEGMResourcesBank  # noqa: PLC0415

    original, arguments = _r_produced_program_arguments(
        declarations=_r_null_declarations()
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    nodes = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["node_operands"])
    children = cast("Mapping[str, EGMCarry]", arguments["next_regime_to_continuation"])

    # keyword-only-exempt: library-callback=jax.jvp
    def evaluate(
        shift: FloatND, *, wrapped: bool
    ) -> BoundedResult[tuple[PytreeValue, ...]]:
        base = cast("FloatND", nodes["income"]["base"])
        actual = {
            **arguments,
            "node_operands": {
                **nodes,
                "income": {
                    **nodes["income"],
                    "base": base + 0.125 * shift,
                    "weights": jnp.stack(
                        (1.0 - probability - shift, probability + shift)
                    ),
                },
            },
            "next_regime_to_continuation": {
                "done": replace(
                    children["done"], value=children["done"].value + 0.25 * shift
                )
            },
        }
        if wrapped:
            return call(**actual)
        operands = cast(
            "NBCaseROperands",
            tuple(
                actual[name]
                for name in (
                    "liquid_grid",
                    "savings_nodes",
                    "next_regime_to_continuation",
                    "group_params",
                    "node_operands",
                    "indices",
                    "pool",
                )
            ),
        )
        return original.evaluate(operands=operands)

    primal, direction = jax.jvp(
        partial(evaluate, wrapped=True),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )
    reference, reference_direction = jax.jvp(
        partial(evaluate, wrapped=False),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )
    for result in (primal, reference, direction, reference_direction):
        assert isinstance(result.output[2], NBEGMResourcesBank)
    np.testing.assert_array_equal(
        primal.output[2].resources_support, reference.output[2].resources_support
    )
    for actual, expected in zip(
        jax.tree.leaves(direction.output[2]),
        jax.tree.leaves(reference_direction.output[2]),
        strict=True,
    ):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    if probability > 0:
        for actual, expected in zip(
            jax.tree.leaves(direction),
            jax.tree.leaves(reference_direction),
            strict=True,
        ):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


@categorical(ordered=False)
class _CurrentRShock:
    low: ScalarInt
    high: ScalarInt


def _r_categorical_resources(*, wealth: FloatND, eps: ScalarInt) -> FloatND:
    return wealth + 2 * eps


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_nbegm_r_null_categorical_law_parameter_keeps_boundary_direction() -> (
    None
):
    """An original law probability parameter retains the finite null-category value."""
    declarations = _r_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                functions={**child.functions, "resources": _r_categorical_resources},
            ),
        },
        shocks={
            "income": ShockGroup(
                shocks={"eps": Discrete(categories=_CurrentRShock, probs=None)}
            )
        },
    )
    original, arguments = _r_produced_program_arguments(
        declarations=declarations,
        params={**_nb_params(), "income": {"eps__probs": np.asarray([1.0, 0.0])}},
    )
    call = cast("Callable[..., BoundedResult[tuple[PytreeValue, ...]]]", original)
    groups = cast("Mapping[str, Mapping[str, PytreeValue]]", arguments["group_params"])

    def values(probability: FloatND) -> FloatND:
        result = call(
            **{
                **arguments,
                "group_params": {
                    **groups,
                    "income": {
                        **groups["income"],
                        "eps__probs": jnp.stack((1.0 - probability, probability)),
                    },
                },
            }
        )
        projected = result.output[1]
        assert isinstance(projected, ProjectedEGMCarry)
        return jnp.stack((projected.value[1], projected.economic_marginal[1]))

    primal, direction = jax.jvp(values, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_array_almost_equal(
        primal, [2 * math.log(0.5), 2.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        direction, [2 * math.log(3.0), -4 / 3], decimal=DECIMAL_PRECISION
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_optional_callbacks_survive_construction_and_replace() -> None:
    """Original recipe and publisher identities survive immutable specialization."""

    scalar = _model()
    kernel = scalar._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    core = kernel.point.core
    assert isinstance(core, _NBEGMContinuousCore)
    assert core.interval_recipe is None
    assert core._evaluate_preimages is None
    recipe = _query(actions=False).model.intervals
    bound = replace(core, interval_recipe=recipe)
    assert bound.interval_recipe is recipe
    assert bound._evaluate_preimages is recipe.evaluate_preimages
    rebound = replace(bound, retain_query=not bound.retain_query)
    assert rebound.interval_recipe is recipe
    assert rebound._evaluate_preimages is recipe.evaluate_preimages
    assert replace(bound, interval_recipe=None)._evaluate_preimages is None
    with pytest.raises(ValueError, match="init=False"):
        replace(bound, _evaluate_preimages=recipe.evaluate_preimages)

    actual = balanced_model()._regimes["child"].solution.period_kernels[1]
    assert isinstance(actual, NativeCaseUPeriodKernel)
    function = actual.core_programs()["main"].function
    assert isinstance(function, CaseUExactFunction)
    assert function.balanced is not None
    publisher = function._publish_balanced
    assert isinstance(publisher, MethodType)
    assert publisher.__self__ is function.balanced
    assert publisher.__func__ is function.balanced.publish.__func__
    changed = replace(function, capacity=function.capacity)
    assert changed.balanced is function.balanced
    assert isinstance(changed._publish_balanced, MethodType)
    assert changed._publish_balanced.__self__ is function.balanced
    assert changed._publish_balanced.__func__ is publisher.__func__
    assert replace(function, balanced=None)._publish_balanced is None
    with pytest.raises(ValueError, match="init=False"):
        replace(function, _publish_balanced=publisher)
