"""Original scalar NB candidates survive private Model solve and physical replay."""

import math
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY, CaseUReplayChildren
from _lcm.egm.current_iid_nbegm import NBCaseUArguments, NBCaseUProgram
from _lcm.egm.preferences import _PreferencesBuilder
from _lcm.egm.projected_continuation import (
    PROJECTED_EGM_CONTINUATION,
    ProjectedEGMCarry,
)
from _lcm.egm.upper_envelope.certified_sign import backend_flushes_subnormals
from _lcm.execution.core_program import CoreBuildContext
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.current_iid_case_u_build import guard_case_u_shape
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactFunction,
)
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
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
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import ArtifactRef, ArtifactStore, ResultRetention
from lcm.solvers import DCEGM, NBEGM
from lcm.typing import ContinuousAction, FloatND, ReferenceName, UserParams, ValueND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.simulation.test_case_u_reconstruction import _ReplayRegime

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


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
