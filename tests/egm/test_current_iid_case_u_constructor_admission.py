"""Entry-owned original U constructor schemas, copies and retained roots."""

import inspect
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from types import MappingProxyType, MethodType
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.api import Inline

from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.current_iid_case_u_replay import CASE_U_FUTURE_REPLAY
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import _EgmKernelPieces
from _lcm.egm.upper_envelope._exact_affine import ffi as exact_affine_ffi
from _lcm.egm.validation_stages import ValidationCaptureHasher
from _lcm.execution.core_program import CoreBuildContext
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.solution.current_iid_case_u_build import (
    CaseUProgramFunction,
    _case_u_original_roots,
    guard_case_u_shape,
)
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactFunction,
    NativeCaseUPeriodKernel,
    acquire_installed_case_u_date,
)
from _lcm.solution.generation import (
    CanonicalModelGeneration,
    canonical_generation_roots,
)
from _lcm.solution.retained_buffers import retained_artifact_authority_buffers
from _lcm.typing import PytreeValue
from lcm import ConsumptionSavingsRegime, ExecutionConfig, Model
from lcm._solver_api.authority import _snapshot_artifact_template_once
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.shocks import UserNodes
from lcm.solvers import DCEGM
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_installation import _utility_u
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations, _params
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.egm.test_current_iid_parent_demands import _resources

pytestmark = [
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
]


def test_original_u_constructor_retains_templates_and_captures_after_entry() -> None:
    """Use the original log(c)+eps*wealth source with the actual bounded owner."""
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    assert isinstance(child.solver, DCEGM)
    child = replace(
        child,
        solver=replace(child.solver, current_iid_projection_capacity=8),
        functions={**child.functions, "utility": _utility_u, "resources": _resources},
    )
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": child},
        execution_config=ExecutionConfig(
            devices=(jax.devices("cpu")[0].id,), device_memory_bytes=1 << 28
        ),
    )
    model = Model._from_call_owned_params(declarations=declarations, params=_params())
    _, generation = model._prepare_call(params=_params())
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    source = generation.solver_preparation(regime_name="child", period=1)
    assert isinstance(source.step, _EGMStep)
    assert isinstance(prepared.program.function, CaseUExactFunction)
    assert prepared.program.function.point.pieces is source.step.pieces
    _reject_copied_u_pieces(generation=generation)
    roots = canonical_generation_roots(declarations=generation.roots)
    arguments = prepared.program.argument_builder
    assert isinstance(arguments, CaseUExactArguments)
    assert arguments.trace_capture_roots
    for leaf in arguments.trace_capture_roots:
        assert any(root is leaf for root in roots)
    authority = generation.execution._regimes["child"].solution.artifact_authorities[
        CASE_U_FUTURE_REPLAY
    ]
    banks = retained_artifact_authority_buffers(authority=authority)
    assert banks
    for leaf in banks:
        assert any(root is leaf for root in roots)
    # The constructor entry has closed. The next entry must still see the exact
    # authenticated private/public banks held by the genuine generation.
    next_owner = _owner(roots=generation.roots)
    try:
        held = jax.tree.leaves(next_owner.solve_input_roots())
        for leaf in banks:
            assert any(root is leaf for root in held)
    finally:
        next_owner.close()


def test_native_template_host_upload_refuses_before_copy() -> None:
    """The optional copier grants no implicit host-array producer admission."""
    owner = _owner()
    try:
        with pytest.raises(ExecutionPlanningError, match="host uploads"):
            _snapshot_artifact_template_once(
                template=(np.array([1.0]),),
                payload_runtime_type=tuple,
                array_copier=owner.copy_solution_leaf,
            )
        assert not owner._foreign_copies
    finally:
        owner.close()


@pytest.fixture
def shape_capture_owner() -> Iterator[
    tuple[CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction]
]:
    model = Model._from_call_owned_params(
        declarations=_actual_declarations(), params=_params()
    )
    flat, generation = model._prepare_call(params=_params())
    prepared = generation.solver_preparation(regime_name="child", period=1)
    context = CoreBuildContext(
        state_action_space=prepared.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params=flat,
        period=1,
        ages=prepared.context.ages,
    )
    assert isinstance(prepared.step, _EGMStep)
    assert prepared.plan is not None
    function = CaseUProgramFunction(
        pieces=prepared.step.pieces,
        plan=prepared.plan,
        period=1,
        savings_point_width=None,
        stochastic_node_width=None,
        envelope_cell_width=1,
    )
    owner = _owner(roots=generation.roots)
    try:
        yield context, owner, function
    finally:
        owner.close()


def _unsafe_eager_capture(*, value: FloatND) -> FloatND:
    with jax.ensure_compile_time_eval():
        offset = jnp.ones(())
    return value + offset


def test_abstract_shape_capture_refuses_explicit_eager_escape(
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
) -> None:
    """Abstract tracing grants no authority to execute compile-time callbacks."""
    context, owner, _ = shape_capture_owner
    with pytest.raises(ExecutionPlanningError, match="compile-time eager"):
        guard_case_u_shape(
            function=_unsafe_eager_capture,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


def test_abstract_shape_capture_refuses_hidden_host_array_before_copy(
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hidden host buffer cannot enter through an uncharged callback constant."""
    context, owner, _ = shape_capture_owner
    hidden = np.array([2.0])

    def captured(*, value: FloatND) -> FloatND:
        return value + hidden[0]

    def forbid_copy(*args: PytreeValue, **kwargs: PytreeValue) -> None:
        del args, kwargs
        pytest.fail("The capture census must not convert captured array contents.")

    monkeypatch.setattr(np, "asarray", forbid_copy)
    with pytest.raises(UnsupportedOperationError, match="host-array capture"):
        guard_case_u_shape(
            function=captured,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


def test_abstract_shape_user_early_capture_retains_device_buffer_without_copy(
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A safe original user EARLY helper retains its actual referenced buffer."""
    context, owner, _ = shape_capture_owner
    hidden = jnp.array([2.0])

    @partial(jax.jit, inline=Inline.JAX_EARLY)
    def captured(*, value: FloatND) -> FloatND:
        return value + jnp.isfinite(hidden[0])

    def forbid_copy(*args: PytreeValue, **kwargs: PytreeValue) -> None:
        del args, kwargs
        pytest.fail("The capture census must retain the original device buffer.")

    with pytest.raises(ExecutionPlanningError, match="JAX_EARLY"):
        ValidationCaptureHasher().visit(value=captured)
    monkeypatch.setattr(np, "asarray", forbid_copy)
    held = guard_case_u_shape(
        function=captured,
        context=context,
        fixed_flat_params=MappingProxyType({}),
        materializer=owner,
    )
    assert any(array is hidden for array in held)
    assert any(array is hidden for array in jax.tree.leaves(owner.solve_input_roots()))


def _consume_user_rule(*, rule: UserNodes, value: FloatND) -> FloatND:
    return value + rule.nodes[0, 0]


def _consume_user_plan(*, plan: CurrentIIDPlan, value: FloatND) -> FloatND:
    rule = plan.bindings[0][1].group.nodes
    if not isinstance(rule, UserNodes):
        raise TypeError("Original user rule changed.")
    return value + rule.nodes[0, 0]


@pytest.mark.parametrize("capture", ["device", "host", "plan"])
def test_abstract_shape_tree_mapping_preserves_callback_capture_rules(
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    capture: Literal["device", "host", "plan"],
) -> None:
    context, owner, function = shape_capture_owner
    if capture == "plan":
        plan = function.plan

        def mapper(value: FloatND) -> FloatND:
            return _consume_user_plan(plan=plan, value=value)

    else:
        offset = jnp.asarray([2.0]) if capture == "device" else np.asarray([2.0])

        def mapper(value: FloatND) -> FloatND:
            return value + offset[0]

    def captured(*, value: FloatND) -> FloatND:
        return jax.tree.leaves(jax.tree.map(mapper, (value,)))[0]

    if capture == "device":
        held = guard_case_u_shape(
            function=captured,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
        assert any(array is offset for array in held)
        assert any(
            array is offset for array in jax.tree.leaves(owner.solve_input_roots())
        )
    else:
        with pytest.raises(
            (UnsupportedOperationError, TypeError),
            match=r"host-array capture|direct object dependency",
        ):
            guard_case_u_shape(
                function=captured,
                context=context,
                fixed_flat_params=MappingProxyType({}),
                materializer=owner,
            )
        assert not owner._pending


def _consume_user_point(*, point: CaseUProgramFunction, value: FloatND) -> FloatND:
    return _consume_user_plan(plan=point.plan, value=value)


def _consume_user_realiser(*, realiser: MethodType, value: FloatND) -> FloatND:
    plan = realiser.__self__
    if not isinstance(plan, CurrentIIDPlan):
        raise TypeError("Original plan receiver changed.")
    return _consume_user_plan(plan=plan, value=value)


def _consume_user_pieces(*, pieces: _EgmKernelPieces, value: FloatND) -> FloatND:
    return value + pieces.savings_nodes[0]


@pytest.mark.parametrize(
    "alias",
    [
        "nodes",
        "weights",
        "rule",
        "plan",
        "pieces",
        "point_path",
        "whole_point",
        "cloned_plan",
        "plan_method",
        "cloned_plan_method",
    ],
)
def test_abstract_shape_resolved_rule_does_not_admit_callback_aliases(  # noqa: C901 - distinguish genuine alias paths through one original owner
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    alias: str,
) -> None:
    """The original rule's host buffers remain unsupported callback constants."""
    context, owner, function = shape_capture_owner
    rule = function.plan.bindings[0][1].group.nodes
    assert isinstance(rule, UserNodes)
    if alias == "rule":

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_rule(rule=rule, value=value)

    elif alias == "plan":
        plan = function.plan

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_plan(plan=plan, value=value)

    elif alias == "point_path":

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_plan(plan=function.plan, value=value)

    elif alias == "whole_point":

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_point(point=function, value=value)

    elif alias == "cloned_plan":
        cloned = replace(function.plan)

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_plan(plan=cloned, value=value)

    elif alias in ("plan_method", "cloned_plan_method"):
        original = function.plan if alias == "plan_method" else replace(function.plan)
        realiser = original.realise_nodes
        if not isinstance(realiser, MethodType):
            raise TypeError("Original current realiser is not a bound method.")

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_realiser(realiser=realiser, value=value)

    elif alias == "pieces":
        pieces = function.pieces

        def captured(*, value: FloatND) -> FloatND:
            return _consume_user_pieces(pieces=pieces, value=value)

    else:
        host = rule.nodes if alias == "nodes" else rule.weights

        def captured(*, value: FloatND) -> FloatND:
            return value + host[0]

    function = replace(function, pieces=replace(function.pieces, utility_func=captured))
    with pytest.raises(
        (UnsupportedOperationError, TypeError),
        match=(
            r"host-array capture|direct object dependency "
            r".*(?:UserNodes|CurrentIIDPlan|_EgmKernelPieces|CaseUProgramFunction|tuple);"
        ),
    ):
        guard_case_u_shape(
            function=function,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


class _NominalCurrentIIDPlan(CurrentIIDPlan):
    """A structurally identical declaration has no exact engine metadata role."""


def test_abstract_shape_plan_subclass_has_no_metadata_role(
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
) -> None:
    context, owner, function = shape_capture_owner
    nominal = _NominalCurrentIIDPlan(
        bindings=function.plan.bindings, indices=function.plan.indices
    )
    with pytest.raises(TypeError, match="lost its original current plan"):
        guard_case_u_shape(
            function=replace(function, plan=nominal),
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


@pytest.mark.parametrize("method", ["realise_nodes", "realise_information_nodes"])
def test_abstract_shape_plan_method_binding_is_original(
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    context, owner, function = shape_capture_owner
    monkeypatch.setattr(CurrentIIDPlan, method, _unsafe_eager_capture)
    with pytest.raises(
        TypeError, match="Original Case-U plan method authority changed"
    ):
        guard_case_u_shape(
            function=function,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


@pytest.mark.parametrize("corruption", ["wrapper", "raw", "export", "unregistered"])
def test_abstract_shape_native_boundary_refuses_changed_authority(
    *,
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """A numerical primitive seal cannot survive changed source or preflight."""
    context, owner, function = shape_capture_owner
    original = exact_affine_ffi.exact_affine_read
    if corruption == "wrapper":
        monkeypatch.setattr(original, "__code__", _unsafe_eager_capture.__code__)
    elif corruption == "raw":
        raw = inspect.unwrap(original)
        monkeypatch.setattr(raw, "__code__", _unsafe_eager_capture.__code__)
    elif corruption == "export":
        monkeypatch.setattr(
            exact_affine_ffi, "exact_affine_read", _unsafe_eager_capture
        )
    else:
        monkeypatch.setattr(exact_affine_ffi, "_REGISTERED", False)
    with pytest.raises(TypeError, match=r"Case-U native .* (?:changed|absent)"):
        guard_case_u_shape(
            function=function,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


def test_abstract_shape_native_loader_is_not_a_numerical_primitive(
    shape_capture_owner: tuple[
        CoreBuildContext, SimulationEntryAllocations, CaseUProgramFunction
    ],
) -> None:
    """Native numerical authority does not exempt registration or file loaders."""
    context, owner, _ = shape_capture_owner
    with pytest.raises(TypeError, match=r"direct object dependency pathlib[.]"):
        guard_case_u_shape(
            function=exact_affine_ffi._ensure_registered,
            context=context,
            fixed_flat_params=MappingProxyType({}),
            materializer=owner,
        )
    assert not owner._pending


def test_original_capture_recipe_keeps_numeric_operand_order_and_identity() -> None:
    """Actual JAX keyword-tree canonicalization must not reorder owner roots."""
    original = tuple(jnp.asarray(float(index)) for index in range(12))
    owner = _owner(roots=original)
    try:
        held = owner.evaluate_validation(
            function=_case_u_original_roots,
            arguments={
                f"operand_{index}": original[index]
                for index in reversed(range(len(original)))
            },
        )
        assert all(left is right for left, right in zip(original, held, strict=True))
    finally:
        owner.close()


def _reject_copied_u_pieces(*, generation: CanonicalModelGeneration) -> None:
    """The actual dated validator still refuses clones and foreign economics."""
    dated = generation.execution._dated_generation
    assert dated is not None
    numerical = dated.preparation.numerical
    solution = numerical.kernels["child"]
    kernel = solution.period_kernels[1]
    assert isinstance(kernel, NativeCaseUPeriodKernel)
    function = kernel.prepared.program.function
    assert isinstance(function, CaseUExactFunction)
    for foreign in (
        replace(function.point.pieces),
        replace(function.point.pieces, utility_func=_resources),
    ):
        substituted = replace(function, point=replace(function.point, pieces=foreign))
        changed_kernel = replace(
            kernel,
            prepared=replace(
                kernel.prepared,
                program=replace(kernel.prepared.program, function=substituted),
            ),
        )
        changed_solution = replace(
            solution,
            period_kernels=MappingProxyType(
                {**solution.period_kernels, 1: changed_kernel}
            ),
        )
        changed = replace(
            numerical,
            kernels=MappingProxyType({**numerical.kernels, "child": changed_solution}),
        )
        with pytest.raises(TypeError, match="Installed U envelope substituted"):
            changed.validate_state_utility()
