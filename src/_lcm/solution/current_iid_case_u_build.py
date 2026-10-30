"""Private genuine Case-U enclosing program assembly and materialization.

The program evaluates original CaseUPointSolver integration at every actual
liquid coordinate. It publishes its real economic/status tree, not a carry or
replay facade. Public Model admission remains closed. The actual generation
owner and complete compiler/resident budget admission are separate prerequisites.
"""

import dis
import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.continuation import ContinuationPayload
from _lcm.current_iid import (
    CURRENT_IID_ARGUMENT,
    CURRENT_IID_NODES_ARGUMENT,
    CURRENT_IID_ROWS_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDPlan,
    NodeOperands,
)
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u import CaseUPointSolver, realise_case_u_nodes
from _lcm.egm.current_iid_evaluation import (
    CurrentResourcesEvaluation,
    _unflatten_current_resources_evaluation,
)
from _lcm.egm.projected_continuation import ProjectedEGMCarry
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import _EgmKernelPieces
from _lcm.egm.upper_envelope import get_bracket_finder
from _lcm.egm.upper_envelope._exact_affine import ffi as exact_affine_ffi
from _lcm.egm.upper_envelope._exact_affine import hermite_critical
from _lcm.egm.upper_envelope.certified_sign import backend_flushes_subnormals
from _lcm.egm.upper_envelope.query import (
    _derived_subnormal_possible,
    _subnormal_operand_present,
)
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    ProgramScope,
    ValueRead,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.footprint import ResidentInventory, concrete_device_bytes
from _lcm.execution.output_layout import OutputRoleTree, StateAxesLeading
from _lcm.execution.workspace_planning import (
    WorkspacePlan,
    compiler_memory_reservation,
    plan_axis_free_workspace,
)
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.shock_registry import ShockBinding
from _lcm.solution.continuation_reads import (
    continuation_leaf_reads,
)
from _lcm.solution.continuation_target import union_fixed_params
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.dcegm import EGMStepBuild, _carry_subset, _DCEGMArgumentBuilder
from _lcm.solution.feasibility import (
    BoundedResult,
    BoundInputStatus,
    _unflatten_query_input_status,
)
from _lcm.solution.fingerprint import (
    _native_numeric_callable_kind,
    _Reachable,
    _unwrap_exact_beartype_wrapper,
)
from _lcm.solution.generation import (
    CanonicalModelGeneration,
    GenerationSolverPreparation,
    canonical_generation_roots,
)
from _lcm.solution.preparation_clock import prepare_case_u_clock
from _lcm.time import ModelTime, TimeAxis
from _lcm.typing import (
    DataclassInstance,
    EconFunctionArg,
    FlatParams,
    PytreeValue,
    ShapeDtypePytree,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import NodeSet, ShockGroup, UserNodes
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarBool,
    ScalarFloat,
)

CASE_U_ECONOMIC_CORE = "case_u_economic"
_JAX_CONFIG_READ = type(jax.config).read
_ORIGINAL_CANONICAL_FLOAT_DTYPE = inspect.unwrap(canonical_float_dtype)
# The original envelope's memoised backend probe measures the compiled backend,
# never captured data; a callback reaching the same probe stays refused.
_ORIGINAL_BACKEND_SUBNORMAL_PROBE_CALLERS = tuple(
    inspect.unwrap(caller)
    for caller in (
        _subnormal_operand_present,
        _derived_subnormal_possible,
    )
)

_CASE_U_NATIVE_EXPORTS = tuple(
    (module, *export)
    for module in (exact_affine_ffi, hermite_critical)
    for export in module._ORIGINAL_NUMERICAL_EXPORTS  # noqa: SLF001
)
_CASE_U_PLAN_METHODS = tuple(
    (name, method, method.__code__, raw, raw.__code__)
    for name in ("realise_nodes", "realise_information_nodes")
    for method in (getattr(CurrentIIDPlan, name),)
    for raw in (_unwrap_exact_beartype_wrapper(method) or method,)
)


def _require_case_u_plan_methods() -> None:
    """Keep the resolved-plan metadata role on its original numerical methods."""
    for name, method, code, raw, raw_code in _CASE_U_PLAN_METHODS:
        if (
            vars(CurrentIIDPlan).get(name) is not method
            or method.__code__ is not code
            or (_unwrap_exact_beartype_wrapper(method) or method) is not raw
            or raw.__code__ is not raw_code
        ):
            raise TypeError("Original Case-U plan method authority changed.")


def _require_case_u_native_preflight(*, registered: bool) -> None:
    """Read completed DCEGM capability authority without loading or registering.

    Args:
        registered: Whether the route traces the native kernels, so they must
            already be registered. An ordinary route only reaches the exports
            by reference; their identity is still checked.
    """
    if (
        (registered and exact_affine_ffi._REGISTERED is not True)  # noqa: SLF001
        or exact_affine_ffi._TARGETS is not exact_affine_ffi._ORIGINAL_TARGETS  # noqa: SLF001
        or exact_affine_ffi.EXACT_AFFINE_HANDLER_SYMBOLS
        is not exact_affine_ffi._ORIGINAL_TARGETS  # noqa: SLF001
        or jax.ffi.ffi_call is not exact_affine_ffi._ORIGINAL_FFI_CALL[0]  # noqa: SLF001
        or jax.ffi.ffi_call.__code__ is not exact_affine_ffi._ORIGINAL_FFI_CALL[1]  # noqa: SLF001
    ):
        raise TypeError("Case-U native target preflight is absent or changed.")
    for module, name, function, code, raw, raw_code in (
        *_CASE_U_NATIVE_EXPORTS,
        (
            exact_affine_ffi,
            "_ensure_registered",
            *exact_affine_ffi._ORIGINAL_REGISTRATION,  # noqa: SLF001
        ),
    ):
        if (
            vars(module).get(name) is not function
            or function.__code__ is not code
            or (_unwrap_exact_beartype_wrapper(function) or function) is not raw
            or raw.__code__ is not raw_code
        ):
            raise TypeError("Original Case-U native export authority changed.")


_UNADMITTED_U_TRACE_FACTORIES = (
    np.array,
    np.asarray,
    np.arange,
    np.empty,
    np.empty_like,
    np.zeros,
    np.zeros_like,
    np.ones,
    np.ones_like,
    np.full,
    np.full_like,
    np.linspace,
    np.logspace,
    np.geomspace,
    np.eye,
    np.identity,
    jax.device_put,
    jax.random.PRNGKey,
    jax.random.key,
)


def _refuse_case_u_local_imports(*, code: types.CodeType) -> None:
    """Refuse unresolved import authorities in this actual callback's code."""
    if any(
        instruction.opname in {"IMPORT_NAME", "IMPORT_FROM", "IMPORT_STAR"}
        for instruction in dis.get_instructions(code)
    ):
        raise UnsupportedOperationError(
            "Case-U shape callback local import requires its exact producer "
            "authority before tracing."
        )
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            _refuse_case_u_local_imports(code=constant)


def _case_u_original_roots(**operands: jax.Array) -> tuple[jax.Array, ...]:
    """Retain already-existing inputs without a numerical dispatch or allocation."""
    return tuple(operands[f"operand_{index}"] for index in range(len(operands)))


def guard_case_u_shape(  # noqa: C901, PLR0912, PLR0915 - keep exact roles and pretrace checks together
    *,
    function: Callable[..., PytreeValue],
    context: CoreBuildContext,
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer,
) -> tuple[jax.Array, ...]:
    """Retain original inputs/captures and refuse unsafe callbacks before trace."""
    from _lcm.egm import current_iid_nnbegm as nnbegm_module  # noqa: PLC0415
    from _lcm.egm.current_iid_nbegm import (  # noqa: PLC0415 - exact original scalar NB callbacks
        NBCaseUPoint,
        NBCaseUProgram,
    )
    from _lcm.egm.current_iid_negm import (  # noqa: PLC0415
        _ORIGINAL_NEGM_CASE_U_METHODS,
        _ORIGINAL_NEGM_CASE_U_PROPERTIES,
        NEGMCaseUPoint,
    )
    from _lcm.egm.current_iid_negm_program import (  # noqa: PLC0415
        _ORIGINAL_NEGM_CASE_U_PROGRAM_METHODS,
        NEGMCaseUProgram,
    )
    from _lcm.egm.current_iid_nnbegm import (  # noqa: PLC0415
        _ORIGINAL_NNBEGM_HELPERS,
        _ORIGINAL_NNBEGM_IMPORTS,
        _ORIGINAL_NNBEGM_METHODS,
        _ORIGINAL_NNBEGM_PROPERTIES,
        NNBEGMCaseUPoint,
        NNBEGMInnerSource,
        require_original_nnbegm_case_u_methods,
    )
    from _lcm.egm.validation_stages import ValidationCaptureHasher  # noqa: PLC0415
    from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
        _ORIGINAL_CASE_U_PUBLISHER_METHODS,
        CaseUExactFunction,
        CaseURideExactFunction,
        PassiveCellsPublisher,
        _same_numerical_publisher,
    )
    from _lcm.solution.dcegm import _DCEGMPeriodKernel  # noqa: PLC0415
    from _lcm.solution.nbegm import (  # noqa: PLC0415
        _NBEGMContinuousCore,
        _NBEGMQueryIntervalEvaluators,
        _NBEGMTiledCore,
        _RideAlongNBEGMPeriodKernel,
    )
    from _lcm.solution.negm import _NEGMPeriodKernel  # noqa: PLC0415
    from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel  # noqa: PLC0415

    point = (
        function
        if type(function) is CaseUProgramFunction
        else function.point
        if (
            type(function) is CaseUExactFunction
            or type(function) is CaseURideExactFunction
        )
        else None
    )
    # The ride publisher's traced cell body reads the same point and demands.
    ride = function if type(function) is CaseURideExactFunction else None
    publisher = None if ride is None else ride.publisher

    # A missing ModelTime cache would create its clock array here, before admission.
    ages = context.ages
    if ages is None or (
        isinstance(ages, ModelTime) and ages.ages is None and "values" not in vars(ages)
    ):
        raise UnsupportedOperationError(
            "Case-U shape preparation requires its existing original clock array."
        )
    nested = point.pieces if type(point) is NEGMCaseUProgram else None
    if nested is not None and type(nested) not in (NEGMCaseUPoint, NNBEGMCaseUPoint):
        raise UnsupportedOperationError(
            "Cold nested shape capture requires its genuine NEGM/DC or NNBEGM/NB point."
        )
    nn_nested = type(nested) is NNBEGMCaseUPoint
    nn_inners = (
        (nested.keeper, nested.adjuster) if type(nested) is NNBEGMCaseUPoint else ()
    )
    nn_evaluators: list[_NBEGMQueryIntervalEvaluators] = []
    for inner in nn_inners:
        callback = inner.recipe.evaluate_preimages
        if (
            not isinstance(callback, types.MethodType)
            or type(callback.__self__) is not _NBEGMQueryIntervalEvaluators
        ):
            raise TypeError("Original NNBEGM query evaluator authority changed.")
        nn_evaluators.append(callback.__self__)
    nested_methods = (
        tuple(
            original[1:]
            for original in _ORIGINAL_NNBEGM_METHODS
            if original[0] is NNBEGMCaseUPoint
        )
        if nn_nested
        else _ORIGINAL_NEGM_CASE_U_METHODS
    )
    nested_properties = (
        tuple(
            original[1:]
            for original in _ORIGINAL_NNBEGM_PROPERTIES
            if original[0] is NNBEGMCaseUPoint
        )
        if nn_nested
        else _ORIGINAL_NEGM_CASE_U_PROPERTIES
    )
    if nn_nested:
        require_original_nnbegm_case_u_methods()
    if nested is not None:
        for owner, name, method, code, raw, raw_code in (
            *_ORIGINAL_CASE_U_PUBLISHER_METHODS,
            *(
                (NEGMCaseUProgram, *original)
                for original in _ORIGINAL_NEGM_CASE_U_PROGRAM_METHODS
            ),
        ):
            if (
                vars(owner).get(name) is not method
                or method.__code__ is not code
                or (_unwrap_exact_beartype_wrapper(method) or method) is not raw
                or raw.__code__ is not raw_code
            ):
                raise TypeError(
                    "Original NEGM publisher/program method authority changed."
                )
        for name, method, code, raw, raw_code in nested_methods:
            if (
                vars(type(nested)).get(name) is not method
                or method.__code__ is not code
                or (_unwrap_exact_beartype_wrapper(method) or method) is not raw
                or raw.__code__ is not raw_code
            ):
                raise TypeError("Original NEGM shape method authority changed.")
        for (
            name,
            descriptor,
            getter,
            code,
            raw,
            raw_code,
        ) in nested_properties:
            if (
                vars(type(nested)).get(name) is not descriptor
                or descriptor.fget is not getter
                or getter.__code__ is not code
                or (_unwrap_exact_beartype_wrapper(getter) or getter) is not raw
                or raw.__code__ is not raw_code
            ):
                raise TypeError("Original NEGM shape property authority changed.")
    if type(point) is CaseUProgramFunction:
        engine_receivers = (
            (point, CaseUProgramFunction.__call__),
            (point, CaseUProgramFunction.evaluate_with_law),
            (function, CaseUExactFunction.__call__),
        )
    elif type(point) is NBCaseUProgram:
        engine_receivers = (
            (point, NBCaseUProgram.evaluate_with_law),
            (point.pieces, NBCaseUPoint.node),
            (point.pieces.core, _NBEGMContinuousCore.construct_point),
            (function, CaseUExactFunction.__call__),
        )
    elif type(point) is NEGMCaseUProgram and type(nested) in (
        NEGMCaseUPoint,
        NNBEGMCaseUPoint,
    ):
        engine_receivers = (
            (point, NEGMCaseUProgram.evaluate_with_law),
            (
                function,
                CaseUExactFunction.__call__
                if type(function) is CaseUExactFunction
                else CaseURideExactFunction.__call__,
            ),
            *(
                ((publisher, PassiveCellsPublisher.__call__),)
                if publisher is not None
                else ()
            ),
            *((nested, original[1]) for original in nested_methods),
            *((nested, original[2]) for original in nested_properties),
            *(
                (receiver, original[2])
                for receiver in nn_inners
                for original in _ORIGINAL_NNBEGM_METHODS
                if original[0] is NNBEGMInnerSource
            ),
            *(
                (receiver, original[3])
                for receiver in nn_inners
                for original in _ORIGINAL_NNBEGM_PROPERTIES
                if original[0] is NNBEGMInnerSource
            ),
            *(
                (receiver, original[2])
                for receiver in (
                    *tuple(inner.core.envelope_build.bind_cell for inner in nn_inners),
                    *nn_evaluators,
                )
                for original in _ORIGINAL_NNBEGM_METHODS
                if original[0] is type(receiver)
            ),
        )
    else:
        engine_receivers = ()
    plan = None
    if (
        type(point) is CaseUProgramFunction
        or type(point) is NBCaseUProgram
        or type(point) is NEGMCaseUProgram
    ):
        _require_case_u_plan_methods()
        plan = point.plan
    engine_bound_methods = tuple(
        (id(_unwrap_exact_beartype_wrapper(method) or method), id(receiver))
        for receiver, method in engine_receivers
    ) + tuple(
        (id(raw), id(plan))
        for _name, _method, _code, raw, _raw_code in _CASE_U_PLAN_METHODS
        if plan is not None
    )
    # Only certified arithmetic and ExactEnvelope trace the native kernels.
    if type(point) is NBCaseUProgram:
        cores = (point.pieces.core,)
    elif nn_nested:
        cores = tuple(inner.core for inner in nn_inners)
    else:
        cores = None
    traces_native = cores is None or any(
        (
            core.envelope_arithmetic
            if isinstance(core, _NBEGMContinuousCore)
            else core.statics.envelope_arithmetic
        )
        == "certified"
        for core in cores
    )
    function_depth = 0
    function_frames: list[types.FunctionType] = []
    roots = list(
        canonical_generation_roots(
            declarations=(context, fixed_flat_params, ages.values)
        )
    )

    # NNBEGMInnerSource.__post_init__ checks its recipe evaluators by code identity.
    nn_recipe_evaluators = tuple(
        vars(nnbegm_module)[name]
        for name in (
            "_ORIGINAL_PREIMAGES",
            "_ORIGINAL_SIDES",
            "_ORIGINAL_RAW_PREIMAGES",
            "_ORIGINAL_RAW_SIDES",
        )
    )

    def in_original_inner_post_init() -> bool:
        return bool(function_frames) and any(
            original[0] is NNBEGMInnerSource
            and original[1] == "__post_init__"
            and any(function_frames[-1] is method for method in original[2::2])
            for original in _ORIGINAL_NNBEGM_METHODS
        )

    class CaptureGuard(ValidationCaptureHasher):
        # Abstract enclosing-kernel tracing preserves no separate eager stages.
        _preserve_eager_boundary = False

        def visit(
            self, *, value: _Reachable, _ignore_beartype_guards: bool = False
        ) -> None:
            for (
                _module,
                _name,
                original,
                _code,
                raw,
                _raw_code,
            ) in _CASE_U_NATIVE_EXPORTS:
                if value is original or value is raw:
                    _require_case_u_native_preflight(registered=traces_native)
                    self.frame(label="abstract-original-native-numerical-abi")
                    self._visit_native_python_function_seal(raw)
                    return
            if isinstance(value, jax.Array):
                roots.append(value)
                self.frame(
                    label="abstract-capture-descriptor",
                    payload=repr((value.shape, value.dtype.str)).encode(),
                )
                return
            if isinstance(value, np.ndarray):
                raise UnsupportedOperationError(
                    "Case-U shape callback host-array capture requires its "
                    "original producer admission before tracing."
                )
            if isinstance(value, types.FunctionType):
                original_import_method = (
                    nn_nested
                    and any(
                        value is original[4]
                        and value.__code__ is original[5]
                        and (original[0].__name__, original[1])
                        in {
                            (NNBEGMCaseUPoint.__name__, "__post_init__"),
                            ("_NBEGMCellContinuation", "_cell_rows_for_pool"),
                            ("_NBEGMBranchSolver", "solve"),
                            (
                                _ORIGINAL_NNBEGM_HELPERS[0][0].__name__,
                                "_solve_ride_along_cell_step",
                            ),
                            (
                                _ORIGINAL_NNBEGM_HELPERS[0][0].__name__,
                                "_nbegm_cell_breakpoints",
                            ),
                            (
                                _ORIGINAL_NNBEGM_HELPERS[0][0].__name__,
                                "_cell_breakpoint",
                            ),
                        }
                        for original in (
                            *_ORIGINAL_NNBEGM_METHODS,
                            *_ORIGINAL_NNBEGM_HELPERS,
                        )
                    )
                    and all(
                        any(
                            frame is candidate
                            for original in (
                                *_ORIGINAL_NNBEGM_METHODS,
                                *_ORIGINAL_NNBEGM_HELPERS,
                            )
                            for candidate in (original[2], original[4])
                        )
                        for frame in function_frames
                    )
                )
                if not original_import_method:
                    _refuse_case_u_local_imports(code=value.__code__)
            # Fingerprinting a host factory's version does not admit its allocation.
            if _native_numeric_callable_kind(value) in {
                "numpy-ufunc",
                "numpy-array-function",
            } or any(value is producer for producer in _UNADMITTED_U_TRACE_FACTORIES):
                raise UnsupportedOperationError(
                    "Case-U abstract callback numerical factory requires its "
                    "original producer admission before tracing."
                )
            super().visit(value=value, _ignore_beartype_guards=_ignore_beartype_guards)

        def _visit_dataclass(self, value: DataclassInstance) -> None:  # noqa: C901, PLR0912 - original persistent numerical records and closed metadata roles
            if not function_depth and nn_nested:
                for inner in nn_inners:
                    if value is inner:
                        self._visit_nn_inner_identity(inner=inner)
                        for declaration in fields(inner):
                            if declaration.name != "kernel":
                                self.visit(value=getattr(inner, declaration.name))
                        self._visit_referenced_bound_method(inner.__post_init__)
                        return
                    if value is inner.core:
                        # NN reads this original numerical record; it does not call
                        # the standalone tiled publisher or compiled CorePrograms.
                        for declaration in fields(inner.core):
                            self.visit(value=getattr(inner.core, declaration.name))
                        return
            if not function_depth and nested is not None and value is nested:
                self._visit_nested_identity()
                for declaration in fields(nested):
                    if declaration.name in {"owner", "context", "plan"}:
                        continue
                    self.frame(label="nested-field", payload=declaration.name.encode())
                    self.visit(value=getattr(nested, declaration.name))
                self._visit_referenced_bound_method(nested.__post_init__)
                return
            if (
                not function_depth
                and ride is not None
                and publisher is not None
                and value is publisher
            ):
                # A retained publisher's point differs from the function's only in
                # projected declarations its body never reads; capture the latter.
                if not _same_numerical_publisher(
                    retained=publisher,
                    current=PassiveCellsPublisher(
                        point=ride.point, capacity=ride.capacity, demands=ride.demands
                    ),
                ):
                    raise TypeError("Passive U publisher lost its function's point.")
                self.frame(label="abstract-original-ride-publisher")
                self.visit(value=ride.point)
                self.visit(value=ride.capacity)
                self.visit(value=ride.demands)
                self._visit_referenced_bound_method(publisher.__call__)
                return
            if function_depth or point is None or value is not point:
                super()._visit_dataclass(value)
                return
            self.frame(label="abstract-original-point")
            for declaration in fields(point):
                self.frame(label="point-field", payload=declaration.name.encode())
                if declaration.name == "plan":
                    self._visit_original_plan(plan=point.plan)
                else:
                    self.visit(value=getattr(point, declaration.name))
            if type(point) in (NBCaseUProgram, NEGMCaseUProgram):
                self._visit_referenced_bound_method(point.evaluate_with_law)
            elif type(point) is CaseUProgramFunction:
                self._visit_referenced_bound_method(point.__call__)

        def _visit_program_reference(self, *, path: tuple[str, ...]) -> None:
            if type(point) is not NEGMCaseUProgram:
                raise TypeError("Nested program role lost its exact source program.")
            if path == ("pieces",):
                self.visit(value=point.pieces)
            elif path[:1] == ("pieces",):
                self._visit_nested_reference(path=path[1:])
            elif path == ("plan",):
                self._visit_original_plan(plan=point.plan)
            else:
                super()._visit_object_reference(
                    value=point, attribute_paths=frozenset({path})
                )

        def _visit_nested_reference(self, *, path: tuple[str, ...]) -> None:
            if not isinstance(nested, (NEGMCaseUPoint, NNBEGMCaseUPoint)):
                raise TypeError("Nested receiver role lost its exact source point.")
            if (
                nn_nested
                and path
                in {
                    ("owner", "liquid_state_name"),
                    ("owner", "inner_action"),
                    ("owner", "outer_action"),
                    ("owner", "outer_grid_values", "shape"),
                    ("context",),
                    ("context", "transitions"),
                }
            ) or path in {
                ("owner",),
                ("owner", "regime_name"),
                ("owner", "keeper_kernel"),
                ("owner", "adjuster_kernel"),
                ("owner", "outer_post_decision"),
                ("owner", "transition_target_names"),
                ("context", "regime_name"),
                ("context", "current_iid_plans", "get"),
                ("context", "certainty_equivalent"),
            }:
                self._visit_nested_identity()
            elif path[:1] in {("keeper",), ("adjuster",)}:
                if len(path) == 1:
                    self.visit(value=getattr(nested, path[0]))
                elif nn_nested:
                    self._visit_nn_inner_reference(
                        inner=getattr(nested, path[0]), path=path[1:]
                    )
                else:
                    super()._visit_object_reference(
                        value=nested, attribute_paths=frozenset({path})
                    )
            elif path == ("plan",):
                self._visit_original_plan(plan=nested.plan)
            elif len(path) == 1 and any(
                path[0] == original[0] for original in nested_properties
            ):
                for original in nested_properties:
                    if original[0] == path[0]:
                        self._visit_referenced_bound_method(
                            types.MethodType(original[2], nested)
                        )
            else:
                super()._visit_object_reference(
                    value=nested, attribute_paths=frozenset({path})
                )

        def _visit_nn_inner_identity(self, *, inner: NNBEGMInnerSource) -> None:
            if (
                type(inner) is not NNBEGMInnerSource
                or type(inner.core) is not _NBEGMTiledCore
                or type(inner.kernel) is not _RideAlongNBEGMPeriodKernel
                or inner.kernel.core_programs()["main"].function is not inner.core
            ):
                raise TypeError("NNBEGM capture lost its genuine original inner core.")
            self.frame(label="abstract-original-nn-inner-owner")

        def _visit_nn_inner_reference(  # noqa: C901 - original inner receiver paths stay together
            self, *, inner: NNBEGMInnerSource, path: tuple[str, ...]
        ) -> None:
            self._visit_nn_inner_identity(inner=inner)
            if path in {("kernel",), ("kernel", "core_programs")}:
                self.frame(label="original-nn-inner-graph-identity")
            elif path == ("core",):
                self.visit(value=inner.core)
            elif path == ("core", "envelope_build", "bind_cell"):
                self.visit(value=inner.core.envelope_build.bind_cell)
            elif path == ("core", "continuation_plan"):
                self.visit(value=inner.core.continuation_plan)
            elif path == ("core", "schedule_spec", "sources"):
                self.visit(value=inner.core.schedule_spec.sources)
            elif path == ("recipe", "source_identity"):
                self.visit(value=inner.recipe.source_identity)
            elif path[:1] == ("continuation_plan",) and len(path) > 1:
                self._visit_nn_inner_reference(inner=inner, path=("continuation_plan",))
                super()._visit_object_reference(
                    value=inner.core.continuation_plan,
                    attribute_paths=frozenset({path[1:]}),
                )
            elif len(path) == 1 and any(
                original[0] is NNBEGMInnerSource and original[1] == path[0]
                for original in _ORIGINAL_NNBEGM_PROPERTIES
            ):
                for original in _ORIGINAL_NNBEGM_PROPERTIES:
                    if original[0] is NNBEGMInnerSource and original[1] == path[0]:
                        self._visit_referenced_bound_method(
                            types.MethodType(original[3], inner)
                        )
            else:
                super()._visit_object_reference(
                    value=inner, attribute_paths=frozenset({path})
                )

        def _visit_nested_identity(self) -> None:
            if type(nested) is NNBEGMCaseUPoint:
                if (
                    type(nested.owner) is not _FiniteNNBEGMPeriodKernel
                    or nested.owner.regime_name != nested.context.regime_name
                    or nested.context.current_iid_plans.get(nested.period)
                    is not nested.plan
                    or nested.owner.keeper_kernel is not nested.keeper.kernel
                    or nested.owner.adjuster_kernel is not nested.adjuster.kernel
                ):
                    raise TypeError(
                        "NNBEGM capture lost its genuine original owner/plan."
                    )
                for inner in nn_inners:
                    self._visit_nn_inner_identity(inner=inner)
                self.visit(
                    value=(
                        nested.context.regime_name,
                        tuple(nested.context.transitions),
                        nested.context.certainty_equivalent,
                        nested.owner.regime_name,
                        nested.owner.outer_post_decision,
                        nested.owner.liquid_state_name,
                        nested.owner.inner_action,
                        nested.owner.outer_action,
                    )
                )
                self.visit(value=nested.owner.outer_grid_values)
                self._visit_original_plan(plan=nested.plan)
                return
            if type(nested) is not NEGMCaseUPoint:
                raise TypeError("Nested shape capture lost its exact source point.")
            # Read original identity checks without executing a callback before
            # admission. The sealed with_operands method repeats full validation.
            if (
                type(nested.owner) is not _NEGMPeriodKernel
                or nested.owner.regime_name != nested.context.regime_name
                or nested.context.current_iid_plans.get(nested.period)
                is not nested.plan
            ):
                raise TypeError("Nested shape capture lost its original owner/plan.")
            for kernel, pieces in (
                (nested.owner.keeper_kernel, nested.keeper),
                (nested.owner.adjuster_kernel, nested.adjuster),
            ):
                if (
                    type(kernel) is not _DCEGMPeriodKernel
                    or type(kernel.demand_step) is not _EGMStep
                    or kernel.demand_step.pieces is not pieces
                    or pieces.continuation_plan.certainty_equivalent
                    is not nested.context.certainty_equivalent
                ):
                    raise TypeError("Nested shape capture lost original inner pieces.")
            self.frame(label="abstract-original-nested-owner")
            self.visit(
                value=(
                    nested.context.regime_name,
                    nested.context.certainty_equivalent,
                    nested.owner.regime_name,
                    nested.owner.outer_post_decision,
                    nested.owner.transition_target_names,
                )
            )
            self._visit_original_plan(plan=nested.plan)

        def _visit_original_plan(self, *, plan: CurrentIIDPlan) -> None:
            if type(plan) is not CurrentIIDPlan:
                raise TypeError("Case-U capture lost its original current plan.")
            self.frame(label="abstract-original-current-plan")
            self.visit(value=plan.indices)
            self.visit(value=plan.node_operands)
            for name, binding, nodes in plan.bindings:
                if (
                    type(binding) is not ShockBinding
                    or type(binding.group) is not ShockGroup
                    or type(nodes) is not NodeSet
                ):
                    raise TypeError(
                        "Case-U capture lost its original resolved binding."
                    )
                self.visit(value=name)
                self.visit(value=nodes)
                for declaration in fields(binding):
                    self.frame(label="binding-field", payload=declaration.name.encode())
                    if declaration.name != "group":
                        self.visit(value=getattr(binding, declaration.name))
                        continue
                    for member in fields(binding.group):
                        self.frame(label="group-field", payload=member.name.encode())
                        rule = binding.group.nodes
                        if member.name == "nodes" and type(rule) is UserNodes:
                            # Only the resolved rule declaration is metadata;
                            # direct callback reads still reach the host-array refusal.
                            for array in (rule.nodes, rule.weights):
                                if (
                                    type(array) is not np.ndarray
                                    or array.flags.writeable
                                ):
                                    raise TypeError(
                                        "Original UserNodes metadata lost its owned "
                                        "read-only host declaration."
                                    )
                                self.visit(value=(array.shape, array.dtype))
                            self.visit(value=rule.signed)
                        else:
                            self.visit(value=getattr(binding.group, member.name))

        def _visit_function(
            self,
            *,
            function: types.FunctionType,
            ignore_beartype_guards: bool = False,
        ) -> None:
            nonlocal function_depth
            function_depth += 1
            function_frames.append(function)
            try:
                super()._visit_function(
                    function=function, ignore_beartype_guards=ignore_beartype_guards
                )
            finally:
                function_frames.pop()
                function_depth -= 1

        def _visit_direct_global_reference(self, value: _Reachable) -> None:
            if (
                value is backend_flushes_subnormals
                and function_frames
                and any(
                    inspect.unwrap(function_frames[-1]) is caller
                    for caller in _ORIGINAL_BACKEND_SUBNORMAL_PROBE_CALLERS
                )
            ):
                self.frame(label="original-backend-subnormal-probe")
                return
            if (
                nn_nested
                and isinstance(value, types.CodeType)
                and in_original_inner_post_init()
                and any(
                    value is evaluator.__code__ for evaluator in nn_recipe_evaluators
                )
            ):
                self.frame(
                    label="original-nn-recipe-code-identity",
                    payload=value.co_qualname.encode(),
                )
                return
            super()._visit_direct_global_reference(value)

        def _visit_referenced_bound_method(self, method: types.MethodType) -> None:
            if (
                nn_nested
                and method.__self__ is jax.config
                and method.__func__ is _JAX_CONFIG_READ
                and function_frames
                and function_frames[-1] is _ORIGINAL_CANONICAL_FLOAT_DTYPE
            ):
                # The original active-precision read; JAX's config holds no
                # captured data.
                self.frame(label="original-canonical-float-dtype-config-read")
                return
            if (
                publisher is not None
                and method.__self__ is publisher
                and method.__func__ is vars(PassiveCellsPublisher)["traced"]
            ):
                # `traced` only wraps the publisher's own call in one inlined jit.
                self.frame(label="original-traced-passive-cells")
                self.visit(value=publisher)
                return
            super()._visit_referenced_bound_method(method)

        def _visit_object_reference(  # noqa: C901, PLR0912 - exact original engine receiver boundary
            self,
            *,
            value: _Reachable,
            attribute_paths: frozenset[tuple[str, ...]],
        ) -> None:
            if (
                nn_nested
                and value is object
                and attribute_paths == frozenset({("__setattr__",)})
                and any(
                    original[1] == "__init__"
                    and function_frames
                    and function_frames[-1] is original[4]
                    for original in _ORIGINAL_NNBEGM_METHODS
                )
            ):
                if (
                    inspect.getattr_static(object, "__setattr__")
                    is not object.__setattr__
                ):
                    raise TypeError("Original frozen dataclass setter changed.")
                self.frame(label="original-frozen-dataclass-object-setattr")
                return
            if (
                nn_nested
                and ("__code__",) in attribute_paths
                and attribute_paths <= {(), ("__code__",)}
                and isinstance(value, types.FunctionType)
                and in_original_inner_post_init()
                and any(value is evaluator for evaluator in nn_recipe_evaluators)
            ):
                # The original inner owner's code-identity check of its recipe
                # evaluators; the evaluators themselves are hashed as the recipe.
                self.frame(
                    label="original-nn-recipe-code-identity",
                    payload=value.__qualname__.encode(),
                )
                attribute_paths = attribute_paths - {("__code__",)}
                if not attribute_paths:
                    return
            if function_depth or not any(
                identity in self._active_bound_methods
                for identity in engine_bound_methods
                if identity[1] == id(value)
            ):
                super()._visit_object_reference(
                    value=value, attribute_paths=attribute_paths
                )
                return
            for path in sorted(attribute_paths):
                if type(value) is NNBEGMInnerSource and any(
                    value is inner for inner in nn_inners
                ):
                    self._visit_nn_inner_reference(inner=value, path=path)
                elif (
                    type(value) is _NBEGMQueryIntervalEvaluators
                    and any(value is evaluator for evaluator in nn_evaluators)
                    and path == ("sources",)
                ):
                    self.visit(value=value.sources)
                elif (
                    nested is not None
                    and (value is function or value is publisher)
                    and path[:1] == ("point",)
                ):
                    if path == ("point",):
                        self.visit(value=point)
                    else:
                        self._visit_program_reference(path=path[1:])
                elif nested is not None and value is point:
                    self._visit_program_reference(path=path)
                elif (
                    publisher is not None
                    and value is function
                    and path == ("publisher",)
                ):
                    self.visit(value=publisher)
                elif nested is not None and value is nested:
                    self._visit_nested_reference(path=path)
                elif (point is not None and value is point and path == ("plan",)) or (
                    point is not None and value is point.plan and path == ("bindings",)
                ):
                    self._visit_original_plan(plan=point.plan)
                elif point is not None and value is point and path == ("pieces",):
                    self.visit(value=point.pieces)
                elif (
                    (
                        type(function) is CaseUExactFunction
                        or type(function) is CaseURideExactFunction
                    )
                    and (value is function or value is publisher)
                    and path == ("demands",)
                ):
                    self.visit(value=function.demands)
                else:
                    super()._visit_object_reference(
                        value=value, attribute_paths=frozenset({path})
                    )

    recorder = CaptureGuard(array_recorder=roots.append)
    if nn_nested:
        for owner, name, _method, _code, raw, _raw_code in _ORIGINAL_NNBEGM_METHODS:
            if name == "__init__" or (owner.__name__, name) in {
                ("_NBEGMCellBinder", "__call__"),
                ("_NBEGMCellContinuation", "__post_init__"),
                ("_NBEGMCellContinuation", "__call__"),
                ("_NBEGMCellContinuation", "_cell_rows_for_pool"),
                ("_NBEGMCellSolver", "construct_owned_query_row"),
                ("_NBEGMBranchSolver", "solve"),
                ("_BoundScalarFunction", "__call__"),
            }:
                recorder.visit(value=raw, _ignore_beartype_guards=True)
        for (
            _module,
            _original_module,
            _name,
            original,
            _code,
            _raw,
            _raw_code,
        ) in _ORIGINAL_NNBEGM_IMPORTS:
            recorder.visit(value=original)
    recorder.visit(value=function)
    # The existing validator identity recipe has no primitive/output allocation;
    # it publishes ALL original roots before any clock upload or possible copy.
    held = materializer.evaluate_validation(
        function=_case_u_original_roots,
        arguments={f"operand_{index}": value for index, value in enumerate(roots)},
    )
    if any(left is not right for left, right in zip(roots, held, strict=True)):
        raise TypeError("Case-U capture custody substituted an original array.")
    for value in held:
        actual = materializer(
            value=value, dtype=np.dtype(value.dtype), name="case_u_trace_capture"
        )
        if actual is not value:
            raise UnsupportedOperationError(
                "Case-U shape capture requires its original admitted array "
                "identity; remapped capture placement remains unsupported."
            )

    return held


def trace_case_u_shape(
    *,
    function: Callable[..., PytreeValue],
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
) -> ShapeDtypePytree:
    """Run only the original abstract computation after capture/operand admission."""
    return jax.eval_shape(function, **arguments)


def materialize_case_u_schema(
    *, schema: ShapeDtypePytree, materializer: DatedGenerationMaterializer | None
) -> PytreeValue:
    """Materialize the original schema through its applicable allocation route."""
    from _lcm.solution.current_iid_case_r import _zeros_for_template  # noqa: PLC0415

    if materializer is None:
        return _zeros_for_template(schema=schema)
    return materializer.produce(
        function=partial(_zeros_for_template, schema=schema), arguments={}
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUArguments:
    """Project actual dated DC operands plus explicit current/code-grid banks."""

    base: _DCEGMArgumentBuilder
    plan: CurrentIIDPlan
    pieces: _EgmKernelPieces
    period: int
    ages: TimeAxis
    fixed_flat_params: FlatParams = MappingProxyType({})
    trace_capture_roots: tuple[jax.Array, ...] = ()
    """Original referenced callback buffers retained beyond the entry scope."""

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        if context.period != self.period or context.ages is not self.ages:
            raise TypeError(
                "Case-U materialization requires its original source clock."
            )
        if context.state_action_space is None:
            raise TypeError("Case-U materialization requires its actual state arrays.")
        return self._arguments(context=context, base=lambda: self.base(context))

    def prepare(
        self, *, context: CoreBuildContext, materializer: DatedGenerationMaterializer
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Use the same argument assembly with an admitted original clock."""
        if context.period != self.period or context.ages is not self.ages:
            raise TypeError(
                "Case-U materialization requires its original source clock."
            )
        if context.state_action_space is None:
            raise TypeError("Case-U materialization requires its actual state arrays.")
        return self._arguments(
            context=context,
            base=lambda: self.base.with_clock(
                context=context,
                clock=lambda: prepare_case_u_clock(
                    period=context.period, ages=self.ages, materializer=materializer
                ),
            ),
        )

    def _arguments(
        self,
        *,
        context: CoreBuildContext,
        base: Callable[[], Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]],
    ) -> Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]:
        """Share the original fixed/free/node/read operand assembly unchanged."""
        fixed = union_fixed_params(
            fixed_flat_params=self.fixed_flat_params,
            regime_name=self.base.regime_name,
            transition_target_names=self.base.transition_target_names,
        )
        arguments = {**fixed, **base()}
        arguments[CURRENT_IID_ARGUMENT] = {
            name: {
                **self.fixed_flat_params.get(name, {}),
                **context.flat_params.get(name, {}),
            }
            for name, _, _ in self.plan.bindings
        }
        arguments[CURRENT_IID_NODES_ARGUMENT] = self.plan.node_operands
        arguments[CURRENT_IID_ROWS_ARGUMENT] = self.plan.indices
        arguments["_lcm_case_u_savings_nodes"] = self.pieces.savings_nodes
        arguments["_lcm_case_u_borrowing_limit"] = self.pieces.borrowing_limit
        arguments["_lcm_case_u_floor_status"] = self.pieces.floor_status
        arguments["_lcm_case_u_child_node_values"] = tuple(
            read.stochastic_node_values
            for read in self.pieces.continuation_plan.child_reads.values()
        )
        return MappingProxyType(arguments)


def _bind_numeric_pieces(
    *,
    original: _EgmKernelPieces,
    savings_nodes: FloatND,
    borrowing_limit: ScalarFloat,
    floor_status: BoundInputStatus | None,
    child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
) -> _EgmKernelPieces:
    """Bind explicit numeric leaves under their original immutable read code."""
    reads = original.continuation_plan.child_reads
    if len(child_node_values) != len(reads):
        raise TypeError("Case-U dynamic child grids lost their actual read roster.")
    rebound = {}
    for (target, read), nodes in zip(reads.items(), child_node_values, strict=True):
        if tuple((node.shape, node.dtype) for node in nodes) != tuple(
            (node.shape, node.dtype) for node in read.stochastic_node_values
        ):
            raise TypeError("Case-U dynamic child grids changed the original schema.")
        rebound[target] = replace(read, stochastic_node_values=nodes)
    if savings_nodes.shape != original.savings_nodes.shape:
        raise TypeError("Case-U savings operands changed the original node shape.")
    return replace(
        original,
        savings_nodes=savings_nodes,
        borrowing_limit=borrowing_limit,
        floor_status=floor_status,
        continuation_plan=replace(
            original.continuation_plan, child_reads=MappingProxyType(rebound)
        ),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class CaseUProgramFunction:
    """The real enclosing state/current-node/savings/read computation."""

    pieces: _EgmKernelPieces
    plan: CurrentIIDPlan
    period: int
    savings_point_width: int | None
    stochastic_node_width: int | None
    envelope_cell_width: int

    def __call__(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_case_u_savings_nodes: FloatND,
        _lcm_case_u_borrowing_limit: ScalarFloat,
        _lcm_case_u_floor_status: BoundInputStatus | None,
        _lcm_case_u_child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        **kwargs: EconFunctionArg,
    ) -> BoundedResult[CurrentResourcesEvaluation]:
        return self.evaluate_with_law(
            next_regime_to_continuation=next_regime_to_continuation,
            _lcm_current_iid=_lcm_current_iid,
            _lcm_current_iid_nodes=_lcm_current_iid_nodes,
            _lcm_current_iid_rows=_lcm_current_iid_rows,
            _lcm_case_u_savings_nodes=_lcm_case_u_savings_nodes,
            _lcm_case_u_borrowing_limit=_lcm_case_u_borrowing_limit,
            _lcm_case_u_floor_status=_lcm_case_u_floor_status,
            _lcm_case_u_child_node_values=_lcm_case_u_child_node_values,
            **kwargs,
        )[0]

    def evaluate_with_law(
        self,
        *,
        next_regime_to_continuation: MappingProxyType[RegimeName, ContinuationPayload],
        _lcm_current_iid: AdmittedGroupParams,
        _lcm_current_iid_nodes: NodeOperands,
        _lcm_current_iid_rows: IntND,
        _lcm_case_u_savings_nodes: FloatND,
        _lcm_case_u_borrowing_limit: ScalarFloat,
        _lcm_case_u_floor_status: BoundInputStatus | None,
        _lcm_case_u_child_node_values: tuple[tuple[FloatND | IntND, ...], ...],
        **kwargs: EconFunctionArg,
    ) -> tuple[BoundedResult[CurrentResourcesEvaluation], BoolND]:
        """Return the same original source evaluation and its actual law witness."""
        state = self.pieces.euler_state_name
        liquid = kwargs[state]
        if not isinstance(liquid, jax.Array) or liquid.ndim != 1:
            raise UnsupportedOperationError(
                "Case-U enclosing core requires its scalar liquid axis."
            )
        pool = {name: value for name, value in kwargs.items() if name != state}
        pieces = _bind_numeric_pieces(
            original=self.pieces,
            savings_nodes=_lcm_case_u_savings_nodes,
            borrowing_limit=_lcm_case_u_borrowing_limit,
            floor_status=_lcm_case_u_floor_status,
            child_node_values=_lcm_case_u_child_node_values,
        )
        solver = CaseUPointSolver(
            pieces=pieces,
            pool=pool,
            children=_carry_subset(
                next_regime_to_continuation=next_regime_to_continuation,
                stateful_targets=frozenset(
                    (
                        *pieces.continuation_plan.stateful_targets,
                        *pieces.continuation_plan.scalar_targets,
                    )
                ),
            ),
            savings_point_width=self.savings_point_width,
            stochastic_node_width=self.stochastic_node_width,
            envelope_cell_width=self.envelope_cell_width,
        )

        def integrate(
            coordinate: ScalarFloat,
        ) -> tuple[BoundedResult[CurrentResourcesEvaluation], ScalarBool]:
            nodes = realise_case_u_nodes(
                plan=self.plan,
                liquid=coordinate,
                liquid_state=state,
                arguments=pool,
                group_params=_lcm_current_iid,
                node_operands=_lcm_current_iid_nodes,
                indices=_lcm_current_iid_rows,
            )
            return solver.integrate(liquid=coordinate, nodes=nodes), nodes.law_valid

        result, law_valid = jax.vmap(integrate)(liquid)
        if result.query_status is None:
            raise TypeError("Case-U enclosing core lost its original source status.")
        date_valid = jnp.asarray(kwargs["period"]) == self.period
        own = result.query_status[-1]
        return BoundedResult(
            output=_unflatten_current_resources_evaluation(
                None,
                (
                    result.output.value,
                    result.output.economic_marginal,
                    result.output.empty,
                    result.output.valid & date_valid,
                ),
            ),
            bound_status=result.bound_status,
            query_status=(
                *result.query_status[:-1],
                _unflatten_query_input_status(
                    None,
                    (
                        own.nonfinite_floor,
                        own.invalid_shifted_grid,
                        own.coverage_missing | ~date_valid,
                    ),
                ),
            ),
            continuation_read_counts=result.continuation_read_counts,
        ), law_valid


def case_u_output_roles(
    *,
    template: BoundedResult[CurrentResourcesEvaluation],
    liquid_state: ReferenceName,
) -> OutputRoleTree:
    """Use the actual abstract result's trusted tree reconstruction for roles."""

    def role(leaf: jax.Array | jax.ShapeDtypeStruct) -> StateAxesLeading:
        if len(leaf.shape) != 1:
            raise UnsupportedOperationError(
                "Case-U scalar enclosing output has an unrepresented axis."
            )
        return StateAxesLeading(state_names=(liquid_state,), shape=(), dtype=leaf.dtype)

    return jax.tree.map(role, template)


@dataclass(frozen=True, kw_only=True, eq=False)
class OwnedCaseUProgram:
    """A real generation's private enclosing program, not an installed publisher."""

    preparation: GenerationSolverPreparation
    program: CoreProgram
    template: BoundedResult[CurrentResourcesEvaluation]

    def materialize(self, *, context: CoreBuildContext) -> MaterializedCoreProgram:
        self.preparation.validate_owner()
        if context.period != self.preparation.period:
            raise TypeError("Case-U materialization changed its genuine producer date.")
        return materialize_core_program(program=self.program, context=context)

    @property
    def roots(self) -> tuple[jax.Array, ...]:
        """Actual code/grid and live generation owners through profile and dispatch."""
        self.preparation.validate_owner()
        return canonical_generation_roots(
            declarations=(
                self.preparation.generation.live_roots(),
                self.preparation.code_roots,
                self.program,
            )
        )

    def require_model_installation(self) -> None:
        """Do not turn a privately materialized program into Model admission."""
        self.preparation.validate_owner()
        actual = core_program_graph(kernel=self.preparation.installed_kernel)
        if actual.get(CASE_U_ECONOMIC_CORE) is not self.program:
            raise UnsupportedOperationError(
                "Case-U enclosing program is not installed in its actual Model "
                "schedule/publisher/parent/forward graph; private profiling "
                "does not grant Model admission."
            )


def _admit_scalar_owner(*, preparation: GenerationSolverPreparation) -> None:
    preparation.validate_owner()
    context = preparation.context
    pieces = preparation.step.pieces
    if preparation.plan is None:
        raise UnsupportedOperationError(
            "Case-U requires the actual dated current-IID plan."
        )
    if context.current_iid_plans.get(preparation.period) is not preparation.plan:
        raise TypeError(
            "Case-U source plan differs from its real retained preparation."
        )
    if preparation.build.steps[preparation.period] is not preparation.step:
        raise TypeError("Case-U source step differs from its real dated build.")
    if (
        pieces.continuation_plan.certainty_equivalent
        is not context.certainty_equivalent
    ):
        raise TypeError(
            "Case-U CE declaration differs from its genuine source context."
        )
    if tuple(context.state_action_space.states) != (pieces.euler_state_name,):
        raise UnsupportedOperationError(
            "Case-U enclosing core has not represented passive/discrete state axes."
        )
    if (
        pieces.combo_names
        or pieces.query_spec is not None
        or pieces.lower_func is not None
        or context.state_action_space.discrete_actions
        or context.sharded_state_names
        or context.co_map_state_names
        or context.action_partitions != 1
        or pieces._prepared_nested_floor is not None  # noqa: SLF001
        or pieces._prepared_nested_axes  # noqa: SLF001
    ):
        raise UnsupportedOperationError(
            "Case-U enclosing source still requires its passive/choice/query/"
            "moving-floor/nested/sharded economic installation."
        )


def build_owned_case_u_program(
    *,
    generation: CanonicalModelGeneration,
    regime_name: RegimeName,
    period: int,
    context: CoreBuildContext,
) -> OwnedCaseUProgram:
    """Acquire the actual retained Model build, then assemble its real body.

    Abstract evaluation runs the genuine original computation's shape trace; it
    is not a hand-written template or production allocation. Runtime preparation
    requires a separately released bounded window.
    """
    preparation = generation.solver_preparation(regime_name=regime_name, period=period)
    _admit_scalar_owner(preparation=preparation)
    dated_parameters = generation.execution._dated_parameters  # noqa: SLF001 - genuine generation owns its fixed-parameter preparation
    if dated_parameters is None:
        raise UnsupportedOperationError(
            "Case-U source lacks actual normalized fixed-parameter metadata."
        )
    plan = preparation.plan
    if plan is None:
        raise TypeError("Admitted Case-U source lost its actual current plan.")
    program, template = prepare_case_u_point_program(
        source=preparation.context,
        build=preparation.build,
        period=period,
        call_context=context,
        fixed_flat_params=dated_parameters.fixed.granular,
        preparation=preparation,
    )
    return OwnedCaseUProgram(
        preparation=preparation, program=program, template=template
    )


def _case_u_call_reads(
    *,
    source: SolverBuildContext,
    pieces: _EgmKernelPieces,
    period: int,
    call_context: CoreBuildContext,
) -> tuple[ValueRead, ...]:
    """Read actual dated child payloads, not a provisional schema or fake carry."""
    reads = []
    plan = pieces.continuation_plan
    for target in (*plan.stateful_targets, *plan.scalar_targets):
        child = call_context.next_regime_to_continuation[target]
        if not isinstance(child, (EGMCarry, ProjectedEGMCarry)):
            raise UnsupportedOperationError(
                "Case-U child lacks its actual typed continuation schema."
            )
        reads.extend(
            continuation_leaf_reads(
                template=child,
                artifact_key=child.artifact_key,
                target=target,
                source_regime=source.regime_name,
                source_period=period,
                core_key=CASE_U_ECONOMIC_CORE,
            )
        )
    return tuple(reads)


def prepare_case_u_point_program(
    *,
    source: SolverBuildContext,
    build: EGMStepBuild,
    period: int,
    call_context: CoreBuildContext,
    fixed_flat_params: FlatParams,
    preparation: GenerationSolverPreparation | None = None,
    materializer: DatedGenerationMaterializer | None = None,
) -> tuple[CoreProgram, BoundedResult[CurrentResourcesEvaluation]]:
    """Assemble from original dated pass objects shared with the final generation."""
    step = build.steps[period]
    plan = source.current_iid_plans.get(period)
    if not isinstance(step, _EGMStep) or plan is None:
        raise TypeError("Case-U preparation requires its original dated EGM step/plan.")
    if call_context.period != period or call_context.ages is not source.ages:
        raise TypeError("Case-U preparation lost its actual source clock.")
    if preparation is None:
        authority = build.native_case_u_sources.get(period)
        if authority is None:
            raise TypeError(
                "Case-U point core lacks its original dated numerical authority."
            )
        authority.require(context=source, solver=authority.solver, period=period)
        original_solver = authority.solver
    else:
        preparation.validate_owner()
        _admit_scalar_owner(preparation=preparation)
        if (
            preparation.context is not source
            or preparation.build is not build
            or preparation.step is not step
            or preparation.period != period
            or preparation.plan is not plan
            or preparation.solver is not source.user_regimes[source.regime_name].solver
        ):
            raise TypeError(
                "Private Case-U point core lost its original generation owner."
            )
        original_solver = preparation.solver
    regime_name = source.regime_name
    pieces = step.pieces
    if period not in build.native_case_u_sources:
        # A private owned experiment can use a non-U installed source. Actual
        # native U already owns these exact publication pieces in its dated step.
        pieces = replace(
            pieces,
            refine_to_bracket=get_bracket_finder(
                solver=original_solver,
                n_refined=pieces.n_pad,
                publish_provenance=True,
            ),
            check_query_construction=True,
            collect_continuation_read_counts=True,
        )
    targets = frozenset(
        (
            *pieces.continuation_plan.stateful_targets,
            *pieces.continuation_plan.scalar_targets,
        )
    )
    arguments = CaseUArguments(
        base=_DCEGMArgumentBuilder(
            regime_name=regime_name,
            stateful_targets=targets,
            transition_target_names=tuple(source.transitions),
        ),
        plan=plan,
        pieces=pieces,
        period=period,
        ages=source.ages,
        fixed_flat_params=fixed_flat_params,
    )
    function = CaseUProgramFunction(
        pieces=pieces,
        plan=plan,
        period=period,
        savings_point_width=None,
        stochastic_node_width=None,
        envelope_cell_width=1,
    )
    if materializer is not None:
        captures = guard_case_u_shape(
            function=function,
            context=call_context,
            fixed_flat_params=fixed_flat_params,
            materializer=materializer,
        )
        arguments = replace(arguments, trace_capture_roots=captures)
    actual_arguments = (
        arguments(call_context)
        if materializer is None
        else arguments.prepare(context=call_context, materializer=materializer)
    )
    template = trace_case_u_shape(function=function, arguments=actual_arguments)
    if materializer is not None:
        template = materialize_case_u_schema(schema=template, materializer=materializer)
    if not isinstance(template, BoundedResult) or not isinstance(
        template.output, CurrentResourcesEvaluation
    ):
        raise TypeError(
            "Case-U abstract trace did not preserve its genuine economic output."
        )
    program = CoreProgram(
        name=CASE_U_ECONOMIC_CORE,
        function=function,
        argument_builder=arguments,
        requirements=CoreExecutionRequirements(
            value_reads=_case_u_call_reads(
                source=source, pieces=pieces, period=period, call_context=call_context
            )
        ),
        output_roles=case_u_output_roles(
            template=template, liquid_state=pieces.euler_state_name
        ),
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason=(
            "Original liquid-grid/current-node/asset solve with a complete "
            "transient economic CE bank."
        ),
        donation_candidates=(),
        scope=ProgramScope.VALUES_ONLY,
    )
    return program, template


@dataclass(frozen=True, kw_only=True, eq=False)
class PrivateCaseUWorkspace:
    """Retain the exact private probe's admitted arguments and held owners."""

    owned: OwnedCaseUProgram
    context: CoreBuildContext
    materialized: MaterializedCoreProgram
    workspace: WorkspacePlan[jax.stages.Compiled]
    inventory: ResidentInventory
    held_roots: tuple[jax.Array, ...]


def plan_private_case_u_workspace(
    *,
    owned: OwnedCaseUProgram,
    context: CoreBuildContext,
    budget_bytes: int,
) -> PrivateCaseUWorkspace:
    """Admit only this authentic private materializer probe's real held roots.

    This is deliberately not the missing Model scheduler. All actual retained
    generation/code/input roots stay live and conservatively count in full;
    compiler argument overlap is not subtracted from their census. The exact
    enclosing program is lowered without donation or surrogate callbacks.
    Concrete bank/output allocation and dispatch must wait for the returned
    plan. Caller must retain owned/context/plan for their complete lifetime.
    """
    materialized = owned.materialize(context=context)
    roots = canonical_generation_roots(
        declarations=(owned.roots, materialized.arguments)
    )
    fixed = concrete_device_bytes(tree=roots)
    device_ids = tuple(device.id for device in jax.devices())
    inventory = ResidentInventory(
        device_ids=device_ids,
        live=MappingProxyType({}),
        peer_bytes=MappingProxyType(dict.fromkeys(device_ids, 0)),
        declared_inputs=(),
        fixed_bytes=fixed,
    )

    def compile_actual() -> jax.stages.Compiled:
        return jax.jit(materialized.function).lower(**materialized.arguments).compile()

    workspace = plan_axis_free_workspace(
        compile_candidate=compile_actual,
        memory_for=lambda compiled: compiler_memory_reservation(
            compiled=compiled, widths={}
        ),
        budget_bytes=budget_bytes,
        resident_bytes=inventory.resident_bytes(consumes=()),
    )
    return PrivateCaseUWorkspace(
        owned=owned,
        context=context,
        materialized=materialized,
        workspace=workspace,
        inventory=inventory,
        held_roots=roots,
    )
