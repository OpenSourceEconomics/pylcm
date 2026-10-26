"""Bounded recipes of the original validator's separately rounded JAX stages.

Only the listed genuine primitive objects are executable. An original nested JIT
remains one opaque dispatch stage. Code descriptors contain no device-array
owners; closed numerical constants stay in the call-owned recipe environment.
"""

import inspect
import types
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax._src import config as jax_config
from jax._src import pjit, traceback_util
from jax._src.api import Inline
from jax._src.core import JaxprEqnContext
from jax._src.lax.lax import unstack_p
from jax._src.mesh import AbstractMesh
from jax._src.named_sharding import UnspecifiedValue
from jax.extend.core import (
    DropVar,
    Jaxpr,
    JaxprEqn,
    Literal,
    Primitive,
    Var,
    check_jaxpr,
)
from jax.extend.core import primitives as primitive_objects
from jax.lax import GatherDimensionNumbers, GatherScatterMode, ScatterDimensionNumbers

from _lcm.grids import DiscreteGrid
from _lcm.simulation.process_grids import (
    ProcessValue,
    _aval_schema,
    _ProcessOperand,
    _root_equation_schema,
    _validate_attached_process_value,
)
from _lcm.solution.fingerprint import (
    _JAX_PJIT_FUNCTION_TYPE,
    _is_captured_jax_pjit,
    _native_numeric_callable_kind,
    _Reachable,
    _SemanticHasher,
)
from _lcm.typing import PytreeValue
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ReferenceName

type _Static = (
    bool
    | int
    | float
    | str
    | np.generic
    | np.dtype[np.generic]
    | GatherScatterMode
    | Inline
    | GatherDimensionNumbers
    | ScatterDimensionNumbers
    | UnspecifiedValue
    | AbstractMesh
    | Jaxpr
    | tuple[_Static, ...]
    | None
)


# None means a fixed equation-specific arity checked by check_jaxpr, not a
# value-dependent number of inputs. No name-based fallback is permitted.
_PRIMITIVE_ARITIES: Mapping[Primitive, int | None] = MappingProxyType(
    {
        primitive_objects.add_p: 2,
        primitive_objects.sub_p: 2,
        primitive_objects.mul_p: 2,
        primitive_objects.div_p: 2,
        primitive_objects.neg_p: 1,
        primitive_objects.abs_p: 1,
        primitive_objects.log_p: 1,
        primitive_objects.exp_p: 1,
        primitive_objects.sqrt_p: 1,
        primitive_objects.nextafter_p: 2,
        primitive_objects.integer_pow_p: 1,
        primitive_objects.eq_p: 2,
        primitive_objects.ne_p: 2,
        primitive_objects.lt_p: 2,
        primitive_objects.le_p: 2,
        primitive_objects.gt_p: 2,
        primitive_objects.ge_p: 2,
        primitive_objects.max_p: 2,
        primitive_objects.min_p: 2,
        primitive_objects.and_p: 2,
        primitive_objects.or_p: 2,
        primitive_objects.not_p: 1,
        primitive_objects.select_n_p: None,
        primitive_objects.convert_element_type_p: 1,
        primitive_objects.broadcast_in_dim_p: 1,
        primitive_objects.reshape_p: 1,
        primitive_objects.squeeze_p: 1,
        unstack_p: 1,
        primitive_objects.slice_p: 1,
        primitive_objects.dynamic_slice_p: None,
        primitive_objects.gather_p: 2,
        primitive_objects.iota_p: 0,
        primitive_objects.scatter_add_p: 3,
        primitive_objects.concatenate_p: None,
        primitive_objects.reduce_max_p: 1,
        primitive_objects.reduce_sum_p: 1,
        primitive_objects.reduce_or_p: 1,
        primitive_objects.is_finite_p: 1,
        primitive_objects.argmax_p: 1,
        primitive_objects.sort_p: None,
        primitive_objects.jit_p: None,
    }
)


_UNSAFE_TRACE_CONTEXTS = (
    jax.ensure_compile_time_eval,
    inspect.unwrap(jax.ensure_compile_time_eval),
    jax.core.eval_context,
    inspect.unwrap(jax.core.eval_context),
)


def _capture_jax_code(
    *, factory: _Reachable, nested_name: str
) -> tuple[types.FunctionType, types.CodeType, types.CodeType]:
    """Capture the genuine factory and unique nested code, never a filename seal."""
    if not isinstance(factory, types.FunctionType):
        raise ExecutionPlanningError("Original JIT factory is not inspectable.")
    codes = tuple(
        item
        for item in factory.__code__.co_consts
        if isinstance(item, types.CodeType) and item.co_name == nested_name
    )
    if len(codes) != 1:
        raise ExecutionPlanningError("Original JIT callback code is not represented.")
    return factory, factory.__code__, codes[0]


def _capture_cache_miss_property() -> property:
    descriptor = inspect.getattr_static(_JAX_PJIT_FUNCTION_TYPE, "_cache_miss", None)
    if (
        not isinstance(descriptor, property)
        or type(descriptor) is not property
        or descriptor.fget is None
        or descriptor.fset is not None
        or descriptor.fdel is not None
    ):
        raise ExecutionPlanningError("Original JIT callback getter is not readonly.")
    return descriptor


_BOUNDARY_FACTORY, _BOUNDARY_FACTORY_CODE, _BOUNDARY_CODE = _capture_jax_code(
    factory=traceback_util.api_boundary,
    nested_name="reraise_with_filtered_traceback",
)
_CPP_PJIT_FACTORY, _CPP_PJIT_FACTORY_CODE, _CACHE_MISS_CODE = _capture_jax_code(
    factory=vars(pjit).get("_cpp_pjit"),
    nested_name="cache_miss",
)
_CACHE_MISS_PROPERTY = _capture_cache_miss_property()
_PJIT_INFO_TYPE = pjit.PjitInfo
# Single elementary primitives with only dtype/shape promotion; other EARLY APIs refuse.
_CAPTURED_ELEMENTARY_LOG = jnp.log
_CAPTURED_ELEMENTARY_NEXTAFTER = jnp.nextafter
_CAPTURED_ELEMENTARY_ISNAN = jnp.isnan


def _original_closure_value(*, function: types.FunctionType, name: str) -> _Reachable:
    cells = function.__closure__ or ()
    if len(cells) != len(function.__code__.co_freevars):
        raise ExecutionPlanningError("Original JIT closure is malformed.")
    for cell_name, cell in zip(function.__code__.co_freevars, cells, strict=True):
        if cell_name == name:
            try:
                return cell.cell_contents
            except ValueError as error:
                raise ExecutionPlanningError(
                    "Original JIT closure is empty."
                ) from error
    raise ExecutionPlanningError("Original JIT closure binding is not represented.")


def _require_original_jit_policy(
    *, value: _Reachable, preserve_eager_boundary: bool = True
) -> None:
    """Read actual cache-miss closure authority before tracing a referenced JIT."""
    if (
        traceback_util.api_boundary is not _BOUNDARY_FACTORY
        or _BOUNDARY_FACTORY.__code__ is not _BOUNDARY_FACTORY_CODE
        or vars(pjit).get("_cpp_pjit") is not _CPP_PJIT_FACTORY
        or _CPP_PJIT_FACTORY.__code__ is not _CPP_PJIT_FACTORY_CODE
        or pjit.PjitInfo is not _PJIT_INFO_TYPE
        or inspect.getattr_static(type(value), "_cache_miss", None)
        is not _CACHE_MISS_PROPERTY
    ):
        raise ExecutionPlanningError("Original JIT origin bindings changed.")
    # Reading the captured native property does not invoke its returned callback.
    outer = _CACHE_MISS_PROPERTY.__get__(value, type(value))
    if (
        not isinstance(outer, types.FunctionType)
        or outer.__code__ is not _BOUNDARY_CODE
    ):
        raise ExecutionPlanningError(
            "Original JIT boundary callback is not represented."
        )
    inner = _original_closure_value(function=outer, name="fun")
    if (
        not isinstance(inner, types.FunctionType)
        or inner.__code__ is not _CACHE_MISS_CODE
    ):
        raise ExecutionPlanningError(
            "Original JIT cache-miss callback is not represented."
        )
    original_fun = _original_closure_value(function=inner, name="fun")
    original_info = _original_closure_value(function=inner, name="jit_info")
    if (
        not isinstance(original_fun, types.FunctionType)
        or not isinstance(original_info, pjit.PjitInfo)
        or type(original_info) is not _PJIT_INFO_TYPE
        or vars(value).get("_fun") is not original_fun
        or vars(value).get("_jit_info") is not original_info
    ):
        raise ExecutionPlanningError("Original JIT function/metadata bindings changed.")
    if type(original_info.inline) is not Inline:
        raise ExecutionPlanningError("Original JIT inline policy is not represented.")
    if original_info.inline is Inline.JAX_EARLY:
        if not preserve_eager_boundary:
            return
        if (
            (
                value is _CAPTURED_ELEMENTARY_LOG
                or value is _CAPTURED_ELEMENTARY_NEXTAFTER
                or value is _CAPTURED_ELEMENTARY_ISNAN
            )
            and _is_captured_jax_pjit(value)
            and jax_config.error_checking_behavior_nan.value == "ignore"
        ):
            return
        raise ExecutionPlanningError(
            "Validator JAX_EARLY JIT would erase its original eager boundary "
            "before tracing."
        )


class ValidationCaptureHasher(_SemanticHasher):
    """Reuse the genuine binding walk while refusing eager trace escape hatches."""

    _preserve_eager_boundary = True

    def visit(
        self, *, value: _Reachable, _ignore_beartype_guards: bool = False
    ) -> None:
        if any(value is forbidden for forbidden in _UNSAFE_TRACE_CONTEXTS):
            raise ExecutionPlanningError(
                "Validator compile-time eager evaluation requires admission "
                "before tracing and remains unsupported."
            )
        if (
            isinstance(value, DiscreteGrid)
            and value.values is not None
            and value.materialized_values is None
        ):
            raise ExecutionPlanningError(
                "Referenced validator grid needs admitted precision-cache construction."
            )
        if _native_numeric_callable_kind(value) == "jax-pjit":
            _require_original_jit_policy(
                value=value, preserve_eager_boundary=self._preserve_eager_boundary
            )
        super().visit(value=value, _ignore_beartype_guards=_ignore_beartype_guards)


@dataclass(frozen=True, kw_only=True, eq=False)
class ValidationStage:
    """Array-free code and original context for one eager dispatch boundary."""

    primitive: Primitive
    parameters: Mapping[ReferenceName, _Static]
    context: JaxprEqnContext
    identity: Hashable
    """Canonical dataflow/parameter/context identity, never argument values."""

    def __call__(
        self, *, parameters: Mapping[ReferenceName, jax.Array]
    ) -> tuple[jax.Array, ...]:
        operands = tuple(parameters[f"operand_{i}"] for i in range(len(parameters)))
        with self.context.manager:
            output = self.primitive.bind(*operands, **self.parameters)
        return tuple(output) if self.primitive.multiple_results else (output,)


@dataclass(frozen=True, kw_only=True)
class ValidationRecipe:
    """One actual callback graph with numerical owners outside cached code."""

    initial_values: tuple[ProcessValue, ...]
    stages: tuple[
        tuple[ValidationStage, tuple[_ProcessOperand, ...], tuple[Var | DropVar, ...]],
        ...,
    ]
    output_operands: tuple[_ProcessOperand, ...]
    output_tree: jax.tree_util.PyTreeDef
    output_contracts: tuple[tuple[tuple[int, ...], str, bool], ...]


def _dimension_indices(values: Sequence[int]) -> tuple[int, ...]:
    """Detach an original exact integer tuple without accepting mutable metadata."""
    if type(values) is not tuple or any(type(value) is not int for value in values):
        raise ExecutionPlanningError(
            "Validator dimension indices need exact integer tuples."
        )
    return tuple(values)


def _static_value(value: _Static) -> _Static:
    """Detach only genuine immutable static parameters, refusing array captures."""
    if value is None:
        return None
    if isinstance(value, bool | int | float | str) and type(value) in (
        bool,
        int,
        float,
        str,
    ):
        return value
    if isinstance(
        value, np.generic | np.dtype | GatherScatterMode | Inline | UnspecifiedValue
    ):
        return value
    if isinstance(value, GatherDimensionNumbers):
        return GatherDimensionNumbers(
            offset_dims=_dimension_indices(value.offset_dims),
            collapsed_slice_dims=_dimension_indices(value.collapsed_slice_dims),
            start_index_map=_dimension_indices(value.start_index_map),
            operand_batching_dims=_dimension_indices(value.operand_batching_dims),
            start_indices_batching_dims=_dimension_indices(
                value.start_indices_batching_dims
            ),
        )
    if isinstance(value, ScatterDimensionNumbers):
        return ScatterDimensionNumbers(
            update_window_dims=_dimension_indices(value.update_window_dims),
            inserted_window_dims=_dimension_indices(value.inserted_window_dims),
            scatter_dims_to_operand_dims=_dimension_indices(
                value.scatter_dims_to_operand_dims
            ),
            operand_batching_dims=_dimension_indices(value.operand_batching_dims),
            scatter_indices_batching_dims=_dimension_indices(
                value.scatter_indices_batching_dims
            ),
        )
    if isinstance(value, tuple):
        items = value
        if type(value) is tuple:
            return tuple(_static_value(item) for item in items)
    raise ExecutionPlanningError(
        "Validator stage has unsupported or numerical static parameters."
    )


def _nested_parameter(value: _Static) -> _Static:
    if isinstance(value, Jaxpr):
        return _check_nested_graph(graph=value)
    if isinstance(value, GatherDimensionNumbers | ScatterDimensionNumbers):
        return _static_value(value)
    if isinstance(value, tuple):
        items = value
        if type(value) is tuple:
            return tuple(_nested_parameter(item) for item in items)
    return _static_value(value)


def _checked_context(context: JaxprEqnContext) -> JaxprEqnContext:
    if not context.cur_abstract_mesh.empty:
        raise ExecutionPlanningError(
            "Validator stage has an unsupported nonempty mesh context."
        )
    for value in (context.xla_metadata or {}).values():
        _static_value(value)
    return context


def _check_nested_atom(*, atom: Var | DropVar | Literal) -> None:
    _aval_schema(atom)
    if isinstance(atom, Literal) and (
        isinstance(atom.val, jax.Array)
        or (isinstance(atom.val, np.ndarray) and atom.val.ndim != 0)
    ):
        raise ExecutionPlanningError(
            "Validator nested JIT hides an array-valued literal owner."
        )


def _snapshot_nested_literal(
    *, atom: Var | DropVar | Literal
) -> Var | DropVar | Literal:
    if isinstance(atom, Literal) and isinstance(atom.val, np.ndarray):
        value = atom.val.copy()
        value.flags.writeable = False
        return Literal(value, atom.aval)
    return atom


def _checked_jit_parameters(*, equation: JaxprEqn) -> dict[ReferenceName, _Static]:
    expected = {
        "jaxpr",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "name",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
    if set(equation.params) != expected:
        raise ExecutionPlanningError(
            "Validator JIT has an unsupported static contract."
        )
    if (
        any(equation.params["donated_invars"])
        or any(
            value is not None
            for name in ("in_layouts", "out_layouts")
            for value in equation.params[name]
        )
        or any(
            not isinstance(value, UnspecifiedValue)
            for name in ("in_shardings", "out_shardings")
            for value in equation.params[name]
        )
        or not equation.params["ctx_mesh"].empty
        or equation.params["compiler_options_kvs"]
    ):
        raise ExecutionPlanningError(
            "Validator JIT donation, layout, sharding or compiler options "
            "need their original execution contract."
        )
    return {
        name: (
            _check_nested_graph(graph=value)
            if name == "jaxpr"
            else value
            if name == "ctx_mesh"
            else _static_value(value)
        )
        for name, value in equation.params.items()
    }


def _check_nested_graph(*, graph: Jaxpr) -> Jaxpr:
    """Keep an original opaque JIT, refusing hidden owners and unsafe metadata."""
    if graph.consts or graph.constvars or graph.effects:
        raise ExecutionPlanningError(
            "Validator nested JIT captures numerical constants or effects; "
            "its exact ownership class still requires support."
        )
    check_jaxpr(graph)
    for atom in (*graph.invars, *graph.outvars):
        _check_nested_atom(atom=atom)
    equations = []
    for equation in graph.eqns:
        if (
            equation.effects
            or equation.primitive.ref_primitive
            or equation.primitive.ref_allocating
        ):
            raise ExecutionPlanningError(
                "Validator nested JIT contains effects or references."
            )
        _checked_context(equation.ctx)
        parameters = (
            _checked_jit_parameters(equation=equation)
            if equation.primitive is primitive_objects.jit_p
            else {
                name: _nested_parameter(value)
                for name, value in equation.params.items()
            }
        )
        for atom in (*equation.invars, *equation.outvars):
            _check_nested_atom(atom=atom)
        invars = [_snapshot_nested_literal(atom=atom) for atom in equation.invars]
        equations.append(
            equation.replace(
                invars=invars,
                outvars=list(equation.outvars),
                params=parameters,
            )
        )
    return graph.replace(
        invars=list(graph.invars),
        outvars=[_snapshot_nested_literal(atom=atom) for atom in graph.outvars],
        eqns=tuple(equations),
    )


def _stage(*, equation: JaxprEqn) -> ValidationStage:
    primitive = equation.primitive
    if primitive not in _PRIMITIVE_ARITIES or equation.effects:
        raise ExecutionPlanningError(
            f"Validator uses unsupported eager primitive {primitive.name!r}."
        )
    arity = _PRIMITIVE_ARITIES[primitive]
    if arity is not None and len(equation.invars) != arity:
        raise ExecutionPlanningError("Validator primitive has an unsupported arity.")
    if not equation.outvars or any(
        not isinstance(v, Var | DropVar) for v in equation.outvars
    ):
        raise ExecutionPlanningError("Validator stage has unsupported output bindings.")
    for atom in (*equation.invars, *equation.outvars):
        shape, dtype, _ = _aval_schema(atom)
        if (
            any(type(n) is not int or n < 0 for n in shape)
            or np.dtype(dtype).kind not in "biufc"
        ):
            raise ExecutionPlanningError(
                "Validator stage needs static ordinary numeric arrays."
            )
    context = _checked_context(equation.ctx)
    parameters = (
        _checked_jit_parameters(equation=equation)
        if primitive is primitive_objects.jit_p
        else {name: _nested_parameter(value) for name, value in equation.params.items()}
    )
    return ValidationStage(
        primitive=primitive,
        parameters=MappingProxyType(parameters),
        context=context,
        identity=(
            "original_validator_stage",
            _root_equation_schema(
                equation.replace(invars=[Var(atom.aval) for atom in equation.invars])
            ),
            repr(context),
        ),
    )


def _call_validation(
    *values: ProcessValue,
    function: Callable[..., PytreeValue],
    input_tree: jax.tree_util.PyTreeDef,
) -> PytreeValue:
    return function(**input_tree.unflatten(values))


def _validation_operand(
    *,
    atom: Var | Literal,
    variables: Mapping[Var, int],
    literal_values: list[ProcessValue],
    literal_base: int,
) -> _ProcessOperand:
    if isinstance(atom, Literal):
        actual = jax.typeof(atom.val)
        if (actual.shape, str(actual.dtype), actual.weak_type) != _aval_schema(
            atom
        ) or np.dtype(actual.dtype).kind not in "biufc":
            raise ExecutionPlanningError(
                "Validator literal changed its original dtype, shape or weak type."
            )
        literal_values.append(atom.val)
        return literal_base + len(literal_values) - 1
    if atom not in variables:
        raise ExecutionPlanningError("Validator graph reads an undeclared binding.")
    return variables[atom]


def _literal_bindings(
    *,
    graph: Jaxpr,
    initial_values: tuple[ProcessValue, ...],
    variables: Mapping[Var, int],
) -> tuple[tuple[ProcessValue, ...], dict[int, int]]:
    """Keep original literal values in dynamic ownership, outside cached code."""
    literals: list[ProcessValue] = []
    literal_base = len(initial_values)
    operands: dict[int, int] = {}
    atoms = (
        *(atom for equation in graph.eqns for atom in equation.invars),
        *graph.outvars,
    )
    for atom in atoms:
        if isinstance(atom, Literal) and id(atom) not in operands:
            operands[id(atom)] = cast(
                "int",
                _validation_operand(
                    atom=atom,
                    variables=variables,
                    literal_values=literals,
                    literal_base=literal_base,
                ),
            )
    return (*initial_values, *literals), operands


def prepare_validation_recipe(
    *,
    function: Callable[..., PytreeValue],
    arguments: Mapping[ReferenceName, PytreeValue],
) -> ValidationRecipe:
    """Validate the entire source graph before any stage can dispatch."""
    values, tree = jax.tree.flatten(dict(arguments))
    descriptors = tuple(
        jax.ShapeDtypeStruct(kind.shape, kind.dtype, weak_type=kind.weak_type)
        for kind in map(jax.typeof, values)
    )
    closed, output_shape = jax.make_jaxpr(
        partial(_call_validation, function=function, input_tree=tree),
        return_shape=True,
    )(*descriptors)
    check_jaxpr(closed)
    if closed.effects or len(closed.invars) != len(values) or not closed.outvars:
        raise ExecutionPlanningError(
            "Validator callback must return pure numerical arrays."
        )
    initial_values = (*closed.consts, *values)
    variables = {
        variable: i for i, variable in enumerate((*closed.constvars, *closed.invars))
    }
    for value, variable in zip(
        initial_values, (*closed.constvars, *closed.invars), strict=True
    ):
        _validate_attached_process_value(
            value=value, variable=variable, host_constant=False
        )
    initial_values, literal_operands = _literal_bindings(
        graph=closed, initial_values=initial_values, variables=variables
    )

    def read(atom: Var | Literal) -> _ProcessOperand:
        if isinstance(atom, Literal):
            return literal_operands[id(atom)]
        return variables[atom]

    stages = []
    next_output = len(initial_values)
    for equation in closed.eqns:
        stage = _stage(equation=equation)
        operands = tuple(read(atom) for atom in equation.invars)
        for index, variable in enumerate(equation.outvars):
            if isinstance(variable, Var):
                variables[variable] = next_output + index
        next_output += len(equation.outvars)
        stages.append((stage, operands, tuple(equation.outvars)))
    return ValidationRecipe(
        initial_values=initial_values,
        stages=tuple(stages),
        output_operands=tuple(read(atom) for atom in closed.outvars),
        output_tree=jax.tree.structure(output_shape),
        output_contracts=tuple(_aval_schema(atom) for atom in closed.outvars),
    )


def validation_identity(
    *, parameters: Mapping[ReferenceName, jax.Array]
) -> tuple[jax.Array, ...]:
    """Place literal/identity callback outputs through the same admitted compiler."""
    return tuple(parameters[f"operand_{i}"] for i in range(len(parameters)))
