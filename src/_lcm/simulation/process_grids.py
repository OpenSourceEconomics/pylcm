"""Admit runtime process supports with exact eager arithmetic.

Normal endpoint arithmetic retains its separate rounded stages because exact support
coordinates enter durable solution fingerprints. Composite supports replay their
top-level eager JAXPR stages through the same admitted operation boundary. Call-owned
bindings, temporary arrays, and grids never enter the code cache.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import (
    DropVar,
    Jaxpr,
    JaxprEqn,
    Var,
    check_jaxpr,
)
from jax.extend.core import (
    Literal as JaxprLiteral,
)
from jax.typing import DTypeLike

from _lcm.dtypes import canonical_float_dtype
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.processes.ar1 import (
    RouwenhorstAR1Process,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import (
    LogNormalIIDProcess,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    UniformIIDProcess,
)
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    _operation_memory,
    _OperationCompiler,
)
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.solution.backward_induction import _lowering_key
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import Float1D, ScalarFloat, ScalarInt, ValueND

type _GridStage = Literal[
    "uniform",
    "multiply",
    "subtract",
    "add",
    "normal",
    "mul",
    "sub",
    "div",
    "neg",
    "integer_pow",
    "sqrt",
    "exp",
    "jit",
    "convert_element_type",
]

_STAGED_PROCESS_TYPES = (
    LogNormalIIDProcess,
    NormalMixtureIIDProcess,
    TauchenAR1Process,
    RouwenhorstAR1Process,
    TauchenNormalMixtureAR1Process,
)
_PROCESS_STAGE_ARITIES = MappingProxyType(
    {
        "mul": 2,
        "sub": 2,
        "add": 2,
        "div": 2,
        "neg": 1,
        "integer_pow": 1,
        "sqrt": 1,
        "exp": 1,
        "jit": 2,
        "convert_element_type": 1,
    }
)

_UNIFORM_GRID_OPERATIONS = ProfiledSimulationOperations()


def _shared_uniform_operations() -> ProfiledSimulationOperations:
    """Share only executable code and abstract numerical signatures."""
    return _UNIFORM_GRID_OPERATIONS


@dataclass(frozen=True, kw_only=True)
class _GridBinding:
    """Retain immutable operands so an identity key cannot name a recycled object."""

    parameters: Mapping[str, ScalarFloat | ScalarInt]
    """Exact runtime operands held for this binding."""
    key: tuple[object, ...]
    """Canonical support identity, independent of equivalent parameter copies."""


@dataclass(frozen=True, kw_only=True)
class _LiteralOperand:
    """One validated scalar or vector literal embedded in a traced stage."""

    value: object


type _ProcessOperand = int | _LiteralOperand


@dataclass(frozen=True, kw_only=True)
class _ProcessStage:
    """One immutable admitted stage extracted from a validated eager graph."""

    operation: _GridStage
    operands: tuple[_ProcessOperand, ...]
    n_points: int
    exponent: int
    dtype: str
    weak_type: bool


@dataclass(frozen=True, kw_only=True)
class _ProcessRecipe:
    """Validated producer dataflow, detached from the mutable traced JAXPR."""

    initial_values: tuple[object, ...]
    stages: tuple[_ProcessStage, ...]
    result: _ProcessOperand


@dataclass(kw_only=True)
class SimulationProcessGrids:
    """Keep admitted supports and exact bindings alive within one call."""

    live_footprint: Callable[[], DeviceBufferFootprint]
    """Observe all other currently owned entry buffers."""
    devices: tuple[jax.Device, ...]
    """Selected devices covered by the per-device budget."""
    budget_bytes: int
    """Per-device compiler peak and retained payload ceiling."""
    operations: ProfiledSimulationOperations = field(
        default_factory=_shared_uniform_operations
    )
    """Shared executable profiles that retain no concrete bindings."""
    grids: dict[tuple[object, ...], Float1D] = field(default_factory=dict)
    """Ready supports held through all entry, solve and forward consumers."""
    bindings: dict[tuple[object, ...], _GridBinding] = field(default_factory=dict)
    """Exact operand owners for the synchronization-free identity lookup."""
    sealed: bool = False
    """Whether every supported producer has completed entry admission."""
    temporary_roots: list[object] = field(default_factory=list)
    """Placed operands and completed stages held until support publication."""

    def supports(self, spec: _ContinuousStochasticProcess) -> bool:
        """Select every exact built-in runtime process support producer."""
        return (
            type(spec) is UniformIIDProcess
            or (type(spec) is NormalIIDProcess and not spec.gauss_hermite)
            or (type(spec) is NormalIIDProcess and spec.gauss_hermite)
            or type(spec) in _STAGED_PROCESS_TYPES
        )

    @property
    def array_roots(self) -> tuple[object, ...]:
        """Expose every support and identity-binding owner to memory accounting."""
        return (
            tuple(self.grids.values()),
            tuple(binding.parameters for binding in self.bindings.values()),
            tuple(self.temporary_roots),
        )

    def __call__(  # noqa: C901
        self,
        *,
        spec: _ContinuousStochasticProcess,
        parameters: Mapping[str, ScalarFloat | ScalarInt],
        required: jax.sharding.Sharding,
    ) -> Float1D:
        """Admit a support once and reuse its exact coordinates for later reads.

        `parameters` contains only runtime operands. Fixed endpoint values are
        converted on the host before their first admitted device placement.
        """
        if not self.supports(spec):
            raise ExecutionPlanningError(
                "This process has no supported grid allocation profile."
            )
        binding_key = (
            spec,
            required,
            tuple((name, id(value)) for name, value in sorted(parameters.items())),
        )
        if type(spec) is NormalIIDProcess and not spec.gauss_hermite:
            binding_key = (binding_key, _normal_fixed_identity(spec))
        elif type(spec) is not UniformIIDProcess:
            binding_key = (binding_key, _process_fixed_identity(spec))
        binding = self.bindings.get(binding_key)
        if binding is not None:
            return self.grids[binding.key]
        complete = _complete_process_parameters(spec=spec, parameters=parameters)
        if type(spec) is UniformIIDProcess:
            complete = _uniform_parameters(spec=spec, parameters=parameters)
        elif type(spec) is NormalIIDProcess and not spec.gauss_hermite:
            complete = _normal_parameters(spec=spec, parameters=parameters)
        key = (
            spec.n_points,
            required,
            tuple(
                (name, _parameter_bytes(value))
                for name, value in sorted(complete.items())
            ),
        )
        if type(spec) is NormalIIDProcess and not spec.gauss_hermite:
            key = (
                NormalIIDProcess,
                key,
                tuple(
                    (
                        name,
                        type(value) in (bool, int, float)
                        or getattr(value, "weak_type", False),
                    )
                    for name, value in sorted(complete.items())
                ),
            )
        elif type(spec) is not UniformIIDProcess:
            key = (
                type(spec),
                getattr(spec, "gauss_hermite", None),
                key,
                tuple(
                    (name, _staged_parameter_is_weak(value))
                    for name, value in sorted(complete.items())
                ),
            )
        if key not in self.grids:
            if self.sealed:
                raise ExecutionPlanningError(
                    "Process support changed after entry admission."
                )
            self.grids[key] = (
                self._produce_normal(
                    parameters=complete, n_points=spec.n_points, required=required
                )
                if type(spec) is NormalIIDProcess and not spec.gauss_hermite
                else self._produce(
                    parameters=complete, n_points=spec.n_points, required=required
                )
                if type(spec) is UniformIIDProcess
                else self._produce_staged(
                    spec=spec, parameters=complete, required=required
                )
            )
        self.bindings[binding_key] = _GridBinding(
            parameters=MappingProxyType(dict(parameters)), key=key
        )
        return self.grids[key]

    def seal(self) -> None:
        """Require later consumers to reuse support admitted at entry."""
        self.sealed = True

    def close(self) -> None:
        """Release all call-owned support and identity bindings."""
        self.grids.clear()
        self.bindings.clear()
        self.temporary_roots.clear()

    def _produce_normal(
        self,
        *,
        parameters: Mapping[str, object],
        n_points: int,
        required: jax.sharding.Sharding,
    ) -> Float1D:
        """Admit the five eager normal stages with cumulative temporary ownership."""
        execution_devices = tuple(
            device for device in self.devices if device in required.device_set
        )
        current = self.snapshot()
        placed = place_simulation_arguments(
            arguments=parameters,
            subject_arg_names=(),
            value_reads=(),
            devices=execution_devices,
            budget_bytes=self.budget_bytes,
            live_footprint=current,
            budget_devices=tuple(dict.fromkeys((*self.devices, *current.spans))),
        )
        self.temporary_roots.append(placed)
        try:
            endpoints: dict[str, object] = {}
            for endpoint, stage in (("start", "subtract"), ("stop", "add")):
                offset = self._produce(
                    parameters={"left": placed["n_std"], "right": placed["sigma"]},
                    n_points=n_points,
                    required=required,
                    stage="multiply",
                )
                self.temporary_roots.append(offset)
                value = self._produce(
                    parameters={"left": placed["mu"], "right": offset},
                    n_points=n_points,
                    required=required,
                    stage=cast("_GridStage", stage),
                )
                self.temporary_roots.append(value)
                endpoints[endpoint] = value
            return self._produce(
                parameters=endpoints,
                n_points=n_points,
                required=required,
                stage="normal",
            )
        finally:
            self.temporary_roots.clear()

    def _produce_staged(
        self,
        *,
        spec: _ContinuousStochasticProcess,
        parameters: Mapping[str, object],
        required: jax.sharding.Sharding,
    ) -> Float1D:
        """Replay exact eager primitive boundaries through admitted operations."""
        parameter_names = tuple(sorted(parameters))
        parameter_values = tuple(parameters[name] for name in parameter_names)
        closed = _trace_process_jaxpr(
            spec=spec,
            parameter_names=parameter_names,
            parameter_values=parameter_values,
        )
        recipe = _validated_process_recipe(
            closed=closed,
            parameter_values=parameter_values,
            n_points=spec.n_points,
        )
        environment = list(recipe.initial_values)
        try:
            for stage in recipe.stages:
                output = self._produce(
                    parameters={
                        f"operand_{index}": _read_process_operand(
                            operand=operand, environment=environment
                        )
                        for index, operand in enumerate(stage.operands)
                    },
                    n_points=stage.n_points,
                    required=required,
                    stage=stage.operation,
                    exponent=stage.exponent,
                    dtype=stage.dtype,
                    weak_type=stage.weak_type,
                    process_stage=True,
                )
                environment.append(output)
                self.temporary_roots.append(output)
            result = _read_process_operand(
                operand=recipe.result, environment=environment
            )
            return cast("Float1D", result)
        finally:
            self.temporary_roots.clear()

    def _produce(
        self,
        *,
        parameters: Mapping[str, object],
        n_points: int,
        required: jax.sharding.Sharding,
        stage: _GridStage = "uniform",
        exponent: int = 1,
        dtype: str = "float32",
        weak_type: bool = False,
        process_stage: bool = False,
    ) -> ValueND:
        execution_devices = tuple(
            device for device in self.devices if device in required.device_set
        )
        current = self.snapshot()
        budget_devices = tuple(dict.fromkeys((*self.devices, *current.spans)))
        placed = place_simulation_arguments(
            arguments={"parameters": parameters},
            subject_arg_names=(),
            value_reads=(),
            devices=execution_devices,
            budget_bytes=self.budget_bytes,
            live_footprint=current,
            budget_devices=budget_devices,
        )
        argument_buffers = measure_buffer_footprint(tree=placed)
        live = union_buffer_footprints(footprints=(self.snapshot(), argument_buffers))
        abstract = jax.tree.map(_abstract_grid_parameter, dict(placed))
        static: Mapping[str, object] = MappingProxyType({"n_points": n_points})
        function: Callable[..., ValueND] = _compute_uniform_grid
        if process_stage:
            function = _compute_process_stage
            static = MappingProxyType(
                {
                    "n_points": n_points,
                    "stage": stage,
                    "exponent": exponent,
                    "dtype": dtype,
                    "weak_type": weak_type,
                }
            )
        elif stage in {"multiply", "subtract", "add", "normal"}:
            function = _compute_normal_stage
            static = MappingProxyType({"n_points": n_points, "stage": stage})
        compiler_key = _lowering_key(
            program_identity=function,
            arguments=abstract,
            specialization_key=tuple(static.items()),
            layout_key=("uniform_process_grid", required),
        )
        external = resident_bytes_by_device(
            live=live, arguments=argument_buffers, devices=execution_devices
        )
        plan = plan_workspace(
            axes=(),
            compile_candidate=_OperationCompiler(
                owner=self.operations,
                key=compiler_key,
                function=function,
                arguments=abstract,
                static_arguments=static,
                output_sharding=required,
            ),
            budget_bytes=self.budget_bytes,
            resident_bytes=max(external.values()),
            memory_for=_operation_memory,
        )
        return cast(
            "ValueND",
            plan.compiled.executable(**placed).block_until_ready(),
        )

    def snapshot(self) -> DeviceBufferFootprint:
        """Include every supported process owner in subsequent producer admission."""
        return union_buffer_footprints(
            footprints=(
                self.live_footprint(),
                measure_buffer_footprint(tree=self.array_roots),
            )
        )


def _uniform_parameters(
    *, spec: UniformIIDProcess, parameters: Mapping[str, ScalarFloat | ScalarInt]
) -> Mapping[str, object]:
    """Mirror process scalar dtypes on the host before admitted device placement."""
    complete: dict[str, object] = dict(parameters)
    for name in ("start", "stop"):
        value = getattr(spec, name)
        if value is not None:
            dtype = (
                np.int32 if isinstance(value, bool | int) else canonical_float_dtype()
            )
            complete[name] = np.asarray(value, dtype=dtype)
    return MappingProxyType(complete)


def _normal_parameters(
    *, spec: NormalIIDProcess, parameters: Mapping[str, ScalarFloat | ScalarInt]
) -> Mapping[str, object]:
    """Preserve weak fixed floats and canonical strong fixed integers on upload."""
    complete: dict[str, object] = dict(parameters)
    for name in ("mu", "sigma", "n_std"):
        value = getattr(spec, name)
        if value is not None:
            complete[name] = (
                np.asarray(value, dtype=np.int32)
                if isinstance(value, bool | int)
                else value
            )
    return MappingProxyType(complete)


def _normal_fixed_identity(spec: NormalIIDProcess) -> tuple[object, ...]:
    """Distinguish fixed scalar types and signed zero before binding reuse."""
    return tuple(
        (name, type(value), None if value is None else _parameter_bytes(value))
        for name in ("mu", "sigma", "n_std")
        for value in (getattr(spec, name),)
    )


def _complete_process_parameters(
    *,
    spec: _ContinuousStochasticProcess,
    parameters: Mapping[str, ScalarFloat | ScalarInt],
) -> Mapping[str, object]:
    """Complete runtime operands with fixed host scalars before device admission."""
    complete: dict[str, object] = dict(parameters)
    for name in spec._param_field_names:  # noqa: SLF001
        value = getattr(spec, name)
        if value is not None:
            complete[name] = (
                np.asarray(value, dtype=np.int32)
                if isinstance(value, bool | int)
                else value
            )
    return MappingProxyType(complete)


def _process_fixed_identity(
    spec: _ContinuousStochasticProcess,
) -> tuple[object, ...]:
    """Distinguish fixed scalar types and bytes before binding reuse."""
    return tuple(
        (name, type(value), None if value is None else _parameter_bytes(value))
        for name in spec._param_field_names  # noqa: SLF001
        for value in (getattr(spec, name),)
    )


def _staged_parameter_is_weak(value: object) -> bool:
    """Preserve host and JAX weak typing in composite support identity."""
    return type(value) in (bool, int, float) or bool(getattr(value, "weak_type", False))


def _process_grid_call(
    *values: object,
    spec: _ContinuousStochasticProcess,
    parameter_names: tuple[str, ...],
) -> Float1D:
    """Rebind positional trace values to their declared parameter names.

    Module-level so the beartype claw decorates it once at import instead of on
    every `_trace_process_jaxpr` call; see `_lcm/utils/functools.py`.
    """
    arguments = cast(
        "dict[str, ScalarFloat | ScalarInt]",
        dict(zip(parameter_names, values, strict=True)),
    )
    return spec.compute_gridpoints(**arguments)


def _trace_process_jaxpr(
    *,
    spec: _ContinuousStochasticProcess,
    parameter_names: tuple[str, ...],
    parameter_values: tuple[object, ...],
) -> Jaxpr:
    """Trace an explicit positional binding for deterministic input ordering."""
    bound = partial(_process_grid_call, spec=spec, parameter_names=parameter_names)
    return jax.make_jaxpr(bound)(*parameter_values)


def _validated_process_recipe(  # noqa: C901
    *, closed: Jaxpr, parameter_values: tuple[object, ...], n_points: int
) -> _ProcessRecipe:
    """Validate the complete eager graph before exposing any dispatch recipe."""
    try:
        check_jaxpr(closed)
    except (AssertionError, TypeError, ValueError) as error:
        raise ExecutionPlanningError(
            "A process support has an invalid JAXPR."
        ) from error
    if closed.effects or len(closed.outvars) != 1:
        raise ExecutionPlanningError(
            "A process support must be a pure graph returning one array."
        )
    if len(closed.constvars) != len(closed.consts) or len(closed.invars) != len(
        parameter_values
    ):
        raise ExecutionPlanningError("A process support has an invalid input graph.")
    if any(_aval_shape(var) != () for var in closed.invars):
        raise ExecutionPlanningError("Process parameters must remain scalar inputs.")
    if any(_aval_shape(var) not in {(), (n_points,)} for var in closed.constvars):
        raise ExecutionPlanningError("A process support has an invalid host constant.")
    if _aval_shape(closed.outvars[0]) != (n_points,):
        raise ExecutionPlanningError(
            "A process support must return its declared one-dimensional grid."
        )
    for value, variable in zip(closed.consts, closed.constvars, strict=True):
        _validate_attached_process_value(
            value=value,
            variable=variable,
            host_constant=True,
        )
    for value, variable in zip(parameter_values, closed.invars, strict=True):
        _validate_attached_process_value(
            value=value,
            variable=variable,
            host_constant=False,
        )

    initial_values = (*closed.consts, *parameter_values)
    variables = {
        variable: index
        for index, variable in enumerate((*closed.constvars, *closed.invars))
    }
    stages: list[_ProcessStage] = []
    for equation in closed.eqns:
        operation = equation.primitive.name
        expected_arity = _PROCESS_STAGE_ARITIES.get(operation)
        if (
            expected_arity is None
            or len(equation.invars) != expected_arity
            or len(equation.outvars) != 1
            or type(equation.outvars[0]) is not Var
            or equation.effects
            or _aval_shape(equation.outvars[0]) not in {(), (n_points,)}
        ):
            raise ExecutionPlanningError(
                f"Process support uses unsupported stage {operation!r}."
            )
        operands = tuple(
            _validated_process_operand(
                atom=atom, variables=variables, n_points=n_points
            )
            for atom in equation.invars
        )
        _validate_process_equation(equation=equation, n_points=n_points)
        stages.append(
            _ProcessStage(
                operation=cast("_GridStage", operation),
                operands=operands,
                n_points=(n_points if operation == "jit" else 0),
                exponent=cast("int", equation.params.get("y", 1)),
                dtype=_aval_dtype(equation.outvars[0]),
                weak_type=cast("bool", equation.params.get("weak_type", False)),
            )
        )
        variables[equation.outvars[0]] = len(initial_values) + len(stages) - 1

    result = _validated_process_operand(
        atom=closed.outvars[0], variables=variables, n_points=n_points
    )
    return _ProcessRecipe(
        initial_values=initial_values,
        stages=tuple(stages),
        result=result,
    )


def _validate_attached_process_value(
    *, value: object, variable: Var, host_constant: bool
) -> None:
    """Match each attached value to its binder before any stage can dispatch."""
    if host_constant and (
        isinstance(value, jax.Array)
        or not isinstance(value, bool | int | float | complex | np.generic | np.ndarray)
        or np.asarray(value).dtype.kind not in "biufc"
    ):
        raise ExecutionPlanningError(
            "Process support captured a non-host numerical constant."
        )
    # Re-read this concrete value's contract at every attachment. An identity
    # graph adds no information beyond its existing JAX abstract value.
    observed = jax.typeof(value)
    if (
        getattr(observed, "shape", None),
        str(getattr(observed, "dtype", None)),
        bool(getattr(observed, "weak_type", False)),
    ) != _aval_schema(variable):
        raise ExecutionPlanningError(
            "A process support value does not match its traced input contract."
        )


def _validated_process_operand(
    *, atom: Var | JaxprLiteral, variables: Mapping[Var, int], n_points: int
) -> _ProcessOperand:
    """Resolve one already-declared variable or validated embedded literal."""
    if isinstance(atom, JaxprLiteral):
        if _aval_shape(atom) not in {(), (n_points,)}:
            raise ExecutionPlanningError(
                "Process support contains an invalid embedded constant."
            )
        return _LiteralOperand(value=atom.val)
    if atom not in variables:
        raise ExecutionPlanningError("Process support reads an undeclared value.")
    return variables[atom]


def _validate_process_equation(  # noqa: C901
    *, equation: JaxprEqn, n_points: int
) -> None:
    """Check every semantic parameter consumed by the stage interpreter."""
    operation = equation.primitive.name
    nested = [value for value in equation.params.values() if isinstance(value, Jaxpr)]
    if operation == "integer_pow":
        exponent = equation.params.get("y")
        if set(equation.params) != {"y"} or type(exponent) is not int:
            raise ExecutionPlanningError("Process integer power has no fixed exponent.")
    elif operation == "convert_element_type":
        weak_type = equation.params.get("weak_type")
        output_dtype = _aval_dtype(equation.outvars[0])
        if (
            set(equation.params) != {"new_dtype", "weak_type", "sharding"}
            or str(equation.params.get("new_dtype")) != output_dtype
            or type(weak_type) is not bool
            or weak_type != getattr(equation.outvars[0].aval, "weak_type", None)
            or equation.params.get("sharding") is not None
            or _aval_shape(equation.outvars[0]) != ()
            or (
                weak_type
                and (
                    not np.issubdtype(
                        np.dtype(_aval_dtype(equation.invars[0])), np.integer
                    )
                    or output_dtype != str(np.dtype(canonical_float_dtype()))
                )
            )
        ):
            raise ExecutionPlanningError(
                "Process scalar conversion has unsupported semantics."
            )
    elif operation == "jit":
        if equation.params.get("name") != "_linspace" or len(nested) != 1:
            raise ExecutionPlanningError(
                "Process support uses an unsupported nested JIT stage."
            )
        _validate_linspace_jaxpr(
            equation=equation,
            n_points=n_points,
        )
    elif nested:
        raise ExecutionPlanningError("Process arithmetic contains a nested graph.")
    elif operation == "mul":
        if (
            set(equation.params) - {"out_dtype"}
            or equation.params.get("out_dtype") is not None
        ):
            raise ExecutionPlanningError(
                "Process multiplication requests an output dtype."
            )
    elif operation in {"sqrt", "exp"}:
        if (
            set(equation.params) - {"accuracy"}
            or equation.params.get("accuracy") is not None
        ):
            raise ExecutionPlanningError(
                f"Process {operation} stage requests unsupported accuracy."
            )
    elif equation.params:
        raise ExecutionPlanningError(
            f"Process {operation} stage has unsupported static parameters."
        )


def _validate_linspace_jaxpr(*, equation: JaxprEqn, n_points: int) -> None:
    """Match the nested graph to the eager linspace used by the stage executor."""
    abstract = tuple(
        jax.ShapeDtypeStruct(
            shape=_aval_shape(atom),
            dtype=cast("DTypeLike", getattr(atom.aval, "dtype", None)),
            weak_type=getattr(atom.aval, "weak_type", False),
        )
        for atom in equation.invars
    )
    reference = jax.make_jaxpr(
        lambda lower, upper: jnp.linspace(lower, upper, n_points)
    )(*abstract)
    if len(reference.eqns) != 1 or _root_equation_schema(
        equation
    ) != _root_equation_schema(reference.eqns[0]):
        raise ExecutionPlanningError(
            "Process support uses an unsupported nested linspace graph."
        )


def _jaxpr_schema(jaxpr: Jaxpr) -> tuple[object, ...]:
    """Describe graph dataflow canonically, independent of variable spellings."""
    variables = {
        variable: index
        for index, variable in enumerate((*jaxpr.constvars, *jaxpr.invars))
    }
    equations: list[tuple[object, ...]] = []
    for equation in jaxpr.eqns:
        for variable in equation.outvars:
            if isinstance(variable, Var):
                variables[variable] = len(variables)
        equations.append(_equation_schema(equation=equation, variables=variables))
    return (
        tuple(_aval_schema(var) for var in jaxpr.constvars),
        tuple(_aval_schema(var) for var in jaxpr.invars),
        tuple(sorted(map(repr, jaxpr.effects))),
        tuple(equations),
        tuple(
            _graph_atom_schema(atom=atom, variables=variables) for atom in jaxpr.outvars
        ),
    )


def _root_equation_schema(equation: JaxprEqn) -> tuple[object, ...]:
    """Canonicalize one outer equation and every nested graph it owns."""
    variables: dict[Var, int] = {}
    for atom in equation.invars:
        if isinstance(atom, Var) and atom not in variables:
            variables[atom] = len(variables)
    for variable in equation.outvars:
        if isinstance(variable, Var):
            variables[variable] = len(variables)
    return _equation_schema(equation=equation, variables=variables)


def _equation_schema(
    *, equation: JaxprEqn, variables: Mapping[Var, int]
) -> tuple[object, ...]:
    """Capture one primitive's complete abstract and static contract."""
    return (
        equation.primitive.name,
        tuple(
            _graph_atom_schema(atom=atom, variables=variables)
            for atom in equation.invars
        ),
        tuple(
            _graph_atom_schema(atom=atom, variables=variables)
            for atom in equation.outvars
        ),
        tuple(sorted(map(repr, equation.effects))),
        tuple(
            sorted(
                (
                    name,
                    _jaxpr_schema(value) if isinstance(value, Jaxpr) else repr(value),
                )
                for name, value in equation.params.items()
            )
        ),
    )


def _graph_atom_schema(
    *, atom: Var | DropVar | JaxprLiteral, variables: Mapping[Var, int]
) -> tuple[object, ...]:
    """Name graph variables by canonical binder index and literals by value."""
    if isinstance(atom, JaxprLiteral):
        return "literal", _aval_schema(atom), _parameter_bytes(atom.val)
    if isinstance(atom, DropVar):
        return "drop", _aval_schema(atom)
    if atom not in variables:
        raise ExecutionPlanningError("Process nested graph reads an undeclared value.")
    return "variable", variables[atom], _aval_schema(atom)


def _aval_schema(
    atom: Var | DropVar | JaxprLiteral,
) -> tuple[tuple[int, ...], str, bool]:
    """Return the static array contract attached to one JAXPR atom."""
    return (
        _aval_shape(atom),
        _aval_dtype(atom),
        bool(getattr(atom.aval, "weak_type", False)),
    )


def _aval_shape(atom: Var | DropVar | JaxprLiteral) -> tuple[int, ...]:
    """Require a statically shaped array atom."""
    shape = getattr(atom.aval, "shape", None)
    if shape is None or any(type(dimension) is not int for dimension in shape):
        raise ExecutionPlanningError("Process support has a dynamic array shape.")
    return tuple(shape)


def _aval_dtype(atom: Var | DropVar | JaxprLiteral) -> str:
    """Require an ordinary numerical dtype on one array atom."""
    dtype = getattr(atom.aval, "dtype", None)
    if dtype is None:
        raise ExecutionPlanningError("Process support has a non-array value.")
    return str(dtype)


def _read_process_operand(
    *, operand: _ProcessOperand, environment: list[object]
) -> object:
    """Read one validated environment slot or embedded literal."""
    return (
        operand.value if isinstance(operand, _LiteralOperand) else environment[operand]
    )


def _parameter_bytes(value: object) -> tuple[str, tuple[int, ...], bytes]:
    """Read canonical support content once per previously unseen operand binding."""
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes()


def _abstract_grid_parameter(value: object) -> object:
    """Retain exact placed shape and weak type without a concrete array owner."""
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=value.sharding, weak_type=value.weak_type
        )
    return value


def _compute_uniform_grid(
    *, parameters: Mapping[str, ScalarFloat | ScalarInt], n_points: int
) -> Float1D:
    """Compute equally spaced support with the Uniform process law."""
    return UniformIIDProcess(n_points=n_points).compute_gridpoints(**parameters)


def _compute_normal_stage(
    *,
    parameters: Mapping[str, ScalarFloat | ScalarInt],
    n_points: int,
    stage: _GridStage,
) -> ValueND:
    """Keep each eager rounding boundary in its own exact-layout executable."""
    if stage == "multiply":
        return parameters["left"] * parameters["right"]
    if stage == "subtract":
        return parameters["left"] - parameters["right"]
    if stage == "add":
        return parameters["left"] + parameters["right"]
    if stage == "normal":
        return jnp.linspace(
            start=parameters["start"], stop=parameters["stop"], num=n_points
        )
    raise ExecutionPlanningError("Unknown normal support stage.")


def _compute_process_stage(  # noqa: C901, PLR0911
    *,
    parameters: Mapping[str, ValueND],
    n_points: int,
    stage: _GridStage,
    exponent: int,
    dtype: str,
    weak_type: bool,
) -> ValueND:
    """Execute one top-level eager process primitive without cross-stage fusion."""
    operands = tuple(parameters[f"operand_{index}"] for index in range(len(parameters)))
    if stage in {"mul", "sub", "add", "div"}:
        left, right = operands
        if stage == "mul":
            return left * right
        if stage == "sub":
            return left - right
        if stage == "add":
            return left + right
        return left / right
    value = operands[0]
    if stage == "neg":
        return -value
    if stage == "integer_pow":
        return value**exponent
    if stage == "sqrt":
        return jnp.sqrt(value)
    if stage == "exp":
        return jnp.exp(value)
    if stage == "jit":
        return jnp.linspace(start=value, stop=operands[1], num=n_points)
    if stage == "convert_element_type":
        return value * 1.0 if weak_type else jnp.asarray(value, dtype=jnp.dtype(dtype))
    raise ExecutionPlanningError(f"Unknown process support stage {stage!r}.")
