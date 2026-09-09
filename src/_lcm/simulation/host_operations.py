"""Profile exact pure forward operations against current owned device payloads.

Only module-level functions with immutable static bindings belong here. The cache
owns abstract signatures and executable code; argument arrays and live-footprint
providers belong to the calling simulation unit. No compiler options are supplied.
"""

import dataclasses
import inspect
import struct
import threading
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future
from functools import partial
from types import FunctionType, MappingProxyType

import jax

from _lcm.execution.workspace_planning import compiler_peak_bytes, plan_workspace
from _lcm.simulation.operand_placement import (
    place_simulation_arguments,
    subject_operand_sharding,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.solution.backward_induction import _lowering_key
from lcm.exceptions import ExecutionPlanningError

type StaticArgument = bool | int | float | str | tuple[StaticArgument, ...] | None


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ProfiledOperation:
    """One executable and its actual compiler-reported peak, without admission."""

    executable: jax.stages.Compiled
    peak_bytes: int


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ProfiledSimulationOperations:
    """Reuse compiled pure operations while admitting every call independently."""

    cache: dict[Hashable, _ProfiledOperation] = dataclasses.field(
        default_factory=dict, repr=False
    )
    in_flight: dict[Hashable, Future[_ProfiledOperation]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    def dispatch(
        self,
        *,
        function: Callable[..., object],
        arguments: Mapping[str, object],
        subject_arg_names: tuple[str, ...],
        devices: tuple[jax.Device, ...],
        live_footprint: Callable[[], DeviceBufferFootprint],
        budget_devices: tuple[jax.Device, ...],
        budget_bytes: int,
        static_arguments: Mapping[str, object] = MappingProxyType({}),
        subject_outputs: bool = False,
    ) -> object:
        """Place once, inspect current residency, and execute the admitted code."""
        static = _validated_static_arguments(
            function=function,
            arguments=arguments,
            static_arguments=static_arguments,
            subject_outputs=subject_outputs,
        )
        if not devices or not set(devices).issubset(budget_devices):
            raise ExecutionPlanningError(
                "The operation budget omits executing devices."
            )
        placed = place_simulation_arguments(
            arguments=MappingProxyType(dict(sorted(arguments.items()))),
            subject_arg_names=subject_arg_names,
            value_reads=(),
            devices=devices,
            budget_bytes=budget_bytes,
            live_footprint=live_footprint(),
            budget_devices=budget_devices,
        )
        argument_buffers = measure_buffer_footprint(tree=placed)
        live = union_buffer_footprints(
            footprints=(
                live_footprint(),
                measure_buffer_footprint(tree=arguments),
                argument_buffers,
            )
        )
        require_transfer_headroom(
            live=live,
            destination_bytes={},
            scratch_bytes={},
            budget_bytes=budget_bytes,
            devices=budget_devices,
        )
        external = resident_bytes_by_device(
            live=live, arguments=argument_buffers, devices=devices
        )
        abstract = jax.tree.map(_abstract_operand, dict(placed))
        key = _operation_key(
            function=function,
            arguments=abstract,
            static_arguments=static,
            subject_outputs=subject_outputs,
            devices=devices,
        )
        plan = plan_workspace(
            axes=(),
            compile_candidate=_OperationCompiler(
                owner=self,
                key=key,
                function=function,
                arguments=abstract,
                static_arguments=static,
                output_sharding=(
                    subject_operand_sharding(devices=devices)
                    if subject_outputs
                    else None
                ),
            ),
            budget_bytes=budget_bytes,
            resident_bytes=max(external.values()),
            peak_bytes_for=_operation_peak,
        )
        result = plan.compiled.executable(**placed)
        jax.block_until_ready(result)
        return result

    def prepare_abstract(
        self,
        *,
        function: Callable[..., object],
        arguments: Mapping[str, object],
        subject_arg_names: tuple[str, ...],
        devices: tuple[jax.Device, ...],
        static_arguments: Mapping[str, object] = MappingProxyType({}),
        subject_outputs: bool = False,
    ) -> _ProfiledOperation:
        """Profile already-placed shape descriptors without allocating or admitting.

        Each operand must declare its exact required layout. Concrete inputs are
        refused; shared containers are canonicalized exactly as at dispatch. Only
        executable code and its compiler peak enter the shared cache.
        """
        static = _validated_static_arguments(
            function=function,
            arguments=arguments,
            static_arguments=static_arguments,
            subject_outputs=subject_outputs,
        )
        subject = subject_operand_sharding(devices=devices)
        shared = simulation_value_sharding(stored_sharding=subject, devices=devices)
        abstract = {
            name: _abstract_operation_tree(
                tree=value,
                required=subject if name in subject_arg_names else shared,
            )
            for name, value in sorted(arguments.items())
        }
        return self.compile_candidate(
            key=_operation_key(
                function=function,
                arguments=abstract,
                static_arguments=static,
                subject_outputs=subject_outputs,
                devices=devices,
            ),
            function=function,
            arguments=abstract,
            static_arguments=static,
            output_sharding=subject if subject_outputs else None,
        )

    def compile_candidate(
        self,
        *,
        key: Hashable,
        function: Callable[..., object],
        arguments: Mapping[str, object],
        static_arguments: Mapping[str, object],
        output_sharding: jax.sharding.Sharding | None = None,
    ) -> _ProfiledOperation:
        """Compile an abstract signature once, without holding the cache lock."""
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            future = self.in_flight.get(key)
            owns_compilation = future is None
            if future is None:
                future = Future()
                self.in_flight[key] = future
        if not owns_compilation:
            return future.result()
        try:
            # Bind only validated immutable metadata. A fresh callable also keeps
            # JAX's own static-argument cache from conflating signed float zeros.
            bound = partial(function, **static_arguments)
            # Residency excludes arguments charged through the compiler's peak.
            # Keep shape-only inputs in that report while their callers own them.
            jitted = (
                jax.jit(bound, keep_unused=True)
                if output_sharding is None
                else jax.jit(bound, keep_unused=True, out_shardings=output_sharding)
            )
            executable = jitted.lower(**arguments).compile()
            compiled = _ProfiledOperation(
                executable=executable,
                peak_bytes=compiler_peak_bytes(compiled=executable, widths={}),
            )
        except BaseException as error:
            with self.lock:
                del self.in_flight[key]
                future.set_exception(error)
            raise
        with self.lock:
            self.cache[key] = compiled
            del self.in_flight[key]
            future.set_result(compiled)
        return compiled


@dataclasses.dataclass(frozen=True, kw_only=True)
class _OperationCompiler:
    """Transient planner callback holding abstract operands and immutable metadata."""

    owner: ProfiledSimulationOperations
    key: Hashable
    function: Callable[..., object]
    arguments: Mapping[str, object]
    static_arguments: Mapping[str, object]
    output_sharding: jax.sharding.Sharding | None

    def __call__(self, widths: Mapping[str, int]) -> _ProfiledOperation:
        """Compile the single, axis-free host operation."""
        if widths:
            raise ExecutionPlanningError(
                "A pure host operation declares no width axes."
            )
        return self.owner.compile_candidate(
            key=self.key,
            function=self.function,
            arguments=self.arguments,
            static_arguments=self.static_arguments,
            output_sharding=self.output_sharding,
        )


def _abstract_operand(value: object) -> object:
    """Preserve exact placed shape, weak type and ordered device layout."""
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(
            value.shape,
            value.dtype,
            sharding=value.sharding,
            weak_type=getattr(value, "weak_type", False),
        )
    return value


def _validated_static_arguments(
    *,
    function: Callable[..., object],
    arguments: Mapping[str, object],
    static_arguments: Mapping[str, object],
    subject_outputs: bool,
) -> Mapping[str, object]:
    """Use one pure-function and immutable-binding contract for both entry paths."""
    if type(subject_outputs) is not bool:
        raise ExecutionPlanningError("Subject-output metadata must be a bool.")
    original = inspect.unwrap(function)
    if (
        not isinstance(original, FunctionType)
        or original.__closure__
        or "<locals>" in original.__qualname__
    ):
        raise ExecutionPlanningError(
            "Profiled simulation operations require module-level pure functions."
        )
    for default in (
        *tuple(original.__defaults__ or ()),
        *tuple((original.__kwdefaults__ or {}).values()),
    ):
        _static_identity(default)
    if any(type(name) is not str for name in (*arguments, *static_arguments)):
        raise ExecutionPlanningError("Operation argument names must be strings.")
    static = MappingProxyType(dict(sorted(static_arguments.items())))
    for value in static.values():
        _static_identity(value)
    if arguments.keys() & static.keys():
        raise ExecutionPlanningError("Dynamic and static operation arguments overlap.")
    return static


def _operation_key(
    *,
    function: Callable[..., object],
    arguments: Mapping[str, object],
    static_arguments: Mapping[str, object],
    subject_outputs: bool,
    devices: tuple[jax.Device, ...],
) -> Hashable:
    """Identify the same abstract executable at preparation and concrete dispatch."""
    return _lowering_key(
        program_identity=function,
        arguments=arguments,
        specialization_key=tuple(
            (name, _static_identity(value)) for name, value in static_arguments.items()
        ),
        layout_key=("simulation_host_operation", subject_outputs),
        placement_key=devices,
    )


def _abstract_operation_tree(
    *, tree: object, required: jax.sharding.Sharding
) -> object:
    """Mirror placed containers while verifying every abstract leaf's layout."""
    if isinstance(tree, Mapping):
        return MappingProxyType(
            {
                name: _abstract_operation_tree(tree=value, required=required)
                for name, value in tree.items()
            }
        )
    if isinstance(tree, tuple | list):
        values = [
            _abstract_operation_tree(tree=value, required=required) for value in tree
        ]
        return tuple(values) if isinstance(tree, tuple) else values
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return dataclasses.replace(
            tree,
            **{
                field.name: _abstract_operation_tree(
                    tree=getattr(tree, field.name), required=required
                )
                for field in dataclasses.fields(tree)
                if field.init
            },
        )
    if not isinstance(tree, jax.ShapeDtypeStruct):
        raise ExecutionPlanningError(
            "Abstract operation operands must be shape descriptors."
        )
    if tree.sharding != required:
        raise ExecutionPlanningError(
            "Abstract operation operand has the wrong required sharding."
        )
    return tree


def _static_identity(value: object) -> Hashable:
    """Refuse owners and distinguish equal scalar values with different types."""
    if type(value) is float:
        return (float, struct.pack("!d", value))
    if value is None or type(value) in (bool, int, str):
        return (type(value), value)
    if type(value) is tuple:
        return (tuple, tuple(_static_identity(item) for item in value))
    raise ExecutionPlanningError(
        "Operation static arguments must be immutable scalars or tuples of scalars."
    )


def _operation_peak(compiled: _ProfiledOperation) -> int:
    """Return the actual executable report, not a guessed output multiplier."""
    return compiled.peak_bytes
