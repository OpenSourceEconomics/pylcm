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
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
    resident_bytes_by_device,
    union_buffer_footprints,
)
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
    ) -> object:
        """Place once, inspect current residency, and execute the admitted code."""
        original = inspect.unwrap(function)
        if not isinstance(original, FunctionType) or original.__closure__:
            raise ExecutionPlanningError(
                "Profiled simulation operations require module-level pure functions."
            )
        if "<locals>" in original.__qualname__:
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
        static_key = tuple(
            (name, _static_identity(value)) for name, value in static.items()
        )
        if arguments.keys() & static.keys():
            raise ExecutionPlanningError(
                "Dynamic and static operation arguments overlap."
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
        key = _lowering_key(
            program_identity=function,
            arguments=placed,
            specialization_key=static_key,
            layout_key="simulation_host_operation",
            placement_key=devices,
        )
        abstract = jax.tree.map(_abstract_operand, dict(placed))
        plan = plan_workspace(
            axes=(),
            compile_candidate=_OperationCompiler(
                owner=self,
                key=key,
                function=function,
                arguments=abstract,
                static_arguments=static,
            ),
            budget_bytes=budget_bytes,
            resident_bytes=max(external.values()),
            peak_bytes_for=_operation_peak,
        )
        result = plan.compiled.executable(**placed)
        jax.block_until_ready(result)
        return result

    def compile_candidate(
        self,
        *,
        key: Hashable,
        function: Callable[..., object],
        arguments: Mapping[str, object],
        static_arguments: Mapping[str, object],
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
            executable = jax.jit(bound).lower(**arguments).compile()
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
