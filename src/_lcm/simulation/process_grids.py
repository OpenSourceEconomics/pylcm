"""Admit UniformIIDProcess support while preserving composite process arithmetic.

Only uniform grids enter this producer profile. Composite process families use eager
numerical stages because exact support coordinates enter durable solution
fingerprints. Call-owned bindings and grids never enter the code cache.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import jax
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import UniformIIDProcess
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    _operation_peak,
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
from lcm.typing import Float1D, ScalarFloat, ScalarInt

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


@dataclass(kw_only=True)
class SimulationProcessGrids:
    """Keep admitted uniform supports and exact bindings alive within one call."""

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

    def supports(self, spec: _ContinuousStochasticProcess) -> bool:
        """Select the built-in producer whose numerical law is one linspace call."""
        return type(spec) is UniformIIDProcess

    @property
    def array_roots(self) -> tuple[object, ...]:
        """Expose every support and identity-binding owner to memory accounting."""
        return (
            tuple(self.grids.values()),
            tuple(binding.parameters for binding in self.bindings.values()),
        )

    def __call__(
        self,
        *,
        spec: _ContinuousStochasticProcess,
        parameters: Mapping[str, ScalarFloat | ScalarInt],
        required: jax.sharding.Sharding,
    ) -> Float1D:
        """Admit a uniform grid once and reuse its exact support for later reads.

        `parameters` contains only runtime operands. Fixed endpoint values are
        converted on the host before their first admitted device placement.
        """
        if not self.supports(spec):
            raise ExecutionPlanningError(
                "This process has no uniform-grid allocation profile."
            )
        spec = cast("UniformIIDProcess", spec)
        binding_key = (
            spec,
            required,
            tuple((name, id(value)) for name, value in sorted(parameters.items())),
        )
        binding = self.bindings.get(binding_key)
        if binding is not None:
            return self.grids[binding.key]
        complete = _uniform_parameters(spec=spec, parameters=parameters)
        key = (
            spec.n_points,
            required,
            tuple(
                (name, _parameter_bytes(value))
                for name, value in sorted(complete.items())
            ),
        )
        if key not in self.grids:
            if self.sealed:
                raise ExecutionPlanningError(
                    "Uniform process support changed after entry admission."
                )
            self.grids[key] = self._produce(
                parameters=complete, n_points=spec.n_points, required=required
            )
        self.bindings[binding_key] = _GridBinding(
            parameters=MappingProxyType(dict(parameters)), key=key
        )
        return self.grids[key]

    def seal(self) -> None:
        """Require later consumers to reuse uniform support admitted at entry."""
        self.sealed = True

    def close(self) -> None:
        """Release all call-owned support and identity bindings."""
        self.grids.clear()
        self.bindings.clear()

    def _produce(
        self,
        *,
        parameters: Mapping[str, object],
        n_points: int,
        required: jax.sharding.Sharding,
    ) -> Float1D:
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
        static = MappingProxyType({"n_points": n_points})
        compiler_key = _lowering_key(
            program_identity=_compute_uniform_grid,
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
                function=_compute_uniform_grid,
                arguments=abstract,
                static_arguments=static,
                output_sharding=required,
            ),
            budget_bytes=self.budget_bytes,
            resident_bytes=max(external.values()),
            peak_bytes_for=_operation_peak,
        )
        return cast("Float1D", plan.compiled.executable(**placed).block_until_ready())

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
