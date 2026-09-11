"""Admit uniform and equally spaced normal support with exact eager arithmetic.

Normal endpoint arithmetic retains its separate rounded stages because exact support
coordinates enter durable solution fingerprints. Gauss-Hermite and other composite
families remain outside this producer profile. Call-owned bindings, temporary arrays,
and grids never enter the code cache.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import NormalIIDProcess, UniformIIDProcess
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

type _GridStage = Literal["uniform", "multiply", "subtract", "add", "normal"]

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
    """Placed normal parameters and scalar stages held until support publication."""

    def supports(self, spec: _ContinuousStochasticProcess) -> bool:
        """Select uniform or equally spaced normal support producers."""
        return type(spec) is UniformIIDProcess or (
            type(spec) is NormalIIDProcess and not spec.gauss_hermite
        )

    @property
    def array_roots(self) -> tuple[object, ...]:
        """Expose every support and identity-binding owner to memory accounting."""
        return (
            tuple(self.grids.values()),
            tuple(binding.parameters for binding in self.bindings.values()),
            tuple(self.temporary_roots),
        )

    def __call__(
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
        spec = cast("UniformIIDProcess | NormalIIDProcess", spec)
        binding_key = (
            spec,
            required,
            tuple((name, id(value)) for name, value in sorted(parameters.items())),
        )
        if type(spec) is NormalIIDProcess:
            binding_key = (binding_key, _normal_fixed_identity(spec))
        binding = self.bindings.get(binding_key)
        if binding is not None:
            return self.grids[binding.key]
        complete = (
            _normal_parameters(spec=spec, parameters=parameters)
            if type(spec) is NormalIIDProcess
            else _uniform_parameters(
                spec=cast("UniformIIDProcess", spec), parameters=parameters
            )
        )
        key = (
            spec.n_points,
            required,
            tuple(
                (name, _parameter_bytes(value))
                for name, value in sorted(complete.items())
            ),
        )
        if type(spec) is NormalIIDProcess:
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
        if key not in self.grids:
            if self.sealed:
                raise ExecutionPlanningError(
                    "Process support changed after entry admission."
                )
            self.grids[key] = (
                self._produce_normal(
                    parameters=complete, n_points=spec.n_points, required=required
                )
                if type(spec) is NormalIIDProcess
                else self._produce(
                    parameters=complete, n_points=spec.n_points, required=required
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

    def _produce(
        self,
        *,
        parameters: Mapping[str, object],
        n_points: int,
        required: jax.sharding.Sharding,
        stage: _GridStage = "uniform",
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
        if stage != "uniform":
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
