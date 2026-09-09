"""Admit complete simulation chunks from compiled stages and owned payloads.

Profiles retain executable code and immutable size/layout metadata, never caller
arrays. Future output slots are reservations, not allocations by abstract inputs.
They may conservatively overlap a stage's reported input/output storage. Consequently
selection means widest fitting this bound, not a measured allocator optimum.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import jax

from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.simulation.residency import DeviceBufferFootprint, resident_bytes_by_device
from lcm.exceptions import ExecutionPlanningError


@dataclass(frozen=True, kw_only=True)
class SimulationStageProfile:
    """One actual compiled allocation stage on its ordered execution devices."""

    name: str
    executable: jax.stages.Compiled
    devices: tuple[jax.Device, ...]
    peak_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        """Read the compiler's raw peak separately from external reservations."""
        _validate_devices(devices=self.devices)
        compiled_devices = set().union(
            *(
                leaf.device_set
                for leaf in jax.tree.leaves(
                    (self.executable.input_shardings, self.executable.output_shardings)
                )
                if isinstance(leaf, jax.sharding.Sharding)
            )
        )
        if set(self.devices) != compiled_devices:
            raise ExecutionPlanningError(
                "A simulation chunk stage's devices differ from its compiled placement."
            )
        if not self.name:
            raise ExecutionPlanningError("A simulation chunk stage needs a name.")
        object.__setattr__(
            self,
            "peak_bytes",
            compiler_peak_bytes(compiled=self.executable, widths={}),
        )


@dataclass(frozen=True, kw_only=True)
class SimulationChunkProfile:
    """Compiled stages plus conservative future storage for one chunk extent."""

    n_subjects: int
    padded_population: int
    stages: tuple[SimulationStageProfile, ...]
    fixed_reservation: Mapping[jax.Device, int]
    output_reservation: Mapping[jax.Device, int]
    axis_widths: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    """Common inner specialization, clamped to each declared program's extent."""
    setup_reservation: Mapping[jax.Device, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Part of fixed storage fulfilled by additional padding and population metadata."""
    host_stages: tuple[SimulationStageProfile, ...] = ()
    """CPU assembly profiles reported separately from a GPU memory ceiling."""

    def __post_init__(self) -> None:
        """Freeze independently owned metadata without capturing profile inputs."""
        if (
            type(self.n_subjects) is not int
            or self.n_subjects <= 0
            or type(self.padded_population) is not int
            or self.padded_population < self.n_subjects
            or self.padded_population % self.n_subjects
        ):
            raise ExecutionPlanningError(
                "A simulation chunk must divide its positive padded population."
            )
        if not self.stages:
            raise ExecutionPlanningError("A simulation chunk needs compiled stages.")
        for name in ("fixed_reservation", "output_reservation", "setup_reservation"):
            costs = getattr(self, name)
            if any(type(value) is not int or value < 0 for value in costs.values()):
                raise ExecutionPlanningError(
                    "Simulation chunk reservations need nonnegative integer bytes."
                )
            object.__setattr__(self, name, MappingProxyType(dict(costs)))
        if any(
            value > self.fixed_reservation.get(device, 0)
            for device, value in self.setup_reservation.items()
        ):
            raise ExecutionPlanningError(
                "Chunk setup storage must be included in fixed reservations."
            )
        if any(
            type(name) is not str or not name or type(width) is not int or width <= 0
            for name, width in self.axis_widths.items()
        ):
            raise ExecutionPlanningError(
                "A chunk specialization requires named positive widths."
            )
        object.__setattr__(
            self, "axis_widths", MappingProxyType(dict(self.axis_widths))
        )


@dataclass(frozen=True, kw_only=True)
class SimulationChunkPlan:
    """Selected code and per-device required bytes; feasibility is call-local."""

    profile: SimulationChunkProfile
    required_bytes: Mapping[jax.Device, int]

    def __post_init__(self) -> None:
        """Own the accepted integer inventory rather than a mutable caller mapping."""
        object.__setattr__(
            self, "required_bytes", MappingProxyType(dict(self.required_bytes))
        )


@runtime_checkable
class ChunkProfiler(Protocol):
    """Prepare an extent using the actual numerical bodies without dispatch."""

    def __call__(self, *, n_subjects: int) -> SimulationChunkProfile:
        """Return real compiled stages and their complete future owner inventory."""
        ...


def plan_simulation_chunks(
    *,
    candidates: tuple[int, ...],
    profile_candidate: ChunkProfiler,
    live: DeviceBufferFootprint,
    budget_bytes: int,
    devices: tuple[jax.Device, ...],
) -> SimulationChunkPlan:
    """Choose the first fitting declared extent using fresh actual residency.

    No input spans are subtracted here: abstract profiles cannot establish identity
    with future runtime buffers. Each actual dispatch performs its own more precise
    compiler-input overlap accounting against the still-current owners.
    """
    _validate_devices(devices=devices)
    if type(budget_bytes) is not int or budget_bytes <= 0:
        raise ExecutionPlanningError("A simulation chunk budget must be positive.")
    if (
        not candidates
        or any(type(width) is not int or width <= 0 for width in candidates)
        or tuple(sorted(set(candidates), reverse=True)) != candidates
    ):
        raise ExecutionPlanningError(
            "Simulation chunk candidates must be distinct positive decreasing extents."
        )
    resident = resident_bytes_by_device(
        live=live, arguments=DeviceBufferFootprint(spans={}), devices=devices
    )
    if max(resident.values()) > budget_bytes:
        raise ExecutionPlanningError(
            "Simulation chunk admission fails on existing retained storage."
        )
    for width in candidates:
        profile = profile_candidate(n_subjects=width)
        if profile.n_subjects != width:
            raise ExecutionPlanningError(
                "The simulation chunk profile does not match its requested extent."
            )
        required = _required_bytes(profile=profile, resident=resident, devices=devices)
        if all(value <= budget_bytes for value in required.values()):
            return SimulationChunkPlan(profile=profile, required_bytes=required)
    raise ExecutionPlanningError(
        f"No declared simulation chunk fits the {budget_bytes}-byte device budget."
    )


def _required_bytes(
    *,
    profile: SimulationChunkProfile,
    resident: Mapping[jax.Device, int],
    devices: tuple[jax.Device, ...],
) -> Mapping[jax.Device, int]:
    """Sum owners on each actual device before comparing separate stage peaks."""
    selected = set(devices)
    if profile.host_stages and (
        any(device.platform != "gpu" for device in devices)
        or any(
            device.platform != "cpu"
            for stage in profile.host_stages
            for device in stage.devices
        )
    ):
        raise ExecutionPlanningError(
            "Only GPU chunk offload may exclude separately profiled CPU assembly."
        )
    if any(not set(stage.devices) <= selected for stage in profile.stages):
        raise ExecutionPlanningError(
            "A simulation chunk stage executes outside its budgeted devices."
        )
    if any(
        not set(costs) <= selected
        for costs in (profile.fixed_reservation, profile.output_reservation)
    ):
        raise ExecutionPlanningError(
            "A simulation chunk reservation names an unbudgeted device."
        )
    return MappingProxyType(
        {
            device: resident[device]
            + profile.fixed_reservation.get(device, 0)
            + profile.output_reservation.get(device, 0)
            + max(
                (
                    stage.peak_bytes
                    for stage in profile.stages
                    if device in stage.devices
                ),
                default=0,
            )
            for device in devices
        }
    )


def _validate_devices(*, devices: tuple[jax.Device, ...]) -> None:
    """Keep backend identities distinct and require a nonempty execution inventory."""
    if not devices or len(set(devices)) != len(devices):
        raise ExecutionPlanningError(
            "Simulation chunk admission needs distinct actual devices."
        )
