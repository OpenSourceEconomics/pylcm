"""Diagnostic-only record of the resolved simulation execution plan.

A production run's log otherwise carries no evidence of *which* forward route
actually engaged — subject-sharded or legacy, on which devices, at which
resolved widths and chunking. `SimulationPlanSummary` is built once per
`simulate()` call after chunk admission (when budgeted) has run, and logged
by `simulate()`; see `_lcm.simulation.simulate.simulate`. It reports what the
runtime already resolved and changes no numerical, RNG, ownership, or
admission decision.
"""

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType

import jax

from _lcm.engine import Regime
from _lcm.simulation.chunk_admission import PreparedSimulationChunks
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.typing import RegimeName


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationPlanSummary:
    """Frozen snapshot of one `simulate()` call's resolved execution plan.

    Diagnostic only: every field reports a decision the runtime already made
    (route selection, device binding, width resolution, chunk admission); the
    record itself moves no value, no RNG stream and admits nothing.
    """

    route: str
    """`"legacy"` or `"subjects"`, from `ResolvedExecution.simulation_sharding`."""

    subject_device_ids: tuple[int, ...]
    """Ordered ids of the devices actually evaluating subjects."""

    subject_backend: str
    """Platform of the subject devices (e.g. `"cpu"`, `"gpu"`); `""` if none."""

    axis_widths_by_regime: Mapping[RegimeName, Mapping[str, int]]
    """Resolved planner axis widths this call dispatched, by regime name."""

    outer_chunk_count: int
    """Number of outer subject chunks the (padded) population is split into."""

    admitted_chunk_widths: tuple[int, ...]
    """Outer chunk extents actually admitted, in dispatch order."""

    budget_mode: str
    """`"unbudgeted"` or `"budgeted"`."""

    effective_device_memory_bytes: int | None
    """Effective per-device budget in bytes; `None` when unbudgeted."""

    def summary(self) -> str:
        """Return one line describing the resolved plan, for the progress tier."""
        devices = (
            f"{len(self.subject_device_ids)} {self.subject_backend} "
            f"device(s) {self.subject_device_ids}"
            if self.subject_device_ids
            else "no subject devices resolved"
        )
        budget = (
            "unbudgeted"
            if self.effective_device_memory_bytes is None
            else f"budgeted at {self.effective_device_memory_bytes} bytes"
        )
        return (
            f"Simulation plan: route={self.route}; {devices}; "
            f"outer chunks={self.outer_chunk_count} "
            f"(widths={self.admitted_chunk_widths}); {budget}."
        )

    def details(self) -> str:
        """Return the complete record as one multi-field debug-tier line."""
        widths = "; ".join(
            f"{regime_name}: {dict(widths)}"
            for regime_name, widths in self.axis_widths_by_regime.items()
        )
        return (
            f"Simulation plan detail: route={self.route}; "
            f"subject_devices=({self.subject_backend}) {self.subject_device_ids}; "
            f"axis_widths_by_regime=[{widths}]; "
            f"outer_chunk_count={self.outer_chunk_count}; "
            f"admitted_chunk_widths={self.admitted_chunk_widths}; "
            f"budget_mode={self.budget_mode}; "
            f"effective_device_memory_bytes={self.effective_device_memory_bytes}."
        )


def build_simulation_plan_summary(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    subject_devices: tuple[jax.Device, ...],
    n_subjects: int,
    batch_size: int,
    prepared_chunks: PreparedSimulationChunks | None,
) -> SimulationPlanSummary:
    """Build the resolved-plan record for one `simulate()` call.

    Args:
        regimes: The regimes this call dispatched, sharing one executor.
        subject_devices: The ordered devices actually evaluating subjects, as
            resolved by the caller (independent of solve placement).
        n_subjects: The (already padded) population size this call dispatched.
        batch_size: The outer subject chunk extent this call resolved,
            before chunk admission overrides it (see `prepared_chunks`).
        prepared_chunks: The budgeted chunk-admission result, or `None` for
            an unbudgeted call.

    Returns:
        The frozen plan summary.

    """
    runtime = next(iter(regimes.values())).simulation.programs.executor
    if not isinstance(runtime, SimulationRuntime):
        return SimulationPlanSummary(
            route="legacy",
            subject_device_ids=tuple(device.id for device in subject_devices),
            subject_backend=(subject_devices[0].platform if subject_devices else ""),
            axis_widths_by_regime=MappingProxyType({}),
            outer_chunk_count=math.ceil(n_subjects / batch_size) if batch_size else 1,
            admitted_chunk_widths=(batch_size,)
            * (math.ceil(n_subjects / batch_size) if batch_size else 1),
            budget_mode="unbudgeted",
            effective_device_memory_bytes=None,
        )
    execution = runtime.execution
    chunk_override = (
        MappingProxyType({})
        if prepared_chunks is None
        else prepared_chunks.plan.profile.axis_widths
    )
    axis_widths_by_regime = MappingProxyType(
        {
            regime_name: MappingProxyType(
                {**execution.widths_for(regime_name=regime_name), **chunk_override}
            )
            for regime_name in regimes
        }
    )
    if prepared_chunks is not None:
        profile = prepared_chunks.plan.profile
        chunk_width = profile.n_subjects
        outer_chunk_count = profile.padded_population // chunk_width
    else:
        chunk_width = n_subjects if batch_size == 0 else min(batch_size, n_subjects)
        outer_chunk_count = math.ceil(n_subjects / chunk_width) if chunk_width else 1
    return SimulationPlanSummary(
        route=execution.simulation_sharding,
        subject_device_ids=tuple(device.id for device in subject_devices),
        subject_backend=subject_devices[0].platform if subject_devices else "",
        axis_widths_by_regime=axis_widths_by_regime,
        outer_chunk_count=outer_chunk_count,
        admitted_chunk_widths=(chunk_width,) * outer_chunk_count,
        budget_mode="unbudgeted"
        if execution.device_memory_bytes is None
        else "budgeted",
        effective_device_memory_bytes=execution.device_memory_bytes,
    )
