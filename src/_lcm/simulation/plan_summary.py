"""Diagnostic-only record of the resolved simulation execution plan.

A production run's log otherwise carries no evidence of *which* forward route
actually engaged — agent-sharded or legacy, on which devices, at which
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
from _lcm.simulation.agent_groups import AgentGroupPlan
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
    """`"legacy"` or `"agents"`, from `ResolvedExecution.simulation_sharding`."""

    agent_device_ids: tuple[int, ...]
    """Ordered ids of the devices actually evaluating agents."""

    agent_backend: str
    """Platform of the agent devices (e.g. `"cpu"`, `"gpu"`); `""` if none."""

    axis_widths_by_regime: Mapping[RegimeName, Mapping[str, int]]
    """Resolved planner axis widths this call dispatched, by regime name."""

    outer_chunk_count: int
    """Number of outer agent chunks the (padded) population is split into."""

    admitted_chunk_widths: tuple[int, ...]
    """Outer chunk extents actually admitted, in dispatch order."""

    budget_mode: str
    """`"unbudgeted"` or `"budgeted"`."""

    effective_device_memory_bytes: int | None
    """Effective per-device budget in bytes; `None` when unbudgeted."""

    agent_grouping: str | None = None
    """The invariant state agents were grouped by, or `None` when ungrouped."""

    def summary(self) -> str:
        """Return one line describing the resolved plan, for the progress tier."""
        devices = (
            f"{len(self.agent_device_ids)} {self.agent_backend} "
            f"device(s) {self.agent_device_ids}"
            if self.agent_device_ids
            else "no agent devices resolved"
        )
        budget = (
            "unbudgeted"
            if self.effective_device_memory_bytes is None
            else f"budgeted at {self.effective_device_memory_bytes} bytes"
        )
        grouping = (
            ""
            if self.agent_grouping is None
            else f" grouped by {self.agent_grouping!r}"
        )
        return (
            f"Simulation plan: route={self.route}; {devices}; "
            f"outer chunks={self.outer_chunk_count}{grouping} "
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
            f"agent_devices=({self.agent_backend}) {self.agent_device_ids}; "
            f"axis_widths_by_regime=[{widths}]; "
            f"outer_chunk_count={self.outer_chunk_count}; "
            f"admitted_chunk_widths={self.admitted_chunk_widths}; "
            f"budget_mode={self.budget_mode}; "
            f"effective_device_memory_bytes={self.effective_device_memory_bytes}; "
            f"agent_grouping={self.agent_grouping}."
        )


def build_simulation_plan_summary(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    agent_devices: tuple[jax.Device, ...],
    n_agents: int,
    batch_size: int,
    prepared_chunks: PreparedSimulationChunks | None,
    group_plan: AgentGroupPlan | None = None,
) -> SimulationPlanSummary:
    """Build the resolved-plan record for one `simulate()` call.

    Args:
        regimes: The regimes this call dispatched, sharing one executor.
        agent_devices: The ordered devices actually evaluating agents, as
            resolved by the caller (independent of solve placement).
        n_agents: The (already padded) population size this call dispatched.
        batch_size: The outer agent chunk extent this call resolved,
            before chunk admission overrides it (see `prepared_chunks`).
        prepared_chunks: The budgeted chunk-admission result, or `None` for
            an unbudgeted call.
        group_plan: The grouped chunks this call dispatched, or `None` when
            agents are not grouped.

    Returns:
        The frozen plan summary.

    """
    runtime = next(iter(regimes.values())).simulation.programs.executor
    if not isinstance(runtime, SimulationRuntime):
        return SimulationPlanSummary(
            route="legacy",
            agent_device_ids=tuple(device.id for device in agent_devices),
            agent_backend=(agent_devices[0].platform if agent_devices else ""),
            axis_widths_by_regime=MappingProxyType({}),
            outer_chunk_count=math.ceil(n_agents / batch_size) if batch_size else 1,
            admitted_chunk_widths=(batch_size,)
            * (math.ceil(n_agents / batch_size) if batch_size else 1),
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
        chunk_width = profile.n_agents
        outer_chunk_count = profile.padded_population // chunk_width
    else:
        chunk_width = n_agents if batch_size == 0 else min(batch_size, n_agents)
        outer_chunk_count = math.ceil(n_agents / chunk_width) if chunk_width else 1
    if group_plan is not None:
        # A plan of selected codes no agent holds dispatches no chunk.
        chunk_width = (
            len(group_plan.chunks[0].rows) if group_plan.chunks else chunk_width
        )
        outer_chunk_count = len(group_plan.chunks)
    grouping = next(iter(regimes.values())).simulation.programs.grouping
    return SimulationPlanSummary(
        route=execution.simulation_sharding,
        agent_device_ids=tuple(device.id for device in agent_devices),
        agent_backend=agent_devices[0].platform if agent_devices else "",
        axis_widths_by_regime=axis_widths_by_regime,
        outer_chunk_count=outer_chunk_count,
        admitted_chunk_widths=(chunk_width,) * outer_chunk_count,
        budget_mode="unbudgeted"
        if execution.device_memory_bytes is None
        else "budgeted",
        effective_device_memory_bytes=execution.device_memory_bytes,
        agent_grouping=None if grouping is None else grouping.state_name,
    )
