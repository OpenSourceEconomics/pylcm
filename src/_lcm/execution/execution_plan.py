"""Resolve a public `ExecutionConfig` against what a model declares.

The devices a model may use are read here and nowhere else in the package: a
model resolves them once when it is built and every phase reads the resolved
ids, so two phases of one model can never disagree about the hardware they run
on.
"""

import dataclasses
import json
import logging
import math
import operator
import os
import re
from collections.abc import Collection, Iterable, Mapping
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import jax

from _lcm.execution.core_program import CoreProgram
from _lcm.execution.value_transfer import TransferCost, TransferOperationClass
from _lcm.typing import JSONValue, RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import (
    AxisWidth,
    ExecutionConfig,
    InvariantBlockSchedule,
    WidthSearchPolicy,
)

logger = logging.getLogger(__name__)

_BUDGET_REMEDIES = (
    "To fit, lower `ExecutionConfig.device_memory_headroom_fraction` or raise "
    "`ExecutionConfig.device_memory_bytes`, cap widths with `axis_width_ceilings` "
    "or `axis_widths`, or shard over more devices with `sharded_states` or "
    "`devices`. `device_memory_bytes=None` disables admission: every axis then "
    "compiles at its bootstrap width, which is slower and may exhaust device "
    "memory."
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedExecution:
    """Hardware-local facts every phase of one model reads."""

    device_ids: tuple[int, ...]
    """Visible device ids the model uses, ascending."""

    sharded_states: frozenset[StateName]
    """States carrying a device axis."""

    axis_widths: MappingProxyType[str, int]
    """Fixed planner widths by axis name, for every regime declaring the axis."""

    axis_widths_by_regime: MappingProxyType[RegimeName, MappingProxyType[str, int]] = (
        MappingProxyType({})
    )
    """Widths that override the model-wide ones, by regime name then axis name."""

    axis_width_ceilings: MappingProxyType[str, int] = MappingProxyType({})
    """Widest block the planner may compile each named axis at; empty means none."""

    covered_axes: frozenset[str] = frozenset()
    """Axis names whose conservative seed covers the whole extent; solve only."""

    device_memory_bytes: int | None
    """Effective per-device workspace budget every phase admits against, or `None`.

    The requested budget reduced to what the selected devices' allocator pools
    leave once their headroom is kept free.
    """

    requested_device_memory_bytes: int | None = None
    """Budget the configuration asked for, before the device headroom.

    The smallest selected pool limit when the budget is derived from the
    devices. `None` on a resolution built without a request, so an unbudgeted
    model and one whose request was never recorded read alike.
    """

    budget_source: Literal["explicit", "device", "none"] = "none"
    """Where the effective budget came from.

    - `"explicit"`: the configuration passed an integer; routes that cannot be
      budgeted refuse it.
    - `"device"`: the default, derived from the selected devices' pool limits;
      routes that cannot be budgeted refuse it, as they refuse an explicit one.
    - `"none"`: no budget applies.
    """

    device_memory_headroom_fraction: float = 0.15
    """Share of each device's pool the effective budget keeps free."""

    device_pool_limit_bytes: MappingProxyType[int, int | None] = MappingProxyType({})
    """Allocator pool limit of each selected device, `None` where unreported.

    Empty on an unbudgeted model: the resolver keeps no limit when no budget
    applies, although construction still reads the visible pool statistics once.
    """

    simulation_sharding: Literal["legacy", "subjects"] = "legacy"
    """Forward placement and local-loop policy, separate from solve-state axes."""

    continuous_sharded_state: StateName | None = None
    """Internal capability set only after Model validates continuous GridSearch."""

    donate_buffers: bool = True
    """Whether eligible solve inputs may be donated to a compiled executable."""

    halve_on_materialised_gather: bool = False
    """Whether GridSearch cell widths are halved while a gather materialises."""

    width_search: WidthSearchPolicy = dataclasses.field(
        default_factory=WidthSearchPolicy
    )
    """Which width search a budgeted solve's compilation waves are driven by."""

    invariant_block_widths: MappingProxyType[StateName, int] = MappingProxyType({})
    """Invariant states solved one block of codes at a time, with the block width."""

    invariant_block_schedule: InvariantBlockSchedule = (
        InvariantBlockSchedule.PERIOD_MAJOR
    )
    """Whether a blocked solve runs period by period or code by code."""

    action_partitions: MappingProxyType[RegimeName, int] = MappingProxyType({})
    """Regimes whose action product is shared by several devices, with the count."""

    axis_width_ceilings_by_regime: MappingProxyType[
        RegimeName, MappingProxyType[str, int]
    ] = MappingProxyType({})
    """Ceilings one regime's programs are planned under on top of the model-wide
    ones, by regime name then axis name; the tighter of the two binds."""

    def action_partitions_for(self, *, regime_name: RegimeName) -> int:
        """Return how many devices share one regime's action product; one by default."""
        return self.action_partitions.get(regime_name, 1)

    def ceilings_for(self, *, regime_name: RegimeName) -> MappingProxyType[str, int]:
        """Return the width ceilings one regime's programs are planned under.

        Args:
            regime_name: The regime whose programs are being planned.

        Returns:
            The model-wide ceilings, each lowered to this regime's own ceiling
            where it declares a tighter one.

        """
        own = self.axis_width_ceilings_by_regime.get(regime_name)
        if not own:
            return self.axis_width_ceilings
        merged = dict(self.axis_width_ceilings)
        for axis_name, ceiling in own.items():
            merged[axis_name] = min(ceiling, merged.get(axis_name, ceiling))
        return MappingProxyType(merged)

    def widths_for(self, *, regime_name: RegimeName) -> MappingProxyType[str, int]:
        """Return the fixed widths one regime's programs are planned against.

        Args:
            regime_name: The regime whose programs are being planned.

        Returns:
            The model-wide widths, with this regime's overrides applied.

        """
        override = self.axis_widths_by_regime.get(regime_name)
        if not override:
            return self.axis_widths
        return MappingProxyType({**self.axis_widths, **override})

    def device_memory_budget_summary(self) -> str:
        """Return one line stating how the effective budget was arrived at.

        Names where the budget came from — the request, or the device default —
        the headroom fraction, each selected device's pool limit with the bytes
        that fraction keeps free, the effective budget, and whether the devices
        capped an explicit request.
        """
        if self.device_memory_bytes is None:
            return "Device-memory budget: none requested; admission is unbudgeted."
        requested = self.requested_device_memory_bytes
        fraction = self.device_memory_headroom_fraction
        limits = "; ".join(
            f"device {device_id}: "
            + (
                "no reported limit"
                if limit is None
                else f"limit {limit} bytes, "
                f"headroom {_headroom_bytes(limit=limit, fraction=fraction)} bytes"
            )
            for device_id, limit in self.device_pool_limit_bytes.items()
        )
        if self.budget_source == "device":
            origin = "derived from the device pool limit (default)"
            verdict = ""
        else:
            origin = (
                f"requested "
                f"{self.device_memory_bytes if requested is None else requested} bytes"
            )
            verdict = (
                "; capped by device headroom"
                if self._device_headroom_capped_the_request()
                else "; no cap applied"
            )
        return (
            f"Device-memory budget: {origin}; headroom fraction {fraction}; "
            f"per-device pool limits: {limits or 'none consulted'}; "
            f"effective {self.device_memory_bytes} bytes{verdict}."
        )

    def device_memory_cap_note(self) -> str:
        """Return the clause a budget refusal appends: the ceiling and the remedies.

        Names the effective budget and where it came from — the request, the
        request capped by the device headroom, or the device default — and then
        what a caller can change to fit. Empty without a budget. Leading space
        included, so a caller appends it to a sentence.
        """
        if self.device_memory_bytes is None:
            return ""
        fraction = self.device_memory_headroom_fraction
        if self.budget_source == "device":
            origin = (
                f"The budget is {self.device_memory_bytes} bytes, derived from the "
                "device pool limit (default): the smallest selected pool, "
                f"{self.requested_device_memory_bytes} bytes, less a device-memory "
                f"headroom fraction of {fraction}."
            )
        elif self._device_headroom_capped_the_request():
            origin = (
                f"The effective budget is {self.device_memory_bytes} bytes of the "
                f"requested {self.requested_device_memory_bytes} bytes after a "
                f"device-memory headroom fraction of {fraction}."
            )
        else:
            origin = f"The budget is the requested {self.device_memory_bytes} bytes."
        return f" {origin} {_BUDGET_REMEDIES}"

    def _device_headroom_capped_the_request(self) -> bool:
        """Report whether the selected devices reduced the requested budget."""
        return (
            self.requested_device_memory_bytes is not None
            and self.device_memory_bytes is not None
            and self.device_memory_bytes < self.requested_device_memory_bytes
        )


def _split_axis_widths(
    *, axis_widths: Mapping[str, AxisWidth]
) -> tuple[
    MappingProxyType[str, int], MappingProxyType[RegimeName, MappingProxyType[str, int]]
]:
    """Split one declaration into model-wide widths and per-regime overrides.

    Args:
        axis_widths: The user's declaration, already validated by `ExecutionConfig`.

    Returns:
        The widths that hold for every regime, and the per-regime overrides
        keyed by regime name so a planner serving one regime reads one mapping.

    """
    model_wide: dict[str, int] = {}
    by_regime: dict[RegimeName, dict[str, int]] = {}
    for axis_name, width in axis_widths.items():
        if isinstance(width, Mapping):
            for regime_name, regime_width in width.items():
                by_regime.setdefault(regime_name, {})[axis_name] = regime_width
        else:
            model_wide[axis_name] = width
    return (
        MappingProxyType(model_wide),
        MappingProxyType(
            {name: MappingProxyType(widths) for name, widths in by_regime.items()}
        ),
    )


def resolve_execution_config(
    *,
    config: ExecutionConfig,
    visible_device_ids: tuple[int, ...],
    state_names: frozenset[StateName],
    regime_names: frozenset[RegimeName] = frozenset(),
    device_pool_limit_bytes: Mapping[int, int | None] = MappingProxyType({}),
) -> ResolvedExecution:
    """Check a configuration against the model and freeze it.

    The axis names are not checked here: they are legal exactly when a core
    program declares them, and the programs do not exist until the regimes are
    built. `fail_if_axis_widths_name_undeclared_axes` is that gate, and runs
    once the programs are in hand.

    Args:
        config: The user's configuration.
        visible_device_ids: Ids of the devices JAX reports at model build.
        state_names: Every state name any regime declares.
        regime_names: Every regime name the model declares, which a per-regime
            axis width may name. Empty admits no per-regime width.
        device_pool_limit_bytes: Allocator pool limit by visible device id,
            `None` where the backend reports none. A device absent from the
            mapping contributes no cap, so a caller with no limits in hand
            leaves the requested budget untouched.

    Returns:
        The resolved facts, with the requested device-memory budget reduced to
        what the selected devices' pools leave once their headroom is free.

    Raises:
        ExecutionPlanningError: A state, regime or device the model cannot serve,
            or a device default over a pool that is not preallocated.

    """
    for name in config.sharded_states:
        if name not in state_names:
            msg = (
                f"ExecutionConfig.sharded_states names {name!r}, which no regime "
                f"declares as a state; declared states are {sorted(state_names)!r}."
            )
            raise ExecutionPlanningError(msg)
    model_wide_widths, widths_by_regime = _split_axis_widths(
        axis_widths=config.axis_widths
    )
    _fail_if_a_width_names_an_unknown_regime(
        widths_by_regime=widths_by_regime, regime_names=regime_names
    )
    device_ids = visible_device_ids if config.devices is None else config.devices
    for device_id in device_ids:
        if device_id not in visible_device_ids:
            msg = (
                f"ExecutionConfig.devices: device id {device_id} is not visible; "
                f"visible ids are {visible_device_ids!r}."
            )
            raise ExecutionPlanningError(msg)
    selected_ids = tuple(sorted(device_ids))
    selected_limits = (
        MappingProxyType({})
        if config.device_memory_bytes is None
        else MappingProxyType(
            {
                device_id: device_pool_limit_bytes.get(device_id)
                for device_id in selected_ids
            }
        )
    )
    requested_bytes, budget_source = _requested_budget(
        requested=config.device_memory_bytes, pool_limits=selected_limits
    )
    if budget_source == "device":
        _fail_if_the_pool_grows_in_regions()
    if requested_bytes is None:
        selected_limits = MappingProxyType({})
    resolved = ResolvedExecution(
        device_ids=selected_ids,
        sharded_states=frozenset(config.sharded_states),
        axis_widths=model_wide_widths,
        axis_widths_by_regime=widths_by_regime,
        axis_width_ceilings=MappingProxyType(dict(config.axis_width_ceilings)),
        covered_axes=frozenset(config.covered_axes),
        device_memory_bytes=_effective_device_memory_bytes(
            requested_bytes=requested_bytes,
            headroom_fraction=config.device_memory_headroom_fraction,
            pool_limits=selected_limits,
        ),
        requested_device_memory_bytes=requested_bytes,
        budget_source=budget_source,
        device_memory_headroom_fraction=config.device_memory_headroom_fraction,
        device_pool_limit_bytes=selected_limits,
        donate_buffers=config.donate_buffers,
        halve_on_materialised_gather=config.halve_on_materialised_gather,
        width_search=config.width_search,
        simulation_sharding=config.simulation_sharding,
        invariant_block_widths=MappingProxyType(dict(config.invariant_block_widths)),
        invariant_block_schedule=config.invariant_block_schedule,
        action_partitions=MappingProxyType(dict(config.action_partitions)),
    )
    if requested_bytes is not None:
        summary = resolved.device_memory_budget_summary()
        if (
            budget_source == "explicit"
            and resolved.device_memory_bytes != requested_bytes
        ):
            logger.warning("%s", summary)
        else:
            logger.info("%s", summary)
    return resolved


def _requested_budget(
    *,
    requested: int | Literal["device"] | None,
    pool_limits: Mapping[int, int | None],
) -> tuple[int | None, Literal["explicit", "device", "none"]]:
    """Return the budget request before the headroom, and where it came from.

    The device default requests the smallest reported pool limit, so the
    headroom applied afterwards is the only reduction; with no reported limit
    there is nothing to derive a budget from and the model runs unbudgeted.
    """
    if requested is None:
        return None, "none"
    if requested != "device":
        return requested, "explicit"
    reported = [limit for limit in pool_limits.values() if limit is not None]
    if not reported:
        return None, "none"
    return min(reported), "device"


def _fail_if_the_pool_grows_in_regions() -> None:
    """Refuse the device default when the allocator pool is not preallocated.

    Without preallocation the BFC allocator grows its pool in separate regions,
    so free bytes below the pool limit need not form one contiguous block and a
    single buffer the budget admits can still fail to allocate. The async
    allocator, selected with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, does not
    grow that way; JAX does not read `TF_GPU_ALLOCATOR`.
    """
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower()
    if preallocate not in {"false", "0"}:
        return
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "").lower() == "cuda_async":
        return
    msg = (
        "ExecutionConfig.device_memory_bytes='device' derives the budget from the "
        "device pool limit, but XLA_PYTHON_CLIENT_PREALLOCATE is "
        f"{os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']!r}: the allocator then "
        "grows its pool in separate regions, so the limit does not promise one "
        "contiguous block. Set XLA_PYTHON_CLIENT_PREALLOCATE=true or "
        "XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async before JAX initialises, or pass an "
        "explicit ExecutionConfig(device_memory_bytes=...) (or "
        "device_memory_bytes=None to disable admission)."
    )
    raise ExecutionPlanningError(msg)


def _effective_device_memory_bytes(
    *,
    requested_bytes: int | None,
    headroom_fraction: float,
    pool_limits: Mapping[int, int | None],
) -> int | None:
    """Return the ceiling admission checks against, given what the devices hold.

    The minimum of the request and every selected device's pool limit less its
    headroom, so an already-conservative request is never reduced a second time
    and a device reporting no limit contributes no cap.

    Args:
        requested_bytes: The configured budget, or `None` for unbudgeted.
        headroom_fraction: Share of a pool limit kept out of the ceiling.
        pool_limits: Allocator pool limit of each selected device.

    Returns:
        The effective budget, at least one byte, or `None` when unbudgeted.

    """
    if requested_bytes is None:
        return None
    caps = [
        limit - _headroom_bytes(limit=limit, fraction=headroom_fraction)
        for limit in pool_limits.values()
        if limit is not None
    ]
    if not caps:
        return requested_bytes
    # A pool small enough that its headroom consumes all of it still leaves a
    # positive ceiling: admission refuses every width there, which is the
    # honest outcome, whereas a non-positive budget is not a budget at all.
    return max(min(requested_bytes, *caps), 1)


def _headroom_bytes(*, limit: int, fraction: float) -> int:
    """Return the bytes one pool limit keeps free, rounded up."""
    return math.ceil(fraction * limit)


def _fail_if_a_width_names_an_unknown_regime(
    *,
    widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
    regime_names: frozenset[RegimeName],
) -> None:
    """Reject a per-regime axis width for a regime the model does not declare."""
    for regime_name, widths in widths_by_regime.items():
        if regime_name in regime_names:
            continue
        axis_name = min(widths)
        msg = (
            f"ExecutionConfig.axis_widths[{axis_name!r}] names regime "
            f"{regime_name!r}, which the model does not declare; declared regimes "
            f"are {sorted(regime_names)!r}."
        )
        raise ExecutionPlanningError(msg)


def fail_if_per_regime_widths_name_non_solve_axes(
    *,
    axis_widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
    solve_programs: Iterable[CoreProgram],
) -> None:
    """Reject a per-regime width for an axis the solve phase does not declare.

    The solve planner is the one that plans per regime, so it is the only
    consumer a per-regime override can reach. An axis only simulation programs
    declare would silently keep its planned width, so it is refused instead.

    Args:
        axis_widths_by_regime: The per-regime overrides, by regime then axis.
        solve_programs: Every core program the solve phase declares.

    Raises:
        ExecutionPlanningError: An override names an axis no solve program declares.

    """
    if not axis_widths_by_regime:
        return
    declared = frozenset(
        name for program in solve_programs for name in program.requirements.axis_names
    )
    for regime_name, widths in axis_widths_by_regime.items():
        for axis_name in sorted(widths):
            if axis_name not in declared:
                msg = (
                    f"ExecutionConfig.axis_widths[{axis_name!r}] fixes a width for "
                    f"regime {regime_name!r}, but no solve program declares that "
                    "axis; only the solve phase plans per regime, so this axis "
                    "takes a single width for the whole model."
                )
                raise ExecutionPlanningError(msg)


def fail_if_axis_widths_name_undeclared_axes(
    *,
    axis_widths: Collection[str],
    program_collections: tuple[Iterable[CoreProgram], ...],
    label: str = "axis_widths",
) -> None:
    """Reject an axis width for a name none of the model's programs declares.

    The legal set is derived, never listed: it is the union of the axis names
    over every program in every collection, so a solver that declares a new axis
    is configurable the moment it declares it. Each phase contributes one
    collection, which is why the collections arrive as a tuple rather than
    already merged.

    Args:
        axis_widths: The axis names the user declared, as width keys or a tuple.
        program_collections: One collection of core programs per phase whose
            axes the widths may name.
        label: The `ExecutionConfig` field the declaration came from, named in
            the error a rejected axis raises.

    Raises:
        ExecutionPlanningError: A width names an axis no program declares.

    """
    if not axis_widths:
        return
    declared = frozenset(
        name
        for programs in program_collections
        for program in programs
        for name in program.requirements.axis_names
    )
    for name in axis_widths:
        if name not in declared:
            msg = (
                f"ExecutionConfig.{label} names {name!r}, which no core program "
                f"declares; declared axes are {sorted(declared)!r}."
            )
            raise ExecutionPlanningError(msg)


def visible_devices() -> tuple[jax.Device, ...]:
    """Return every device JAX reports, ascending by id."""
    return tuple(sorted(jax.devices(), key=operator.attrgetter("id")))


def visible_device_ids() -> tuple[int, ...]:
    """Return the ids of every device JAX reports, ascending."""
    return tuple(device.id for device in visible_devices())


@runtime_checkable
class SupportsMemoryStats(Protocol):
    """A device that carries an id and may report allocator counters."""

    @property
    def id(self) -> int:
        """Return the device id."""

    def memory_stats(self) -> Mapping[str, int] | None:
        """Return the backend's allocator counters, or `None`."""


def visible_device_pool_limits(
    *, devices: Iterable[SupportsMemoryStats] | None = None
) -> MappingProxyType[int, int | None]:
    """Return each device's allocator pool limit in bytes, by device id.

    The query is tolerant because reporting is backend-specific: a device whose
    backend returns no counters, omits `bytes_limit`, reports a limit of 0 (an
    on-demand pool such as the async allocator's), or fails the query
    contributes `None`, which places no cap on a requested budget.

    Args:
        devices: The devices to query; every device JAX reports when omitted.

    Returns:
        The pool limit of each device by id, `None` where unreported.

    """
    queried = visible_devices() if devices is None else tuple(devices)
    return MappingProxyType(
        {device.id: _pool_limit_bytes(device=device) for device in queried}
    )


def _pool_limit_bytes(*, device: SupportsMemoryStats) -> int | None:
    """Return one device's reported pool limit, or `None` if it reports none."""
    try:
        stats = device.memory_stats()
    except Exception:  # noqa: BLE001 - backends differ in what a query may raise
        return None
    if stats is None:
        return None
    limit = stats.get("bytes_limit")
    return limit if isinstance(limit, int) and limit > 0 else None


def execution_over_visible_devices() -> ResolvedExecution:
    """Return the inert configuration resolved against every visible device.

    The resolution a caller that builds canonical regimes outside a `Model` —
    a test, or a tool inspecting one regime — runs under.
    """
    return resolve_execution_config(
        config=ExecutionConfig(device_memory_bytes=None),
        visible_device_ids=visible_device_ids(),
        state_names=frozenset(),
        device_pool_limit_bytes=visible_device_pool_limits(),
    )


type AxisDispatch = Literal["dense", "streamed"]

# A plan record field as `to_json` reads it: plain data, tuples, and mappings keyed
# by names or device ids.
type _RecordField = (
    bool
    | int
    | float
    | str
    | tuple[_RecordField, ...]
    | Mapping[str, _RecordField]
    | Mapping[int, _RecordField]
    | None
)

# Optimized-HLO opcodes that move data between devices, with their asynchronous
# start halves; the matching `-done` halves are not counted a second time.
_HLO_COLLECTIVE = re.compile(
    r"[\s)](?P<opcode>all-gather|all-reduce|all-to-all|reduce-scatter|"
    r"collective-permute|collective-broadcast)(?:-start)?\("
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CorePlanRecord:
    """What width selection decided for one solve core, as metadata only.

    Built from planning results that already exist when the widths are chosen:
    no array is read and no device is synchronized. Byte counts are per device
    unless named logical. The logical bytes a planned transfer gathers are a
    planning quantity; the collectives the compiled program contains are what
    the compiler emitted. Neither is the communication a device performed,
    which only a profiler observes. An interpolation `gather` inside a kernel is
    a local read and is not counted as a collective.
    """

    regime: RegimeName
    """Regime whose core this is."""

    period: int
    """Period the core solves."""

    core: str
    """Name of the core within the regime's period program."""

    state_extents: MappingProxyType[StateName, int]
    """Extent of each named state axis of the published value, in axis order.

    Empty when the value's leading axes cannot be matched to state names.
    """

    selected_block: MappingProxyType[StateName, tuple[int, int]] | None
    """Half-open coordinate interval evaluated per blocked state, or `None` when
    every state is evaluated over its whole extent."""

    logical_value_shape: tuple[int, ...]
    """Shape of the whole value the core publishes; one block's for a bound core."""

    physical_value_shape: tuple[int, ...]
    """Shape of the value shard one device holds."""

    device_ids: tuple[int, ...]
    """Ascending ids of the devices the value is laid out on."""

    axis_extents: MappingProxyType[str, int]
    """Points in each planner axis: the action product or the state cells."""

    widths: MappingProxyType[str, int]
    """Selected width of each planner axis."""

    dispatch: MappingProxyType[str, AxisDispatch]
    """`dense` where the width covers the whole axis, `streamed` otherwise."""

    stored_owner_bytes: MappingProxyType[int, int]
    """Solve-lifetime concrete owners charged on each device."""

    resident_bytes: int | None
    """Bytes admission found already resident at this core, or `None` unbudgeted."""

    active_replica_bytes: int
    """Bytes the core's non-local value transfers place on each device."""

    logical_gathered_bytes: int
    """Whole-value bytes of the core's collective value transfers."""

    transfer_workspace_bytes: int
    """Operator scratch the core's value transfers hold on each device."""

    period_transfer_scratch_bytes: MappingProxyType[int, int]
    """Declared whole-period transfer scratch charged on each endpoint device."""

    compiler_peak_bytes: int | None
    """Raw compiler peak of the selected executable, or `None` when unreported."""

    compiler_reservation_bytes: int | None
    """Represented reservation admission used, or `None` without a budget."""

    compiled_collectives: MappingProxyType[str, int]
    """Collective operations in the selected executable's optimized HLO."""

    def to_json(self) -> str:
        """Return the record as one compact JSON object."""
        return json.dumps(
            {
                field.name: _jsonable(getattr(self, field.name))
                for field in dataclasses.fields(self)
            },
            separators=(",", ":"),
            sort_keys=True,
        )


def build_core_plan_record(
    *,
    triple: tuple[RegimeName, int, str],
    state_names: tuple[StateName, ...],
    value_shape: tuple[int, ...],
    value_sharding: jax.sharding.Sharding,
    axis_extents: Mapping[str, int],
    widths: Mapping[str, int],
    transfer_costs: Iterable[TransferCost],
    stored_owner_bytes: Mapping[int, int],
    resident_bytes: int | None,
    period_transfer_scratch_bytes: Mapping[int, int],
    compiler_peak_bytes: int | None,
    compiler_reservation_bytes: int | None,
    hlo_text: str | None,
    selected_block: Mapping[StateName, tuple[int, int]] | None = None,
) -> CorePlanRecord:
    """Assemble one core's plan record from planning results already in hand.

    Args:
        triple: The core's regime, period and name.
        state_names: State names of the published value's leading axes.
        value_shape: Shape of the whole published value.
        value_sharding: Layout the value is published on.
        axis_extents: Points in each planner axis.
        widths: Selected width of each planner axis.
        transfer_costs: Costs of the core's planned value transfers.
        stored_owner_bytes: Solve-lifetime owners charged on each device.
        resident_bytes: Bytes admission found resident, or `None` unbudgeted.
        period_transfer_scratch_bytes: Declared period transfer scratch by device.
        compiler_peak_bytes: Raw compiler peak, or `None` when unreported.
        compiler_reservation_bytes: Admitted reservation, or `None` unbudgeted.
        hlo_text: Optimized HLO of the selected executable, or `None`.
        selected_block: Half-open interval a bound core evaluates per blocked
            state, or `None` for a core evaluating every state whole.

    Returns:
        The record.

    """
    transfers = summarize_transfer_costs(costs=transfer_costs)
    named = len(state_names) <= len(value_shape)
    return CorePlanRecord(
        regime=triple[0],
        period=triple[1],
        core=triple[2],
        state_extents=MappingProxyType(
            dict(zip(state_names, value_shape, strict=False)) if named else {}
        ),
        selected_block=(
            None if selected_block is None else MappingProxyType(dict(selected_block))
        ),
        logical_value_shape=tuple(value_shape),
        physical_value_shape=tuple(value_sharding.shard_shape(tuple(value_shape))),
        device_ids=tuple(sorted(device.id for device in value_sharding.device_set)),
        axis_extents=MappingProxyType(dict(axis_extents)),
        widths=MappingProxyType(dict(widths)),
        dispatch=MappingProxyType(
            {
                name: "dense" if widths.get(name, 1) >= extent else "streamed"
                for name, extent in axis_extents.items()
            }
        ),
        stored_owner_bytes=MappingProxyType(dict(stored_owner_bytes)),
        resident_bytes=resident_bytes,
        active_replica_bytes=transfers["active_replica_bytes"],
        logical_gathered_bytes=transfers["logical_gathered_bytes"],
        transfer_workspace_bytes=transfers["transfer_workspace_bytes"],
        period_transfer_scratch_bytes=MappingProxyType(
            dict(period_transfer_scratch_bytes)
        ),
        compiler_peak_bytes=compiler_peak_bytes,
        compiler_reservation_bytes=compiler_reservation_bytes,
        compiled_collectives=count_hlo_collectives(hlo_text=hlo_text),
    )


def summarize_transfer_costs(
    *, costs: Iterable[TransferCost]
) -> MappingProxyType[str, int]:
    """Sum what a core's planned value transfers hold and gather.

    - `active_replica_bytes`: per-device result bytes of every non-local transfer;
    - `logical_gathered_bytes`: whole-value bytes of every collective transfer;
    - `transfer_workspace_bytes`: per-device operator scratch of every transfer.

    Args:
        costs: The costs of one core's planned value transfers.

    Returns:
        Immutable mapping of the three sums by name.

    """
    replica = gathered = workspace = 0
    for cost in costs:
        workspace += cost.temporary_bytes
        if cost.operation_class is TransferOperationClass.LOCAL:
            continue
        replica += cost.per_device_bytes
        if cost.operation_class is TransferOperationClass.COLLECTIVE:
            gathered += cost.logical_bytes
    return MappingProxyType(
        {
            "active_replica_bytes": replica,
            "logical_gathered_bytes": gathered,
            "transfer_workspace_bytes": workspace,
        }
    )


def count_hlo_collectives(*, hlo_text: str | None) -> MappingProxyType[str, int]:
    """Count the inter-device collective operations in optimized HLO text.

    Args:
        hlo_text: One optimized module's text, or `None` when unavailable.

    Returns:
        Immutable mapping of collective opcode to its number of occurrences;
        empty when the text is unavailable or holds none.

    """
    counts: dict[str, int] = {}
    for match in _HLO_COLLECTIVE.finditer(hlo_text or ""):
        counts[match["opcode"]] = counts.get(match["opcode"], 0) + 1
    return MappingProxyType(dict(sorted(counts.items())))


def _jsonable(value: _RecordField) -> JSONValue:
    """Convert a record field into JSON-serializable plain data."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
