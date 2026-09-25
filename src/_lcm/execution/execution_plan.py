"""Resolve a public `ExecutionConfig` against what a model declares.

The devices a model may use are read here and nowhere else in the package: a
model resolves them once when it is built and every phase reads the resolved
ids, so two phases of one model can never disagree about the hardware they run
on.
"""

import dataclasses
import logging
import math
import operator
from collections.abc import Collection, Iterable, Mapping
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import jax

from _lcm.execution.core_program import CoreProgram
from _lcm.typing import RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import AxisWidth, ExecutionConfig, WidthSearchPolicy

logger = logging.getLogger(__name__)


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

    `None` on a resolution built without a request, so an unbudgeted model and
    one whose request was never recorded read alike.
    """

    device_memory_headroom_fraction: float = 0.15
    """Share of each device's pool the effective budget keeps free."""

    device_pool_limit_bytes: MappingProxyType[int, int | None] = MappingProxyType({})
    """Allocator pool limit of each selected device, `None` where unreported.

    Empty on an unbudgeted model: the resolver keeps no limit when no budget is
    set, although construction still reads the visible pool statistics once.
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

        Names the requested budget, the headroom fraction, each selected
        device's pool limit with the bytes that fraction keeps free, the
        effective budget, and whether the devices capped the request.
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
        verdict = (
            "capped by device headroom"
            if self._device_headroom_capped_the_request()
            else "no cap applied"
        )
        return (
            f"Device-memory budget: requested "
            f"{self.device_memory_bytes if requested is None else requested} bytes; "
            f"headroom fraction {fraction}; per-device pool limits: "
            f"{limits or 'none consulted'}; "
            f"effective {self.device_memory_bytes} bytes; {verdict}."
        )

    def device_memory_cap_note(self) -> str:
        """Return a clause for a diagnostic, empty unless the devices capped.

        Leading space included, so a caller appends it to a sentence.
        """
        if not self._device_headroom_capped_the_request():
            return ""
        return (
            f" The effective budget is {self.device_memory_bytes} bytes of the "
            f"requested {self.requested_device_memory_bytes} bytes after a "
            "device-memory headroom fraction of "
            f"{self.device_memory_headroom_fraction}."
        )

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
        ExecutionPlanningError: A state, regime or device the model cannot serve.

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
    resolved = ResolvedExecution(
        device_ids=selected_ids,
        sharded_states=frozenset(config.sharded_states),
        axis_widths=model_wide_widths,
        axis_widths_by_regime=widths_by_regime,
        axis_width_ceilings=MappingProxyType(dict(config.axis_width_ceilings)),
        covered_axes=frozenset(config.covered_axes),
        device_memory_bytes=_effective_device_memory_bytes(
            requested_bytes=config.device_memory_bytes,
            headroom_fraction=config.device_memory_headroom_fraction,
            pool_limits=selected_limits,
        ),
        requested_device_memory_bytes=config.device_memory_bytes,
        device_memory_headroom_fraction=config.device_memory_headroom_fraction,
        device_pool_limit_bytes=selected_limits,
        donate_buffers=config.donate_buffers,
        halve_on_materialised_gather=config.halve_on_materialised_gather,
        width_search=config.width_search,
        simulation_sharding=config.simulation_sharding,
    )
    if config.device_memory_bytes is not None:
        summary = resolved.device_memory_budget_summary()
        if resolved.device_memory_bytes != config.device_memory_bytes:
            logger.warning("%s", summary)
        else:
            logger.info("%s", summary)
    return resolved


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
    backend returns no counters, omits `bytes_limit`, or fails the query
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
    return limit if isinstance(limit, int) else None


def execution_over_visible_devices() -> ResolvedExecution:
    """Return the inert configuration resolved against every visible device.

    The resolution a caller that builds canonical regimes outside a `Model` —
    a test, or a tool inspecting one regime — runs under.
    """
    return resolve_execution_config(
        config=ExecutionConfig(),
        visible_device_ids=visible_device_ids(),
        state_names=frozenset(),
        device_pool_limit_bytes=visible_device_pool_limits(),
    )
