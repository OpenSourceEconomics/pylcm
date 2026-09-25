"""Public execution-policy configuration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

from lcm.typing import RegimeName, StateName

# The declared width of one execution axis: one width for every regime, or one
# width per named regime.
type AxisWidth = int | Mapping[RegimeName, int]


class WidthSearch(Enum):
    """How much of the width frontier a budgeted solve is willing to compile."""

    EXHAUSTIVE = "exhaustive"
    """Walk the ranked frontier widest-first and keep the first admitted candidate."""
    BOUNDED = "bounded"
    """Seed, shrink and refine within one evaluation budget per core."""


@dataclass(frozen=True, kw_only=True)
class WidthSearchPolicy:
    """How a budgeted solve chooses a width when its first candidate is refused.

    - `EXHAUSTIVE` walks the ranked frontier widest-first and keeps the first
      admitted candidate: the widest feasible point, at the cost of compiling
      every ranked candidate above it. `max_evaluations`, `refinement_share`,
      `seed` and `hints` are ignored.
    - `BOUNDED` evaluates a seed, shrinks decisively on refusal and refines
      within one evaluation budget; keeps the widest admitted candidate found.
      The result may be narrower than the widest feasible point, and the
      refusal on exhaustion says the search, not the model, ran out.
    """

    kind: WidthSearch = WidthSearch.EXHAUSTIVE
    """Which of the two searches a budgeted solve runs."""
    max_evaluations: int = 24
    """Distinct width candidates evaluated by the memory-admission search per core,
    cache hits included. The optional post-selection materialised-gather pass
    has a separate, finite halving walk and is not charged to this count.
    Leave `halve_on_materialised_gather=False` (the default) when this
    admission-search count must also be the total width-evaluation ceiling.
    """
    refinement_share: int = 8
    """Of `max_evaluations`, how many may be spent widening after admission."""
    seed: Literal["conservative", "widest"] = "conservative"
    """Start from the conservative bootstrap anchor, or from the widest candidate."""
    hints: Mapping[RegimeName, Mapping[str, int]] = MappingProxyType({})
    """Regime name to a width mapping tried first; an incompatible hint is skipped."""

    def __post_init__(self) -> None:
        """Reject unusable counts, seeds and hints at construction."""
        _fail_if_width_search_kind_invalid(kind=self.kind)
        _fail_if_evaluation_counts_invalid(
            max_evaluations=self.max_evaluations,
            refinement_share=self.refinement_share,
        )
        _fail_if_seed_invalid(seed=self.seed)
        object.__setattr__(
            self, "hints", MappingProxyType(_normalized_hints(hints=self.hints))
        )


def _fail_if_width_search_kind_invalid(*, kind: WidthSearch) -> None:
    """Require one of the two declared search kinds."""
    if not isinstance(kind, WidthSearch):
        raise TypeError("WidthSearchPolicy.kind must be a WidthSearch member.")


def _fail_if_evaluation_counts_invalid(
    *, max_evaluations: int, refinement_share: int
) -> None:
    """Require a positive evaluation budget holding a non-negative refinement share."""
    for label, value in (
        ("max_evaluations", max_evaluations),
        ("refinement_share", refinement_share),
    ):
        if type(value) is not int:
            raise TypeError(f"WidthSearchPolicy.{label} must be an exact int.")
    if max_evaluations < 1:
        raise ValueError("WidthSearchPolicy.max_evaluations must be at least one.")
    if refinement_share < 0:
        raise ValueError("WidthSearchPolicy.refinement_share must not be negative.")
    if refinement_share > max_evaluations:
        msg = (
            "WidthSearchPolicy.refinement_share must not exceed max_evaluations; got "
            f"{refinement_share} of {max_evaluations}."
        )
        raise ValueError(msg)


def _fail_if_seed_invalid(*, seed: str) -> None:
    """Require one of the two declared seed rules."""
    if type(seed) is not str:
        raise TypeError("WidthSearchPolicy.seed must be an exact str.")
    if seed not in ("conservative", "widest"):
        msg = f"WidthSearchPolicy.seed must be conservative or widest; got {seed!r}."
        raise ValueError(msg)


def _normalized_hints(
    *, hints: Mapping[RegimeName, Mapping[str, int]]
) -> dict[RegimeName, Mapping[str, int]]:
    """Validate every hinted width and freeze each regime's mapping."""
    normalized: dict[RegimeName, Mapping[str, int]] = {}
    for regime_name, widths in hints.items():
        if type(regime_name) is not str or not regime_name:
            msg = "WidthSearchPolicy.hints keys must be non-empty regime names."
            raise TypeError(msg)
        if not isinstance(widths, Mapping):
            msg = (
                f"WidthSearchPolicy.hints[{regime_name!r}] must map axis names to "
                "widths."
            )
            raise TypeError(msg)
        normalized[regime_name] = MappingProxyType(
            _validated_hint_widths(regime_name=regime_name, widths=widths)
        )
    return normalized


def _validated_hint_widths(
    *, regime_name: RegimeName, widths: Mapping[str, int]
) -> dict[str, int]:
    """Require positive exact widths keyed by non-empty axis names."""
    validated: dict[str, int] = {}
    for axis_name, width in widths.items():
        if type(axis_name) is not str or not axis_name:
            msg = (
                f"WidthSearchPolicy.hints[{regime_name!r}] keys must be non-empty "
                "axis names."
            )
            raise TypeError(msg)
        label = f"WidthSearchPolicy.hints[{regime_name!r}][{axis_name!r}]"
        if type(width) is not int:
            raise TypeError(f"{label} must be an exact int.")
        if width <= 0:
            raise ValueError(f"{label} must be positive.")
        validated[axis_name] = width
    return validated


@dataclass(frozen=True, kw_only=True)
class ExecutionConfig:
    """Hardware-local controls for solving and simulation.

    None of these values enters a model's durable fingerprint. Without a
    device-memory budget, omitted widths use conservative bootstrap choices.
    An axis width fixes the execution block of every program declaring that name;
    a sharded state spreads its grid axis over its regime's assigned devices.
    """

    device_memory_bytes: int | None = None
    """Per-device ceiling for represented compiler reservation plus residency.

    The reservation enforces the raw peak and represented argument, output,
    alias, and temporary allocations. Runtime storage omitted by the compiler
    remains outside this accounting scope. `None` disables budget admission.

    A caller may pass the device's whole allocator pool limit here: the model
    resolves the ceiling it plans against by taking
    `device_memory_headroom_fraction` off every selected device's pool limit
    and keeping the smaller of that and this request.
    """

    device_memory_headroom_fraction: float = 0.15
    """Share of each device's allocator pool kept out of the budget ceiling.

    An operational safety margin for pool pressure that lives outside the
    represented accounting: collective-communication buffers, library
    workspaces such as cuBLAS, the driver context, and allocator
    fragmentation. It is a policy awaiting workload validation, not a measured
    requirement, so a caller who has measured their own envelope may set it to
    `0.0` and plan against the whole pool.

    It moves only the ceiling a plan is admitted against; every compiler
    reservation and residency figure the admission compares to that ceiling is
    unchanged. Must be an exact float in `[0, 1)`.
    """

    sharded_states: tuple[StateName, ...] = ()
    """States whose grid axis is spread over the regime's devices.

    Sharding is resolved per regime, from the state the regime actually carries:

    - A regime whose DAG reads the state carries its grid axis and runs on the
      submesh that axis defines.
    - A regime that never reads the state — so reachability prunes it there —
      publishes a value without the axis and runs on a single device. This is
      an ordinary plan for a terminal and a non-terminal regime alike, and the
      values crossing between the two placements are moved by the planner.

    A state every regime prunes leaves no axis to spread and is refused.

    A discrete state always qualifies. A continuous state qualifies only on a
    narrow route: it must be the sole sharded state, a model-level
    `LinSpacedGrid` retained in every regime, solved everywhere with
    `GridSearch` under a singleton hard-max, and the regimes must carry no
    stakeholders, gated edges, value constraints, same-period references or
    taste shocks. Model construction checks every requirement and the error
    lists what a rejected model violates.
    """

    axis_widths: Mapping[str, AxisWidth] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Planner axis name to the block width it is compiled at; empty means planned.

    A width takes one of two forms, and the two may be mixed across axes:

    - A bare integer fixes that axis in every regime declaring it.
    - A mapping from regime name to width fixes it only in the regimes it
      names; every other regime keeps the width the planner chooses for it.

    The per-regime form serves the solve phase, where a regime's shape drives
    the choice: each regime is fixed independently, so two regimes of opposite
    shape need not share a width. Simulation plans without a regime in hand, so
    an axis only its programs declare takes the bare-integer form.
    """

    axis_width_ceilings: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Planner axis name to the widest block it may be compiled at; empty means none.

    A ceiling is an upper bound, not an override: the planner keeps every legal
    candidate at or below it and drops the wider ones. It leaves the axis extent,
    the output shape, the alignment and the floor of the axis untouched, and it
    binds only the axes it names. An axis whose `axis_widths` entry already fixes
    it is narrowed to the ceiling as well.

    A ceiling below an axis's narrowest legal width is refused when the axis is
    planned, naming the axis and the value.
    """

    covered_axes: tuple[str, ...] = ()
    """Planner axis names whose conservative seed covers the whole extent.

    Without a budget, and as the conservative seed of a bounded search, an axis
    is lowered at the largest power of two below its extent. For an axis named
    here whose extent is at most the bootstrap cap and not a multiple of that
    power of two, the seed is the full extent instead, so the map over the axis
    needs no remainder program. A fixed width or a ceiling below the extent
    still binds. A bounded search refused at the covered width proposes the
    power-of-two width next. An empty tuple covers nothing.

    Only solve planning reads this option. Forward simulation plans its programs
    without it, so naming an axis that only simulation programs declare is
    accepted but covers nothing.
    """

    devices: tuple[int, ...] | None = None
    """Device ids the model may use, or `None` for every device JAX reports."""

    simulation_sharding: Literal["legacy", "subjects"] = "legacy"
    """Use solve-derived placement, or shard forward subjects on every device.

    The subjects mode puts each compiled subject tile loop inside a device shard.
    Its subject width is a per-device upper bound, independent of solve axes.
    """

    donate_buffers: bool = True
    """Allow eligible owned inputs to be donated by compiled solve programs."""

    halve_on_materialised_gather: bool = False
    """Opt in to halving a GridSearch cell width when its gather materialises.

    Disabled by default. Enable only for a measured workload and compiler/device
    configuration: unsuccessful trials still cost compilation time, and a fused
    program is not necessarily faster. Disabling this pass does not disable the
    ordinary memory-admission search.

    Past a device- and program-dependent cell count, XLA writes the continuation
    lookup table to device memory and the reduction reads it back, instead of
    recomputing it inside the reduction. The planner reads each compiled GridSearch
    solve program and, while one of its reduce fusions reads a gather table another
    fusion wrote, recompiles it at half the cell width. A width fixed through
    `axis_widths` is kept as given. A narrower width is kept only when its compiled
    program is proved fused. When the walk reaches the narrowest width its cell
    axis admits and every program still materialises, or reaches a program whose
    structure the planner does not read completely, the solve keeps the width that
    passed memory admission, with its materialised gather and its already compiled
    program, and logs why no narrower width was kept. An unreadable program is
    diagnosed once and is never assumed fused. Under a device-memory budget, a
    narrower candidate is admitted like any other width: one that exceeds the
    budget raises `ExecutionPlanningError`. Each halving is a trial beyond
    `WidthSearchPolicy.max_evaluations`, and is compiled only when no earlier
    program shares its lowering. Other solvers and simulation are not checked.

    Halving is a bounded heuristic over the compiled executable, not a guarantee
    that the fused program is faster: the compiler's choice need not be monotone in
    the width, so the walk can miss a useful width or give up at its floor although
    an unvisited width fuses.
    """

    width_search: WidthSearchPolicy = WidthSearchPolicy()
    """How much of the width frontier a budgeted solve is willing to compile.

    The default walks the ranked frontier widest-first, so a solve keeps the
    widest feasible width and pays for every ranked candidate above it. The
    bounded policy trades that guarantee for an evaluation budget per core:
    the budgeted compilation waves take each core's next width from its search
    instead of from the ranked frontier.
    """

    def __post_init__(self) -> None:
        """Reject ambiguous or unusable values at construction."""
        _fail_if_budget_invalid(device_memory_bytes=self.device_memory_bytes)
        _fail_if_headroom_fraction_invalid(
            device_memory_headroom_fraction=self.device_memory_headroom_fraction
        )
        if not isinstance(self.width_search, WidthSearchPolicy):
            msg = "ExecutionConfig.width_search must be a WidthSearchPolicy."
            raise TypeError(msg)
        if type(self.donate_buffers) is not bool:
            raise TypeError("ExecutionConfig.donate_buffers must be an exact bool.")
        if type(self.halve_on_materialised_gather) is not bool:
            raise TypeError(
                "ExecutionConfig.halve_on_materialised_gather must be an exact bool."
            )
        if type(self.simulation_sharding) is not str:
            raise TypeError("ExecutionConfig.simulation_sharding must be an exact str.")
        if self.simulation_sharding not in ("legacy", "subjects"):
            raise ValueError(
                "ExecutionConfig.simulation_sharding must be legacy or subjects."
            )
        widths = _normalized_axis_widths(axis_widths=self.axis_widths)
        object.__setattr__(self, "axis_widths", MappingProxyType(widths))
        object.__setattr__(
            self,
            "axis_width_ceilings",
            _normalized_axis_width_ceilings(
                axis_width_ceilings=self.axis_width_ceilings
            ),
        )
        _fail_if_covered_axes_invalid(covered_axes=self.covered_axes)
        sharded = tuple(self.sharded_states)
        _fail_if_sharded_states_invalid(sharded_states=sharded)
        object.__setattr__(self, "sharded_states", sharded)
        if self.devices is not None:
            devices = tuple(self.devices)
            _fail_if_devices_invalid(devices=devices)
            object.__setattr__(self, "devices", devices)


def _fail_if_budget_invalid(*, device_memory_bytes: int | None) -> None:
    """Require a positive exact-integer byte ceiling when one is supplied."""
    if device_memory_bytes is None:
        return
    if type(device_memory_bytes) is not int:
        raise TypeError("ExecutionConfig.device_memory_bytes must be an exact int.")
    if device_memory_bytes <= 0:
        raise ValueError("ExecutionConfig.device_memory_bytes must be positive.")


def _fail_if_headroom_fraction_invalid(
    *, device_memory_headroom_fraction: float
) -> None:
    """Require an exact float share of the pool in `[0, 1)`."""
    if type(device_memory_headroom_fraction) is not float:
        msg = "ExecutionConfig.device_memory_headroom_fraction must be an exact float."
        raise TypeError(msg)
    if math.isnan(device_memory_headroom_fraction):
        msg = "ExecutionConfig.device_memory_headroom_fraction must not be NaN."
        raise ValueError(msg)
    if not 0.0 <= device_memory_headroom_fraction < 1.0:
        msg = (
            "ExecutionConfig.device_memory_headroom_fraction must lie in [0, 1); got "
            f"{device_memory_headroom_fraction!r}."
        )
        raise ValueError(msg)


def _normalized_axis_widths(
    *, axis_widths: Mapping[str, AxisWidth]
) -> dict[str, AxisWidth]:
    """Validate both declaration forms and freeze the per-regime mappings.

    Args:
        axis_widths: The widths the caller declared, by axis name.

    Returns:
        The same declaration with every per-regime mapping copied into a
        read-only mapping, so a later mutation of the caller's dict cannot
        reach the configuration.

    Raises:
        TypeError: An axis name, regime name or width has the wrong type.
        ValueError: A width is not positive, or a mapping names no regime.

    """
    normalized: dict[str, AxisWidth] = {}
    for name, width in axis_widths.items():
        if type(name) is not str or not name:
            msg = "ExecutionConfig.axis_widths keys must be non-empty axis names."
            raise TypeError(msg)
        if isinstance(width, Mapping):
            normalized[name] = MappingProxyType(
                _validated_per_regime_widths(axis_name=name, widths=width)
            )
        else:
            _fail_if_width_invalid(label=f"axis_widths[{name!r}]", width=width)
            normalized[name] = width
    return normalized


def _normalized_axis_width_ceilings(
    *, axis_width_ceilings: Mapping[str, int]
) -> MappingProxyType[str, int]:
    """Validate the one declaration form a ceiling takes and freeze it.

    Args:
        axis_width_ceilings: The ceilings the caller declared, by axis name.

    Returns:
        The same declaration in a read-only mapping.

    Raises:
        TypeError: An axis name is not a non-empty string, or a ceiling is not
            an exact integer — a per-regime mapping among them.
        ValueError: A ceiling is not positive.

    """
    validated: dict[str, int] = {}
    for name, ceiling in axis_width_ceilings.items():
        if type(name) is not str or not name:
            msg = (
                "ExecutionConfig.axis_width_ceilings keys must be non-empty axis names."
            )
            raise TypeError(msg)
        _fail_if_width_invalid(label=f"axis_width_ceilings[{name!r}]", width=ceiling)
        validated[name] = ceiling
    return MappingProxyType(validated)


def _fail_if_covered_axes_invalid(*, covered_axes: tuple[str, ...]) -> None:
    """Require distinct non-empty axis names."""
    if not all(covered_axes):
        msg = "ExecutionConfig.covered_axes entries must be non-empty axis names."
        raise ValueError(msg)
    if len(set(covered_axes)) != len(covered_axes):
        msg = f"ExecutionConfig.covered_axes names an axis twice: {covered_axes!r}."
        raise ValueError(msg)


def _validated_per_regime_widths(
    *, axis_name: str, widths: Mapping[RegimeName, int]
) -> dict[RegimeName, int]:
    """Require at least one regime, each named once with a positive exact width."""
    if not widths:
        msg = (
            f"ExecutionConfig.axis_widths[{axis_name!r}] names no regime; give a "
            "bare integer to fix the axis in every regime."
        )
        raise ValueError(msg)
    validated: dict[RegimeName, int] = {}
    for regime_name, width in widths.items():
        if type(regime_name) is not str or not regime_name:
            msg = (
                f"ExecutionConfig.axis_widths[{axis_name!r}] keys must be non-empty "
                "regime names."
            )
            raise TypeError(msg)
        _fail_if_width_invalid(
            label=f"axis_widths[{axis_name!r}][{regime_name!r}]", width=width
        )
        validated[regime_name] = width
    return validated


def _fail_if_width_invalid(*, label: str, width: object) -> None:
    """Require a positive exact integer, naming the declaration that carries it."""
    if type(width) is not int:
        raise TypeError(f"ExecutionConfig.{label} must be an exact int.")
    if width <= 0:
        raise ValueError(f"ExecutionConfig.{label} must be positive.")


def _fail_if_sharded_states_invalid(*, sharded_states: tuple[StateName, ...]) -> None:
    """Require distinct, non-empty state names."""
    if any(type(name) is not str or not name for name in sharded_states):
        msg = "ExecutionConfig.sharded_states must contain non-empty state names."
        raise TypeError(msg)
    if len(set(sharded_states)) != len(sharded_states):
        msg = "ExecutionConfig.sharded_states must be distinct."
        raise ValueError(msg)


def _fail_if_devices_invalid(*, devices: tuple[int, ...]) -> None:
    """Require at least one distinct, non-negative exact-integer device id."""
    if any(type(device_id) is not int or device_id < 0 for device_id in devices):
        msg = "ExecutionConfig.devices must contain non-negative exact ints."
        raise TypeError(msg)
    if len(set(devices)) != len(devices):
        msg = "ExecutionConfig.devices must be distinct."
        raise ValueError(msg)
    if not devices:
        msg = "ExecutionConfig.devices must name at least one device."
        raise ValueError(msg)
