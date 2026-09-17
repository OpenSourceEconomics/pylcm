"""Public execution-policy configuration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from lcm.typing import RegimeName, StateName

# The declared width of one execution axis: one width for every regime, or one
# width per named regime.
type AxisWidth = int | Mapping[RegimeName, int]


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
    the choice: two regimes of opposite shape no longer share the width one of
    them needs. Simulation plans without a regime in hand, so an axis only its
    programs declare takes the bare-integer form.
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

    def __post_init__(self) -> None:
        """Reject ambiguous or unusable values at construction."""
        _fail_if_budget_invalid(device_memory_bytes=self.device_memory_bytes)
        _fail_if_headroom_fraction_invalid(
            device_memory_headroom_fraction=self.device_memory_headroom_fraction
        )
        if type(self.donate_buffers) is not bool:
            raise TypeError("ExecutionConfig.donate_buffers must be an exact bool.")
        if type(self.simulation_sharding) is not str:
            raise TypeError("ExecutionConfig.simulation_sharding must be an exact str.")
        if self.simulation_sharding not in ("legacy", "subjects"):
            raise ValueError(
                "ExecutionConfig.simulation_sharding must be legacy or subjects."
            )
        widths = _normalized_axis_widths(axis_widths=self.axis_widths)
        object.__setattr__(self, "axis_widths", MappingProxyType(widths))
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
