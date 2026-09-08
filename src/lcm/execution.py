"""Public execution-policy configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from lcm.typing import StateName


@dataclass(frozen=True, kw_only=True)
class ExecutionConfig:
    """Hardware-local controls for solving and simulation.

    None of these values enters a model's durable fingerprint. A missing
    device-memory budget leaves execution unconstrained; an axis width fixes the
    compiled block width of every program declaring that axis name; a sharded
    state spreads its grid axis over the devices its regime is placed on.
    """

    device_memory_bytes: int | None = None
    """Per-device byte ceiling for compiler-reported peak workspace, or `None`."""

    sharded_states: tuple[StateName, ...] = ()
    """States whose grid axis is spread over the regime's devices."""

    axis_widths: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    """Planner axis name to the block width it is compiled at; empty means planned."""

    devices: tuple[int, ...] | None = None
    """Device ids the model may use, or `None` for every device JAX reports."""

    donate_buffers: bool = True
    """Allow eligible owned inputs to be donated by compiled solve programs."""

    def __post_init__(self) -> None:
        """Reject ambiguous or unusable values at construction."""
        _fail_if_budget_invalid(device_memory_bytes=self.device_memory_bytes)
        if type(self.donate_buffers) is not bool:
            raise TypeError("ExecutionConfig.donate_buffers must be an exact bool.")
        widths = dict(self.axis_widths)
        _fail_if_axis_widths_invalid(axis_widths=widths)
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


def _fail_if_axis_widths_invalid(*, axis_widths: Mapping[str, int]) -> None:
    """Require every axis name to be non-empty and every width a positive int."""
    for name, width in axis_widths.items():
        if type(name) is not str or not name:
            msg = "ExecutionConfig.axis_widths keys must be non-empty axis names."
            raise TypeError(msg)
        if type(width) is not int:
            msg = f"ExecutionConfig.axis_widths[{name!r}] must be an exact int."
            raise TypeError(msg)
        if width <= 0:
            msg = f"ExecutionConfig.axis_widths[{name!r}] must be positive."
            raise ValueError(msg)


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
