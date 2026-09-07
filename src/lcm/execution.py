"""Public execution-policy configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, kw_only=True)
class ExecutionConfig:
    """Hardware-local controls for solving and simulation.

    None of these values enters a model's durable fingerprint. A missing
    device-memory budget leaves execution unconstrained; an axis width fixes the
    compiled block width of every program declaring that axis name.
    """

    device_memory_bytes: int | None = None
    """Per-device byte ceiling for compiler-reported peak workspace, or `None`."""

    axis_widths: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    """Planner axis name to the block width it is compiled at; empty means planned."""

    def __post_init__(self) -> None:
        """Reject ambiguous or unusable values at construction."""
        _fail_if_budget_invalid(device_memory_bytes=self.device_memory_bytes)
        widths = dict(self.axis_widths)
        for name, width in widths.items():
            if type(name) is not str or not name:
                msg = "ExecutionConfig.axis_widths keys must be non-empty axis names."
                raise TypeError(msg)
            if type(width) is not int:
                msg = f"ExecutionConfig.axis_widths[{name!r}] must be an exact int."
                raise TypeError(msg)
            if width <= 0:
                msg = f"ExecutionConfig.axis_widths[{name!r}] must be positive."
                raise ValueError(msg)
        object.__setattr__(self, "axis_widths", MappingProxyType(widths))


def _fail_if_budget_invalid(*, device_memory_bytes: int | None) -> None:
    """Require a positive exact-integer byte ceiling when one is supplied."""
    if device_memory_bytes is None:
        return
    if type(device_memory_bytes) is not int:
        raise TypeError("ExecutionConfig.device_memory_bytes must be an exact int.")
    if device_memory_bytes <= 0:
        raise ValueError("ExecutionConfig.device_memory_bytes must be positive.")
