"""Public execution-policy configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ExecutionConfig:
    """Hardware-local controls for solving and simulation.

    A missing device-memory budget leaves execution unconstrained. When supplied,
    the budget is a per-device byte ceiling for compiler-reported peak workspace.
    """

    device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unusable byte budgets at construction."""
        if self.device_memory_bytes is None:
            return
        if type(self.device_memory_bytes) is not int:
            raise TypeError("ExecutionConfig.device_memory_bytes must be an exact int.")
        if self.device_memory_bytes <= 0:
            raise ValueError("ExecutionConfig.device_memory_bytes must be positive.")
