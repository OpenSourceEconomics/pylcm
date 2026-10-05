"""Immutable requests and results for a production candidate's unoptimized IR."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from beartype import beartype

from _lcm.beartype_conf import MODEL_CONF
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import RegimeName


@beartype(conf=MODEL_CONF)
@dataclass(frozen=True, kw_only=True)
class PeriodCandidate:
    """Select an exact primary width candidate from the production ranked frontier."""

    regime: RegimeName
    """Regime name in the model's solve graph."""
    period: int
    """Zero-based period in the unchanged model horizon."""
    core: str
    """Named core selected by the solve's retention policy."""
    widths: Mapping[str, int]
    """Exact execution-axis widths; copied to an immutable mapping."""

    def __post_init__(self) -> None:
        if type(self.period) is not int or self.period < 0:
            raise ExecutionPlanningError(
                "Candidate period must be a nonnegative integer."
            )
        if not self.regime or not self.core:
            raise ExecutionPlanningError("Candidate regime and core must be nonempty.")
        if any(type(width) is not int or width <= 0 for width in self.widths.values()):
            raise ExecutionPlanningError("Candidate widths must be positive integers.")
        object.__setattr__(self, "widths", MappingProxyType(dict(self.widths)))


@dataclass(frozen=True, kw_only=True)
class LoweredPeriodCandidate:
    """Own unoptimized IR bytes and immutable facts, without a live JAX program.

    Lowering does not establish compiler memory, admission or physical residency.
    Current preparation may initialize a backend and allocate zero templates.
    """

    manifest: MappingProxyType[str, object]
    """Copied descriptor facts, including explicit unavailable measurements."""
    stablehlo: bytes
    """Exact UTF-8 StableHLO text, with no normalization of source locations."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(
                {key: _freeze_descriptor(value) for key, value in self.manifest.items()}
            ),
        )
        if type(self.stablehlo) is not bytes:
            raise ExecutionPlanningError("StableHLO must be owned immutable bytes.")


def _freeze_descriptor(value: object) -> object:
    """Copy immutable descriptor containers and reject every live payload type."""
    if value is None or type(value) in (str, bool, int, bytes):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _freeze_descriptor(key): _freeze_descriptor(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_freeze_descriptor(child) for child in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_descriptor(child) for child in value)
    raise ExecutionPlanningError(f"Unsupported live descriptor payload: {type(value)}")
