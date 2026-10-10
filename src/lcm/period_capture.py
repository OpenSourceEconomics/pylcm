"""Select and inspect production-period captures without serializing a model."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import jax
import numpy as np

from _lcm.typing import JSONValue


@dataclass(frozen=True, kw_only=True)
class PeriodCapture:
    """Persist selected ordinary GridSearch entries and completed references.

    Each target gets a new `regime@period` directory. Existing targets are refused.
    The caller's source identity complements the model, grid, parameter, installed
    source and runtime identities computed by pylcm. No model or callable is pickled.
    GPU capture and replay require actual serialized buffer-assignment metadata.
    Missing runtime metadata is refused before entry publication or replay dispatch.
    Startup cache-off and compiler debug metadata do not guarantee runtime support.
    """

    directory: Path
    """Parent directory for the selected captures."""

    periods: tuple[tuple[str, int], ...]
    """Distinct `(regime name, zero-based period)` targets."""

    source_identity: Mapping[str, str]
    """Explicit revision or content digests of the consuming model and inputs."""

    def __post_init__(self) -> None:
        """Validate selection and own the caller's identity mapping."""
        targets = tuple(self.periods)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("PeriodCapture periods must be nonempty and distinct.")
        for name, period in targets:
            if (
                type(name) is not str
                or not name
                or any(character in name for character in "/\\@")
                or name in {".", ".."}
                or type(period) is not int
                or period < 0
            ):
                raise ValueError("Invalid PeriodCapture regime or period.")
        identity = dict(self.source_identity.items())
        if not identity or any(
            type(key) is not str or not key or type(value) is not str or not value
            for key, value in identity.items()
        ):
            raise ValueError(
                "source_identity requires nonempty string names and values."
            )
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "periods", targets)
        object.__setattr__(self, "source_identity", MappingProxyType(identity))


@dataclass(frozen=True, kw_only=True)
class PeriodCaptureRecord:
    """Inspect a checksum-verified entry and its optional completed reference."""

    metadata: Mapping[str, JSONValue]
    """Identity, layout, widths, compiler admission and optimized HLO records."""

    reference: np.ndarray | None
    """Completed in-context value on the host, absent for an interrupted entry."""

    in_context_seconds: float | None
    """Synchronized dispatch wall time, excluding capture serialization."""

    @property
    def completed(self) -> bool:
        """Return whether a completed reference is available."""
        return self.reference is not None


@dataclass(frozen=True, kw_only=True)
class CapturedPeriodReplay:
    """Report one strict recorded-layout and recorded-admission replay.

    Admission is the recorded represented-allocation contract, not a restoration
    of unrelated live production buffers or a measured physical memory peak.
    """

    value: jax.Array | np.ndarray
    """The value returned by the production period adapter."""

    reference_matches: bool | None
    """Exact value bytes and nonfinite masks, or None without a reference."""

    optimized_hlo_matches: bool
    """Whether canonical optimized HLO identities match the captured executables."""

    in_context_seconds: float | None
    """Captured synchronized dispatch wall, absent for entry-only evidence."""

    replay_seconds: float
    """Synchronized replay dispatch wall, excluding loading and compilation."""

    capture: PeriodCaptureRecord
    """The verified source record used for this replay."""
