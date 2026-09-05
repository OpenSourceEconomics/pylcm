"""Continuation specs: what a regime publishes for the previous period's kernels.

The engine treats a continuation as an opaque `ContinuationArtifact` identified by
its key. `ContinuationSpec` pairs the all-finite template the loop rolls and lowers
with that key; the EGM family publishes the concrete `EGMContinuationSpec`, which
adds every layout property a reading parent needs.
"""

from dataclasses import dataclass

from _lcm.egm.carry import EGMCarry
from lcm.solver_api import EGM_CONTINUATION, ArtifactKey, ContinuationArtifact

# Backward induction stores and forwards this channel without reading its fields.
type ContinuationPayload = ContinuationArtifact


@dataclass(frozen=True, kw_only=True)
class EGMContinuationLayout:
    """Static interpretation of the leading axes in one EGM carry."""

    retains_discrete_action_rows: bool = True
    """Whether a parent must aggregate child discrete-action rows."""

    rows_share_state_grid: bool = False
    """Whether every row uses the child's own state grid as abscissae."""

    n_stacked_candidates: int = 0
    """Length of a stacked outer-candidate axis; zero means no such axis."""


@dataclass(frozen=True, kw_only=True)
class ContinuationSpec:
    """Template and identity of the continuation a regime publishes."""

    template: ContinuationArtifact
    """All-finite payload with the exact shapes used for rolling and lowering."""

    artifact_key: ArtifactKey
    """Versioned identity under which the kernel publishes this payload."""

    def __post_init__(self) -> None:
        if self.template.artifact_key != self.artifact_key:
            msg = (
                f"The continuation template carries key {self.template.artifact_key}, "
                f"not the declared {self.artifact_key}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class EGMContinuationSpec(ContinuationSpec):
    """Concrete EGM continuation template bundled with its static layout."""

    template: EGMCarry
    """All-finite payload with the exact shapes used for rolling and lowering."""

    artifact_key: ArtifactKey = EGM_CONTINUATION
    """Versioned identity under which an EGM kernel publishes this payload."""

    layout: EGMContinuationLayout = EGMContinuationLayout()
    """How a reading parent interprets the template and produced payloads."""
