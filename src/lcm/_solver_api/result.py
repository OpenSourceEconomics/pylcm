"""Solution metadata, artifact-contract comparison, and the labelled solution result."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    TypeAlias,
    cast,
)

from lcm._solver_api.authority import (
    ArtifactAuthority,
)
from lcm._solver_api.contract import (
    ArtifactRef,
    OmissionReason,
    SolutionMetadata,
)
from lcm._solver_api.stores import (
    ArtifactStore,
    ValueStore,
    _ArtifactStoreBoundary,
    _FloatValueBoundary,
    _RegimeNameBoundary,
    _require_exact_artifact_ref,
    _traverse_public_mapping_items,
    _ValuePeriodBoundary,
)
from lcm.typing import FloatND, RegimeName

if TYPE_CHECKING:
    _SolutionValuesInput: TypeAlias = (  # noqa: UP040
        Mapping[int, Mapping[RegimeName, FloatND]] | ValueStore
    )
    _SolutionOmissionsInput: TypeAlias = Mapping[  # noqa: UP040
        ArtifactRef, OmissionReason
    ]
else:
    # The public static contract stays precise above. At runtime the package-wide
    # beartype claw must not traverse these mappings: a ValueStore can contain lazy
    # archive entries whose checksum and payload validation belong to explicit
    # materialization, while omission validation belongs to result/save preflight.
    _SolutionValuesInput = object
    _SolutionOmissionsInput = object


@dataclass(frozen=True, kw_only=True)
class SolutionResult:
    """Labelled value functions, retained artifacts, and omission records."""

    values: _SolutionValuesInput
    """Value function of every solved cell, keyed by period then regime."""
    metadata: SolutionMetadata
    """Identity, retention, and schema facts of the solve."""
    retained_continuations: _ArtifactStoreBoundary = field(
        default_factory=ArtifactStore
    )
    """Continuation payloads kept for persistence, addressed by cell and key."""
    replay_artifacts: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    """Payloads simulation replays decisions from, addressed by cell and key."""
    auxiliary_artifacts: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    """Additional solver-published payloads, addressed by cell and key."""
    omissions: _SolutionOmissionsInput = field(default_factory=dict)
    """Why each accounted-for artifact that is absent was left out."""
    diagnostics: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    """Solver diagnostics kept according to the solve's log level."""
    _artifact_authority: Mapping[ArtifactRef, ArtifactAuthority] = field(
        default_factory=lambda: MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )
    _engine_view: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    """The producing model's by-reference view of this result, when the engine
    built it in this process; `None` for every other provenance and for any
    copy made through `dataclasses.replace`."""
    _consumed_views: dict[object, object] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    """Validated engine views keyed by the consuming model and parameters, so a
    result from elsewhere is validated and materialized once per consumer."""

    def __post_init__(self) -> None:
        for field_name, store in (
            ("retained_continuations", self.retained_continuations),
            ("replay_artifacts", self.replay_artifacts),
            ("auxiliary_artifacts", self.auxiliary_artifacts),
            ("diagnostics", self.diagnostics),
        ):
            if type(store) is not ArtifactStore:
                raise TypeError(
                    f"SolutionResult.{field_name} must be an exact ArtifactStore."
                )
        values = (
            self.values
            if type(self.values) is ValueStore
            else ValueStore(cast("Mapping", self.values))
        )
        object.__setattr__(self, "values", values)
        omissions: dict[ArtifactRef, OmissionReason] = {}
        for raw_ref, reason in _traverse_public_mapping_items(
            mapping=self.omissions, label="SolutionResult omissions"
        ):
            ref = _require_exact_artifact_ref(raw_ref)
            if type(reason) is not OmissionReason:
                raise TypeError(
                    "SolutionResult omission reasons must be exact OmissionReason "
                    "values."
                )
            if ref in omissions:
                raise ValueError(
                    f"SolutionResult omission address {ref!r} appears twice."
                )
            omissions[ref] = reason
        object.__setattr__(self, "omissions", MappingProxyType(omissions))

    def value(
        self,
        *,
        period: _ValuePeriodBoundary,
        regime: _RegimeNameBoundary,
    ) -> _FloatValueBoundary:
        """Return one value-function array by its explicit coordinates."""
        return self.values[period][regime]

    def save(self, *, path: Path) -> Path:
        """Persist this complete result atomically to a versioned archive."""
        from lcm.persistence import save_solution  # noqa: PLC0415

        return save_solution(solution=self, path=path)
