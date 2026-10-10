"""Solution metadata, artifact-contract comparison, and the labelled solution result."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
from beartype import beartype

from lcm._solver_api.authority import (
    ArtifactAuthority,
)
from lcm._solver_api.beartype_conf import SOLVER_API_CONF
from lcm._solver_api.contract import (
    ArtifactRef,
    OmissionReason,
    SolutionMetadata,
)
from lcm._solver_api.stores import (
    ArtifactStore,
    ValueStore,
    _FloatValueBoundary,
    _require_exact_artifact_ref,
    _traverse_public_mapping_items,
)
from lcm.typing import FloatND, RegimeName

if TYPE_CHECKING:
    from _lcm.solution.artifacts import OwnedSolutionView
    from lcm.model import _ResolvedSolution

    type _SolutionValuesInput = Mapping[int, Mapping[RegimeName, FloatND]] | ValueStore
    type _SolutionOmissionsInput = Mapping[ArtifactRef, OmissionReason]
    type _EngineView = OwnedSolutionView
    # Engine replay inputs a consuming model resolved from this result, keyed by
    # that model's instance and the parameter fingerprint it consumed under. The
    # model fills this memo as it consumes the result, so it is a mutable dict.
    type _ConsumedViews = dict[tuple[str, str], _ResolvedSolution]
else:
    # The public static contract stays precise above. At runtime the package-wide
    # beartype claw must not traverse these mappings: a ValueStore can contain lazy
    # archive entries whose checksum and payload validation belong to explicit
    # materialization, while omission validation belongs to result/save preflight.
    type _SolutionValuesInput = object
    type _SolutionOmissionsInput = object
    # The engine's view class lives in the engine, which the solver API does not
    # import at runtime.
    type _EngineView = object
    # The resolved replay inputs are defined by the model, which imports the solver
    # API.
    type _ConsumedViews = object


@beartype(conf=SOLVER_API_CONF)
@dataclass(frozen=True, kw_only=True)
class SolutionResult:
    """Labelled value functions, retained artifacts, and omission records."""

    values: _SolutionValuesInput
    """Value function of every solved cell, keyed by period then regime."""
    metadata: SolutionMetadata
    """Identity, retention, and schema facts of the solve."""
    retained_continuations: ArtifactStore = field(default_factory=ArtifactStore)
    """Continuation payloads kept for persistence, addressed by cell and key."""
    replay_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    """Payloads simulation replays decisions from, addressed by cell and key."""
    auxiliary_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    """Additional solver-published payloads, addressed by cell and key."""
    omissions: _SolutionOmissionsInput = MappingProxyType({})
    """Why each accounted-for artifact that is absent was left out."""
    diagnostics: ArtifactStore = field(default_factory=ArtifactStore)
    """Solver diagnostics kept according to the solve's log level."""
    _artifact_authority: MappingProxyType[ArtifactRef, ArtifactAuthority] = field(
        default=MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )
    _engine_view: _EngineView | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    """The producing model's by-reference view of this result, when the engine
    built it in this process; `None` for every other provenance and for any
    copy made through `dataclasses.replace`."""
    _consumed_views: _ConsumedViews = field(
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
            self.values if type(self.values) is ValueStore else ValueStore(self.values)
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
        period: int,
        regime: RegimeName,
    ) -> _FloatValueBoundary:
        """Return one value-function array by its explicit coordinates."""
        return self.values[period][regime]

    def save(self, *, path: Path) -> Path:
        """Persist this complete result atomically to a versioned archive."""
        from lcm.persistence import save_solution  # noqa: PLC0415

        return save_solution(solution=self, path=path)

    def value_frame(
        self, *, period: int, regime: RegimeName, use_labels: bool = True
    ) -> pd.DataFrame:
        """Return grid values in long form, with state columns followed by `V`.

        Columns follow the stored value schema's axis order, including a
        `stakeholder` column for collective values. With `use_labels=True`,
        discrete states use pandas categoricals matching simulation labels;
        otherwise they retain integer codes. Resolved process and period-specific
        grid nodes are those used by this solve.
        Axis names must be unique and must not use the value column name `V`.

        This explicit call materializes one row per grid point and stakeholder,
        so large grids produce large frames. For values between nodes, use
        `Model.lookup_policy`, which uses simulation's interpolation.
        """
        value = np.asarray(self.value(period=period, regime=regime))
        schema = self.metadata.value_schemas[period, regime]
        if "V" in schema.axis_names or len(set(schema.axis_names)) != len(
            schema.axis_names
        ):
            raise ValueError(
                "Value-frame axes must be unique and must not be named 'V'."
            )
        if len(schema.named_axes) != value.ndim:
            raise ValueError("This value schema does not retain grid coordinates.")
        if value.shape != schema.shape:
            raise ValueError("Value shape differs from its stored schema.")
        frame = pd.DataFrame(index=pd.RangeIndex(value.size))
        for index, axis in enumerate(schema.named_axes):
            repeats = int(np.prod(value.shape[index + 1 :], dtype=np.int64))
            tiles = int(np.prod(value.shape[:index], dtype=np.int64))
            coordinates = np.tile(np.repeat(axis.coordinates, repeats), tiles)
            frame[axis.name] = coordinates
            domain = schema.categorical_domains.get(axis.name)
            if use_labels and domain is not None:
                frame[axis.name] = pd.Categorical.from_codes(
                    pd.Index(domain.codes).get_indexer(pd.Index(coordinates)),
                    categories=pd.Index(domain.labels),
                    ordered=domain.ordered,
                )
        frame["V"] = value.reshape(-1).copy()
        return frame
