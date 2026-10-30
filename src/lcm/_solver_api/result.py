"""Solution metadata, artifact-contract comparison, and the labelled solution result."""

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
import xarray as xr
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
    from lcm._solver_api.entries import _LazyEntry
    from lcm.model import _ResolvedSolution

    type _SolutionValuesInput = Mapping[int, Mapping[RegimeName, FloatND]] | ValueStore
    type _SolutionOmissionsInput = Mapping[ArtifactRef, OmissionReason]
    type _EngineView = OwnedSolutionView
    # Engine replay inputs a consuming model resolved from this result, keyed by
    # that model's instance and the parameter fingerprint it consumed under. The
    # model fills this memo as it consumes the result, so it is a mutable dict.
    type _ConsumedViews = dict[tuple[str, str], _ResolvedSolution]
    # Stored arrival surfaces by period, publishing regime and leaf name: arrays
    # for a solve in this process, lazy archive entries for a loaded result.
    type _ArrivalSurfaces = Mapping[
        int, Mapping[RegimeName, Mapping[str, FloatND | _LazyEntry]]
    ]
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
    # Lazy archive entries validate on explicit materialization, as values do.
    type _ArrivalSurfaces = object


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
    _arrival_surfaces: _ArrivalSurfaces = field(
        default=MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )
    """Stored arrival surfaces arrival replay reads, persisted beside the values;
    empty for a model without nested arrivals and for any `dataclasses.replace`
    copy."""

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

    def value_array(
        self,
        *,
        period: int,
        regime: RegimeName,
        use_categorical_codes: bool = False,
    ) -> xr.DataArray:
        """Return one value-function array with its state coordinates, named `V`.

        Dimensions follow the stored value schema's axis order, including a
        trailing `stakeholder` dimension for collective values; a regime without
        states yields a zero-dimensional array. Coordinates are the resolved
        process and period-specific grid nodes used by this solve. By default, a
        category-class state carries its labels as an (ordered) categorical index
        in declaration order, matching simulation labels, and a state declared
        with `values` carries those numbers, so `.sel(experience=4)` picks a node
        by its value. With `use_categorical_codes=True` every discrete state
        carries its integer codes. Axis names must be unique and must not be `V`.

        Coordinates, like every other boundary (initial conditions, params,
        `to_dataframe`, `lookup_policy`), use a value grid's values by default.
        `Model.simulate(shock_paths=...)` is the exception: it takes category
        codes even for a value grid, because a code names its node exactly and
        so replays a path exactly and losslessly. A code maps to its value as
        `grid.values[code]`, and values map to codes with `grid.encode`:

        ```python
        grid = DiscreteGrid(values=jnp.array([0, 4, 10]))
        grid.values[2]  # 10
        ```

        The array wraps the stored value as is, so a device-resident value stays
        on its device until it is converted. Select coordinates with `.sel`
        before converting to a table to keep the conversion small.
        """
        value = self.value(period=period, regime=regime)
        schema = self.metadata.value_schemas[period, regime]
        if "V" in schema.axis_names or len(set(schema.axis_names)) != len(
            schema.axis_names
        ):
            raise ValueError("Value axes must be unique and must not be named 'V'.")
        if len(schema.named_axes) != value.ndim:
            raise ValueError("This value schema does not retain grid coordinates.")
        if value.shape != schema.shape:
            raise ValueError("Value shape differs from its stored schema.")
        coordinates: dict[str, pd.Index] = {}
        for axis in schema.named_axes:
            nodes = pd.Index(np.asarray(axis.coordinates))
            domain = schema.categorical_domains.get(axis.name)
            if not use_categorical_codes and domain is not None:
                positions = pd.Index(domain.codes).get_indexer(nodes)
                values = _domain_values(domain.labels)
                nodes = (
                    pd.CategoricalIndex(
                        pd.Categorical.from_codes(
                            positions,
                            categories=pd.Index(domain.labels),
                            ordered=domain.ordered,
                        )
                    )
                    if values is None
                    else pd.Index(np.asarray(values)[positions])
                )
            coordinates[axis.name] = nodes
        return xr.DataArray(value, dims=schema.axis_names, coords=coordinates, name="V")

    def value_frame(
        self,
        *,
        period: int,
        regime: RegimeName,
        use_categorical_codes: bool = False,
    ) -> pd.DataFrame:
        """Return grid values in long form, with state columns followed by `V`.

        The frame is `value_array(...)` in long form: one column per dimension,
        in the same order and with the same coordinates and categorical labels,
        followed by `V`.

        This explicit call materializes one row per grid point and stakeholder,
        so large grids produce large frames; select a slice of `value_array`
        first to convert only part of it. For values between nodes, use
        `Model.lookup_policy`, which uses simulation's interpolation.
        """
        array = self.value_array(
            period=period,
            regime=regime,
            use_categorical_codes=use_categorical_codes,
        )
        if not array.dims:
            return pd.DataFrame({"V": np.asarray(array).reshape(-1)})
        return array.to_dataframe().reset_index()


def _domain_values(labels: tuple[str, ...]) -> tuple[int | float, ...] | None:
    """Return the values of a value grid's categories, or `None` for labels.

    A value grid's categories are the `repr` of its numbers. A category class's
    are field names, which are identifiers and so never read as a number.
    """
    values = []
    for label in labels:
        try:
            value = ast.literal_eval(label)
        except ValueError, SyntaxError:
            return None
        if type(value) not in (int, float):
            return None
        values.append(value)
    return tuple(values)
