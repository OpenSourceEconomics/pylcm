"""Public, lightweight types for labelled solver artifacts and solutions.

This module begins pylcm's dependency-safe solver extension boundary. Its public
definitions cover result identity and retention without referring to engine-private
``_lcm`` types or concrete built-in solver payloads.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from jaxtyping import Float

from lcm.typing import FloatND, RegimeName

_SHA256_HEX_LENGTH = 64


class ResultRetention(StrEnum):
    """Artifacts a caller asks to keep after backward induction."""

    VALUES = "values"
    VALUES_AND_REPLAY = "values_and_replay"
    ALL_PERSISTABLE_ARTIFACTS = "all_persistable_artifacts"

    @property
    def retains_replay(self) -> bool:
        """Whether replay artifacts remain available after the solve."""
        return self is not ResultRetention.VALUES


@dataclass(frozen=True, order=True, kw_only=True)
class ArtifactKey:
    """Versioned identity of one artifact payload schema.

    ``type_id`` is a qualified, globally meaningful name such as
    ``"pylcm.simulation.policy"`` or ``"example_solver.euler_residuals"``.
    Changing a payload's interpretation requires a new ``schema_version``.
    """

    type_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, str):
            raise TypeError("ArtifactKey.type_id must be a str.")
        if not self.type_id:
            raise ValueError("ArtifactKey.type_id must not be empty.")
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError("ArtifactKey.schema_version must be an int.")
        if self.schema_version < 1:
            raise ValueError("ArtifactKey.schema_version must be at least 1.")


# Built-in identities live on the public transport boundary. The engine imports
# these same singleton objects rather than defining private lookalikes, so an
# installed solver and pylcm always address the same schema.
EGM_CONTINUATION = ArtifactKey(type_id="pylcm.egm.continuation", schema_version=1)
SIMULATION_POLICY = ArtifactKey(type_id="pylcm.simulation.policy", schema_version=1)
DISSOLUTION_FLAG = ArtifactKey(
    type_id="pylcm.collective.dissolution_flag", schema_version=1
)
SOLVER_DIAGNOSTICS = ArtifactKey(type_id="pylcm.solver.diagnostics", schema_version=1)


@dataclass(frozen=True, kw_only=True)
class KernelOutput:
    """One solver kernel's value and explicitly typed artifact channels.

    This is the dependency-safe producer envelope for solver extensions. Artifact
    identity is carried by :class:`ArtifactKey`, while the engine decides which
    declared artifacts it understands and consumes. The mappings are copied at
    construction and exposed as immutable views so a producer cannot mutate a
    published kernel result after returning it.

    Numerical diagnostics are intentionally not a field of this initial public
    contract. Existing in-tree solvers that publish ``SolverDiagnostics`` continue
    to use the legacy engine-private result while that artifact schema is designed.
    """

    value: FloatND | Float[np.ndarray, "*shape"]
    """The regime's value-function array on its exogenous state grid."""

    continuations: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Cross-period artifacts required while backward induction is running."""

    solve_time_artifacts: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Other artifacts consumed by the solve before the period rolls."""

    replay: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Artifacts a later simulation or policy replay may consume."""

    auxiliary: Mapping[ArtifactKey, object] = field(default_factory=dict)
    """Optional, solver-defined artifacts for inspection or persistence."""

    def __post_init__(self) -> None:
        if not hasattr(self.value, "shape") or not hasattr(self.value, "dtype"):
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf."
            )
        try:
            value_dtype = np.dtype(self.value.dtype)
        except TypeError as error:
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf."
            ) from error
        if not np.issubdtype(value_dtype, np.floating):
            raise TypeError(
                "KernelOutput.value must be one floating JAX or NumPy array leaf; "
                f"got dtype {value_dtype}."
            )

        key_to_channel: dict[ArtifactKey, str] = {}
        for field_name in (
            "continuations",
            "solve_time_artifacts",
            "replay",
            "auxiliary",
        ):
            entries = getattr(self, field_name)
            if not all(isinstance(key, ArtifactKey) for key in entries):
                raise TypeError(f"KernelOutput.{field_name} keys must be ArtifactKey.")
            for key in entries:
                if previous_channel := key_to_channel.get(key):
                    raise ValueError(
                        f"Artifact '{key.type_id}' version "
                        f"{key.schema_version} appears "
                        f"in both KernelOutput.{previous_channel} and "
                        f"KernelOutput.{field_name}; one artifact identity must belong "
                        "to exactly one semantic channel."
                    )
                key_to_channel[key] = field_name
            object.__setattr__(self, field_name, MappingProxyType(dict(entries)))


@dataclass(frozen=True, order=True, kw_only=True)
class ArtifactRef:
    """Address of one artifact in a regime-period solution cell."""

    period: int
    regime: RegimeName
    key: ArtifactKey

    def __post_init__(self) -> None:
        if self.period < 0:
            raise ValueError("ArtifactRef.period must be non-negative.")
        if not self.regime:
            raise ValueError("ArtifactRef.regime must not be empty.")


@dataclass(frozen=True, eq=False)
class ArtifactStore(Mapping[ArtifactRef, object]):
    """Immutable store of explicitly addressed solution artifacts.

    The mapping interface keeps artifacts solver-extensible. ``project`` is the
    compatibility adapter for consumers that need pylcm's existing nested
    ``period -> regime -> payload`` representation for one known artifact key.
    """

    _entries: Mapping[ArtifactRef, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_entries", MappingProxyType(dict(self._entries)))

    def __getitem__(self, ref: ArtifactRef) -> object:
        return self._entries[ref]

    def __iter__(self) -> Iterator[ArtifactRef]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def project(self, key: ArtifactKey) -> Mapping[int, Mapping[RegimeName, object]]:
        """Project one artifact schema to an immutable nested period mapping."""
        projected: dict[int, dict[RegimeName, object]] = {}
        for ref, payload in self._entries.items():
            if ref.key == key:
                projected.setdefault(ref.period, {})[ref.regime] = payload
        return MappingProxyType(
            {
                period: MappingProxyType(regime_to_payload)
                for period, regime_to_payload in sorted(projected.items())
            }
        )


class OmissionReason(StrEnum):
    """Why an otherwise identifiable solution artifact is absent."""

    NOT_APPLICABLE = "not_applicable"
    NOT_REQUESTED = "not_requested"
    UNSUPPORTED = "unsupported"
    NOT_PERSISTED = "not_persisted"


@dataclass(frozen=True, order=True, kw_only=True)
class ValueArraySchema:
    """Logical identity of one stored value-function array.

    ``axis_names`` names the canonical state axes in order and, for a
    collective regime, the trailing ``"stakeholder"`` axis. The schema is
    deliberately lightweight: it describes an in-memory result array without
    importing any engine-private grid or layout type.
    """

    shape: tuple[int, ...]
    dtype: str
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(size < 0 for size in self.shape):
            raise ValueError("ValueArraySchema.shape entries must be non-negative.")
        if not self.dtype:
            raise ValueError("ValueArraySchema.dtype must not be empty.")
        if len(self.axis_names) != len(self.shape):
            raise ValueError(
                "ValueArraySchema.axis_names must name every array dimension."
            )


@dataclass(frozen=True, kw_only=True)
class SolutionMetadata:
    """In-memory identity and retention facts for one solve.

    ``model_instance_id`` intentionally binds a result to one ``Model``
    instance (including a pickle round trip); it is not a durable model
    fingerprint. ``params_fingerprint`` binds the result to the canonical flat
    parameter values used by that solve.
    """

    retention: ResultRetention
    n_periods: int
    regime_names: tuple[RegimeName, ...]
    solver_types: Mapping[RegimeName, str]
    model_instance_id: str
    params_fingerprint: str
    value_schemas: Mapping[tuple[int, RegimeName], ValueArraySchema]
    solver_api_version: int = 1
    solution_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.n_periods < 1:
            raise ValueError("SolutionMetadata.n_periods must be positive.")
        if self.solution_schema_version < 1:
            raise ValueError(
                "SolutionMetadata.solution_schema_version must be at least 1."
            )
        if self.solver_api_version < 1:
            raise ValueError("SolutionMetadata.solver_api_version must be at least 1.")
        if set(self.solver_types) != set(self.regime_names):
            raise ValueError(
                "SolutionMetadata.solver_types must cover exactly regime_names."
            )
        if not self.model_instance_id:
            raise ValueError("SolutionMetadata.model_instance_id must not be empty.")
        if len(self.params_fingerprint) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in self.params_fingerprint
        ):
            raise ValueError(
                "SolutionMetadata.params_fingerprint must be a lowercase SHA-256 "
                "hex digest."
            )
        if any(
            period < 0 or regime not in self.regime_names
            for period, regime in self.value_schemas
        ):
            raise ValueError(
                "SolutionMetadata.value_schemas contains an invalid coordinate."
            )
        object.__setattr__(
            self, "solver_types", MappingProxyType(dict(self.solver_types))
        )
        object.__setattr__(
            self, "value_schemas", MappingProxyType(dict(self.value_schemas))
        )


@dataclass(frozen=True, kw_only=True)
class SolutionResult:
    """Labelled value functions, retained artifacts, and omission records."""

    values: Mapping[int, Mapping[RegimeName, FloatND]]
    metadata: SolutionMetadata
    retained_continuations: ArtifactStore = field(default_factory=ArtifactStore)
    replay_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    auxiliary_artifacts: ArtifactStore = field(default_factory=ArtifactStore)
    omissions: Mapping[ArtifactRef, OmissionReason] = field(default_factory=dict)
    diagnostics: ArtifactStore = field(default_factory=ArtifactStore)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                {
                    period: MappingProxyType(dict(regime_to_value))
                    for period, regime_to_value in self.values.items()
                }
            ),
        )
        object.__setattr__(self, "omissions", MappingProxyType(dict(self.omissions)))

    def value(self, *, period: int, regime: RegimeName) -> FloatND:
        """Return one value-function array by its explicit coordinates."""
        return self.values[period][regime]


__all__ = [
    "DISSOLUTION_FLAG",
    "EGM_CONTINUATION",
    "SIMULATION_POLICY",
    "SOLVER_DIAGNOSTICS",
    "ArtifactKey",
    "ArtifactRef",
    "ArtifactStore",
    "KernelOutput",
    "OmissionReason",
    "ResultRetention",
    "SolutionMetadata",
    "SolutionResult",
    "ValueArraySchema",
]
