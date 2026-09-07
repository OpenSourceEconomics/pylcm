"""Solution metadata, addressed artifact references, and exact artifact-contract
comparison."""

import dataclasses
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType
from typing import (
    cast,
)

from lcm._solver_api.descriptors import (
    ArtifactDescriptor,
)
from lcm._solver_api.identity import (
    _SHA256_HEX_LENGTH,
    PYLCM_VERSION,
    SOLUTION_SCHEMA_VERSION,
    SOLVER_API_VERSION,
    ArtifactChannel,
    ArtifactKey,
    AxisAuthority,
    AxisDescriptor,
    AxisRole,
    CategoryDomain,
    LeafAuthority,
    LeafDescriptor,
    LoadState,
    PersistencePolicy,
    ReplayRouteIdentity,
    ResultRetention,
    SolutionSource,
    SolverIdentity,
)
from lcm.typing import RegimeName

_TRUSTED_ARTIFACT_STATIC_DATACLASSES: list[tuple[type[object], tuple[str, ...]]] = []
_INERT_ARTIFACT_STATIC_SCALAR_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
)


def _artifact_static_metadata_field_names(
    cls: type[object],
) -> tuple[str, ...] | None:
    """Look up a trusted class by identity without invoking metaclass equality."""
    for registered, field_names in _TRUSTED_ARTIFACT_STATIC_DATACLASSES:
        if cls is registered:
            return field_names
    return None


def _register_artifact_static_metadata_dataclass(
    *, cls: type[object], field_names: tuple[str, ...]
) -> None:
    """Register one engine-owned frozen dataclass for inert metadata snapshots."""
    params = getattr(cls, "__dataclass_params__", None)
    if not dataclasses.is_dataclass(cls) or params is None or not params.frozen:
        raise TypeError(
            "Artifact static metadata registrations must be frozen dataclasses."
        )
    if type(field_names) is not tuple or any(
        type(name) is not str or not name for name in field_names
    ):
        raise TypeError("Artifact static metadata field names must be exact strings.")
    actual_fields = tuple(field.name for field in dataclasses.fields(cls))
    if actual_fields != field_names:
        raise ValueError(
            "Artifact static metadata registration fields differ from class."
        )
    previous = _artifact_static_metadata_field_names(cls)
    if previous is not None and previous != field_names:
        raise ValueError(
            "Artifact static metadata class has conflicting registrations."
        )
    if previous is None:
        _TRUSTED_ARTIFACT_STATIC_DATACLASSES.append((cls, field_names))


def _snapshot_inert_pytree_metadata(  # noqa: C901
    *, value: object, active_ids: set[int] | None = None
) -> object:
    """Own one closed-grammar static value without reducers or user constructors."""
    value_type = type(value)
    if any(value_type is allowed for allowed in _INERT_ARTIFACT_STATIC_SCALAR_TYPES):
        return value
    if value_type is Fraction:
        fraction = cast("Fraction", value)
        numerator = fraction.numerator
        denominator = fraction.denominator
        if (
            type(numerator) is not int
            or type(denominator) is not int
            or denominator == 0
        ):
            raise TypeError("Artifact PyTree Fraction metadata is non-canonical.")
        canonical = Fraction(numerator, denominator)
        if canonical.numerator != numerator or canonical.denominator != denominator:
            raise TypeError(
                "Artifact PyTree Fraction metadata must already be normalized."
            )
        return canonical

    field_names = _artifact_static_metadata_field_names(value_type)
    if value_type is not tuple and value_type is not frozenset and field_names is None:
        raise TypeError(
            "Artifact PyTree static metadata has unsupported exact type "
            f"{value_type.__name__!r}."
        )
    if active_ids is None:
        active_ids = set()
    marker = id(value)
    if marker in active_ids:
        raise TypeError("Artifact PyTree static metadata must be acyclic.")
    active_ids.add(marker)
    try:
        if value_type is tuple:
            return tuple(
                _snapshot_inert_pytree_metadata(value=item, active_ids=active_ids)
                for item in cast("tuple[object, ...]", value)
            )
        if value_type is frozenset:
            return frozenset(
                _snapshot_inert_pytree_metadata(value=item, active_ids=active_ids)
                for item in cast("frozenset[object]", value)
            )
        if field_names is None:
            raise TypeError("Artifact static metadata registration disappeared.")
        owned = object.__new__(value_type)
        for name in field_names:
            object.__setattr__(
                owned,
                name,
                _snapshot_inert_pytree_metadata(
                    value=object.__getattribute__(value, name),
                    active_ids=active_ids,
                ),
            )
        return owned
    finally:
        active_ids.remove(marker)


def _same_inert_pytree_metadata(  # noqa: C901, PLR0911
    *, actual: object, expected: object
) -> bool:
    """Compare validated static values with exact types and no custom equality."""
    actual_type = type(actual)
    if actual_type is not type(expected):
        return False
    if actual_type is float:
        return struct.pack("!d", cast("float", actual)) == struct.pack(
            "!d", cast("float", expected)
        )
    if actual_type is complex:
        actual_complex = cast("complex", actual)
        expected_complex = cast("complex", expected)
        return struct.pack(
            "!dd", actual_complex.real, actual_complex.imag
        ) == struct.pack("!dd", expected_complex.real, expected_complex.imag)
    if any(actual_type is allowed for allowed in (type(None), bool, int, str, bytes)):
        return bool(actual == expected)
    if actual_type is Fraction:
        actual_fraction = cast("Fraction", actual)
        expected_fraction = cast("Fraction", expected)
        return (
            actual_fraction.numerator == expected_fraction.numerator
            and actual_fraction.denominator == expected_fraction.denominator
        )
    if actual_type is tuple:
        actual_tuple = cast("tuple[object, ...]", actual)
        expected_tuple = cast("tuple[object, ...]", expected)
        return len(actual_tuple) == len(expected_tuple) and all(
            _same_inert_pytree_metadata(
                actual=actual_item,
                expected=expected_item,
            )
            for actual_item, expected_item in zip(
                actual_tuple, expected_tuple, strict=True
            )
        )
    if actual_type is frozenset:
        unmatched = list(cast("frozenset[object]", expected))
        for actual_item in cast("frozenset[object]", actual):
            for index, expected_item in enumerate(unmatched):
                if _same_inert_pytree_metadata(
                    actual=actual_item,
                    expected=expected_item,
                ):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    field_names = _artifact_static_metadata_field_names(actual_type)
    if field_names is None:
        raise TypeError("Artifact PyTree static metadata escaped validation.")
    return all(
        _same_inert_pytree_metadata(
            actual=object.__getattribute__(actual, name),
            expected=object.__getattribute__(expected, name),
        )
        for name in field_names
    )


class ReplayMode(StrEnum):
    """How a regime's simulation obtains each period's decision."""

    EXACT_REPLAY = "exact_replay"
    """The retained replay artifact names the decision the solve took;
    simulation replays it and never runs the grid argmax."""

    VALID_RECOMPUTATION = "valid_recomputation"
    """Simulation recomputes the decision on the grid, refined by a published
    read where the route declares one."""

    UNSUPPORTED = "unsupported"
    """The solve's decision cannot be reproduced in simulation, so simulating
    the regime is refused."""


@dataclass(frozen=True, order=True, kw_only=True)
class ArtifactRef:
    """Address of one artifact in a regime-period solution cell."""

    period: int
    """Period of the solution cell."""
    regime: RegimeName
    """Name of the regime of the solution cell."""
    key: ArtifactKey
    """Versioned identity of the artifact schema."""

    def __post_init__(self) -> None:
        if type(self.period) is not int:
            raise TypeError("ArtifactRef.period must be an exact int.")
        if self.period < 0:
            raise ValueError("ArtifactRef.period must be non-negative.")
        if type(self.regime) is not str:
            raise TypeError("ArtifactRef.regime must be an exact str.")
        if not self.regime:
            raise ValueError("ArtifactRef.regime must not be empty.")
        if type(self.key) is not ArtifactKey:
            raise TypeError("ArtifactRef.key must be an exact ArtifactKey.")


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
    """Exact array shape."""
    dtype: str
    """NumPy dtype name."""
    axis_names: tuple[str, ...]
    """Canonical axis name behind each array dimension, in order."""

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

    ``model_fingerprint`` is the durable semantic identity used for restored
    results. ``model_instance_id`` remains a separate same-instance guard for
    in-memory results. ``params_fingerprint`` binds the result to the canonical
    solve parameters used by that solve.
    """

    retention: ResultRetention
    """The retention the solve was requested with."""
    n_periods: int
    """Number of periods in the model's lifecycle."""
    regime_names: tuple[RegimeName, ...]
    """Names of every regime, in model order."""
    solver_types: Mapping[RegimeName, str]
    """Qualified class name of each regime's solver."""
    model_instance_id: str
    """Token of the producing model instance; guards in-memory consumption."""
    params_fingerprint: str
    """Digest of the canonical parameters the solve depends on."""
    value_schemas: Mapping[tuple[int, RegimeName], ValueArraySchema]
    """Schema of every stored value array, keyed by period and regime."""
    model_fingerprint: str = "0" * _SHA256_HEX_LENGTH
    """Durable digest of the model's semantics, binding restored results."""
    solver_identities: Mapping[RegimeName, SolverIdentity] = field(default_factory=dict)
    """Package-owned identity of each regime's solver."""
    replay_routes: Mapping[RegimeName, ReplayRouteIdentity | None] = field(
        default_factory=dict
    )
    """Durable identity of each regime's replay route."""
    artifact_descriptors: Mapping[ArtifactRef, ArtifactDescriptor] = field(
        default_factory=dict
    )
    """Descriptor of every artifact the solve accounted for, present or omitted."""
    source: SolutionSource = SolutionSource.IN_MEMORY
    """Whether the result lives in the producing process or was restored."""
    pylcm_version: str = PYLCM_VERSION
    """The pylcm release that produced the result."""
    solver_api_version: int = SOLVER_API_VERSION
    """The public solver API version of that release."""
    solution_schema_version: int = SOLUTION_SCHEMA_VERSION
    """Version of the result container's schema."""

    def __post_init__(self) -> None:  # noqa: C901
        if self.n_periods < 1:
            raise ValueError("SolutionMetadata.n_periods must be positive.")
        if type(self.pylcm_version) is not str or not self.pylcm_version:
            raise ValueError("SolutionMetadata.pylcm_version must be a non-empty str.")
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
        if len(self.model_fingerprint) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in self.model_fingerprint
        ):
            raise ValueError(
                "SolutionMetadata.model_fingerprint must be a lowercase SHA-256 "
                "hex digest."
            )
        value_schemas = dict(self.value_schemas)
        if any(
            type(coordinate) is not tuple
            or len(coordinate) != 2  # noqa: PLR2004
            or type(coordinate[0]) is not int
            or coordinate[0] < 0
            or coordinate[0] >= self.n_periods
            or type(coordinate[1]) is not str
            or coordinate[1] not in self.regime_names
            for coordinate in value_schemas
        ):
            raise ValueError(
                "SolutionMetadata.value_schemas contains an invalid coordinate."
            )
        object.__setattr__(
            self, "solver_types", MappingProxyType(dict(self.solver_types))
        )
        object.__setattr__(self, "value_schemas", MappingProxyType(value_schemas))
        identities = dict(self.solver_identities) or {
            regime: SolverIdentity(
                plugin_id=solver_type,
                plugin_version="unversioned",
                solver_api_version=self.solver_api_version,
            )
            for regime, solver_type in self.solver_types.items()
        }
        if set(identities) != set(self.regime_names):
            raise ValueError(
                "SolutionMetadata.solver_identities must cover exactly regime_names."
            )
        routes = dict(self.replay_routes)
        if routes and set(routes) != set(self.regime_names):
            raise ValueError(
                "SolutionMetadata.replay_routes must cover exactly regime_names."
            )
        descriptors = dict(self.artifact_descriptors)
        if any(
            type(ref) is not ArtifactRef
            or ref.regime not in self.regime_names
            or ref.period < 0
            or ref.period >= self.n_periods
            or type(descriptor) is not ArtifactDescriptor
            or not _same_exact_artifact_contract(
                actual=ref.key,
                expected=descriptor.key,
            )
            for ref, descriptor in descriptors.items()
        ):
            raise ValueError(
                "SolutionMetadata.artifact_descriptors keys must address each "
                "descriptor's ArtifactKey."
            )
        object.__setattr__(self, "solver_identities", MappingProxyType(identities))
        object.__setattr__(self, "replay_routes", MappingProxyType(routes))
        object.__setattr__(self, "artifact_descriptors", MappingProxyType(descriptors))


_ARTIFACT_CONTRACT_ENUM_TYPES = (
    LoadState,
    ArtifactChannel,
    PersistencePolicy,
    SolutionSource,
    ResultRetention,
    AxisRole,
    ReplayMode,
    OmissionReason,
)
_ARTIFACT_CONTRACT_DATACLASS_FIELDS: tuple[
    tuple[type[object], tuple[str, ...]], ...
] = (
    (ArtifactKey, ("type_id", "schema_version")),
    (
        SolverIdentity,
        ("plugin_id", "plugin_version", "solver_api_version"),
    ),
    (ReplayRouteIdentity, ("route_id", "route_version")),
    (CategoryDomain, ("labels", "codes", "ordered")),
    (AxisDescriptor, ("name", "length", "role", "coordinates")),
    (AxisAuthority, ("name", "length", "role", "coordinates")),
    (LeafDescriptor, ("path", "shape", "dtype", "axis_names")),
    (
        LeafAuthority,
        ("path", "runtime_type", "shape", "dtype", "axis_names"),
    ),
    (
        ArtifactDescriptor,
        (
            "key",
            "channel",
            "persistence",
            "payload_type_id",
            "payload_version",
            "leaf_descriptors",
            "named_axes",
            "state_roles",
            "action_roles",
            "categorical_domains",
            "required_for",
            "required",
        ),
    ),
    (ArtifactRef, ("period", "regime", "key")),
    (ValueArraySchema, ("shape", "dtype", "axis_names")),
    (
        SolutionMetadata,
        (
            "retention",
            "n_periods",
            "regime_names",
            "solver_types",
            "model_instance_id",
            "params_fingerprint",
            "value_schemas",
            "model_fingerprint",
            "solver_identities",
            "replay_routes",
            "artifact_descriptors",
            "source",
            "pylcm_version",
            "solver_api_version",
            "solution_schema_version",
        ),
    ),
)


def _artifact_contract_dataclass_fields(
    cls: type[object],
) -> tuple[str, ...] | None:
    """Look up one contract wrapper by class identity."""
    for registered, field_names in _ARTIFACT_CONTRACT_DATACLASS_FIELDS:
        if cls is registered:
            return field_names
    return None


def _same_exact_artifact_contract(  # noqa: C901, PLR0911, PLR0912
    *,
    actual: object,
    expected: object,
    _active_pairs: set[tuple[int, int]] | None = None,
) -> bool:
    """Compare the closed artifact contract without weak or user-defined equality."""
    actual_type = type(actual)
    expected_type = type(expected)
    actual_is_mapping = actual_type is dict or actual_type is MappingProxyType
    expected_is_mapping = expected_type is dict or expected_type is MappingProxyType
    if actual_is_mapping is not expected_is_mapping:
        return False
    if not actual_is_mapping and actual_type is not expected_type:
        return False
    if actual_type is float:
        return struct.pack("!d", cast("float", actual)) == struct.pack(
            "!d", cast("float", expected)
        )
    if actual_type is complex:
        actual_complex = cast("complex", actual)
        expected_complex = cast("complex", expected)
        return struct.pack(
            "!dd", actual_complex.real, actual_complex.imag
        ) == struct.pack("!dd", expected_complex.real, expected_complex.imag)
    if any(
        actual_type is scalar_type
        for scalar_type in (type(None), bool, int, str, bytes)
    ):
        return bool(actual == expected)
    if any(actual_type is enum_type for enum_type in _ARTIFACT_CONTRACT_ENUM_TYPES):
        return actual is expected
    if isinstance(actual, type):
        return actual is expected
    if actual_type is Fraction:
        try:
            actual_fraction = _snapshot_inert_pytree_metadata(value=actual)
            expected_fraction = _snapshot_inert_pytree_metadata(value=expected)
        except TypeError:
            return False
        return _same_inert_pytree_metadata(
            actual=actual_fraction,
            expected=expected_fraction,
        )

    field_names = _artifact_contract_dataclass_fields(actual_type)
    if (
        not actual_is_mapping
        and actual_type is not tuple
        and actual_type is not frozenset
        and field_names is None
    ):
        return False
    if _active_pairs is None:
        _active_pairs = set()
    marker = (id(actual), id(expected))
    if marker in _active_pairs:
        return False
    _active_pairs.add(marker)
    try:
        if actual_is_mapping:
            actual_items = tuple(cast("Mapping[object, object]", actual).items())
            unmatched = list(cast("Mapping[object, object]", expected).items())
            if len(actual_items) != len(unmatched):
                return False
            for actual_key, actual_value in actual_items:
                for index, (expected_key, expected_value) in enumerate(unmatched):
                    if _same_exact_artifact_contract(
                        actual=actual_key,
                        expected=expected_key,
                        _active_pairs=_active_pairs,
                    ) and _same_exact_artifact_contract(
                        actual=actual_value,
                        expected=expected_value,
                        _active_pairs=_active_pairs,
                    ):
                        unmatched.pop(index)
                        break
                else:
                    return False
            return not unmatched
        if actual_type is tuple:
            actual_tuple = cast("tuple[object, ...]", actual)
            expected_tuple = cast("tuple[object, ...]", expected)
            return len(actual_tuple) == len(expected_tuple) and all(
                _same_exact_artifact_contract(
                    actual=actual_item,
                    expected=expected_item,
                    _active_pairs=_active_pairs,
                )
                for actual_item, expected_item in zip(
                    actual_tuple,
                    expected_tuple,
                    strict=True,
                )
            )
        if actual_type is frozenset:
            unmatched = list(cast("frozenset[object]", expected))
            for actual_item in cast("frozenset[object]", actual):
                for index, expected_item in enumerate(unmatched):
                    if _same_exact_artifact_contract(
                        actual=actual_item,
                        expected=expected_item,
                        _active_pairs=_active_pairs,
                    ):
                        unmatched.pop(index)
                        break
                else:
                    return False
            return not unmatched
        if field_names is None:
            return False
        return all(
            _same_exact_artifact_contract(
                actual=object.__getattribute__(actual, name),
                expected=object.__getattribute__(expected, name),
                _active_pairs=_active_pairs,
            )
            for name in field_names
        )
    finally:
        _active_pairs.remove(marker)
