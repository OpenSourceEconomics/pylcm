"""Versioned, atomic persistence for complete labelled solution results.

The archive contains JSON metadata and numerical HDF5 datasets only.  Every
value or artifact leaf has its own address and checksum; no model, callable,
plugin object, Python class, or executable code is serialized.
"""

# Archive parsing deliberately translates every low-level validation failure into the
# public persistence exceptions at its trust boundary.
# ruff: noqa: TRY301

import contextlib
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from beartype.roar import BeartypeCallHintViolation
from h5py import h5o  # ty: ignore[unresolved-import]
from numpy.typing import NDArray

from _lcm import version as _version
from _lcm.solution.result_snapshot import (
    snapshot_artifact_authorities,
    snapshot_artifact_store,
    snapshot_omissions,
    snapshot_solution_metadata,
    snapshot_value_store,
)
from lcm.exceptions import IncompatibleSolutionError, SolutionIntegrityError
from lcm.solver_api import (
    SOLUTION_FORMAT_VERSION,
    SOLUTION_SCHEMA_VERSION,
    SOLVER_API_VERSION,
    SOLVER_DIAGNOSTICS,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    AxisDescriptor,
    AxisRole,
    CategoryDomain,
    LeafDescriptor,
    LoadState,
    OmissionReason,
    PersistencePolicy,
    ReplayRouteIdentity,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    SolutionSource,
    SolverIdentity,
    ValueArraySchema,
    ValueStore,
    _CanonicalArtifactPayload,
    _CanonicalArtifactTemplate,
    _canonicalize_artifact_payload_snapshot,
    _copy_artifact_array_leaf,
    _LazyEntry,
    _normalize_jax_tree_path,
    _reconstruct_artifact_from_template_snapshot,
    _same_exact_artifact_contract,
    _snapshot_artifact_template_once,
)

if TYPE_CHECKING:
    _SolutionResultBoundary: TypeAlias = SolutionResult  # noqa: UP040
    _ValueStoreBoundary: TypeAlias = ValueStore  # noqa: UP040
    _ArtifactStoreBoundary: TypeAlias = ArtifactStore  # noqa: UP040
    _OmissionsBoundary: TypeAlias = MappingProxyType[  # noqa: UP040
        ArtifactRef, OmissionReason
    ]
    _AuthoritiesBoundary: TypeAlias = MappingProxyType[  # noqa: UP040
        ArtifactRef, ArtifactAuthority
    ]
    _StoreTupleBoundary: TypeAlias = tuple[  # noqa: UP040
        tuple[ArtifactChannel, _ArtifactStoreBoundary], ...
    ]
else:
    # Persistence bodies own exact-type and lazy-materialization checks. Runtime
    # annotation traversal must not inspect caller or decoder-controlled stores first.
    _SolutionResultBoundary = object
    _ValueStoreBoundary = object
    _ArtifactStoreBoundary = object
    _OmissionsBoundary = object
    _AuthoritiesBoundary = object
    _StoreTupleBoundary = object


_MANIFEST_DATASET: Final = "manifest"
_PAYLOAD_GROUP: Final = "payloads"
_UNLOADED: Final = object()
_PAYLOAD_ADDRESS_LENGTH: Final = 8
_SHA256_HEX_LENGTH: Final = 64
PYLCM_VERSION: Final = _version.__version__
_UNDESCRIBED_STANDARD_KEYS: Final = frozenset({SOLVER_DIAGNOSTICS})


def _require_exact_str(*, value: object, label: str) -> str:
    """Return one exact JSON string or raise an archive-integrity error."""
    if type(value) is not str:
        raise SolutionIntegrityError(f"Persisted {label} is not an exact string.")
    return value


def _require_nonempty_exact_str(*, value: object, label: str) -> str:
    """Return one nonempty exact JSON string."""
    result = _require_exact_str(value=value, label=label)
    if not result:
        raise SolutionIntegrityError(f"Persisted {label} is empty.")
    return result


def _require_exact_list(*, value: object, label: str) -> list[object]:
    """Return one exact JSON list."""
    if type(value) is not list:
        raise SolutionIntegrityError(f"Persisted {label} is not an exact list.")
    return cast("list[object]", value)


def _require_nonnegative_exact_int(*, value: object, label: str) -> int:
    """Return one nonnegative exact JSON integer."""
    if type(value) is not int or value < 0:
        raise SolutionIntegrityError(
            f"Persisted {label} is not a nonnegative exact integer."
        )
    return value


def _require_positive_exact_int(*, value: object, label: str) -> int:
    """Return one positive exact JSON integer."""
    result = _require_nonnegative_exact_int(value=value, label=label)
    if result == 0:
        raise SolutionIntegrityError(f"Persisted {label} is not positive.")
    return result


def _require_exact_bool(*, value: object, label: str) -> bool:
    """Return one exact JSON Boolean."""
    if type(value) is not bool:
        raise SolutionIntegrityError(f"Persisted {label} is not an exact Boolean.")
    return value


def _require_exact_json_scalar(
    *, value: object, label: str
) -> bool | int | float | str:
    """Return one finite exact JSON scalar without truthy-type coercion."""
    if not any(type(value) is allowed for allowed in (bool, int, float, str)):
        raise SolutionIntegrityError(f"Persisted {label} is not an exact JSON scalar.")
    if type(value) is float and not math.isfinite(value):
        raise SolutionIntegrityError(f"Persisted {label} is not finite.")
    return cast("bool | int | float | str", value)


def _require_exact_dict(*, value: object, label: str) -> dict[str, object]:
    """Return one exact JSON object with exact string keys."""
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise SolutionIntegrityError(f"Persisted {label} is not an exact object.")
    return cast("dict[str, object]", value)


def _require_exact_str_mapping(*, value: object, label: str) -> dict[str, str]:
    """Return one exact JSON string-to-string mapping."""
    raw = _require_exact_dict(value=value, label=label)
    if any(type(item) is not str for item in raw.values()):
        raise SolutionIntegrityError(
            f"Persisted {label} does not contain exact string values."
        )
    return cast("dict[str, str]", raw)


def _exact_shape(*, value: object, label: str) -> tuple[int, ...]:
    """Return one exact nonnegative JSON shape."""
    raw = _require_exact_list(value=value, label=label)
    if any(type(size) is not int or size < 0 for size in raw):
        raise SolutionIntegrityError(
            f"Persisted {label} is not an exact nonnegative shape."
        )
    return tuple(cast("list[int]", raw))


def _require_numeric_dtype(*, value: object, label: str) -> str:
    """Return one exact numerical-or-Boolean NumPy dtype string."""
    dtype_string = _require_nonempty_exact_str(value=value, label=label)
    try:
        dtype = np.dtype(dtype_string)
    except (TypeError, ValueError) as error:
        raise SolutionIntegrityError(
            f"Persisted {label} is not a valid NumPy dtype."
        ) from error
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        raise SolutionIntegrityError(f"Persisted {label} is not numerical or Boolean.")
    return dtype_string


def _require_sha256(*, value: object, label: str) -> str:
    """Return one exact lowercase SHA-256 hex digest."""
    digest = _require_exact_str(value=value, label=label)
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SolutionIntegrityError(f"Persisted {label} is not a SHA-256 digest.")
    return digest


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build one JSON object while rejecting duplicate names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object field {key!r}")
        result[key] = value
    return result


@dataclass
class _EntryCache:
    """Mutable synchronization state hidden behind one frozen lazy handle."""

    value: object = _UNLOADED
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True, kw_only=True)
class _LoadedEntryPayload:
    """Private numerical state and optional callback-free reconstruction plan."""

    leaves: tuple[jax.Array, ...]
    template_snapshot: _CanonicalArtifactTemplate | None


@dataclass(frozen=True, kw_only=True)
class _PreparedPayload:
    """One payload copied into a stable numerical representation before writing."""

    identity: MappingProxyType[str, object]
    payload_kind: str
    leaf_paths: tuple[tuple[str, ...], ...]
    leaves: tuple[NDArray[np.generic], ...]


@dataclass(frozen=True, kw_only=True)
class _SaveSnapshot:
    """Complete immutable-by-ownership snapshot consumed by the archive writer."""

    metadata: MappingProxyType[str, object]
    values: tuple[_PreparedPayload, ...]
    artifacts: tuple[_PreparedPayload, ...]
    omissions: tuple[tuple[ArtifactRef, OmissionReason], ...]


@dataclass(frozen=True, kw_only=True)
class _SaveEnvelope:
    """Engine-owned copy of every caller field inspected during save preflight."""

    metadata: SolutionMetadata
    values: _ValueStoreBoundary
    stores: _StoreTupleBoundary
    omissions: _OmissionsBoundary
    authorities: _AuthoritiesBoundary


@dataclass(frozen=True, kw_only=True)
class _LazyHdf5Entry(_LazyEntry):
    """One independently addressed and checksummed archive payload."""

    path: Path
    address: str
    label: str
    payload_kind: str
    identity: MappingProxyType[str, object]
    leaves: tuple[MappingProxyType[str, object], ...]
    _cache: _EntryCache = field(default_factory=_EntryCache, repr=False, compare=False)

    @property
    def load_state(self) -> LoadState:
        """Return the current state without opening the archive."""
        return (
            LoadState.UNLOADED if self._cache.value is _UNLOADED else LoadState.LOADED
        )

    def materialize(self, *, template: object | None = None) -> object:
        """Load through the compatibility path that accepts a template object."""
        return self._materialize(template=template, template_snapshot=None)

    def materialize_from_template_snapshot(
        self,
        *,
        template_snapshot: object,
    ) -> object:
        """Load a PyTree directly from its model-authoritative cached declaration."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Lazy materialization requires an exact template snapshot.")
        return self._materialize(
            template=None,
            template_snapshot=template_snapshot,
        )

    def _materialize(  # noqa: C901
        self,
        *,
        template: object | None,
        template_snapshot: _CanonicalArtifactTemplate | None,
    ) -> object:
        """Cache private numerical state and return a fresh detached graph."""
        if template is not None and template_snapshot is not None:
            raise TypeError("Supply a template or a template snapshot, not both.")
        requested_snapshot = template_snapshot
        if (
            self.payload_kind != "array"
            and requested_snapshot is None
            and template is not None
        ):
            requested_snapshot, _containers = _snapshot_artifact_template_once(
                template=template,
                payload_runtime_type=(
                    jax.Array if isinstance(template, jax.Array) else type(template)
                ),
            )

        with self._cache.lock:
            cached = self._cache.value
            if cached is _UNLOADED:
                if self.payload_kind != "array" and requested_snapshot is None:
                    raise IncompatibleSolutionError(
                        f"Persisted {self.label} is a plugin-defined PyTree. Install "
                        "the matching plugin and replay it through Model.simulate(), "
                        "which supplies the model-authoritative template."
                    )
                arrays = _read_and_verify_leaves(
                    path=self.path,
                    label=self.label,
                    address=self.address,
                    identity=self.identity,
                    leaves=self.leaves,
                )
                if self.payload_kind == "array" and len(arrays) != 1:
                    raise SolutionIntegrityError(
                        f"Persisted {self.label} declares one array but contains "
                        f"{len(arrays)} leaves."
                    )
                if self.payload_kind != "array":
                    if requested_snapshot is None:
                        raise TypeError(
                            "A PyTree cache requires a reconstruction plan."
                        )
                    self._validate_template_snapshot(
                        template_snapshot=requested_snapshot,
                        arrays=arrays,
                    )
                private_leaves = tuple(
                    _to_jax_without_narrowing(
                        array=array,
                        label=f"{self.label} leaf {index}",
                    )
                    for index, array in enumerate(arrays)
                )
                cached = _LoadedEntryPayload(
                    leaves=private_leaves,
                    template_snapshot=requested_snapshot,
                )
                self._cache.value = cached

        if type(cached) is not _LoadedEntryPayload:
            raise TypeError("Lazy payload cache contains unsupported state.")
        if self.payload_kind == "array":
            if cached.template_snapshot is not None or len(cached.leaves) != 1:
                raise TypeError("Lazy array cache is inconsistent.")
            return _copy_artifact_array_leaf(
                leaf=cached.leaves[0],
                label=self.label,
            )

        reconstruction_snapshot = (
            requested_snapshot
            if requested_snapshot is not None
            else cached.template_snapshot
        )
        if reconstruction_snapshot is None:
            raise IncompatibleSolutionError(
                f"Persisted {self.label} has no PyTree reconstruction plan."
            )
        self._validate_template_snapshot(
            template_snapshot=reconstruction_snapshot,
            arrays=cached.leaves,
        )
        fresh_leaves = tuple(
            _copy_artifact_array_leaf(
                leaf=leaf,
                label=f"{self.label} leaf {index}",
            )
            for index, leaf in enumerate(cached.leaves)
        )
        return _reconstruct_artifact_from_template_snapshot(
            template_snapshot=reconstruction_snapshot,
            leaves=fresh_leaves,
        )

    def _validate_template_snapshot(
        self,
        *,
        template_snapshot: _CanonicalArtifactTemplate,
        arrays: tuple[object, ...] | list[NDArray[np.generic]],
    ) -> None:
        """Validate one requested plan against persisted or privately cached leaves."""
        persisted_paths = tuple(
            cast("tuple[str, ...]", leaf["path"]) for leaf in self.leaves
        )
        if template_snapshot.leaf_paths != persisted_paths:
            raise IncompatibleSolutionError(
                f"Persisted {self.label} leaf paths differ from the installed "
                "route's model-authoritative template."
            )
        if len(arrays) != len(template_snapshot.leaves):
            raise IncompatibleSolutionError(
                f"Persisted {self.label} leaf count differs from the installed "
                "route's model-authoritative template."
            )
        for index, (array, expected) in enumerate(
            zip(arrays, template_snapshot.leaves, strict=True)
        ):
            if isinstance(array, jax.Array) and array.is_deleted():
                raise TypeError(f"Persisted {self.label} private leaf was deleted.")
            if tuple(getattr(array, "shape", ())) != tuple(expected.shape) or np.dtype(
                getattr(array, "dtype", None)
            ) != np.dtype(expected.dtype):
                raise IncompatibleSolutionError(
                    f"Persisted {self.label} leaf {index} differs from the installed "
                    "route's model-authoritative template."
                )

    def verify(self) -> None:
        """Verify every leaf without changing load state."""
        _read_and_verify_leaves(
            path=self.path,
            label=self.label,
            address=self.address,
            identity=self.identity,
            leaves=self.leaves,
        )


def save_solution_archive(*, solution: _SolutionResultBoundary, path: Path) -> Path:
    """Write a complete solution archive through an atomic sibling file."""
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

    # Materialize, validate, and copy the entire result before creating even a
    # temporary archive.  The writer therefore cannot observe a different user
    # container or mutable array from the object that passed preflight.
    snapshot = _prepare_solution_for_save(solution=solution)

    tmp_path: Path | None = None
    try:
        descriptor, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        tmp_path = Path(tmp_name)
        _write_archive(snapshot=snapshot, path=tmp_path)
        with tmp_path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())
        tmp_path.replace(path)
        directory_open_flag = getattr(os, "O_DIRECTORY", None)
        if directory_open_flag is not None:
            with contextlib.suppress(OSError):
                directory_fd = os.open(
                    path.parent,
                    os.O_RDONLY | directory_open_flag,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        tmp_path = None
        return path
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def load_solution_archive(  # noqa: C901, PLR0912, PLR0915
    *, path: Path, verify_checksums: bool
) -> _SolutionResultBoundary:
    """Read archive metadata and return a lazy complete solution result."""
    if not path.is_file():
        raise FileNotFoundError(path)
    path = path.resolve()
    try:
        archive_context = h5py.File(path, "r")
    except OSError as error:
        raise SolutionIntegrityError(
            f"Solution archive {path} is not a readable HDF5 file."
        ) from error
    with archive_context as archive:
        try:
            if set(archive) != {_MANIFEST_DATASET, _PAYLOAD_GROUP} or set(
                archive.attrs
            ):
                raise TypeError("archive root membership is not exact")
            manifest_link = archive.get(_MANIFEST_DATASET, getlink=True)
            if not isinstance(manifest_link, h5py.HardLink):
                raise TypeError("manifest is not a local dataset")
            manifest_dataset = archive.get(_MANIFEST_DATASET)
            if (
                not isinstance(manifest_dataset, h5py.Dataset)
                or manifest_dataset.ndim != 1
                or manifest_dataset.dtype != np.dtype(np.uint8)
                or manifest_dataset.is_virtual
                or manifest_dataset.external is not None
            ):
                raise TypeError("manifest is not a dataset")
            if set(manifest_dataset.attrs) != {"sha256"}:
                raise TypeError("manifest attributes are not exact")
            payload_link = archive.get(_PAYLOAD_GROUP, getlink=True)
            if not isinstance(payload_link, h5py.HardLink):
                raise TypeError("payloads is not a local group")
            payload_group = archive.get(_PAYLOAD_GROUP)
            if not isinstance(payload_group, h5py.Group):
                raise TypeError("payloads is not a local group")
            manifest_bytes = bytes(manifest_dataset[()])
            expected_manifest_checksum = _require_sha256(
                value=manifest_dataset.attrs["sha256"],
                label="manifest checksum",
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SolutionIntegrityError(
                f"Solution archive {path} has no valid manifest."
            ) from error
    actual_manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_checksum != expected_manifest_checksum:
        raise SolutionIntegrityError(
            f"Solution archive {path} manifest checksum does not match."
        )
    try:
        manifest = json.loads(
            manifest_bytes,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise SolutionIntegrityError(
            f"Solution archive {path} manifest is not valid JSON."
        ) from error
    if type(manifest) is not dict:
        raise SolutionIntegrityError(
            f"Solution archive {path} manifest must be a JSON object."
        )
    typed_manifest = cast("dict[str, object]", manifest)
    _check_archive_versions(manifest=typed_manifest)
    if set(typed_manifest) != {
        "format_version",
        "pylcm_version",
        "solver_api_version",
        "solution_schema_version",
        "metadata",
        "values",
        "artifacts",
        "omissions",
    }:
        raise SolutionIntegrityError(
            f"Solution archive {path} manifest has invalid top-level fields."
        )

    raw_metadata = typed_manifest.get("metadata")
    if type(raw_metadata) is not dict:
        raise SolutionIntegrityError(
            f"Solution archive {path} metadata is not an object."
        )
    metadata = _metadata_from_manifest(cast("dict[str, object]", raw_metadata))
    if metadata.pylcm_version != PYLCM_VERSION:
        raise IncompatibleSolutionError(
            "Solution archive metadata uses incompatible pylcm_version="
            f"{metadata.pylcm_version!r} (expected {PYLCM_VERSION!r})."
        )

    value_entries: dict[tuple[int, str], object] = {}
    addresses: set[str] = set()
    for raw_entry in _require_exact_list(
        value=typed_manifest.get("values"), label="value manifest"
    ):
        if type(raw_entry) is not dict:
            raise SolutionIntegrityError("Value manifest entry is not an object.")
        entry = cast("dict[str, object]", raw_entry)
        period = _require_nonnegative_exact_int(
            value=entry.get("period"), label="value period"
        )
        regime = _require_nonempty_exact_str(
            value=entry.get("regime"), label="value regime"
        )
        coordinate = (period, regime)
        if period >= metadata.n_periods or regime not in metadata.regime_names:
            raise SolutionIntegrityError(
                f"Persisted value {coordinate!r} has an invalid period or regime."
            )
        if coordinate in value_entries:
            raise SolutionIntegrityError(
                f"Solution archive contains duplicate value coordinate {coordinate!r}."
            )
        identity: dict[str, object] = {
            "kind": "value",
            "period": period,
            "regime": regime,
        }
        lazy = _lazy_entry(
            path=path,
            entry=entry,
            label=f"value at period={period}, regime={regime!r}",
            identity=identity,
        )
        schema = metadata.value_schemas.get(coordinate)
        if (
            schema is None
            or lazy.payload_kind != "array"
            or cast("tuple[int, ...]", lazy.leaves[0]["shape"]) != schema.shape
            or cast("str", lazy.leaves[0]["dtype"]) != np.dtype(schema.dtype).str
        ):
            raise SolutionIntegrityError(
                f"Persisted value at {coordinate!r} does not match its schema."
            )
        if lazy.address in addresses:
            raise SolutionIntegrityError(
                f"Solution archive reuses payload address {lazy.address!r}."
            )
        addresses.add(lazy.address)
        value_entries[coordinate] = lazy
    if set(value_entries) != set(metadata.value_schemas):
        raise SolutionIntegrityError(
            "Solution archive value entries do not match its value schemas."
        )
    values = ValueStore(cast("Mapping[object, object]", value_entries))

    stores: dict[ArtifactChannel, dict[ArtifactRef, object]] = {
        channel: {} for channel in ArtifactChannel
    }
    artifact_refs: set[ArtifactRef] = set()
    for raw_entry in _require_exact_list(
        value=typed_manifest.get("artifacts"), label="artifact manifest"
    ):
        if type(raw_entry) is not dict:
            raise SolutionIntegrityError("Artifact manifest entry is not an object.")
        entry = cast("dict[str, object]", raw_entry)
        ref = _artifact_ref_from_manifest(entry)
        _validate_persisted_ref(ref=ref, metadata=metadata, label="artifact")
        if ref in artifact_refs:
            raise SolutionIntegrityError(
                f"Solution archive contains duplicate artifact address {ref!r}."
            )
        artifact_refs.add(ref)
        try:
            channel = ArtifactChannel(
                _require_exact_str(value=entry.get("channel"), label="artifact channel")
            )
        except ValueError as error:
            raise SolutionIntegrityError(
                f"Artifact {ref!r} has an invalid channel."
            ) from error
        descriptor = metadata.artifact_descriptors.get(ref)
        if (
            descriptor is None
            or descriptor.channel is not channel
            or descriptor.persistence is not PersistencePolicy.MODEL_VERIFIABLE
        ):
            raise SolutionIntegrityError(
                f"Persisted artifact {ref!r} has no matching persistable descriptor."
            )
        if not _retention_keeps_present_artifact(
            retention=metadata.retention,
            descriptor=descriptor,
        ):
            raise SolutionIntegrityError(
                f"Persisted artifact {ref!r} is present although result retention "
                "does not select it."
            )
        identity = {
            "kind": "artifact",
            "period": ref.period,
            "regime": ref.regime,
            "type_id": ref.key.type_id,
            "schema_version": ref.key.schema_version,
            "channel": channel.value,
        }
        lazy = _lazy_entry(
            path=path,
            entry=entry,
            label=(
                f"artifact {ref.key.type_id!r} version {ref.key.schema_version} "
                f"at period={ref.period}, regime={ref.regime!r}"
            ),
            identity=identity,
        )
        actual_leaf_descriptors = tuple(
            (
                cast("tuple[str, ...]", leaf["path"]),
                cast("tuple[int, ...]", leaf["shape"]),
                np.dtype(cast("str", leaf["dtype"])),
            )
            for leaf in lazy.leaves
        )
        declared_leaf_descriptors = tuple(
            (leaf.path, leaf.shape, np.dtype(leaf.dtype))
            for leaf in descriptor.leaf_descriptors
        )
        if actual_leaf_descriptors != declared_leaf_descriptors:
            raise SolutionIntegrityError(
                f"Persisted artifact {ref!r} leaves do not match its descriptor."
            )
        if lazy.address in addresses:
            raise SolutionIntegrityError(
                f"Solution archive reuses payload address {lazy.address!r}."
            )
        addresses.add(lazy.address)
        stores[channel][ref] = lazy

    omission_entries: dict[ArtifactRef, OmissionReason] = {}
    for raw_entry in _require_exact_list(
        value=typed_manifest.get("omissions"), label="omission manifest"
    ):
        if type(raw_entry) is not dict:
            raise SolutionIntegrityError("Omission manifest entry is not an object.")
        entry = cast("dict[str, object]", raw_entry)
        if set(entry) != {
            "period",
            "regime",
            "type_id",
            "schema_version",
            "reason",
        }:
            raise SolutionIntegrityError("Omission manifest entry has invalid fields.")
        ref = _artifact_ref_from_manifest(entry)
        _validate_persisted_ref(ref=ref, metadata=metadata, label="omission")
        if ref in omission_entries:
            raise SolutionIntegrityError(
                f"Solution archive contains duplicate omission address {ref!r}."
            )
        if ref in artifact_refs:
            raise SolutionIntegrityError(
                f"Artifact {ref!r} is both present and explicitly omitted."
            )
        try:
            reason = OmissionReason(
                _require_exact_str(value=entry.get("reason"), label="omission reason")
            )
        except ValueError as error:
            raise SolutionIntegrityError(
                f"Omission {ref!r} has an invalid reason."
            ) from error
        descriptor = metadata.artifact_descriptors.get(ref)
        if descriptor is not None:
            _validate_persisted_omission_semantics(
                ref=ref,
                reason=reason,
                descriptor=descriptor,
                retention=metadata.retention,
            )
        omission_entries[ref] = reason
    descriptor_refs = set(metadata.artifact_descriptors)
    described_omissions = set(omission_entries) & descriptor_refs
    if artifact_refs | described_omissions != descriptor_refs:
        raise SolutionIntegrityError(
            "Solution archive artifact entries and omissions do not exactly cover "
            "its artifact descriptors."
        )
    undescribed_omissions = set(omission_entries) - descriptor_refs
    if any(ref.key not in _UNDESCRIBED_STANDARD_KEYS for ref in undescribed_omissions):
        raise SolutionIntegrityError(
            "Solution archive contains an omission with no artifact descriptor."
        )
    if any(
        omission_entries[ref] is not OmissionReason.NOT_PERSISTED
        for ref in undescribed_omissions
    ):
        raise SolutionIntegrityError(
            "An undescribed standard artifact must use the NOT_PERSISTED omission "
            "reason."
        )
    omissions = MappingProxyType(omission_entries)
    result = SolutionResult(
        values=values,
        metadata=replace(metadata, source=SolutionSource.PERSISTED),
        retained_continuations=ArtifactStore(stores[ArtifactChannel.CONTINUATION]),
        replay_artifacts=ArtifactStore(stores[ArtifactChannel.REPLAY]),
        auxiliary_artifacts=ArtifactStore(stores[ArtifactChannel.AUXILIARY]),
        diagnostics=ArtifactStore(stores[ArtifactChannel.DIAGNOSTIC]),
        omissions=omissions,
    )
    _validate_archive_structure(
        path=path,
        entries=tuple(cast("_LazyHdf5Entry", entry) for entry in value_entries.values())
        + tuple(
            cast("_LazyHdf5Entry", entry)
            for store in stores.values()
            for entry in store.values()
        ),
    )
    if verify_checksums:
        _verify_result_entries(result=result)
    return result


def _prepare_solution_for_save(  # noqa: C901, PLR0912, PLR0915
    *, solution: _SolutionResultBoundary
) -> _SaveSnapshot:
    """Validate and copy one complete result before the archive writer runs."""
    envelope = _snapshot_solution_for_save(solution=solution)
    metadata = envelope.metadata
    values = envelope.values
    stores = envelope.stores
    omissions = envelope.omissions
    authorities = envelope.authorities

    _require_save_compatibility(metadata=metadata)

    metadata_manifest = _metadata_to_manifest(metadata)
    # Reuse the hostile-input parser as a strict check of the in-memory
    # descriptive copy.  It rejects weakly typed dataclass fields before JSON
    # encoding has a chance to coerce them.
    try:
        _metadata_from_manifest(dict(metadata_manifest))
    except SolutionIntegrityError as error:
        raise IncompatibleSolutionError(
            "Solution descriptive metadata is malformed."
        ) from error

    value_coordinates = {
        (period, regime) for period in values for regime in values[period]
    }
    invalid_value_coordinates = tuple(
        sorted(
            (period, regime)
            for period, regime in value_coordinates
            if period >= metadata.n_periods or regime not in metadata.regime_names
        )
    )
    if invalid_value_coordinates:
        raise IncompatibleSolutionError(
            "Solution values contain invalid period or regime coordinates: "
            f"{invalid_value_coordinates!r}."
        )
    if value_coordinates != set(metadata.value_schemas):
        raise IncompatibleSolutionError(
            "Solution value entries do not match the declared value schemas."
        )
    prepared_values: list[_PreparedPayload] = []
    for period, regime in sorted(value_coordinates):
        schema = metadata.value_schemas[(period, regime)]
        identity: dict[str, object] = {
            "kind": "value",
            "period": period,
            "regime": regime,
        }
        prepared = _prepare_payload(
            payload=values[period][regime],
            identity=identity,
        )
        if (
            prepared.payload_kind != "array"
            or len(prepared.leaves) != 1
            or prepared.leaf_paths != ((),)
            or prepared.leaves[0].shape != schema.shape
            or prepared.leaves[0].dtype != np.dtype(schema.dtype)
        ):
            raise IncompatibleSolutionError(
                f"Value at {(period, regime)!r} does not match its declared schema."
            )
        prepared_values.append(prepared)

    descriptors = dict(metadata.artifact_descriptors)
    if set(authorities) != set(descriptors):
        raise IncompatibleSolutionError(
            "Solution artifact descriptors do not have exact model-issued authority "
            "coverage. Save the original SolutionResult returned by Model.solve()."
        )
    for ref, descriptor in descriptors.items():
        authority = authorities[ref]
        if type(authority) is not ArtifactAuthority or not (
            _same_exact_artifact_contract(
                actual=authority.descriptor,
                expected=descriptor,
            )
            or _is_model_derived_not_persisted_enrichment(
                durable=descriptor,
                authority=authority,
            )
        ):
            raise IncompatibleSolutionError(
                f"Descriptive metadata for artifact {ref!r} differs from its "
                "model-issued persistence authority."
            )

    omission_entries: dict[ArtifactRef, OmissionReason] = {}
    for ref, reason in omissions.items():
        _validate_result_ref(ref=ref, metadata=metadata, label="omission")
        if type(reason) is not OmissionReason:
            raise TypeError(f"Omission {ref!r} has a non-canonical reason {reason!r}.")
        descriptor = descriptors.get(ref)
        if (
            reason is OmissionReason.NOT_PERSISTED
            and descriptor is not None
            and descriptor.persistence is not PersistencePolicy.NOT_PERSISTED
        ):
            raise IncompatibleSolutionError(
                f"Omission {ref!r} claims NOT_PERSISTED although its descriptor is "
                "model-verifiable."
            )
        omission_entries[ref] = reason

    prepared_artifacts: list[_PreparedPayload] = []
    present_refs: set[ArtifactRef] = set()
    for channel, store in stores:
        for ref in sorted(store):
            _validate_result_ref(ref=ref, metadata=metadata, label="artifact")
            if ref in present_refs:
                raise IncompatibleSolutionError(
                    f"Artifact {ref!r} occurs in more than one channel."
                )
            if ref in omission_entries:
                raise IncompatibleSolutionError(
                    f"Artifact {ref!r} is both present and explicitly omitted."
                )
            present_refs.add(ref)
            descriptor = descriptors.get(ref)
            authority = authorities.get(ref)
            if descriptor is None or authority is None:
                if ref.key in _UNDESCRIBED_STANDARD_KEYS:
                    omission_entries[ref] = OmissionReason.NOT_PERSISTED
                    continue
                raise IncompatibleSolutionError(
                    f"Artifact {ref!r} has no model-issued persistence authority."
                )
            if descriptor.channel is not channel:
                raise IncompatibleSolutionError(
                    f"Artifact {ref!r} is stored on {channel.value!r}, but its "
                    f"descriptor declares {descriptor.channel.value!r}."
                )
            _validate_present_artifact_semantics(
                ref=ref,
                descriptor=descriptor,
                authority=authority,
                retention=metadata.retention,
            )
            if descriptor.persistence is PersistencePolicy.NOT_PERSISTED:
                omission_entries[ref] = OmissionReason.NOT_PERSISTED
                continue
            try:
                canonical = _canonicalize_artifact_payload_snapshot(
                    payload=store[ref],
                    authority=authority,
                )
                prepared = _prepare_canonical_artifact_payload(
                    canonical=canonical,
                    identity={
                        "kind": "artifact",
                        "period": ref.period,
                        "regime": ref.regime,
                        "type_id": ref.key.type_id,
                        "schema_version": ref.key.schema_version,
                        "channel": channel.value,
                    },
                )
            except IncompatibleSolutionError, SolutionIntegrityError:
                raise
            except Exception as error:
                raise IncompatibleSolutionError(
                    f"Artifact {ref!r} differs from its model-issued persistence "
                    f"authority: {error}"
                ) from error
            _check_prepared_artifact(
                prepared=prepared,
                ref=ref,
                descriptor=descriptor,
            )
            prepared_artifacts.append(prepared)

    described_present = {
        ref
        for ref in present_refs
        if ref in descriptors
        and descriptors[ref].persistence is PersistencePolicy.MODEL_VERIFIABLE
    }
    described_omitted = set(omission_entries) & set(descriptors)
    if described_present | described_omitted != set(descriptors):
        missing = sorted(set(descriptors) - described_present - described_omitted)
        raise IncompatibleSolutionError(
            "Every artifact descriptor must have exactly one present payload or "
            f"omission record; missing {missing!r}."
        )
    undescribed_omissions = set(omission_entries) - set(descriptors)
    if any(ref.key not in _UNDESCRIBED_STANDARD_KEYS for ref in undescribed_omissions):
        raise IncompatibleSolutionError(
            "An omission without an artifact descriptor is allowed only for a "
            "standard, non-persistable diagnostic."
        )
    if any(
        omission_entries[ref] is not OmissionReason.NOT_PERSISTED
        for ref in undescribed_omissions
    ):
        raise IncompatibleSolutionError(
            "An undescribed standard artifact must use the NOT_PERSISTED omission "
            "reason."
        )
    for ref in described_omitted:
        _validate_omission_semantics(
            ref=ref,
            reason=omission_entries[ref],
            descriptor=descriptors[ref],
            authority=authorities[ref],
            retention=metadata.retention,
        )

    return _SaveSnapshot(
        metadata=MappingProxyType(metadata_manifest),
        values=tuple(prepared_values),
        artifacts=tuple(prepared_artifacts),
        omissions=tuple(sorted(omission_entries.items())),
    )


def _snapshot_solution_for_save(  # noqa: C901
    *, solution: _SolutionResultBoundary
) -> _SaveEnvelope:
    """Copy the complete save envelope before any lazy entry can run user code."""
    if type(solution) is not SolutionResult:
        raise TypeError("save_solution requires an exact SolutionResult.")
    supplied_values = solution.values
    supplied_metadata = solution.metadata
    supplied_stores = _solution_stores(solution=solution)
    supplied_omissions = solution.omissions
    supplied_authorities = solution._artifact_authority  # noqa: SLF001

    if type(supplied_values) is not ValueStore:
        raise TypeError("SolutionResult.values must be an exact ValueStore.")
    if type(supplied_metadata) is not SolutionMetadata:
        raise TypeError("SolutionResult.metadata must be exact SolutionMetadata.")
    _require_save_compatibility(metadata=supplied_metadata)
    if type(supplied_values._entries) is not MappingProxyType:  # noqa: SLF001
        raise TypeError("SolutionResult value entries must be immutable.")
    for _channel, store in supplied_stores:
        if type(store) is not ArtifactStore:
            raise TypeError(
                "SolutionResult artifact stores must be exact ArtifactStore objects."
            )
        if type(store._entries) is not MappingProxyType:  # noqa: SLF001
            raise TypeError("SolutionResult artifact entries must be immutable.")
    if type(supplied_omissions) is not MappingProxyType:
        raise TypeError("SolutionResult omissions must be an immutable exact mapping.")
    if type(supplied_authorities) is not MappingProxyType:
        raise TypeError(
            "SolutionResult artifact authority must be an immutable exact mapping."
        )
    metadata_mappings = (
        supplied_metadata.solver_types,
        supplied_metadata.value_schemas,
        supplied_metadata.solver_identities,
        supplied_metadata.replay_routes,
        supplied_metadata.artifact_descriptors,
    )
    if any(type(mapping) is not MappingProxyType for mapping in metadata_mappings):
        raise TypeError("SolutionResult metadata mappings must be immutable and exact.")

    try:
        metadata = snapshot_solution_metadata(supplied_metadata)
        authorities = snapshot_artifact_authorities(supplied_authorities)
        values = snapshot_value_store(supplied_values)
        stores = tuple(
            (
                channel,
                snapshot_artifact_store(store=store, authorities=authorities),
            )
            for channel, store in supplied_stores
        )
        omissions = snapshot_omissions(supplied_omissions)
    except (BeartypeCallHintViolation, TypeError, ValueError) as error:
        raise IncompatibleSolutionError(
            "Solution save envelope cannot be copied into exact engine-owned "
            "containers."
        ) from error
    return _SaveEnvelope(
        metadata=metadata,
        values=values,
        stores=stores,
        omissions=omissions,
        authorities=authorities,
    )


def _require_save_compatibility(*, metadata: SolutionMetadata) -> None:
    """Reject incompatible versions before authority PyTree callbacks can run."""
    if (
        type(metadata.solver_api_version) is not int
        or metadata.solver_api_version != SOLVER_API_VERSION
    ):
        raise IncompatibleSolutionError(
            "Cannot save a solution with solver_api_version="
            f"{metadata.solver_api_version!r}; expected {SOLVER_API_VERSION}."
        )
    if (
        type(metadata.solution_schema_version) is not int
        or metadata.solution_schema_version != SOLUTION_SCHEMA_VERSION
    ):
        raise IncompatibleSolutionError(
            "Cannot save a solution with solution_schema_version="
            f"{metadata.solution_schema_version!r}; expected "
            f"{SOLUTION_SCHEMA_VERSION}."
        )
    if (
        type(metadata.pylcm_version) is not str
        or metadata.pylcm_version != PYLCM_VERSION
    ):
        raise IncompatibleSolutionError(
            f"Cannot save a solution with pylcm_version={metadata.pylcm_version!r}; "
            f"expected {PYLCM_VERSION!r}."
        )


def _is_model_derived_not_persisted_enrichment(
    *, durable: ArtifactDescriptor, authority: ArtifactAuthority
) -> bool:
    """Recognize the one safe asymmetry for solve-generated private authority.

    Adaptive replay binds its generated candidate coordinates and numerical leaf
    schema into the model's private authority after solving.  The durable descriptor
    deliberately remains reconstructible without those solve-side facts because the
    payload is never persisted.  Permit that enrichment while requiring every durable
    identity, role, route, category, and already-known axis to remain exact.  An
    ordinary descriptor mismatch, including one for a persistable artifact, still
    fails closed.
    """
    private = authority.descriptor
    if (
        durable.persistence is not PersistencePolicy.NOT_PERSISTED
        or private.persistence is not PersistencePolicy.NOT_PERSISTED
        or authority.template is None
        or durable.leaf_descriptors
        or not private.leaf_descriptors
    ):
        return False

    # Only the solve-derived leaves and axes may enrich the private descriptor.
    enriched = replace(
        private,
        leaf_descriptors=durable.leaf_descriptors,
        named_axes=durable.named_axes,
    )
    if not _same_exact_artifact_contract(
        actual=enriched,
        expected=durable,
    ):
        return False
    private_axes = {axis.name: axis for axis in private.named_axes}
    return all(
        _same_exact_artifact_contract(
            actual=private_axes.get(axis.name),
            expected=axis,
        )
        for axis in durable.named_axes
    )


def _prepare_payload(
    *, payload: object, identity: dict[str, object]
) -> _PreparedPayload:
    """Copy a numerical PyTree into independent contiguous NumPy arrays."""
    with_paths, tree = jax.tree_util.tree_flatten_with_path(payload)
    if not with_paths:
        raise TypeError(f"Persisted {identity!r} contains no numerical leaves.")
    arrays: list[NDArray[np.generic]] = []
    leaf_paths: list[tuple[str, ...]] = []
    for path, leaf in with_paths:
        leaf_paths.append(_normalize_jax_tree_path(path))
        array = np.array(np.asarray(leaf), copy=True, order="C", subok=False)
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(
                f"Persisted {identity!r} contains an object-dtype or other "
                "non-numerical leaf; only numerical or Boolean arrays and scalars "
                "are supported."
            )
        array.flags.writeable = False
        arrays.append(array)
    return _PreparedPayload(
        identity=MappingProxyType(dict(identity)),
        payload_kind=(
            "array" if tree.num_leaves == 1 and tuple(leaf_paths) == ((),) else "pytree"
        ),
        leaf_paths=tuple(leaf_paths),
        leaves=tuple(arrays),
    )


def _prepare_canonical_artifact_payload(
    *,
    canonical: _CanonicalArtifactPayload,
    identity: dict[str, object],
) -> _PreparedPayload:
    """Copy exactly the leaves returned by artifact canonicalization."""
    if not canonical.leaves:
        raise TypeError(f"Persisted {identity!r} contains no numerical leaves.")
    if len(canonical.leaf_paths) != len(canonical.leaves):
        raise TypeError("Canonical artifact paths and leaves differ in length.")
    if canonical.payload_kind not in {"array", "pytree"}:
        raise TypeError("Canonical artifact has an invalid payload kind.")

    arrays: list[NDArray[np.generic]] = []
    for leaf in canonical.leaves:
        array = np.array(np.asarray(leaf), copy=True, order="C", subok=False)
        if not (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        ):
            raise TypeError(
                f"Persisted {identity!r} contains an object-dtype or other "
                "non-numerical leaf; only numerical or Boolean arrays and scalars "
                "are supported."
            )
        array.flags.writeable = False
        arrays.append(array)
    return _PreparedPayload(
        identity=MappingProxyType(dict(identity)),
        payload_kind=canonical.payload_kind,
        leaf_paths=canonical.leaf_paths,
        leaves=tuple(arrays),
    )


def _check_prepared_artifact(
    *,
    prepared: _PreparedPayload,
    ref: ArtifactRef,
    descriptor: ArtifactDescriptor,
) -> None:
    """Ensure canonicalization did not alter any model-authoritative leaf dtype."""
    declared_leaves = descriptor.leaf_descriptors
    if len(declared_leaves) != len(prepared.leaves):
        raise IncompatibleSolutionError(
            f"Artifact {ref!r} does not match its declared leaf count."
        )
    for index, (path, array, declared) in enumerate(
        zip(
            prepared.leaf_paths,
            prepared.leaves,
            declared_leaves,
            strict=True,
        )
    ):
        if (
            path != declared.path
            or array.shape != declared.shape
            or array.dtype != np.dtype(declared.dtype)
        ):
            raise IncompatibleSolutionError(
                f"Artifact {ref!r} leaf {index} differs from its descriptor."
            )


def _validate_result_ref(
    *, ref: ArtifactRef, metadata: SolutionMetadata, label: str
) -> None:
    """Validate one in-memory artifact address without coercing weak types."""
    if (
        type(ref) is not ArtifactRef
        or type(ref.period) is not int
        or ref.period < 0
        or ref.period >= metadata.n_periods
        or type(ref.regime) is not str
        or ref.regime not in metadata.regime_names
    ):
        raise IncompatibleSolutionError(
            f"Solution {label} has an invalid period or regime address: {ref!r}."
        )


def _validate_omission_semantics(
    *,
    ref: ArtifactRef,
    reason: OmissionReason,
    descriptor: ArtifactDescriptor,
    authority: ArtifactAuthority,
    retention: ResultRetention,
) -> None:
    """Require an omission reason to agree with authority and retention."""
    selected = _retention_selects_artifact(
        retention=retention,
        descriptor=descriptor,
    )
    if not authority.applicable:
        expected = OmissionReason.NOT_APPLICABLE
    elif not selected:
        expected = OmissionReason.NOT_REQUESTED
    elif descriptor.persistence is PersistencePolicy.NOT_PERSISTED:
        expected = OmissionReason.NOT_PERSISTED
    elif authority.required:
        raise IncompatibleSolutionError(
            f"Required model-verifiable artifact {ref!r} is absent."
        )
    else:
        expected = OmissionReason.UNSUPPORTED
    if reason is not expected:
        raise IncompatibleSolutionError(
            f"Artifact {ref!r} has omission reason {reason.value!r}; expected "
            f"{expected.value!r} from model authority and retention."
        )


def _validate_present_artifact_semantics(
    *,
    ref: ArtifactRef,
    descriptor: ArtifactDescriptor,
    authority: ArtifactAuthority,
    retention: ResultRetention,
) -> None:
    """Require a present artifact to be both applicable and selected."""
    if not authority.applicable:
        raise IncompatibleSolutionError(
            f"Artifact {ref!r} is present although it is not applicable."
        )
    if not _retention_keeps_present_artifact(
        retention=retention,
        descriptor=descriptor,
    ):
        raise IncompatibleSolutionError(
            f"Artifact {ref!r} is present although result retention does not select it."
        )


def _retention_keeps_present_artifact(
    *, retention: ResultRetention, descriptor: ArtifactDescriptor
) -> bool:
    """Return whether one payload belongs in an in-memory retention ledger."""
    return (
        retention is ResultRetention.VALUES_AND_REPLAY
        and descriptor.channel is ArtifactChannel.REPLAY
    ) or (
        retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
        and descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
    )


def _retention_selects_artifact(
    *, retention: ResultRetention, descriptor: ArtifactDescriptor
) -> bool:
    """Return whether one descriptor's channel is selected for retention."""
    return retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS or (
        retention is ResultRetention.VALUES_AND_REPLAY
        and descriptor.channel is ArtifactChannel.REPLAY
    )


def _write_archive(*, snapshot: _SaveSnapshot, path: Path) -> None:
    """Write one preflighted complete archive to a path that is not yet public."""
    manifest: dict[str, object] = {
        "format_version": SOLUTION_FORMAT_VERSION,
        "pylcm_version": PYLCM_VERSION,
        "solver_api_version": SOLVER_API_VERSION,
        "solution_schema_version": SOLUTION_SCHEMA_VERSION,
        "metadata": dict(snapshot.metadata),
        "values": [],
        "artifacts": [],
        "omissions": [],
    }
    with h5py.File(path, "w") as archive:
        payloads = archive.create_group(_PAYLOAD_GROUP)
        counter = 0
        value_entries = cast("list[object]", manifest["values"])
        for prepared in snapshot.values:
            address = f"{counter:08d}"
            counter += 1
            value_entries.append(
                _write_payload_entry(
                    payloads=payloads,
                    address=address,
                    prepared=prepared,
                )
            )

        artifact_entries = cast("list[object]", manifest["artifacts"])
        for prepared in snapshot.artifacts:
            address = f"{counter:08d}"
            counter += 1
            artifact_entries.append(
                _write_payload_entry(
                    payloads=payloads,
                    address=address,
                    prepared=prepared,
                )
            )

        manifest["omissions"] = [
            {
                "period": ref.period,
                "regime": ref.regime,
                "type_id": ref.key.type_id,
                "schema_version": ref.key.schema_version,
                "reason": reason.value,
            }
            for ref, reason in snapshot.omissions
        ]
        manifest_bytes = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        manifest_dataset = archive.create_dataset(
            _MANIFEST_DATASET,
            data=np.frombuffer(manifest_bytes, dtype=np.uint8),
        )
        manifest_dataset.attrs["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        archive.flush()


def _write_payload_entry(
    *,
    payloads: h5py.Group,
    address: str,
    prepared: _PreparedPayload,
) -> dict[str, object]:
    """Write one preflighted payload and return its non-executable manifest entry."""
    identity = dict(prepared.identity)
    group = payloads.create_group(address)
    leaf_entries: list[dict[str, object]] = []
    for index, array in enumerate(prepared.leaves):
        dataset_name = f"leaf_{index:04d}"
        group.create_dataset(dataset_name, data=array)
        leaf_path = list(prepared.leaf_paths[index])
        leaf_identity = dict(identity) | {
            "address": address,
            "leaf": index,
            "path": leaf_path,
        }
        leaf_entries.append(
            {
                "dataset": f"{_PAYLOAD_GROUP}/{address}/{dataset_name}",
                "path": leaf_path,
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "identity": leaf_identity,
                "sha256": _array_checksum(identity=leaf_identity, array=array),
            }
        )
    return identity | {
        "address": address,
        "payload_kind": prepared.payload_kind,
        "leaves": leaf_entries,
    }


def _lazy_entry(  # noqa: C901, PLR0912
    *,
    path: Path,
    entry: dict[str, object],
    label: str,
    identity: dict[str, object],
) -> _LazyHdf5Entry:
    """Construct one checked lazy handle from manifest metadata."""
    try:
        if set(entry) != set(identity) | {"address", "payload_kind", "leaves"}:
            raise ValueError("invalid payload entry fields")
        if any(
            type(entry.get(name)) is not type(expected) or entry.get(name) != expected
            for name, expected in identity.items()
        ):
            raise ValueError("payload identity differs from its address")
        address = _require_exact_str(
            value=entry.get("address"), label=f"{label} address"
        )
        if (
            len(address) != _PAYLOAD_ADDRESS_LENGTH
            or not address.isascii()
            or not address.isdigit()
        ):
            raise ValueError("invalid address")
        payload_kind = _require_exact_str(
            value=entry.get("payload_kind"), label=f"{label} payload kind"
        )
        if payload_kind not in {"array", "pytree"}:
            raise ValueError("invalid payload kind")
        raw_leaves = _require_exact_list(
            value=entry.get("leaves"), label=f"{label} leaves"
        )
        leaves: list[MappingProxyType[str, object]] = []
        for index, raw_leaf in enumerate(raw_leaves):
            if type(raw_leaf) is not dict:
                raise TypeError("leaf is not an object")
            leaf = cast("dict[str, object]", raw_leaf)
            leaf_path = tuple(
                _require_nonempty_exact_str(
                    value=component, label=f"{label} leaf {index} path component"
                )
                for component in _require_exact_list(
                    value=cast("dict[str, object]", raw_leaf).get("path"),
                    label=f"{label} leaf {index} path",
                )
            )
            expected_leaf_identity = identity | {
                "address": address,
                "leaf": index,
                "path": list(leaf_path),
            }
            expected_dataset = f"{_PAYLOAD_GROUP}/{address}/leaf_{index:04d}"
            if set(leaf) != {
                "dataset",
                "path",
                "shape",
                "dtype",
                "identity",
                "sha256",
            }:
                raise ValueError("invalid leaf fields")
            if leaf.get("dataset") != expected_dataset:
                raise ValueError("leaf dataset does not match its address")
            raw_leaf_identity = _require_exact_dict(
                value=leaf.get("identity"), label=f"{label} leaf {index} identity"
            )
            if set(raw_leaf_identity) != set(expected_leaf_identity) or any(
                type(raw_leaf_identity[name]) is not type(expected)
                or raw_leaf_identity[name] != expected
                for name, expected in expected_leaf_identity.items()
            ):
                raise ValueError("leaf identity does not match its parent")
            shape = _require_exact_list(
                value=leaf.get("shape"), label=f"{label} leaf {index} shape"
            )
            if any(type(size) is not int or size < 0 for size in shape):
                raise TypeError("leaf shape is not exact")
            dtype = _require_numeric_dtype(
                value=leaf.get("dtype"), label=f"{label} leaf {index} dtype"
            )
            checksum = _require_sha256(
                value=leaf.get("sha256"), label=f"{label} leaf {index} checksum"
            )
            leaves.append(
                MappingProxyType(
                    {
                        "dataset": expected_dataset,
                        "path": leaf_path,
                        "shape": tuple(cast("list[int]", shape)),
                        "dtype": dtype,
                        "identity": MappingProxyType(expected_leaf_identity),
                        "sha256": checksum,
                    }
                )
            )
        if not leaves:
            raise ValueError("no leaves")
        if len({cast("tuple[str, ...]", leaf["path"]) for leaf in leaves}) != len(
            leaves
        ):
            raise ValueError("duplicate leaf path")
        if payload_kind == "array" and len(leaves) != 1:
            raise ValueError("array payload does not have exactly one leaf")
        if payload_kind == "array" and leaves[0]["path"] != ():
            raise ValueError("array payload leaf is not at the root path")
        if payload_kind == "pytree" and len(leaves) == 1 and leaves[0]["path"] == ():
            raise ValueError("single-root-leaf payload must use array kind")
        return _LazyHdf5Entry(
            path=path,
            address=address,
            label=label,
            payload_kind=payload_kind,
            identity=MappingProxyType(dict(identity)),
            leaves=tuple(leaves),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SolutionIntegrityError(
            f"Persisted {label} has invalid metadata."
        ) from error


def _read_and_verify_leaves(
    *,
    path: Path,
    label: str,
    address: str,
    identity: MappingProxyType[str, object],
    leaves: tuple[MappingProxyType[str, object], ...],
) -> tuple[NDArray[np.generic], ...]:
    """Read and verify all leaves of one independently addressed payload."""
    arrays: list[NDArray[np.generic]] = []
    try:
        archive_context = h5py.File(path, "r")
    except OSError as error:
        raise SolutionIntegrityError(
            f"Persisted {label} cannot reopen its solution archive."
        ) from error
    with archive_context as archive:
        try:
            if set(archive) != {_MANIFEST_DATASET, _PAYLOAD_GROUP} or set(
                archive.attrs
            ):
                raise TypeError("archive root membership is not exact")
            _require_local_dataset(
                parent=archive,
                name=_MANIFEST_DATASET,
                label="manifest",
                allowed_attributes=frozenset({"sha256"}),
            )
            payloads = _require_local_group(
                parent=archive,
                name=_PAYLOAD_GROUP,
                label="payload root",
            )
            group = _require_local_group(
                parent=payloads,
                name=address,
                label=f"{label} payload group",
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SolutionIntegrityError(
                f"Persisted {label} has an invalid HDF5 ancestor."
            ) from error
        expected_members = {f"leaf_{index:04d}" for index in range(len(leaves))}
        if set(group) != expected_members or set(group.attrs):
            raise SolutionIntegrityError(
                f"Persisted {label} payload group membership is invalid."
            )
        for index, leaf in enumerate(leaves):
            dataset_name = cast("str", leaf["dataset"])
            expected_dataset = f"{_PAYLOAD_GROUP}/{address}/leaf_{index:04d}"
            expected_identity = dict(identity) | {
                "address": address,
                "leaf": index,
                "path": list(cast("tuple[str, ...]", leaf["path"])),
            }
            if (
                dataset_name != expected_dataset
                or dict(cast("MappingProxyType[str, object]", leaf["identity"]))
                != expected_identity
            ):
                raise SolutionIntegrityError(
                    f"Persisted {label} leaf {index} has an invalid logical address."
                )
            try:
                dataset = _require_local_dataset(
                    parent=group,
                    name=f"leaf_{index:04d}",
                    label=f"{label} leaf {index}",
                )
                array = np.array(dataset[()], copy=True, order="C", subok=False)
            except (KeyError, OSError, TypeError, ValueError) as error:
                raise SolutionIntegrityError(
                    f"Persisted {label} leaf {index} cannot be read."
                ) from error
            expected_shape = cast("tuple[int, ...]", leaf["shape"])
            expected_dtype = cast("str", leaf["dtype"])
            if array.shape != expected_shape or array.dtype.str != expected_dtype:
                raise SolutionIntegrityError(
                    f"Persisted {label} leaf {index} shape or dtype does not match "
                    "its manifest."
                )
            actual = _array_checksum_from_leaf_metadata(leaf=leaf, array=array)
            if actual != str(leaf["sha256"]):
                raise SolutionIntegrityError(
                    f"Persisted {label} leaf {index} checksum does not match."
                )
            arrays.append(array)
    return tuple(arrays)


def _validate_archive_structure(  # noqa: C901
    *, path: Path, entries: tuple[_LazyHdf5Entry, ...]
) -> None:
    """Reject non-local links, aliases, and undeclared HDF5 objects."""
    by_address = {entry.address: entry for entry in entries}
    if len(by_address) != len(entries):
        raise SolutionIntegrityError("Solution archive reuses a payload address.")
    try:
        with h5py.File(path, "r") as archive:
            if set(archive) != {_MANIFEST_DATASET, _PAYLOAD_GROUP} or set(
                archive.attrs
            ):
                raise TypeError("archive root membership is not exact")
            manifest = _require_local_dataset(
                parent=archive,
                name=_MANIFEST_DATASET,
                label="manifest",
                allowed_attributes=frozenset({"sha256"}),
            )
            payloads = _require_local_group(
                parent=archive,
                name=_PAYLOAD_GROUP,
                label="payload root",
            )
            if set(payloads) != set(by_address) or set(payloads.attrs):
                raise TypeError("payload root membership is not exact")

            object_addresses = {_hdf5_object_address(manifest)}
            payload_root_address = _hdf5_object_address(payloads)
            if payload_root_address in object_addresses:
                raise TypeError("archive objects are hard-link aliases")
            object_addresses.add(payload_root_address)
            for address, entry in by_address.items():
                group = _require_local_group(
                    parent=payloads,
                    name=address,
                    label=f"payload group {address}",
                )
                group_address = _hdf5_object_address(group)
                if group_address in object_addresses:
                    raise TypeError("payload groups are hard-link aliases")
                object_addresses.add(group_address)
                expected_members = {
                    f"leaf_{index:04d}" for index in range(len(entry.leaves))
                }
                if set(group) != expected_members or set(group.attrs):
                    raise TypeError("payload group membership is not exact")
                for index, leaf in enumerate(entry.leaves):
                    dataset = _require_local_dataset(
                        parent=group,
                        name=f"leaf_{index:04d}",
                        label=f"payload {address} leaf {index}",
                    )
                    dataset_address = _hdf5_object_address(dataset)
                    if dataset_address in object_addresses:
                        raise TypeError("payload datasets are hard-link aliases")
                    object_addresses.add(dataset_address)
                    if (
                        dataset.shape != cast("tuple[int, ...]", leaf["shape"])
                        or dataset.dtype.str != cast("str", leaf["dtype"])
                        or dataset.is_virtual
                        or dataset.external is not None
                    ):
                        raise TypeError("payload dataset representation is invalid")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise SolutionIntegrityError(
            "Solution archive HDF5 structure does not exactly match its manifest."
        ) from error


def _require_local_group(*, parent: h5py.Group, name: str, label: str) -> h5py.Group:
    """Return one direct hard-linked HDF5 group."""
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise TypeError(f"{label} is not a direct local group")
    value = parent.get(name)
    if not isinstance(value, h5py.Group):
        raise TypeError(f"{label} is not a direct local group")
    return value


def _require_local_dataset(
    *,
    parent: h5py.Group,
    name: str,
    label: str,
    allowed_attributes: frozenset[str] = frozenset(),
) -> h5py.Dataset:
    """Return one direct hard-linked, internally stored HDF5 dataset."""
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise TypeError(f"{label} is not a direct local dataset")
    value = parent.get(name)
    if (
        not isinstance(value, h5py.Dataset)
        or value.is_virtual
        or value.external is not None
        or set(value.attrs) != set(allowed_attributes)
    ):
        raise TypeError(f"{label} is not a direct local dataset")
    return value


def _hdf5_object_address(value: h5py.Group | h5py.Dataset) -> int:
    """Return the stable file-local object address used to reject hard-link aliases."""
    return int(h5o.get_info(value.id).addr)


def _to_jax_without_narrowing(*, array: NDArray[np.generic], label: str) -> jax.Array:
    """Convert one restored leaf while refusing JAX's implicit x64 narrowing."""
    try:
        result = jnp.asarray(array)
    except (TypeError, ValueError) as error:
        raise IncompatibleSolutionError(
            f"Persisted {label} has dtype {array.dtype!s}, which the active JAX "
            "configuration cannot materialize."
        ) from error
    if np.dtype(result.dtype) != array.dtype:
        raise IncompatibleSolutionError(
            f"Persisted {label} has dtype {array.dtype!s}, but the active JAX "
            f"configuration would materialize it as {result.dtype!s}. Enable the "
            "matching JAX dtype configuration instead of narrowing the solution."
        )
    return result


def _array_checksum(*, identity: dict[str, object], array: NDArray[np.generic]) -> str:
    """Hash one array together with its logical address and representation."""
    digest = hashlib.sha256()
    framed_identity = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    for part in (
        framed_identity,
        json.dumps(list(array.shape), separators=(",", ":")).encode(),
        array.dtype.str.encode(),
        array.tobytes(order="C"),
    ):
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return digest.hexdigest()


def _array_checksum_from_leaf_metadata(
    *, leaf: MappingProxyType[str, object], array: NDArray[np.generic]
) -> str:
    """Recompute a leaf checksum from the identity stored in its manifest entry."""
    identity = dict(cast("MappingProxyType[str, object]", leaf["identity"]))
    return _array_checksum(identity=identity, array=array)


def _solution_stores(*, solution: _SolutionResultBoundary) -> _StoreTupleBoundary:
    """Return each public artifact store paired with its semantic channel."""
    return (
        (ArtifactChannel.CONTINUATION, solution.retained_continuations),
        (ArtifactChannel.REPLAY, solution.replay_artifacts),
        (ArtifactChannel.AUXILIARY, solution.auxiliary_artifacts),
        (ArtifactChannel.DIAGNOSTIC, solution.diagnostics),
    )


def _verify_result_entries(*, result: _SolutionResultBoundary) -> None:
    """Verify all lazy entries without materializing or caching any of them."""
    values = cast("ValueStore", result.values)
    for period in values:
        for regime in values[period]:
            raw = values._raw(period=period, regime=regime)  # noqa: SLF001
            if isinstance(raw, _LazyHdf5Entry):
                raw.verify()
    for _channel, store in _solution_stores(solution=result):
        for ref in store:
            raw = store._raw(ref)  # noqa: SLF001
            if isinstance(raw, _LazyHdf5Entry):
                raw.verify()


def _metadata_to_manifest(metadata: SolutionMetadata) -> dict[str, object]:
    """Encode descriptive metadata without serializing Python implementations."""
    return {
        "pylcm_version": metadata.pylcm_version,
        "retention": metadata.retention.value,
        "n_periods": metadata.n_periods,
        "regime_names": list(metadata.regime_names),
        "solver_types": dict(metadata.solver_types),
        "model_instance_id": metadata.model_instance_id,
        "model_fingerprint": metadata.model_fingerprint,
        "params_fingerprint": metadata.params_fingerprint,
        "solver_identities": {
            regime: {
                "plugin_id": identity.plugin_id,
                "plugin_version": identity.plugin_version,
                "solver_api_version": identity.solver_api_version,
            }
            for regime, identity in metadata.solver_identities.items()
        },
        "replay_routes": {
            regime: (
                None
                if identity is None
                else {
                    "route_id": identity.route_id,
                    "route_version": identity.route_version,
                }
            )
            for regime, identity in metadata.replay_routes.items()
        },
        "value_schemas": [
            {
                "period": period,
                "regime": regime,
                "shape": list(schema.shape),
                "dtype": schema.dtype,
                "axis_names": list(schema.axis_names),
            }
            for (period, regime), schema in sorted(metadata.value_schemas.items())
        ],
        "artifact_descriptors": [
            {
                "period": ref.period,
                "regime": ref.regime,
                "type_id": ref.key.type_id,
                "schema_version": ref.key.schema_version,
                "channel": descriptor.channel.value,
                "persistence": descriptor.persistence.value,
                "payload_type_id": descriptor.payload_type_id,
                "payload_version": descriptor.payload_version,
                "leaf_descriptors": [
                    {
                        "path": list(leaf.path),
                        "shape": list(leaf.shape),
                        "dtype": leaf.dtype,
                        "axis_names": list(leaf.axis_names),
                    }
                    for leaf in descriptor.leaf_descriptors
                ],
                "named_axes": [
                    {
                        "name": axis.name,
                        "length": axis.length,
                        "role": axis.role.value,
                        "coordinates": list(axis.coordinates),
                    }
                    for axis in descriptor.named_axes
                ],
                "state_roles": list(descriptor.state_roles),
                "action_roles": list(descriptor.action_roles),
                "categorical_domains": {
                    name: {
                        "labels": list(domain.labels),
                        "codes": list(domain.codes),
                        "ordered": domain.ordered,
                    }
                    for name, domain in sorted(descriptor.categorical_domains.items())
                },
                "required_for": [
                    {
                        "route_id": route.route_id,
                        "route_version": route.route_version,
                    }
                    for route in sorted(descriptor.required_for)
                ],
                "required": descriptor.required,
            }
            for ref, descriptor in sorted(metadata.artifact_descriptors.items())
        ],
    }


def _leaf_descriptors_from_manifest(value: object) -> tuple[LeafDescriptor, ...]:
    """Decode one exact ordered artifact-leaf schema."""
    result: list[LeafDescriptor] = []
    for index, raw_leaf in enumerate(
        _require_exact_list(value=value, label="artifact leaf descriptors")
    ):
        leaf = _require_exact_dict(
            value=raw_leaf, label=f"artifact leaf descriptor {index}"
        )
        if set(leaf) != {"path", "shape", "dtype", "axis_names"}:
            raise SolutionIntegrityError(
                f"Artifact leaf descriptor {index} has invalid fields."
            )
        result.append(
            LeafDescriptor(
                path=tuple(
                    _require_nonempty_exact_str(
                        value=component,
                        label=f"artifact leaf descriptor {index} path component",
                    )
                    for component in _require_exact_list(
                        value=leaf.get("path"),
                        label=f"artifact leaf descriptor {index} path",
                    )
                ),
                shape=_exact_shape(
                    value=leaf.get("shape"),
                    label=f"artifact leaf descriptor {index} shape",
                ),
                dtype=_require_numeric_dtype(
                    value=leaf.get("dtype"),
                    label=f"artifact leaf descriptor {index} dtype",
                ),
                axis_names=tuple(
                    _require_nonempty_exact_str(
                        value=name,
                        label=f"artifact leaf descriptor {index} axis name",
                    )
                    for name in _require_exact_list(
                        value=leaf.get("axis_names"),
                        label=f"artifact leaf descriptor {index} axis names",
                    )
                ),
            )
        )
    return tuple(result)


def _axis_descriptors_from_manifest(value: object) -> tuple[AxisDescriptor, ...]:
    """Decode one exact ordered named-axis schema."""
    result: list[AxisDescriptor] = []
    for index, raw_axis in enumerate(
        _require_exact_list(value=value, label="artifact named axes")
    ):
        axis = _require_exact_dict(value=raw_axis, label=f"artifact named axis {index}")
        if set(axis) != {"name", "length", "role", "coordinates"}:
            raise SolutionIntegrityError(
                f"Artifact named axis {index} has invalid fields."
            )
        result.append(
            AxisDescriptor(
                name=_require_nonempty_exact_str(
                    value=axis.get("name"), label=f"artifact named axis {index} name"
                ),
                length=_require_nonnegative_exact_int(
                    value=axis.get("length"),
                    label=f"artifact named axis {index} length",
                ),
                role=AxisRole(
                    _require_exact_str(
                        value=axis.get("role"),
                        label=f"artifact named axis {index} role",
                    )
                ),
                coordinates=tuple(
                    _require_exact_json_scalar(
                        value=coordinate,
                        label=f"artifact named axis {index} coordinate",
                    )
                    for coordinate in _require_exact_list(
                        value=axis.get("coordinates"),
                        label=f"artifact named axis {index} coordinates",
                    )
                ),
            )
        )
    return tuple(result)


def _categorical_domains_from_manifest(
    value: object,
) -> dict[str, CategoryDomain]:
    """Decode exact categorical labels, integer codes, and order."""
    result: dict[str, CategoryDomain] = {}
    for name, raw_domain in _require_exact_dict(
        value=value, label="artifact categorical domains"
    ).items():
        if not name:
            raise SolutionIntegrityError("Artifact categorical role name is empty.")
        domain = _require_exact_dict(
            value=raw_domain, label=f"artifact categorical domain {name!r}"
        )
        if set(domain) != {"labels", "codes", "ordered"}:
            raise SolutionIntegrityError(
                f"Artifact categorical domain {name!r} has invalid fields."
            )
        codes_raw = _require_exact_list(
            value=domain.get("codes"),
            label=f"artifact categorical domain {name!r} codes",
        )
        if any(type(code) is not int for code in codes_raw):
            raise SolutionIntegrityError(
                f"Artifact categorical domain {name!r} has non-exact codes."
            )
        result[name] = CategoryDomain(
            labels=tuple(
                _require_nonempty_exact_str(
                    value=label,
                    label=f"artifact categorical domain {name!r} label",
                )
                for label in _require_exact_list(
                    value=domain.get("labels"),
                    label=f"artifact categorical domain {name!r} labels",
                )
            ),
            codes=tuple(cast("list[int]", codes_raw)),
            ordered=_require_exact_bool(
                value=domain.get("ordered"),
                label=f"artifact categorical domain {name!r} ordering",
            ),
        )
    return result


def _required_routes_from_manifest(value: object) -> frozenset[ReplayRouteIdentity]:
    """Decode an exact set of route identities requiring one artifact."""
    raw_routes = _require_exact_list(value=value, label="artifact required routes")
    routes: list[ReplayRouteIdentity] = []
    for index, raw_route in enumerate(raw_routes):
        route = _require_exact_dict(
            value=raw_route, label=f"artifact required route {index}"
        )
        if set(route) != {"route_id", "route_version"}:
            raise SolutionIntegrityError(
                f"Artifact required route {index} has invalid fields."
            )
        routes.append(
            ReplayRouteIdentity(
                route_id=_require_nonempty_exact_str(
                    value=route.get("route_id"),
                    label=f"artifact required route {index} ID",
                ),
                route_version=_require_positive_exact_int(
                    value=route.get("route_version"),
                    label=f"artifact required route {index} version",
                ),
            )
        )
    if len(set(routes)) != len(routes):
        raise SolutionIntegrityError("Artifact required routes contain a duplicate.")
    return frozenset(routes)


def _metadata_from_manifest(  # noqa: C901, PLR0912, PLR0915
    raw: dict[str, object],
) -> SolutionMetadata:
    """Decode descriptive solution metadata into exact immutable public types."""
    try:
        if set(raw) != {
            "pylcm_version",
            "retention",
            "n_periods",
            "regime_names",
            "solver_types",
            "model_instance_id",
            "model_fingerprint",
            "params_fingerprint",
            "solver_identities",
            "replay_routes",
            "value_schemas",
            "artifact_descriptors",
        }:
            raise ValueError("invalid metadata fields")
        regimes = tuple(
            _require_nonempty_exact_str(value=name, label="regime name")
            for name in _require_exact_list(
                value=raw.get("regime_names"), label="regime names"
            )
        )
        if len(set(regimes)) != len(regimes):
            raise ValueError("duplicate regime name")

        value_schemas: dict[tuple[int, str], ValueArraySchema] = {}
        for item in _require_exact_list(
            value=raw.get("value_schemas"), label="value schemas"
        ):
            if type(item) is not dict:
                raise TypeError("value schema is not an object")
            entry = cast("dict[str, object]", item)
            if set(entry) != {"period", "regime", "shape", "dtype", "axis_names"}:
                raise ValueError("invalid value schema fields")
            coordinate = (
                _require_nonnegative_exact_int(
                    value=entry.get("period"), label="value-schema period"
                ),
                _require_nonempty_exact_str(
                    value=entry.get("regime"), label="value-schema regime"
                ),
            )
            if coordinate in value_schemas:
                raise ValueError("duplicate value schema")
            shape_raw = _require_exact_list(
                value=entry.get("shape"), label="value-schema shape"
            )
            if any(type(size) is not int or size < 0 for size in shape_raw):
                raise TypeError("value-schema shape is not exact")
            axis_names_raw = _require_exact_list(
                value=entry.get("axis_names"), label="value-schema axis names"
            )
            value_schemas[coordinate] = ValueArraySchema(
                shape=tuple(cast("list[int]", shape_raw)),
                dtype=_require_numeric_dtype(
                    value=entry.get("dtype"), label="value-schema dtype"
                ),
                axis_names=tuple(
                    _require_nonempty_exact_str(
                        value=name, label="value-schema axis name"
                    )
                    for name in axis_names_raw
                ),
            )

        descriptors: dict[ArtifactRef, ArtifactDescriptor] = {}
        for item in _require_exact_list(
            value=raw.get("artifact_descriptors"), label="artifact descriptors"
        ):
            if type(item) is not dict:
                raise TypeError("artifact descriptor is not an object")
            entry = cast("dict[str, object]", item)
            if set(entry) != {
                "period",
                "regime",
                "type_id",
                "schema_version",
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
            }:
                raise ValueError("invalid artifact descriptor fields")
            key = ArtifactKey(
                type_id=_require_nonempty_exact_str(
                    value=entry.get("type_id"), label="artifact type ID"
                ),
                schema_version=_require_positive_exact_int(
                    value=entry.get("schema_version"), label="artifact schema version"
                ),
            )
            ref = ArtifactRef(
                period=_require_nonnegative_exact_int(
                    value=entry.get("period"), label="artifact-descriptor period"
                ),
                regime=_require_nonempty_exact_str(
                    value=entry.get("regime"), label="artifact-descriptor regime"
                ),
                key=key,
            )
            if ref in descriptors:
                raise ValueError("duplicate artifact descriptor")
            descriptors[ref] = ArtifactDescriptor(
                key=key,
                channel=ArtifactChannel(
                    _require_exact_str(
                        value=entry.get("channel"), label="artifact-descriptor channel"
                    )
                ),
                persistence=PersistencePolicy(
                    _require_exact_str(
                        value=entry.get("persistence"),
                        label="artifact-descriptor persistence",
                    )
                ),
                payload_type_id=_require_nonempty_exact_str(
                    value=entry.get("payload_type_id"),
                    label="artifact payload type ID",
                ),
                payload_version=_require_positive_exact_int(
                    value=entry.get("payload_version"),
                    label="artifact payload version",
                ),
                leaf_descriptors=_leaf_descriptors_from_manifest(
                    entry.get("leaf_descriptors")
                ),
                named_axes=_axis_descriptors_from_manifest(entry.get("named_axes")),
                state_roles=tuple(
                    _require_nonempty_exact_str(value=name, label="artifact state role")
                    for name in _require_exact_list(
                        value=entry.get("state_roles"), label="artifact state roles"
                    )
                ),
                action_roles=tuple(
                    _require_nonempty_exact_str(
                        value=name, label="artifact action role"
                    )
                    for name in _require_exact_list(
                        value=entry.get("action_roles"), label="artifact action roles"
                    )
                ),
                categorical_domains=_categorical_domains_from_manifest(
                    entry.get("categorical_domains")
                ),
                required_for=_required_routes_from_manifest(entry.get("required_for")),
                required=_require_exact_bool(
                    value=entry.get("required"), label="artifact requiredness"
                ),
            )

        solver_types = _require_exact_str_mapping(
            value=raw.get("solver_types"), label="solver types"
        )
        identities_raw = _require_exact_dict(
            value=raw.get("solver_identities"), label="solver identities"
        )
        if set(identities_raw) != set(regimes):
            raise ValueError("solver identity coverage")
        identities: dict[str, SolverIdentity] = {}
        for regime in regimes:
            identity_raw = _require_exact_dict(
                value=identities_raw[regime], label="solver identity"
            )
            if set(identity_raw) != {
                "plugin_id",
                "plugin_version",
                "solver_api_version",
            }:
                raise ValueError("invalid solver identity fields")
            identities[regime] = SolverIdentity(
                plugin_id=_require_nonempty_exact_str(
                    value=identity_raw.get("plugin_id"), label="plugin ID"
                ),
                plugin_version=_require_nonempty_exact_str(
                    value=identity_raw.get("plugin_version"), label="plugin version"
                ),
                solver_api_version=_require_positive_exact_int(
                    value=identity_raw.get("solver_api_version"),
                    label="solver API version",
                ),
            )

        routes_raw = _require_exact_dict(
            value=raw.get("replay_routes"), label="replay routes"
        )
        if routes_raw and set(routes_raw) != set(regimes):
            raise ValueError("replay route coverage")
        routes: dict[str, ReplayRouteIdentity | None] = {}
        for regime, route_value in routes_raw.items():
            if route_value is None:
                routes[regime] = None
                continue
            route_raw = _require_exact_dict(value=route_value, label="replay route")
            if set(route_raw) != {"route_id", "route_version"}:
                raise ValueError("invalid replay route fields")
            routes[regime] = ReplayRouteIdentity(
                route_id=_require_nonempty_exact_str(
                    value=route_raw.get("route_id"), label="replay route ID"
                ),
                route_version=_require_positive_exact_int(
                    value=route_raw.get("route_version"), label="replay route version"
                ),
            )
        return SolutionMetadata(
            pylcm_version=_require_nonempty_exact_str(
                value=raw.get("pylcm_version"), label="pylcm version"
            ),
            retention=ResultRetention(
                _require_exact_str(value=raw.get("retention"), label="retention")
            ),
            n_periods=_require_positive_exact_int(
                value=raw.get("n_periods"), label="period count"
            ),
            regime_names=regimes,
            solver_types=solver_types,
            model_instance_id=_require_nonempty_exact_str(
                value=raw.get("model_instance_id"), label="model instance ID"
            ),
            model_fingerprint=_require_sha256(
                value=raw.get("model_fingerprint"), label="model fingerprint"
            ),
            params_fingerprint=_require_sha256(
                value=raw.get("params_fingerprint"), label="parameter fingerprint"
            ),
            value_schemas=value_schemas,
            solver_identities=identities,
            replay_routes=routes,
            artifact_descriptors=descriptors,
            source=SolutionSource.PERSISTED,
            solver_api_version=SOLVER_API_VERSION,
            solution_schema_version=SOLUTION_SCHEMA_VERSION,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SolutionIntegrityError(
            "Solution archive contains malformed descriptive metadata."
        ) from error


def _artifact_ref_from_manifest(entry: dict[str, object]) -> ArtifactRef:
    """Decode one exact artifact address from an archive manifest entry."""
    try:
        return ArtifactRef(
            period=_require_nonnegative_exact_int(
                value=entry.get("period"), label="artifact period"
            ),
            regime=_require_nonempty_exact_str(
                value=entry.get("regime"), label="artifact regime"
            ),
            key=ArtifactKey(
                type_id=_require_nonempty_exact_str(
                    value=entry.get("type_id"), label="artifact type ID"
                ),
                schema_version=_require_positive_exact_int(
                    value=entry.get("schema_version"), label="artifact schema version"
                ),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SolutionIntegrityError(
            "Archive contains an invalid artifact address."
        ) from error


def _validate_persisted_ref(
    *, ref: ArtifactRef, metadata: SolutionMetadata, label: str
) -> None:
    """Require one persisted address to name a declared period and regime."""
    if ref.period >= metadata.n_periods or ref.regime not in metadata.regime_names:
        raise SolutionIntegrityError(
            f"Persisted {label} {ref!r} has an invalid period or regime."
        )


def _validate_persisted_omission_semantics(
    *,
    ref: ArtifactRef,
    reason: OmissionReason,
    descriptor: ArtifactDescriptor,
    retention: ResultRetention,
) -> None:
    """Check omission facts that are decidable from descriptive metadata alone."""
    selected = _retention_selects_artifact(
        retention=retention,
        descriptor=descriptor,
    )
    if not selected and reason not in {
        OmissionReason.NOT_APPLICABLE,
        OmissionReason.NOT_REQUESTED,
    }:
        raise SolutionIntegrityError(
            f"Omission {ref!r} has a reason inconsistent with result retention."
        )
    if (
        selected
        and descriptor.persistence is PersistencePolicy.NOT_PERSISTED
        and reason not in {OmissionReason.NOT_APPLICABLE, OmissionReason.NOT_PERSISTED}
    ):
        raise SolutionIntegrityError(
            f"Omission {ref!r} has a reason inconsistent with persistence policy."
        )
    if (
        selected
        and descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
        and reason not in {OmissionReason.NOT_APPLICABLE, OmissionReason.UNSUPPORTED}
    ):
        raise SolutionIntegrityError(
            f"Omission {ref!r} has a reason inconsistent with result retention."
        )
    if (
        descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
        and reason is OmissionReason.NOT_PERSISTED
    ):
        raise SolutionIntegrityError(
            f"Omission {ref!r} claims NOT_PERSISTED although its descriptor is "
            "model-verifiable."
        )
    if (
        selected
        and descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
        and descriptor.required
        and reason is OmissionReason.UNSUPPORTED
    ):
        raise SolutionIntegrityError(
            f"Required artifact {ref!r} cannot use the UNSUPPORTED omission reason."
        )


def _check_archive_versions(*, manifest: dict[str, object]) -> None:
    """Reject unsupported archive, solver, and solution schema versions exactly."""
    expected_ints = {
        "format_version": SOLUTION_FORMAT_VERSION,
        "solver_api_version": SOLVER_API_VERSION,
        "solution_schema_version": SOLUTION_SCHEMA_VERSION,
    }
    mismatches = [
        f"{name}={manifest.get(name)!r} (expected {version})"
        for name, version in expected_ints.items()
        if type(manifest.get(name)) is not int or manifest.get(name) != version
    ]
    if (
        type(manifest.get("pylcm_version")) is not str
        or manifest.get("pylcm_version") != PYLCM_VERSION
    ):
        mismatches.append(
            f"pylcm_version={manifest.get('pylcm_version')!r} "
            f"(expected {PYLCM_VERSION!r})"
        )
    if mismatches:
        raise IncompatibleSolutionError(
            "Solution archive uses incompatible versions: " + "; ".join(mismatches)
        )


__all__ = ["load_solution_archive", "save_solution_archive"]
