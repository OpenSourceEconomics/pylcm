"""Public, lightweight types for labelled solver artifacts and solutions.

This module defines pylcm's dependency-safe solver extension boundary. Its public
definitions cover result identity and retention without referring to engine-private
``_lcm`` types or concrete built-in solver payloads.
"""

import dataclasses
import functools
import struct
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from threading import RLock
from types import GetSetDescriptorType, MappingProxyType, MemberDescriptorType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    SupportsIndex,
    TypeAlias,
    cast,
    runtime_checkable,
)

import jax
import numpy as np
from jaxtyping import Float

from lcm.typing import FloatND, IntND, RegimeName
from lcm.version import __version__

_SHA256_HEX_LENGTH = 64
PYLCM_VERSION = __version__

SOLVER_API_VERSION = 1
# Version of the public solver/plugin protocol implemented by this release.

SOLUTION_SCHEMA_VERSION = 2
# Version of the labelled in-memory solution schema.

SOLUTION_FORMAT_VERSION = 1
# Version of the durable solution archive format.


class LoadState(StrEnum):
    """Whether one independently addressed persisted entry is materialized."""

    UNLOADED = "unloaded"
    LOADED = "loaded"


class ArtifactChannel(StrEnum):
    """Semantic channel on which a solver publishes an artifact."""

    CONTINUATION = "continuation"
    REPLAY = "replay"
    AUXILIARY = "auxiliary"
    DIAGNOSTIC = "diagnostic"


class PersistencePolicy(StrEnum):
    """Whether a model can independently authorize an artifact on restoration."""

    MODEL_VERIFIABLE = "model_verifiable"
    NOT_PERSISTED = "not_persisted"


class SolutionSource(StrEnum):
    """Origin of the current result container."""

    IN_MEMORY = "in_memory"
    PERSISTED = "persisted"


class _LazyEntry(ABC):
    """Private implementation hook shared by value and artifact stores."""

    @property
    @abstractmethod
    def load_state(self) -> LoadState:
        """Return the materialization state without loading the entry."""

    @abstractmethod
    def materialize(self, *, template: object | None = None) -> object:
        """Load and verify the entry, optionally rebuilding a declared PyTree."""

    def materialize_from_template_snapshot(
        self,
        *,
        template_snapshot: object,
    ) -> object:
        """Fallback for lazy implementations that only consume a template object."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Lazy materialization requires an exact template snapshot.")
        template = _reconstruct_artifact_from_template_snapshot(
            template_snapshot=template_snapshot,
            leaves=tuple(template_snapshot.leaves),
        )
        return self.materialize(template=template)


def _materialize_entry(
    *,
    entry: object,
    template: object | None = None,
    template_snapshot: object | None = None,
) -> object:
    """Materialize an internal lazy entry while leaving eager objects untouched."""
    if isinstance(entry, _LazyEntry):
        if template is not None and template_snapshot is not None:
            raise TypeError("Supply a template or a template snapshot, not both.")
        if template_snapshot is not None:
            return entry.materialize_from_template_snapshot(
                template_snapshot=template_snapshot
            )
        return entry.materialize(template=template)
    return entry


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
        if type(self.type_id) is not str:
            raise TypeError("ArtifactKey.type_id must be an exact str.")
        if not self.type_id:
            raise ValueError("ArtifactKey.type_id must not be empty.")
        if type(self.schema_version) is not int:
            raise TypeError("ArtifactKey.schema_version must be an exact int.")
        if self.schema_version < 1:
            raise ValueError("ArtifactKey.schema_version must be at least 1.")


@dataclass(frozen=True, order=True, kw_only=True)
class SolverIdentity:
    """Durable identity and compatibility version of an installed solver plugin."""

    plugin_id: str
    plugin_version: str
    solver_api_version: int = SOLVER_API_VERSION

    def __post_init__(self) -> None:
        if type(self.plugin_id) is not str:
            raise TypeError("SolverIdentity.plugin_id must be an exact str.")
        if not self.plugin_id:
            raise ValueError("SolverIdentity.plugin_id must not be empty.")
        if type(self.plugin_version) is not str:
            raise TypeError("SolverIdentity.plugin_version must be an exact str.")
        if not self.plugin_version:
            raise ValueError("SolverIdentity.plugin_version must not be empty.")
        if type(self.solver_api_version) is not int:
            raise TypeError("SolverIdentity.solver_api_version must be an exact int.")
        if self.solver_api_version != SOLVER_API_VERSION:
            raise ValueError(
                "SolverIdentity.solver_api_version is incompatible with this pylcm "
                f"release: got {self.solver_api_version}, expected "
                f"{SOLVER_API_VERSION}."
            )


@dataclass(frozen=True, order=True, kw_only=True)
class ReplayRouteIdentity:
    """Durable identity and schema version of one replay implementation."""

    route_id: str
    route_version: int

    def __post_init__(self) -> None:
        if type(self.route_id) is not str:
            raise TypeError("ReplayRouteIdentity.route_id must be an exact str.")
        if not self.route_id:
            raise ValueError("ReplayRouteIdentity.route_id must not be empty.")
        if type(self.route_version) is not int:
            raise TypeError("ReplayRouteIdentity.route_version must be an exact int.")
        if self.route_version < 1:
            raise ValueError("ReplayRouteIdentity.route_version must be at least 1.")


type TreePath = tuple[str, ...]
# Stable path to one container or numerical leaf in a public artifact PyTree. Each
# component records both the JAX key kind and value, for example ``"attribute:values"``
# or ``"flattened:0"``. The root path is ``()``.


class AxisRole(StrEnum):
    """Mathematical role of one named artifact axis."""

    STATE = "state"
    ACTION = "action"
    CANDIDATE = "candidate"
    STOCHASTIC = "stochastic"
    STAKEHOLDER = "stakeholder"
    OTHER = "other"


@dataclass(frozen=True, order=True, kw_only=True)
class CategoryDomain:
    """Exact labels, integer codes, and ordering of one categorical role."""

    labels: tuple[str, ...]
    codes: tuple[int, ...]
    ordered: bool

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        codes = tuple(self.codes)
        if not labels or len(labels) != len(codes):
            raise ValueError(
                "CategoryDomain labels and codes must be nonempty and have equal "
                "length."
            )
        if any(type(label) is not str or not label for label in labels):
            raise TypeError("CategoryDomain labels must be nonempty exact strs.")
        if any(type(code) is not int for code in codes):
            raise TypeError("CategoryDomain codes must be exact ints.")
        if type(self.ordered) is not bool:
            raise TypeError("CategoryDomain.ordered must be an exact bool.")
        if len(set(labels)) != len(labels) or len(set(codes)) != len(codes):
            raise ValueError("CategoryDomain labels and codes must each be unique.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "codes", codes)


@dataclass(frozen=True, order=True, kw_only=True)
class AxisDescriptor:
    """Descriptive name, length, and mathematical role of one artifact axis."""

    name: str
    length: int
    role: AxisRole
    coordinates: tuple[bool | int | float | str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("AxisDescriptor.name must be a nonempty exact str.")
        if type(self.length) is not int or self.length < 0:
            raise TypeError("AxisDescriptor.length must be a nonnegative exact int.")
        if type(self.role) is not AxisRole:
            raise TypeError("AxisDescriptor.role must be an exact AxisRole.")
        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != self.length:
            raise ValueError(
                f"AxisDescriptor {self.name!r} has {len(coordinates)} coordinates; "
                f"expected {self.length}."
            )
        if (
            not coordinates
            and self.length
            and self.role
            not in {
                AxisRole.STATE,
                AxisRole.ACTION,
            }
        ):
            raise ValueError(
                "Only a model state or action axis may defer descriptive coordinates."
            )
        if any(
            not any(type(value) is allowed for allowed in (bool, int, float, str))
            for value in coordinates
        ):
            raise TypeError(
                "AxisDescriptor coordinates must use exact JSON scalar types."
            )
        if any(
            type(value) is float and not np.isfinite(value) for value in coordinates
        ):
            raise ValueError("AxisDescriptor coordinates must be finite.")
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True, order=True, kw_only=True)
class AxisAuthority:
    """Model-owned axis description plus its exact canonical coordinates."""

    name: str
    length: int
    role: AxisRole
    coordinates: tuple[bool | int | float | str, ...] = ()

    def __post_init__(self) -> None:
        descriptor = AxisDescriptor(
            name=self.name,
            length=self.length,
            role=self.role,
            coordinates=self.coordinates,
        )
        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != self.length:
            raise ValueError(
                f"AxisAuthority {self.name!r} has {len(coordinates)} coordinates; "
                f"expected {self.length}."
            )
        if (
            not coordinates
            and self.length
            and self.role
            not in {
                AxisRole.STATE,
                AxisRole.ACTION,
            }
        ):
            raise ValueError(
                "Only a model state or action axis may defer coordinates until model "
                "authority is bound."
            )
        if any(
            not any(type(value) is allowed for allowed in (bool, int, float, str))
            for value in coordinates
        ):
            raise TypeError(
                "AxisAuthority coordinates must use exact JSON scalar types."
            )
        object.__setattr__(self, "name", descriptor.name)
        object.__setattr__(self, "length", descriptor.length)
        object.__setattr__(self, "role", descriptor.role)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def descriptor(self) -> AxisDescriptor:
        """Return the transport-safe description of this authoritative axis."""
        return AxisDescriptor(
            name=self.name,
            length=self.length,
            role=self.role,
            coordinates=self.coordinates,
        )


@dataclass(frozen=True, order=True, kw_only=True)
class LeafDescriptor:
    """Transport-safe schema of one numerical artifact leaf."""

    path: TreePath
    shape: tuple[int, ...]
    dtype: str
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        path = tuple(self.path)
        shape = tuple(self.shape)
        axis_names = tuple(self.axis_names)
        if any(type(component) is not str or not component for component in path):
            raise TypeError(
                "LeafDescriptor.path components must be nonempty exact strs."
            )
        if any(type(size) is not int or size < 0 for size in shape):
            raise TypeError("LeafDescriptor.shape must contain nonnegative exact ints.")
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("LeafDescriptor.dtype must be a nonempty exact str.")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise TypeError("LeafDescriptor.dtype must name a NumPy dtype.") from error
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            raise TypeError("LeafDescriptor.dtype must be numerical or Boolean.")
        if len(axis_names) != len(shape) or any(
            type(name) is not str or not name for name in axis_names
        ):
            raise ValueError(
                "LeafDescriptor.axis_names must name every dimension exactly once."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(dtype))
        object.__setattr__(self, "axis_names", axis_names)


@dataclass(frozen=True, kw_only=True)
class LeafAuthority:
    """Exact runtime type and schema of one model-authoritative PyTree leaf."""

    path: TreePath
    runtime_type: type[object]
    shape: tuple[int, ...]
    dtype: str
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        descriptor = LeafDescriptor(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            axis_names=self.axis_names,
        )
        if not isinstance(self.runtime_type, type):
            raise TypeError("LeafAuthority.runtime_type must be a type.")
        object.__setattr__(self, "path", descriptor.path)
        object.__setattr__(self, "shape", descriptor.shape)
        object.__setattr__(self, "dtype", descriptor.dtype)
        object.__setattr__(self, "axis_names", descriptor.axis_names)

    @property
    def descriptor(self) -> LeafDescriptor:
        """Return the transport-safe description of this authoritative leaf."""
        return LeafDescriptor(
            path=self.path,
            shape=self.shape,
            dtype=self.dtype,
            axis_names=self.axis_names,
        )


if TYPE_CHECKING:
    _CategoricalDomainsBoundary: TypeAlias = Mapping[  # noqa: UP040
        str, CategoryDomain
    ]
    _ContainerRuntimeTypesBoundary: TypeAlias = Mapping[  # noqa: UP040
        TreePath, type[object]
    ]
    _LeafAuthoritiesBoundary: TypeAlias = Mapping[  # noqa: UP040
        TreePath, LeafAuthority
    ]
else:
    # These public constructors own exact, single-traversal mapping validation.
    # Runtime annotation sampling must not observe a stateful mapping first.
    _CategoricalDomainsBoundary = object
    _ContainerRuntimeTypesBoundary = object
    _LeafAuthoritiesBoundary = object


def _capture_nonempty_mapping_name(value: object) -> str:
    """Canonicalize one exact nonempty mapping-name key before hashing it."""
    if type(value) is not str or not value:
        raise TypeError("Artifact mapping keys must be nonempty exact strs.")
    return value


def _capture_artifact_tree_path(value: object) -> TreePath:
    """Canonicalize one exact TreePath before hashing it."""
    if type(value) is not tuple:
        raise TypeError("Artifact mapping keys must be exact TreePaths.")
    components = value
    if any(type(component) is not str or not component for component in components):
        raise TypeError("Artifact TreePath components must be nonempty exact strs.")
    return cast("TreePath", tuple(component for component in components))


def _capture_category_domain(value: object) -> CategoryDomain:
    """Require one exact categorical-domain value before insertion."""
    if type(value) is not CategoryDomain:
        raise TypeError("Artifact categorical domains must be exact CategoryDomains.")
    return value


def _capture_container_runtime_type(value: object) -> type[object]:
    """Require one runtime-type declaration before insertion."""
    if not isinstance(value, type):
        raise TypeError("Artifact container runtime declarations must be types.")
    return value


def _capture_leaf_authority(value: object) -> LeafAuthority:
    """Require one exact leaf authority and validate its path before insertion."""
    if type(value) is not LeafAuthority:
        raise TypeError("Artifact leaves must be exact LeafAuthority objects.")
    _capture_artifact_tree_path(value.path)
    return value


def _capture_mapping_item_stream_once(
    *,
    mapping: object,
    label: str,
    snapshot_key: Callable[[object], object],
    snapshot_value: Callable[[object], object],
) -> dict[object, object]:
    """Own and canonicalize one mapping through exactly one item iterator."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        iterator = iter(mapping.items())
    except Exception as error:
        raise TypeError(f"{label} cannot be traversed as mapping items.") from error

    copied: dict[object, object] = {}
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise TypeError(f"{label} cannot be traversed as mapping items.") from error
        if type(item) is not tuple or len(item) != 2:  # noqa: PLR2004
            raise TypeError(f"{label} items must be exact key-value pairs.")
        raw_key, raw_value = item
        key = snapshot_key(raw_key)
        value = snapshot_value(raw_value)
        if key in copied:
            raise ValueError(f"{label} keys collide after exact reconstruction.")
        copied[key] = value
    return copied


@dataclass(frozen=True, kw_only=True)
class ArtifactDescriptor:
    """Public description of one versioned artifact schema.

    The descriptor is useful for retention and transport.  It is descriptive,
    not an authentication peer: replay preflight rebuilds an
    :class:`ArtifactAuthority` from the current model and installed route.
    """

    key: ArtifactKey
    channel: ArtifactChannel
    persistence: PersistencePolicy
    payload_type_id: str
    payload_version: int = 1
    leaf_descriptors: tuple[LeafDescriptor, ...] = ()
    named_axes: tuple[AxisDescriptor, ...] = ()
    state_roles: tuple[str, ...] = ()
    action_roles: tuple[str, ...] = ()
    categorical_domains: _CategoricalDomainsBoundary = field(default_factory=dict)
    required_for: frozenset[ReplayRouteIdentity] = frozenset()
    required: bool = False

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        if type(self.key) is not ArtifactKey:
            raise TypeError("ArtifactDescriptor.key must be an exact ArtifactKey.")
        if type(self.channel) is not ArtifactChannel:
            raise TypeError(
                "ArtifactDescriptor.channel must be an exact ArtifactChannel."
            )
        if type(self.persistence) is not PersistencePolicy:
            raise TypeError(
                "ArtifactDescriptor.persistence must be an exact PersistencePolicy."
            )
        if type(self.payload_type_id) is not str or not self.payload_type_id:
            raise ValueError("ArtifactDescriptor.payload_type_id must not be empty.")
        if type(self.payload_version) is not int or self.payload_version < 1:
            raise TypeError(
                "ArtifactDescriptor.payload_version must be a positive exact int."
            )
        leaves = tuple(self.leaf_descriptors)
        axes = tuple(self.named_axes)
        state_roles = tuple(self.state_roles)
        action_roles = tuple(self.action_roles)
        categories = cast(
            "dict[str, CategoryDomain]",
            _capture_mapping_item_stream_once(
                mapping=self.categorical_domains,
                label="ArtifactDescriptor.categorical_domains",
                snapshot_key=_capture_nonempty_mapping_name,
                snapshot_value=_capture_category_domain,
            ),
        )
        required_for = frozenset(self.required_for)
        if any(type(leaf) is not LeafDescriptor for leaf in leaves):
            raise TypeError("ArtifactDescriptor leaves must be exact LeafDescriptors.")
        if len({leaf.path for leaf in leaves}) != len(leaves):
            raise ValueError("ArtifactDescriptor leaf paths must be unique.")
        if any(type(axis) is not AxisDescriptor for axis in axes):
            raise TypeError("ArtifactDescriptor axes must be exact AxisDescriptors.")
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("ArtifactDescriptor axis names must be unique.")
        axis_names = {axis.name for axis in axes}
        if any(set(leaf.axis_names) - axis_names for leaf in leaves):
            raise ValueError("Every leaf axis must name a declared artifact axis.")
        _check_role_names(names=state_roles, label="state")
        _check_role_names(names=action_roles, label="action")
        if set(state_roles) & set(action_roles):
            raise ValueError("Artifact state and action roles must not overlap.")
        if any(
            axis.role is AxisRole.STATE and axis.name not in state_roles
            for axis in axes
        ):
            raise ValueError(
                "Every STATE axis must name one of the artifact's state roles."
            )
        if any(
            axis.role is AxisRole.ACTION and axis.name not in action_roles
            for axis in axes
        ):
            raise ValueError(
                "Every ACTION axis must name one of the artifact's action roles."
            )
        if any(
            axis.name in state_roles and axis.role is not AxisRole.STATE
            for axis in axes
        ):
            raise ValueError(
                "An axis named as an artifact state role must have STATE role."
            )
        if any(
            axis.name in action_roles and axis.role is not AxisRole.ACTION
            for axis in axes
        ):
            raise ValueError(
                "An axis named as an artifact action role must have ACTION role."
            )
        if any(
            type(name) is not str or not name or type(domain) is not CategoryDomain
            for name, domain in categories.items()
        ):
            raise TypeError(
                "ArtifactDescriptor categorical domains must map exact names to "
                "CategoryDomain values."
            )
        if not set(categories) <= set(state_roles) | set(action_roles):
            raise ValueError(
                "Every categorical domain must belong to a declared state or action "
                "role."
            )
        if any(type(route) is not ReplayRouteIdentity for route in required_for):
            raise TypeError(
                "ArtifactDescriptor.required_for must contain ReplayRouteIdentity "
                "values."
            )
        if type(self.required) is not bool:
            raise TypeError("ArtifactDescriptor.required must be an exact bool.")
        object.__setattr__(self, "leaf_descriptors", leaves)
        object.__setattr__(self, "named_axes", axes)
        object.__setattr__(self, "state_roles", state_roles)
        object.__setattr__(self, "action_roles", action_roles)
        object.__setattr__(self, "categorical_domains", MappingProxyType(categories))
        object.__setattr__(self, "required_for", required_for)


@dataclass(frozen=True, slots=True, eq=False)
class _ArtifactLeafToken:
    """Unique declaration-time marker for one ordered numerical child."""

    index: int


@dataclass(frozen=True, slots=True)
class _ArtifactLeafSlot:
    """Callback-free reconstruction slot for one ordered numerical child."""

    index: int


@dataclass(frozen=True, slots=True)
class _ArtifactTuplePlan:
    """Callback-free reconstruction plan for one exact tuple."""

    children: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactDataclassFieldPlan:
    """One captured dataclass field and its exact storage location."""

    name: str
    stored_in_dict: bool
    value: object


@dataclass(frozen=True, slots=True)
class _ArtifactDataclassPlan:
    """Callback-free reconstruction plan for one closed dataclass record."""

    runtime_type: type[object]
    fields: tuple[_ArtifactDataclassFieldPlan, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactStaticPlan:
    """Owned closed-grammar value injected by an unflatten declaration."""

    value: object
    validate_payload: bool


@dataclass(frozen=True, kw_only=True)
class _CanonicalArtifactTemplate:
    """Owned declaration whose later reconstruction never invokes plugin code."""

    payload: object
    tree: jax.tree_util.PyTreeDef
    leaf_paths: tuple[TreePath, ...]
    leaves: tuple[jax.Array, ...]
    construction_plan: object


@dataclass(frozen=True, kw_only=True)
class _ArtifactAuthorityPickleState:
    """Callback-free sealed state used only for trusted Python-object transport."""

    descriptor: ArtifactDescriptor
    payload_runtime_type: type[object]
    template_snapshot: _CanonicalArtifactTemplate | None
    container_runtime_types: Mapping[TreePath, type[object]]
    leaves: Mapping[TreePath, LeafAuthority]
    axes: tuple[AxisAuthority, ...]
    state_roles: tuple[str, ...]
    action_roles: tuple[str, ...]
    categorical_domains: Mapping[str, CategoryDomain]
    consumer_route: ReplayRouteIdentity | None
    applicable: bool
    required: bool


@dataclass(frozen=True, kw_only=True)
class ArtifactAuthority:
    """Model-built validation authority for one artifact in one solution cell."""

    descriptor: ArtifactDescriptor
    payload_runtime_type: type[object]
    template: object | None
    container_runtime_types: _ContainerRuntimeTypesBoundary = field(
        default_factory=dict
    )
    leaves: _LeafAuthoritiesBoundary = field(default_factory=dict)
    axes: tuple[AxisAuthority, ...] = ()
    state_roles: tuple[str, ...] = ()
    action_roles: tuple[str, ...] = ()
    categorical_domains: _CategoricalDomainsBoundary = field(default_factory=dict)
    consumer_route: ReplayRouteIdentity | None = None
    applicable: bool = True
    required: bool = False

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        _assert_artifact_authority_unbound(self)
        if type(self.descriptor) is not ArtifactDescriptor:
            raise TypeError("ArtifactAuthority.descriptor must be exact.")
        if not isinstance(self.payload_runtime_type, type):
            raise TypeError("ArtifactAuthority.payload_runtime_type must be a type.")
        containers = cast(
            "dict[TreePath, type[object]]",
            _capture_mapping_item_stream_once(
                mapping=self.container_runtime_types,
                label="ArtifactAuthority.container_runtime_types",
                snapshot_key=_capture_artifact_tree_path,
                snapshot_value=_capture_container_runtime_type,
            ),
        )
        leaves = cast(
            "dict[TreePath, LeafAuthority]",
            _capture_mapping_item_stream_once(
                mapping=self.leaves,
                label="ArtifactAuthority.leaves",
                snapshot_key=_capture_artifact_tree_path,
                snapshot_value=_capture_leaf_authority,
            ),
        )
        axes = tuple(self.axes)
        state_roles = tuple(self.state_roles)
        action_roles = tuple(self.action_roles)
        categories = cast(
            "dict[str, CategoryDomain]",
            _capture_mapping_item_stream_once(
                mapping=self.categorical_domains,
                label="ArtifactAuthority.categorical_domains",
                snapshot_key=_capture_nonempty_mapping_name,
                snapshot_value=_capture_category_domain,
            ),
        )
        if any(
            type(path) is not tuple
            or any(type(component) is not str or not component for component in path)
            or not isinstance(runtime_type, type)
            for path, runtime_type in containers.items()
        ):
            raise TypeError(
                "ArtifactAuthority container paths and runtime types must be exact."
            )
        if any(
            type(path) is not tuple
            or any(type(component) is not str or not component for component in path)
            or type(leaf) is not LeafAuthority
            for path, leaf in leaves.items()
        ):
            raise TypeError(
                "ArtifactAuthority leaves must map exact paths to authority."
            )
        if any(path != leaf.path for path, leaf in leaves.items()):
            raise ValueError("ArtifactAuthority leaf keys must equal their TreePaths.")
        if any(type(axis) is not AxisAuthority for axis in axes):
            raise TypeError("ArtifactAuthority axes must be exact AxisAuthorities.")
        if not _same_exact_artifact_contract(
            actual=self.descriptor.leaf_descriptors,
            expected=tuple(leaf.descriptor for leaf in leaves.values()),
        ):
            raise ValueError(
                "Artifact descriptive leaves differ from model leaf authority."
            )
        if not _same_exact_artifact_contract(
            actual=self.descriptor.named_axes,
            expected=tuple(axis.descriptor for axis in axes),
        ):
            raise ValueError(
                "Artifact descriptive axes differ from model axis authority."
            )
        if not _same_exact_artifact_contract(
            actual=(state_roles, action_roles),
            expected=(
                self.descriptor.state_roles,
                self.descriptor.action_roles,
            ),
        ):
            raise ValueError("Artifact descriptive roles differ from model authority.")
        if not _same_exact_artifact_contract(
            actual=categories,
            expected=self.descriptor.categorical_domains,
        ):
            raise ValueError(
                "Artifact descriptive categories differ from model authority."
            )
        if (
            self.consumer_route is not None
            and type(self.consumer_route) is not ReplayRouteIdentity
        ):
            raise TypeError(
                "ArtifactAuthority.consumer_route must be a ReplayRouteIdentity or "
                "None."
            )
        expected_required_for = (
            frozenset({self.consumer_route})
            if self.required and self.consumer_route is not None
            else frozenset()
        )
        if not _same_exact_artifact_contract(
            actual=self.descriptor.required_for,
            expected=expected_required_for,
        ):
            raise ValueError(
                "ArtifactDescriptor.required_for differs from model authority."
            )
        if type(self.applicable) is not bool or type(self.required) is not bool:
            raise TypeError(
                "Artifact applicability and requiredness must be exact bools."
            )
        if self.descriptor.required is not self.required:
            raise ValueError(
                "Artifact descriptive requiredness differs from authority."
            )
        _validate_axes_and_leaves(axes=axes, leaves=tuple(leaves.values()))

        template = self.template
        template_snapshot: _CanonicalArtifactTemplate | None = None
        public_template_leaves: tuple[jax.Array, ...] = ()
        if template is None:
            if containers or leaves:
                raise ValueError(
                    "An authority without a materialization template cannot declare "
                    "containers or leaves."
                )
        else:
            template_snapshot = _canonicalize_declared_template_snapshot(
                template=template,
                payload_runtime_type=self.payload_runtime_type,
                containers=containers,
                leaves=leaves,
            )
            public_template_leaves = template_snapshot.leaves
            template = template_snapshot.payload
            template_snapshot = _rebuild_cached_artifact_template(
                template=template,
                snapshot=template_snapshot,
                payload_runtime_type=self.payload_runtime_type,
                containers=containers,
                leaves=leaves,
            )
        object.__setattr__(
            self, "container_runtime_types", MappingProxyType(containers)
        )
        object.__setattr__(self, "leaves", MappingProxyType(leaves))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "state_roles", state_roles)
        object.__setattr__(self, "action_roles", action_roles)
        object.__setattr__(self, "categorical_domains", MappingProxyType(categories))
        object.__setattr__(self, "template", template)
        _bind_artifact_authority_template(
            authority=self,
            snapshot=template_snapshot,
            public_template_leaves=public_template_leaves,
        )

    def __copy__(self) -> ArtifactAuthority:
        """Return an unbound field copy that cannot inherit private authority."""
        return _copy_artifact_authority_without_binding(authority=self)

    def __deepcopy__(self, memo: dict[int, object], /) -> ArtifactAuthority:
        """Return a detached but unbound copy without invoking plugin callbacks."""
        canonical = _restore_artifact_authority_from_pickle(
            _artifact_authority_pickle_state(authority=self)
        )
        copied = _copy_artifact_authority_without_binding(authority=canonical)
        memo[id(self)] = copied
        return copied

    def __reduce_ex__(
        self, protocol: SupportsIndex, /
    ) -> tuple[object, tuple[object, ...]]:
        """Transport a sealed declaration and rebind it without plugin callbacks."""
        del protocol
        return (
            _restore_artifact_authority_from_pickle,
            (_artifact_authority_pickle_state(authority=self),),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArtifactAuthorityTemplateBinding:
    """Private write-once template binding for one live authority identity."""

    authority_ref: weakref.ReferenceType[ArtifactAuthority]
    template: object | None
    payload_runtime_type: type[object]
    container_runtime_types: Mapping[TreePath, type[object]]
    leaves: Mapping[TreePath, LeafAuthority]
    snapshot: _CanonicalArtifactTemplate | None
    public_template_leaves: tuple[jax.Array, ...]


_ARTIFACT_AUTHORITY_TEMPLATE_LOCK = RLock()
_ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS: dict[int, _ArtifactAuthorityTemplateBinding] = {}
_ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING: dict[
    int, weakref.ReferenceType[ArtifactAuthority]
] = {}


# keyword-only-exempt: library-callback=weakref.ref
def _discard_initializing_identity(
    dead_ref: weakref.ReferenceType[ArtifactAuthority], *, identity: int
) -> None:
    """Forget an initializing authority identity once its referent is collected."""
    with _ARTIFACT_AUTHORITY_TEMPLATE_LOCK:
        current_ref = _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING.get(identity)
        if current_ref is dead_ref:
            del _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING[identity]


# keyword-only-exempt: library-callback=weakref.ref
def _discard_binding_identity(
    dead_ref: weakref.ReferenceType[ArtifactAuthority], *, identity: int
) -> None:
    """Forget a template binding once the authority it belongs to is collected."""
    with _ARTIFACT_AUTHORITY_TEMPLATE_LOCK:
        current = _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS.get(identity)
        if current is not None and current.authority_ref is dead_ref:
            del _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS[identity]


def _assert_artifact_authority_unbound(authority: ArtifactAuthority) -> None:
    """Reject constructor re-entry before touching any caller-replaced field."""
    identity = id(authority)
    with _ARTIFACT_AUTHORITY_TEMPLATE_LOCK:
        current = _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS.get(identity)
        if current is not None and current.authority_ref() is authority:
            raise TypeError("Artifact authority template binding is write-once.")
        if current is not None and current.authority_ref() is not None:
            raise TypeError("Artifact authority identity collides with a live binding.")
        initializing = _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING.get(identity)
        if initializing is not None and initializing() is authority:
            raise TypeError("Artifact authority template initialization is re-entrant.")
        if initializing is not None and initializing() is not None:
            raise TypeError(
                "Artifact authority identity collides with live initialization."
            )

        _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING[identity] = weakref.ref(
            authority,
            functools.partial(_discard_initializing_identity, identity=identity),
        )


def _bind_artifact_authority_template(
    *,
    authority: ArtifactAuthority,
    snapshot: _CanonicalArtifactTemplate | None,
    public_template_leaves: tuple[jax.Array, ...],
) -> None:
    """Bind one authority identity exactly once outside caller-visible fields."""
    if type(authority) is not ArtifactAuthority:
        raise TypeError("Only an exact ArtifactAuthority can own a template binding.")
    if snapshot is not None and type(snapshot) is not _CanonicalArtifactTemplate:
        raise TypeError("Artifact template bindings require an exact snapshot.")
    if type(public_template_leaves) is not tuple or any(
        not isinstance(leaf, jax.Array) for leaf in public_template_leaves
    ):
        raise TypeError("Artifact template bindings require exact public leaves.")
    identity = id(authority)

    authority_ref = weakref.ref(
        authority,
        functools.partial(_discard_binding_identity, identity=identity),
    )
    binding = _ArtifactAuthorityTemplateBinding(
        authority_ref=authority_ref,
        template=authority.template,
        payload_runtime_type=authority.payload_runtime_type,
        container_runtime_types=authority.container_runtime_types,
        leaves=authority.leaves,
        snapshot=snapshot,
        public_template_leaves=public_template_leaves,
    )
    with _ARTIFACT_AUTHORITY_TEMPLATE_LOCK:
        current = _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS.get(identity)
        if current is not None and current.authority_ref() is authority:
            raise TypeError("Artifact authority template binding is write-once.")
        if current is not None and current.authority_ref() is not None:
            raise TypeError("Artifact authority identity collides with a live binding.")
        initializing = _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING.get(identity)
        if initializing is not None and initializing() is not authority:
            raise TypeError(
                "Artifact authority identity collides with live initialization."
            )
        if initializing is not None:
            del _ARTIFACT_AUTHORITY_TEMPLATE_INITIALIZING[identity]
        _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS[identity] = binding


def _artifact_authority_template_snapshot(
    authority: ArtifactAuthority,
) -> _CanonicalArtifactTemplate | None:
    """Return a detached declaration from one identity-bound private snapshot."""
    if type(authority) is not ArtifactAuthority:
        raise TypeError("Artifact authorities must be exact ArtifactAuthority objects.")

    # Pin and compare all reconstruction-bearing fields before any mapping traversal.
    template = authority.template
    payload_runtime_type = authority.payload_runtime_type
    containers = authority.container_runtime_types
    leaves = authority.leaves
    identity = id(authority)
    with _ARTIFACT_AUTHORITY_TEMPLATE_LOCK:
        binding = _ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS.get(identity)
        if binding is None or binding.authority_ref() is not authority:
            raise TypeError("Artifact authority has no identity-bound template.")
        if (
            template is not binding.template
            or payload_runtime_type is not binding.payload_runtime_type
            or containers is not binding.container_runtime_types
            or leaves is not binding.leaves
        ):
            raise TypeError(
                "Artifact authority reconstruction fields differ from its binding."
            )
        snapshot = binding.snapshot
        public_template_leaves = binding.public_template_leaves

    if snapshot is None:
        if template is not None or containers or leaves or public_template_leaves:
            raise TypeError("Artifact authority no-template binding is inconsistent.")
        return None
    if template is None:
        raise TypeError("Artifact authority template binding is inconsistent.")
    plan = _snapshot_artifact_construction_plan(
        plan=snapshot.construction_plan,
        leaf_count=len(public_template_leaves),
    )
    _validate_artifact_value_against_plan(
        payload=template,
        plan=plan,
        leaves=public_template_leaves,
    )
    if len(public_template_leaves) != len(leaves):
        raise TypeError("Artifact authority public leaf binding is inconsistent.")
    for path, public_leaf in zip(leaves, public_template_leaves, strict=True):
        declaration = leaves[path]
        if (
            public_leaf.is_deleted()
            or tuple(public_leaf.shape) != declaration.shape
            or np.dtype(public_leaf.dtype) != np.dtype(declaration.dtype)
        ):
            raise TypeError(
                f"Artifact authority public template leaf {path!r} differs from its "
                "binding."
            )
    return _rebuild_cached_artifact_template(
        template=snapshot.payload,
        snapshot=snapshot,
        payload_runtime_type=payload_runtime_type,
        containers=containers,
        leaves=leaves,
    )


def _check_role_names(*, names: tuple[str, ...], label: str) -> None:
    """Reject ambiguous or non-built-in state/action role declarations."""
    if any(type(name) is not str or not name for name in names):
        raise TypeError(f"Artifact {label} roles must be nonempty exact strs.")
    if len(set(names)) != len(names):
        raise ValueError(f"Artifact {label} roles must be unique.")


def _normalize_jax_tree_path(path: tuple[object, ...]) -> TreePath:  # noqa: C901
    """Encode a JAX key path without collapsing distinct key kinds or types."""
    normalized: list[str] = []
    for component in path:
        if type(component) is jax.tree_util.GetAttrKey:
            attribute_key = cast("jax.tree_util.GetAttrKey", component)
            if type(attribute_key.name) is not str or not attribute_key.name:
                raise TypeError("Artifact attribute TreePath keys must be exact strs.")
            normalized.append(f"attribute:{attribute_key.name}")
        elif type(component) is jax.tree_util.SequenceKey:
            sequence_key = cast("jax.tree_util.SequenceKey", component)
            if type(sequence_key.idx) is not int or sequence_key.idx < 0:
                raise TypeError("Artifact sequence TreePath keys must be exact ints.")
            normalized.append(f"sequence:{sequence_key.idx}")
        elif type(component) is jax.tree_util.FlattenedIndexKey:
            flattened_key = cast("jax.tree_util.FlattenedIndexKey", component)
            if type(flattened_key.key) is not int or flattened_key.key < 0:
                raise TypeError("Artifact flattened TreePath keys must be exact ints.")
            normalized.append(f"flattened:{flattened_key.key}")
        elif type(component) is jax.tree_util.DictKey:
            key = cast("jax.tree_util.DictKey", component).key
            if not any(type(key) is allowed for allowed in (bool, int, float, str)):
                raise TypeError(
                    "Artifact mapping TreePath keys must be exact JSON scalars."
                )
            if type(key) is float and not np.isfinite(key):
                raise ValueError("Artifact mapping TreePath float keys must be finite.")
            normalized.append(f"mapping:{type(key).__name__}:{key!r}")
        else:
            raise TypeError(
                f"Unsupported artifact TreePath component {type(component).__name__}."
            )
    return tuple(normalized)


def _collect_container_types(  # noqa: C901
    *,
    node: jax.tree_util.PyTreeDef,
    path: TreePath,
    leaf_offset: int,
    leaf_paths: tuple[TreePath, ...],
    containers: dict[TreePath, type[object]],
) -> int:
    """Record one PyTree node's container class and descend into its children."""
    children = node.children()
    node_data = node.node_data()
    if node_data is None:
        if children or node.num_leaves != 1:
            raise TypeError("Artifact PyTree exposes an invalid numerical leaf.")
        return leaf_offset + 1
    if (
        type(node_data) is not tuple
        or not node_data
        or not isinstance(node_data[0], type)
    ):
        raise TypeError("Artifact PyTree exposes no exact container runtime type.")
    runtime_type = node_data[0]
    if node.num_leaves == 0:
        if runtime_type is not tuple and runtime_type is not type(None):
            raise TypeError(
                "Zero-leaf artifact PyTree node at "
                f"{path!r} must be an exact tuple or NoneType; got "
                f"{runtime_type.__name__}."
            )
        for index, child in enumerate(children):
            child_path = (*path, f"pytree-child:{index}")
            leaf_offset = _collect_container_types(
                leaf_paths=leaf_paths,
                containers=containers,
                node=child,
                path=child_path,
                leaf_offset=leaf_offset,
            )
        return leaf_offset

    if path in containers:
        raise TypeError(f"Artifact PyTree container path {path!r} is ambiguous.")
    containers[path] = runtime_type
    for index, child in enumerate(children):
        child_leaves = child.num_leaves
        if child_leaves == 0:
            child_path = (*path, f"pytree-child:{index}")
        else:
            if leaf_offset >= len(leaf_paths):
                raise TypeError("Artifact PyTree paths do not cover its containers.")
            child_path = leaf_paths[leaf_offset][: len(path) + 1]
            if len(child_path) != len(path) + 1:
                raise TypeError(
                    "Artifact PyTree paths do not identify every container child."
                )
        leaf_offset = _collect_container_types(
            leaf_paths=leaf_paths,
            containers=containers,
            node=child,
            path=child_path,
            leaf_offset=leaf_offset,
        )
    return leaf_offset


def _container_types_from_tree(
    *, tree: jax.tree_util.PyTreeDef, leaf_paths: tuple[TreePath, ...]
) -> dict[TreePath, type[object]]:
    """Derive exact container classes from a PyTreeDef and its ordered leaf paths."""
    containers: dict[TreePath, type[object]] = {}

    consumed = _collect_container_types(
        leaf_paths=leaf_paths, containers=containers, node=tree, path=(), leaf_offset=0
    )
    if consumed != len(leaf_paths):
        raise TypeError("Artifact PyTree paths do not cover its leaves exactly.")
    return containers


def _same_container_runtime_types(
    *,
    actual: Mapping[TreePath, type[object]],
    expected: Mapping[TreePath, type[object]],
) -> bool:
    """Require identical paths and class identities without metaclass equality."""
    return len(actual) == len(expected) and all(
        path in expected and actual_type is expected[path]
        for path, actual_type in actual.items()
    )


def _check_approved_artifact_containers(
    *,
    payload_runtime_type: type[object],
    container_runtime_types: Mapping[TreePath, type[object]],
) -> None:
    """Accept only closed container layouts that can form detached snapshots."""
    if container_runtime_types:
        if container_runtime_types.get(()) is not payload_runtime_type:
            raise TypeError(
                "The root artifact container type differs from payload_runtime_type."
            )
    elif payload_runtime_type is not jax.Array:
        raise TypeError(
            "A non-array artifact payload must declare its root container type."
        )
    for path, runtime_type in container_runtime_types.items():
        if runtime_type is tuple:
            continue
        if _frozen_dataclass_layout(runtime_type) is not None:
            continue
        raise TypeError(
            f"Artifact container at {path!r} must be an exact tuple or a closed "
            f"dataclass record; got unsupported {runtime_type.__name__}."
        )


def _validate_axes_and_leaves(
    *, axes: tuple[AxisAuthority, ...], leaves: tuple[LeafAuthority, ...]
) -> None:
    """Bind every leaf dimension to one exact named authoritative axis."""
    axes_by_name = {axis.name: axis for axis in axes}
    if len(axes_by_name) != len(axes):
        raise ValueError("ArtifactAuthority axis names must be unique.")
    for leaf in leaves:
        for size, axis_name in zip(leaf.shape, leaf.axis_names, strict=True):
            axis = axes_by_name.get(axis_name)
            if axis is None:
                raise ValueError(
                    f"Artifact leaf {leaf.path!r} names undeclared axis {axis_name!r}."
                )
            if axis.length != size:
                raise ValueError(
                    f"Artifact axis {axis_name!r} has length {axis.length}; leaf "
                    f"{leaf.path!r} requires {size}."
                )


def _payload_has_runtime_type(*, payload: object, expected: type[object]) -> bool:
    """Treat the public JAX Array ABC as the one intentionally polymorphic leaf."""
    return (
        isinstance(payload, jax.Array)
        if expected is jax.Array
        else type(payload) is expected
    )


def _copy_artifact_array_leaf(*, leaf: object, label: str) -> jax.Array:
    """Make one independent exact JAX buffer for an owned artifact graph."""
    if not isinstance(leaf, jax.Array):
        raise TypeError(f"{label} is not a JAX array.")
    if leaf.is_deleted():
        raise TypeError(f"{label} has been deleted or donated.")
    try:
        source_shape = tuple(leaf.shape)
        source_dtype = np.dtype(leaf.dtype)
        source_sharding = leaf.sharding
        copied = jax.numpy.array(leaf, copy=True)
    except TypeError:
        raise
    except Exception as error:
        raise TypeError(f"{label} cannot be copied safely.") from error
    if copied is leaf:
        raise TypeError(f"{label} was not copied into an independent array.")
    if (
        tuple(copied.shape) != source_shape
        or np.dtype(copied.dtype) != source_dtype
        or copied.sharding != source_sharding
    ):
        raise TypeError(
            f"{label} cannot be copied with exact shape, dtype, and sharding."
        )
    return copied


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


def _snapshot_pytree_def(
    tree: jax.tree_util.PyTreeDef,
) -> jax.tree_util.PyTreeDef:
    """Rebuild one tree definition with independently owned inert node metadata."""
    node_data = tree.node_data()
    if node_data is None:
        owned_node_data = None
    else:
        if (
            type(node_data) is not tuple
            or len(node_data) != 2  # noqa: PLR2004
            or not isinstance(node_data[0], type)
        ):
            raise TypeError("Artifact PyTree node data has an unsupported structure.")
        owned_node_data = (
            node_data[0],
            _snapshot_inert_pytree_metadata(value=node_data[1]),
        )
    owned_children = tuple(_snapshot_pytree_def(child) for child in tree.children())
    return tree.from_node_data_and_children(
        jax.tree_util.default_registry,
        owned_node_data,
        owned_children,
    )


def _same_pytree_node_structure(
    *, actual: jax.tree_util.PyTreeDef, expected: jax.tree_util.PyTreeDef
) -> bool:
    """Compare node types and child topology without consulting auxiliary equality."""
    actual_data = actual.node_data()
    expected_data = expected.node_data()
    if (actual_data is None) != (expected_data is None):
        return False
    if (
        actual_data is not None
        and expected_data is not None
        and actual_data[0] is not expected_data[0]
    ):
        return False
    actual_children = actual.children()
    expected_children = expected.children()
    return len(actual_children) == len(expected_children) and all(
        _same_pytree_node_structure(actual=a, expected=e)
        for a, e in zip(actual_children, expected_children, strict=True)
    )


def _same_exact_pytree_def(
    *, actual: jax.tree_util.PyTreeDef, expected: jax.tree_util.PyTreeDef
) -> bool:
    """Compare every node's exact inert auxiliary metadata recursively."""
    if not _same_pytree_node_structure(actual=actual, expected=expected):
        return False
    actual_data = actual.node_data()
    expected_data = expected.node_data()
    if (
        actual_data is not None
        and expected_data is not None
        and not _same_inert_pytree_metadata(
            actual=actual_data[1],
            expected=expected_data[1],
        )
    ):
        return False
    return all(
        _same_exact_pytree_def(actual=a, expected=e)
        for a, e in zip(actual.children(), expected.children(), strict=True)
    )


def _frozen_dataclass_layout(  # noqa: C901, PLR0911, PLR0912
    cls: type[object],
) -> tuple[tuple[str, ...], tuple[bool, ...], object | None, dict[str, object]] | None:
    """Return one closed dataclass record's callback-free storage layout."""
    try:
        class_mro = type.__getattribute__(cls, "__mro__")
    except TypeError:
        return None
    dataclass_fields: object | None = None
    dataclass_params: object | None = None
    for base in class_mro:
        namespace = type.__getattribute__(base, "__dict__")
        if dataclass_fields is None and "__dataclass_fields__" in namespace:
            dataclass_fields = namespace["__dataclass_fields__"]
        if dataclass_params is None and "__dataclass_params__" in namespace:
            dataclass_params = namespace["__dataclass_params__"]
    authority_namespace = type.__getattribute__(ArtifactAuthority, "__dict__")
    authority_params = authority_namespace["__dataclass_params__"]
    authority_fields = authority_namespace["__dataclass_fields__"]
    data_field_kind = authority_fields["descriptor"]._field_type  # noqa: SLF001
    if (
        type(dataclass_fields) is not dict
        or type(dataclass_params) is not type(authority_params)
        or cast("Any", dataclass_params).frozen is not True
    ):
        return None
    field_names: list[str] = []
    for key, field_info in dataclass_fields.items():
        if (
            type(key) is not str
            or type(field_info) is not dataclasses.Field
            or type(field_info.name) is not str
            or field_info.name != key
        ):
            return None
        if field_info._field_type is data_field_kind:  # noqa: SLF001
            field_names.append(key)
    if len(set(field_names)) != len(field_names):
        return None

    dict_descriptor: object | None = None
    slot_descriptors: dict[str, object] = {}
    for base in reversed(class_mro):
        namespace = type.__getattribute__(base, "__dict__")
        candidate_dict_descriptor = namespace.get("__dict__")
        if candidate_dict_descriptor is not None:
            if type(candidate_dict_descriptor) is not GetSetDescriptorType:
                return None
            if dict_descriptor is not None:
                return None
            dict_descriptor = candidate_dict_descriptor
        raw_slots = namespace.get("__slots__", ())
        if type(raw_slots) is str:
            slots = (raw_slots,)
        elif type(raw_slots) is tuple and all(type(name) is str for name in raw_slots):
            slots = raw_slots
        else:
            return None
        for name in slots:
            if name in {"__dict__", "__weakref__"}:
                continue
            descriptor = namespace.get(name)
            if type(descriptor) is not MemberDescriptorType:
                return None
            if name in slot_descriptors:
                return None
            slot_descriptors[name] = descriptor

    field_name_set = set(field_names)
    if set(slot_descriptors) - field_name_set:
        return None
    stored_in_dict: list[bool] = []
    for name in field_names:
        if name in slot_descriptors:
            stored_in_dict.append(False)
        elif dict_descriptor is not None:
            stored_in_dict.append(True)
        else:
            return None
    return (
        tuple(field_names),
        tuple(stored_in_dict),
        dict_descriptor,
        slot_descriptors,
    )


def _artifact_dataclass_field_values(
    value: object,
) -> tuple[tuple[str, bool, object], ...]:
    """Read every exact instance field without invoking user attribute methods."""
    runtime_type = type(value)
    layout = _frozen_dataclass_layout(runtime_type)
    if layout is None:
        raise TypeError("Artifact construction requires a closed dataclass record.")
    field_names, stored_in_dict, dict_descriptor, slot_descriptors = layout
    instance_dict: dict[str, object]
    if dict_descriptor is None:
        instance_dict = {}
    else:
        try:
            raw_dict = cast("Any", dict_descriptor).__get__(value, runtime_type)
        except Exception as error:
            raise TypeError(
                "Artifact dataclass storage cannot be read safely."
            ) from error
        if type(raw_dict) is not dict or any(type(key) is not str for key in raw_dict):
            raise TypeError(
                "Artifact dataclass instance storage must be an exact dict."
            )
        instance_dict = cast("dict[str, object]", raw_dict)
    expected_dict_names = {
        name
        for name, is_dict_field in zip(field_names, stored_in_dict, strict=True)
        if is_dict_field
    }
    if set(instance_dict) != expected_dict_names:
        raise TypeError(
            "Artifact dataclass has missing or hidden instance dictionary state."
        )

    result: list[tuple[str, bool, object]] = []
    for name, is_dict_field in zip(field_names, stored_in_dict, strict=True):
        if is_dict_field:
            field_value = instance_dict[name]
        else:
            descriptor = slot_descriptors[name]
            try:
                field_value = cast("Any", descriptor).__get__(value, runtime_type)
            except Exception as error:
                raise TypeError(
                    f"Artifact dataclass slot {name!r} cannot be read safely."
                ) from error
        result.append((name, is_dict_field, field_value))
    return tuple(result)


def _artifact_leaf_slot_from_callback_token(
    *,
    token: _ArtifactLeafToken,
    tokens: tuple[_ArtifactLeafToken, ...],
    seen: list[int],
) -> _ArtifactLeafSlot:
    """Validate one opaque callback token and record its exact leaf slot."""
    if (
        type(token.index) is not int
        or token.index < 0
        or token.index >= len(tokens)
        or token is not tokens[token.index]
    ):
        raise TypeError("Artifact unflatten callback forged a leaf token.")
    seen[token.index] += 1
    return _ArtifactLeafSlot(index=token.index)


def _artifact_static_plan_from_callback_value(
    *, value: object
) -> _ArtifactStaticPlan | None:
    """Own token-free callback metadata, leaving dynamic records structural."""
    value_type = type(value)
    if (
        value_type is not tuple
        and _artifact_static_metadata_field_names(value_type) is None
    ):
        return None
    try:
        static_value = _snapshot_inert_pytree_metadata(value=value)
    except TypeError:
        return None
    return _ArtifactStaticPlan(value=static_value, validate_payload=False)


def _construction_plan_from_callback_value(
    *,
    value: object,
    tokens: tuple[_ArtifactLeafToken, ...],
    seen: list[int],
    active_ids: set[int],
    encountered_ids: set[int],
) -> object:
    """Own one callback result while retaining no callback-returned reference."""
    value_type = type(value)
    if value_type is _ArtifactLeafToken:
        return _artifact_leaf_slot_from_callback_token(
            token=cast("_ArtifactLeafToken", value),
            tokens=tokens,
            seen=seen,
        )

    static_plan = _artifact_static_plan_from_callback_value(value=value)
    if static_plan is not None:
        return static_plan

    if value_type is tuple:
        marker = id(value)
        if value and marker in encountered_ids:
            raise TypeError("Artifact construction graph must not alias containers.")
        if value:
            encountered_ids.add(marker)
        if marker in active_ids:
            raise TypeError("Artifact construction graph must be acyclic.")
        active_ids.add(marker)
        try:
            return _ArtifactTuplePlan(
                children=tuple(
                    _construction_plan_from_callback_value(
                        value=child,
                        tokens=tokens,
                        seen=seen,
                        active_ids=active_ids,
                        encountered_ids=encountered_ids,
                    )
                    for child in cast("tuple[object, ...]", value)
                )
            )
        finally:
            active_ids.remove(marker)

    if _frozen_dataclass_layout(value_type) is not None:
        marker = id(value)
        if marker in encountered_ids:
            raise TypeError("Artifact construction graph must not alias containers.")
        encountered_ids.add(marker)
        if marker in active_ids:
            raise TypeError("Artifact construction graph must be acyclic.")
        active_ids.add(marker)
        try:
            fields = tuple(
                _ArtifactDataclassFieldPlan(
                    name=name,
                    stored_in_dict=stored_in_dict,
                    value=_construction_plan_from_callback_value(
                        value=field_value,
                        tokens=tokens,
                        seen=seen,
                        active_ids=active_ids,
                        encountered_ids=encountered_ids,
                    ),
                )
                for name, stored_in_dict, field_value in (
                    _artifact_dataclass_field_values(value)
                )
            )
            return _ArtifactDataclassPlan(
                runtime_type=value_type,
                fields=fields,
            )
        finally:
            active_ids.remove(marker)

    return _ArtifactStaticPlan(
        value=_snapshot_inert_pytree_metadata(value=value),
        validate_payload=False,
    )


def _mark_static_provenance_node(
    *, node: object, source: object, missing: object
) -> object:
    """Copy one plan node, marking static metadata the declaration payload carried."""
    node_type = type(node)
    if node_type is _ArtifactLeafSlot:
        return node
    if node_type is _ArtifactStaticPlan:
        static = cast("_ArtifactStaticPlan", node)
        represented = False
        if source is not missing:
            try:
                represented = _same_inert_pytree_metadata(
                    actual=source,
                    expected=static.value,
                )
            except TypeError:
                represented = False
        return _ArtifactStaticPlan(
            value=_snapshot_inert_pytree_metadata(value=static.value),
            validate_payload=represented,
        )
    if node_type is _ArtifactTuplePlan:
        tuple_plan = cast("_ArtifactTuplePlan", node)
        source_children = (
            source
            if type(source) is tuple and len(source) == len(tuple_plan.children)
            else None
        )
        return _ArtifactTuplePlan(
            children=tuple(
                _mark_static_provenance_node(
                    missing=missing,
                    node=child,
                    source=(
                        source_children[index]
                        if source_children is not None
                        else missing
                    ),
                )
                for index, child in enumerate(tuple_plan.children)
            )
        )
    if node_type is not _ArtifactDataclassPlan:
        raise TypeError("Artifact construction plan contains an unsupported node.")
    dataclass_plan = cast("_ArtifactDataclassPlan", node)
    source_fields: dict[tuple[str, bool], object] = {}
    if type(source) is dataclass_plan.runtime_type:
        try:
            source_fields = {
                (name, stored_in_dict): value
                for name, stored_in_dict, value in (
                    _artifact_dataclass_field_values(source)
                )
            }
        except TypeError:
            source_fields = {}
    return _ArtifactDataclassPlan(
        runtime_type=dataclass_plan.runtime_type,
        fields=tuple(
            _ArtifactDataclassFieldPlan(
                name=field.name,
                stored_in_dict=field.stored_in_dict,
                value=_mark_static_provenance_node(
                    missing=missing,
                    node=field.value,
                    source=source_fields.get(
                        (field.name, field.stored_in_dict),
                        missing,
                    ),
                ),
            )
            for field in dataclass_plan.fields
        ),
    )


def _mark_artifact_static_provenance(*, plan: object, payload: object) -> object:
    """Mark static fields represented identically by the declaration payload.

    A custom unflatten callback may inject instance fields which its flatten callback
    does not represent. Those fields belong to the sealed construction plan and are
    canonicalized on publication. Static fields already present with the same exact
    inert value in the declaration payload are payload-owned metadata and must match
    on every later publication.
    """
    missing = object()

    return _mark_static_provenance_node(missing=missing, node=plan, source=payload)


def _compile_artifact_construction_plan(
    *,
    tree: jax.tree_util.PyTreeDef,
    payload_runtime_type: type[object],
    declaration_payload: object,
) -> object:
    """Compile one callback result into a closed callback-free construction plan."""
    if tree.node_data() is None:
        if tree.num_leaves != 1 or payload_runtime_type is not jax.Array:
            raise TypeError(
                "A root artifact leaf must declare jax.Array as its runtime type."
            )
        return _ArtifactLeafSlot(index=0)
    tokens = tuple(_ArtifactLeafToken(index=index) for index in range(tree.num_leaves))
    try:
        candidate = jax.tree_util.tree_unflatten(tree, tokens)
    except Exception as error:
        raise TypeError(
            "Artifact template unflatten callback cannot accept opaque leaf tokens."
        ) from error
    if not _payload_has_runtime_type(
        payload=candidate,
        expected=payload_runtime_type,
    ):
        raise TypeError(
            "Artifact template unflatten callback returned a different payload type."
        )
    seen = [0] * len(tokens)
    plan = _construction_plan_from_callback_value(
        value=candidate,
        tokens=tokens,
        seen=seen,
        active_ids=set(),
        encountered_ids=set(),
    )
    if any(count != 1 for count in seen):
        raise TypeError(
            "Artifact unflatten callback must preserve every leaf token exactly once."
        )
    return _mark_artifact_static_provenance(
        plan=plan,
        payload=declaration_payload,
    )


def _snapshot_plan_node(  # noqa: C901, PLR0912
    *, node: object, seen: list[int], active_ids: set[int]
) -> object:
    """Validate and copy one plan node; `seen` counts every leaf slot met."""
    node_type = type(node)
    if node_type is _ArtifactLeafSlot:
        slot = cast("_ArtifactLeafSlot", node)
        if type(slot.index) is not int or slot.index < 0 or slot.index >= len(seen):
            raise TypeError("Artifact construction plan has an invalid leaf slot.")
        seen[slot.index] += 1
        return _ArtifactLeafSlot(index=slot.index)
    if node_type is _ArtifactStaticPlan:
        static = cast("_ArtifactStaticPlan", node)
        if type(static.validate_payload) is not bool:
            raise TypeError(
                "Artifact static plan validation marker must be an exact bool."
            )
        return _ArtifactStaticPlan(
            value=_snapshot_inert_pytree_metadata(value=static.value),
            validate_payload=static.validate_payload,
        )
    if node_type is _ArtifactTuplePlan:
        tuple_plan = cast("_ArtifactTuplePlan", node)
        if type(tuple_plan.children) is not tuple:
            raise TypeError("Artifact tuple plan children must be an exact tuple.")
        marker = id(node)
        if marker in active_ids:
            raise TypeError("Artifact construction plan must be acyclic.")
        active_ids.add(marker)
        try:
            return _ArtifactTuplePlan(
                children=tuple(
                    _snapshot_plan_node(seen=seen, active_ids=active_ids, node=child)
                    for child in tuple_plan.children
                )
            )
        finally:
            active_ids.remove(marker)
    if node_type is _ArtifactDataclassPlan:
        dataclass_plan = cast("_ArtifactDataclassPlan", node)
        if not isinstance(dataclass_plan.runtime_type, type):
            raise TypeError("Artifact dataclass plan runtime type is invalid.")
        layout = _frozen_dataclass_layout(dataclass_plan.runtime_type)
        if layout is None or type(dataclass_plan.fields) is not tuple:
            raise TypeError("Artifact dataclass plan is not structurally valid.")
        field_names, stored_in_dict, _dict_descriptor, _slots = layout
        if len(dataclass_plan.fields) != len(field_names):
            raise TypeError("Artifact dataclass plan fields are incomplete.")
        marker = id(node)
        if marker in active_ids:
            raise TypeError("Artifact construction plan must be acyclic.")
        active_ids.add(marker)
        try:
            copied_fields: list[_ArtifactDataclassFieldPlan] = []
            for field_plan, expected_name, expected_storage in zip(
                dataclass_plan.fields,
                field_names,
                stored_in_dict,
                strict=True,
            ):
                if (
                    type(field_plan) is not _ArtifactDataclassFieldPlan
                    or field_plan.name != expected_name
                    or type(field_plan.stored_in_dict) is not bool
                    or field_plan.stored_in_dict is not expected_storage
                ):
                    raise TypeError(
                        "Artifact dataclass plan fields differ from exact storage."
                    )
                copied_fields.append(
                    _ArtifactDataclassFieldPlan(
                        name=expected_name,
                        stored_in_dict=expected_storage,
                        value=_snapshot_plan_node(
                            seen=seen, active_ids=active_ids, node=field_plan.value
                        ),
                    )
                )
            return _ArtifactDataclassPlan(
                runtime_type=dataclass_plan.runtime_type,
                fields=tuple(copied_fields),
            )
        finally:
            active_ids.remove(marker)
    raise TypeError("Artifact construction plan contains an unsupported node.")


def _snapshot_artifact_construction_plan(*, plan: object, leaf_count: int) -> object:
    """Validate and detach one closed construction plan without plugin callbacks."""
    seen = [0] * leaf_count
    active_ids: set[int] = set()

    copied = _snapshot_plan_node(seen=seen, active_ids=active_ids, node=plan)
    if any(count != 1 for count in seen):
        raise TypeError(
            "Artifact construction plan must contain every leaf slot exactly once."
        )
    return copied


def _reconstruct_plan_node(*, node: object, leaves: tuple[object, ...]) -> object:  # noqa: C901, PLR0912
    """Build one node of an owned payload graph from the plan and its leaves."""
    node_type = type(node)
    if node_type is _ArtifactLeafSlot:
        index = cast("_ArtifactLeafSlot", node).index
        if type(index) is not int or index < 0 or index >= len(leaves):
            raise TypeError("Artifact construction plan has an invalid leaf slot.")
        return leaves[index]
    if node_type is _ArtifactStaticPlan:
        return _snapshot_inert_pytree_metadata(
            value=cast("_ArtifactStaticPlan", node).value
        )
    if node_type is _ArtifactTuplePlan:
        children = cast("_ArtifactTuplePlan", node).children
        if type(children) is not tuple:
            raise TypeError("Artifact tuple plan children must be an exact tuple.")
        return tuple(
            _reconstruct_plan_node(leaves=leaves, node=child) for child in children
        )
    if node_type is not _ArtifactDataclassPlan:
        raise TypeError("Artifact construction plan contains an unsupported node.")

    dataclass_plan = cast("_ArtifactDataclassPlan", node)
    layout = _frozen_dataclass_layout(dataclass_plan.runtime_type)
    if layout is None:
        raise TypeError("Artifact dataclass plan is not structurally valid.")
    field_names, stored_in_dict, dict_descriptor, slot_descriptors = layout
    if type(dataclass_plan.fields) is not tuple or len(dataclass_plan.fields) != len(
        field_names
    ):
        raise TypeError("Artifact dataclass plan fields are incomplete.")
    values: list[tuple[str, bool, object]] = []
    for field_plan, expected_name, expected_storage in zip(
        dataclass_plan.fields,
        field_names,
        stored_in_dict,
        strict=True,
    ):
        if (
            type(field_plan) is not _ArtifactDataclassFieldPlan
            or field_plan.name != expected_name
            or field_plan.stored_in_dict is not expected_storage
        ):
            raise TypeError("Artifact dataclass plan fields differ from exact storage.")
        values.append(
            (
                expected_name,
                expected_storage,
                _reconstruct_plan_node(leaves=leaves, node=field_plan.value),
            )
        )
    try:
        instance = object.__new__(dataclass_plan.runtime_type)
    except Exception as error:
        raise TypeError(
            "Artifact dataclass cannot be allocated without its constructor."
        ) from error
    if type(instance) is not dataclass_plan.runtime_type:
        raise TypeError("Artifact dataclass allocation returned a different type.")
    if dict_descriptor is not None:
        try:
            instance_dict = cast("Any", dict_descriptor).__get__(
                instance,
                dataclass_plan.runtime_type,
            )
        except Exception as error:
            raise TypeError(
                "Artifact dataclass dictionary cannot be initialized safely."
            ) from error
        if type(instance_dict) is not dict or instance_dict:
            raise TypeError(
                "A fresh artifact dataclass has unexpected dictionary state."
            )
        for name, is_dict_field, field_value in values:
            if is_dict_field:
                instance_dict[name] = field_value
    for name, is_dict_field, field_value in values:
        if not is_dict_field:
            try:
                cast("Any", slot_descriptors[name]).__set__(
                    instance,
                    field_value,
                )
            except Exception as error:
                msg = f"Artifact dataclass slot {name!r} cannot be initialized safely."
                raise TypeError(msg) from error
    return instance


def _reconstruct_artifact_from_plan(
    *, plan: object, leaves: tuple[object, ...]
) -> object:
    """Build an owned payload graph without invoking plugin-owned code."""

    return _reconstruct_plan_node(leaves=leaves, node=plan)


@dataclass(slots=True, kw_only=True)
class _LeafExtraction:
    """Working state of one callback-free leaf extraction."""

    seen: list[int]
    """Number of times each leaf slot has been met."""
    extracted: list[object]
    """Leaf value per slot, `missing` until the slot is met."""
    missing: object
    """Sentinel for a slot no payload field has filled."""
    active_ids: set[int]
    """Identities of containers on the current descent path."""
    encountered_ids: set[int]
    """Identities of every non-empty container met, to refuse aliasing."""
    validate_static: bool
    """Whether static metadata is compared against its binding."""


def _extract_plan_leaves(  # noqa: C901, PLR0912
    *, value: object, node: object, state: _LeafExtraction
) -> None:
    """Walk one payload node against its plan node, collecting leaves in `state`."""
    node_type = type(node)
    if node_type is _ArtifactLeafSlot:
        index = cast("_ArtifactLeafSlot", node).index
        if type(index) is not int or index < 0 or index >= len(state.seen):
            raise TypeError("Artifact construction plan has an invalid leaf slot.")
        state.seen[index] += 1
        state.extracted[index] = value
        return
    if node_type is _ArtifactStaticPlan:
        static = cast("_ArtifactStaticPlan", node)
        if (
            state.validate_static or static.validate_payload
        ) and not _same_inert_pytree_metadata(
            actual=value,
            expected=static.value,
        ):
            raise TypeError("Artifact PyTree static metadata differs from its binding.")
        return

    marker = id(value)
    if node_type is _ArtifactTuplePlan:
        if type(value) is not tuple:
            raise TypeError(
                "Artifact template tuple structure differs from its binding."
            )
        if value and marker in state.encountered_ids:
            raise TypeError("Artifact template graph must not alias containers.")
        if value:
            state.encountered_ids.add(marker)
    elif marker in state.encountered_ids:
        raise TypeError("Artifact template graph must not alias containers.")
    else:
        state.encountered_ids.add(marker)
    if marker in state.active_ids:
        raise TypeError("Artifact template construction graph must be acyclic.")
    state.active_ids.add(marker)
    try:
        if node_type is _ArtifactTuplePlan:
            children = cast("_ArtifactTuplePlan", node).children
            if type(value) is not tuple or len(value) != len(children):
                raise TypeError(
                    "Artifact template tuple structure differs from its binding."
                )
            for child_value, child_plan in zip(value, children, strict=True):
                _extract_plan_leaves(state=state, value=child_value, node=child_plan)
            return
        if node_type is not _ArtifactDataclassPlan:
            raise TypeError("Artifact construction plan contains an unsupported node.")
        dataclass_plan = cast("_ArtifactDataclassPlan", node)
        if type(value) is not dataclass_plan.runtime_type:
            raise TypeError(
                "Artifact template dataclass type differs from its binding."
            )
        actual_fields = _artifact_dataclass_field_values(value)
        if len(actual_fields) != len(dataclass_plan.fields):
            raise TypeError(
                "Artifact template dataclass fields differ from its binding."
            )
        for actual_field, expected_field in zip(
            actual_fields,
            dataclass_plan.fields,
            strict=True,
        ):
            name, stored_in_dict, field_value = actual_field
            if (
                type(expected_field) is not _ArtifactDataclassFieldPlan
                or name != expected_field.name
                or stored_in_dict is not expected_field.stored_in_dict
            ):
                raise TypeError(
                    "Artifact template dataclass fields differ from its binding."
                )
            _extract_plan_leaves(
                state=state, value=field_value, node=expected_field.value
            )
    finally:
        state.active_ids.remove(marker)


def _artifact_leaf_values_from_plan(
    *,
    payload: object,
    plan: object,
    leaf_count: int,
    validate_static: bool = True,
) -> tuple[object, ...]:
    """Extract ordered numerical fields through a sealed callback-free plan."""
    missing = object()
    state = _LeafExtraction(
        seen=[0] * leaf_count,
        extracted=[missing] * leaf_count,
        missing=missing,
        active_ids=set(),
        encountered_ids=set(),
        validate_static=validate_static,
    )

    _extract_plan_leaves(state=state, value=payload, node=plan)
    if any(count != 1 for count in state.seen) or any(
        leaf is missing for leaf in state.extracted
    ):
        raise TypeError(
            "Artifact template must expose every bound numerical field exactly once."
        )
    return tuple(state.extracted)


def _validate_artifact_value_against_plan(
    *, payload: object, plan: object, leaves: tuple[object, ...]
) -> None:
    """Validate one exposed graph against private leaf identities, callback-free."""
    actual_leaves = _artifact_leaf_values_from_plan(
        payload=payload,
        plan=plan,
        leaf_count=len(leaves),
    )
    if any(
        actual is not expected
        for actual, expected in zip(actual_leaves, leaves, strict=True)
    ):
        raise TypeError("Artifact template numerical fields differ from its binding.")


def _reconstruct_artifact_from_template_snapshot(
    *,
    template_snapshot: _CanonicalArtifactTemplate,
    leaves: tuple[object, ...],
) -> object:
    """Reconstruct with a detached validated plan and no plugin callbacks."""
    if type(template_snapshot) is not _CanonicalArtifactTemplate:
        raise TypeError("Artifact reconstruction requires an exact template snapshot.")
    plan = _snapshot_artifact_construction_plan(
        plan=template_snapshot.construction_plan,
        leaf_count=len(leaves),
    )
    return _reconstruct_artifact_from_plan(plan=plan, leaves=leaves)


@dataclass(frozen=True, kw_only=True)
class _CanonicalArtifactPayload:
    """Canonical object plus the exact numerical leaves that produced it."""

    payload: object
    leaf_paths: tuple[TreePath, ...]
    leaves: tuple[jax.Array, ...]
    payload_kind: str


def _validate_cached_artifact_template(  # noqa: C901, PLR0912
    *,
    template: object,
    snapshot: _CanonicalArtifactTemplate | None,
    payload_runtime_type: type[object],
    containers: Mapping[TreePath, type[object]],
    leaves: Mapping[TreePath, LeafAuthority],
) -> tuple[
    jax.tree_util.PyTreeDef,
    tuple[TreePath, ...],
    tuple[jax.Array, ...],
    object,
]:
    """Copy a cached declaration without invoking its plugin PyTree callbacks."""
    if type(snapshot) is not _CanonicalArtifactTemplate:
        raise TypeError("Artifact authority has no exact cached PyTree declaration.")
    if not _payload_has_runtime_type(payload=template, expected=payload_runtime_type):
        raise TypeError(
            "Artifact template has a different exact payload runtime type from its "
            "authority."
        )
    if type(snapshot.tree) is not jax.tree_util.PyTreeDef:
        raise TypeError("Artifact cached tree definition must be exact.")
    if type(snapshot.leaf_paths) is not tuple or any(
        type(path) is not tuple
        or any(type(component) is not str or not component for component in path)
        for path in snapshot.leaf_paths
    ):
        raise TypeError("Artifact cached TreePaths must be exact.")
    if type(snapshot.leaves) is not tuple:
        raise TypeError("Artifact cached numerical leaves must be an exact tuple.")

    owned_tree = _snapshot_pytree_def(snapshot.tree)
    paths = tuple(snapshot.leaf_paths)
    if paths != tuple(leaves):
        raise TypeError("Artifact cached TreePaths differ from leaf authority.")
    if owned_tree.num_leaves != len(snapshot.leaves):
        raise TypeError("Artifact cached tree and leaf counts differ.")
    actual_containers = _container_types_from_tree(
        tree=owned_tree,
        leaf_paths=paths,
    )
    if not _same_container_runtime_types(actual=actual_containers, expected=containers):
        raise TypeError(
            "Artifact cached container runtime types differ from authority."
        )
    _check_approved_artifact_containers(
        payload_runtime_type=payload_runtime_type,
        container_runtime_types=containers,
    )
    plan = _snapshot_artifact_construction_plan(
        plan=snapshot.construction_plan,
        leaf_count=len(snapshot.leaves),
    )
    source_leaves = tuple(snapshot.leaves)
    _validate_artifact_value_against_plan(
        payload=snapshot.payload,
        plan=plan,
        leaves=source_leaves,
    )
    if template is not snapshot.payload:
        _validate_artifact_value_against_plan(
            payload=template,
            plan=plan,
            leaves=source_leaves,
        )

    canonical_leaves: list[jax.Array] = []
    for path, leaf in zip(paths, snapshot.leaves, strict=True):
        declaration = leaves[path]
        if declaration.runtime_type is not jax.Array:
            raise TypeError(
                "Artifact numerical leaves must declare jax.Array as runtime_type."
            )
        if not isinstance(leaf, jax.Array):
            raise TypeError(f"Artifact cached leaf {path!r} is not a JAX array.")
        if tuple(leaf.shape) != declaration.shape or np.dtype(leaf.dtype) != np.dtype(
            declaration.dtype
        ):
            raise TypeError(
                f"Artifact cached leaf {path!r} differs from leaf authority."
            )
        canonical_leaves.append(
            _copy_artifact_array_leaf(
                leaf=leaf,
                label=f"Artifact cached leaf {path!r}",
            )
        )
    owned_leaves = tuple(canonical_leaves)
    return owned_tree, paths, owned_leaves, plan


def _rebuild_cached_artifact_template(
    *,
    template: object,
    snapshot: _CanonicalArtifactTemplate | None,
    payload_runtime_type: type[object],
    containers: Mapping[TreePath, type[object]],
    leaves: Mapping[TreePath, LeafAuthority],
) -> _CanonicalArtifactTemplate:
    """Rebuild a detached template solely from its cached declaration."""
    owned_tree, paths, canonical_leaves, plan = _validate_cached_artifact_template(
        template=template,
        snapshot=snapshot,
        payload_runtime_type=payload_runtime_type,
        containers=containers,
        leaves=leaves,
    )
    canonical = _reconstruct_artifact_from_plan(plan=plan, leaves=canonical_leaves)
    return _CanonicalArtifactTemplate(
        payload=canonical,
        tree=owned_tree,
        leaf_paths=paths,
        leaves=canonical_leaves,
        construction_plan=plan,
    )


def _canonicalize_declared_template_snapshot(
    *,
    template: object,
    payload_runtime_type: type[object],
    containers: Mapping[TreePath, type[object]],
    leaves: Mapping[TreePath, LeafAuthority],
) -> _CanonicalArtifactTemplate:
    """Validate and cache one declaration after exactly one flatten observation."""
    if not _payload_has_runtime_type(payload=template, expected=payload_runtime_type):
        raise TypeError(
            "Artifact template has a different exact payload runtime type from its "
            "authority."
        )
    try:
        with_paths, tree = jax.tree_util.tree_flatten_with_path(template)
    except Exception as error:
        raise TypeError("Artifact template flatten callback failed.") from error
    paths = tuple(_normalize_jax_tree_path(path) for path, _leaf in with_paths)
    if paths != tuple(leaves):
        raise ValueError("Artifact template TreePaths differ from leaf authority.")
    actual_containers = _container_types_from_tree(tree=tree, leaf_paths=paths)
    if not _same_container_runtime_types(actual=actual_containers, expected=containers):
        raise ValueError(
            "Artifact template container runtime types differ from authority."
        )
    owned_tree = _snapshot_pytree_def(tree)
    _check_approved_artifact_containers(
        payload_runtime_type=payload_runtime_type,
        container_runtime_types=containers,
    )

    canonical_leaves: list[jax.Array] = []
    for path, leaf in with_paths:
        normalized = _normalize_jax_tree_path(path)
        declaration = leaves[normalized]
        if declaration.runtime_type is not jax.Array:
            raise TypeError(
                "Artifact numerical leaves must declare jax.Array as runtime_type."
            )
        if not (
            isinstance(leaf, jax.Array | np.ndarray | np.generic)
            or any(type(leaf) is allowed for allowed in (bool, int, float))
        ):
            raise TypeError(f"Artifact template leaf {normalized!r} is not numerical.")
        leaf_shape = tuple(getattr(leaf, "shape", ()))
        leaf_dtype = (
            np.dtype(leaf.dtype) if hasattr(leaf, "dtype") else np.asarray(leaf).dtype
        )
        if leaf_shape != declaration.shape or leaf_dtype != np.dtype(declaration.dtype):
            raise ValueError(
                f"Artifact template leaf {normalized!r} differs from leaf authority."
            )
        canonical = jax.numpy.asarray(leaf)
        if tuple(canonical.shape) != declaration.shape or np.dtype(
            canonical.dtype
        ) != np.dtype(declaration.dtype):
            raise ValueError(
                f"The active JAX profile cannot preserve artifact leaf {normalized!r}."
            )
        canonical_leaves.append(
            _copy_artifact_array_leaf(
                leaf=canonical,
                label=f"Artifact template leaf {normalized!r}",
            )
        )
    plan = _compile_artifact_construction_plan(
        tree=owned_tree,
        payload_runtime_type=payload_runtime_type,
        declaration_payload=template,
    )
    owned_leaves = tuple(canonical_leaves)
    canonical = _reconstruct_artifact_from_plan(plan=plan, leaves=owned_leaves)
    return _CanonicalArtifactTemplate(
        payload=canonical,
        tree=owned_tree,
        leaf_paths=paths,
        leaves=owned_leaves,
        construction_plan=plan,
    )


def _snapshot_artifact_template_once(
    *,
    template: object,
    payload_runtime_type: type[object],
) -> tuple[_CanonicalArtifactTemplate, dict[TreePath, type[object]]]:
    """Observe an engine template once and derive its owned numerical layout."""
    if not _payload_has_runtime_type(
        payload=template,
        expected=payload_runtime_type,
    ):
        raise TypeError(
            "Artifact template has a different exact payload runtime type from its "
            "authority."
        )
    try:
        with_paths, tree = jax.tree_util.tree_flatten_with_path(template)
    except Exception as error:
        raise TypeError("Artifact template flatten callback failed.") from error
    paths = tuple(_normalize_jax_tree_path(path) for path, _leaf in with_paths)
    containers = _container_types_from_tree(tree=tree, leaf_paths=paths)
    _check_approved_artifact_containers(
        payload_runtime_type=payload_runtime_type,
        container_runtime_types=containers,
    )
    owned_tree = _snapshot_pytree_def(tree)
    canonical_leaves: list[jax.Array] = []
    for path, leaf in with_paths:
        normalized = _normalize_jax_tree_path(path)
        if not (
            isinstance(leaf, jax.Array | np.ndarray | np.generic)
            or any(type(leaf) is allowed for allowed in (bool, int, float))
        ):
            raise TypeError(f"Artifact template leaf {normalized!r} is not numerical.")
        source_shape = tuple(getattr(leaf, "shape", ()))
        source_dtype = (
            np.dtype(leaf.dtype) if hasattr(leaf, "dtype") else np.asarray(leaf).dtype
        )
        canonical = jax.numpy.asarray(leaf)
        if (
            tuple(canonical.shape) != source_shape
            or np.dtype(canonical.dtype) != source_dtype
        ):
            raise ValueError(
                f"The active JAX profile cannot preserve artifact leaf {normalized!r}."
            )
        canonical_leaves.append(
            _copy_artifact_array_leaf(
                leaf=canonical,
                label=f"Artifact template leaf {normalized!r}",
            )
        )
    plan = _compile_artifact_construction_plan(
        tree=owned_tree,
        payload_runtime_type=payload_runtime_type,
        declaration_payload=template,
    )
    owned_leaves = tuple(canonical_leaves)
    canonical = _reconstruct_artifact_from_plan(plan=plan, leaves=owned_leaves)
    return (
        _CanonicalArtifactTemplate(
            payload=canonical,
            tree=owned_tree,
            leaf_paths=paths,
            leaves=owned_leaves,
            construction_plan=plan,
        ),
        containers,
    )


def _validate_artifact_authority_declarations(  # noqa: C901, PLR0912
    *,
    descriptor: ArtifactDescriptor,
    payload_runtime_type: type[object],
    containers: Mapping[TreePath, type[object]],
    leaves: Mapping[TreePath, LeafAuthority],
    axes: tuple[AxisAuthority, ...],
    state_roles: tuple[str, ...],
    action_roles: tuple[str, ...],
    categories: Mapping[str, CategoryDomain],
    consumer_route: ReplayRouteIdentity | None,
    applicable: bool,
    required: bool,
) -> None:
    """Validate exact authority fields without observing an executable template."""
    if type(descriptor) is not ArtifactDescriptor:
        raise TypeError("ArtifactAuthority.descriptor must be exact.")
    if not isinstance(payload_runtime_type, type):
        raise TypeError("ArtifactAuthority.payload_runtime_type must be a type.")
    if any(
        type(path) is not tuple
        or any(type(component) is not str or not component for component in path)
        or not isinstance(runtime_type, type)
        for path, runtime_type in containers.items()
    ):
        raise TypeError(
            "ArtifactAuthority container paths and runtime types must be exact."
        )
    if any(
        type(path) is not tuple or type(leaf) is not LeafAuthority
        for path, leaf in leaves.items()
    ):
        raise TypeError("ArtifactAuthority leaves must map exact paths to authority.")
    if any(path != leaf.path for path, leaf in leaves.items()):
        raise ValueError("ArtifactAuthority leaf keys must equal their TreePaths.")
    if any(type(axis) is not AxisAuthority for axis in axes):
        raise TypeError("ArtifactAuthority axes must be exact AxisAuthorities.")
    if not _same_exact_artifact_contract(
        actual=descriptor.leaf_descriptors,
        expected=tuple(leaf.descriptor for leaf in leaves.values()),
    ):
        raise ValueError(
            "Artifact descriptive leaves differ from model leaf authority."
        )
    if not _same_exact_artifact_contract(
        actual=descriptor.named_axes,
        expected=tuple(axis.descriptor for axis in axes),
    ):
        raise ValueError("Artifact descriptive axes differ from model axis authority.")
    if not _same_exact_artifact_contract(
        actual=(state_roles, action_roles),
        expected=(descriptor.state_roles, descriptor.action_roles),
    ):
        raise ValueError("Artifact descriptive roles differ from model authority.")
    if not _same_exact_artifact_contract(
        actual=categories,
        expected=descriptor.categorical_domains,
    ):
        raise ValueError("Artifact descriptive categories differ from model authority.")
    if consumer_route is not None and type(consumer_route) is not ReplayRouteIdentity:
        raise TypeError(
            "ArtifactAuthority.consumer_route must be a ReplayRouteIdentity or None."
        )
    expected_required_for = (
        frozenset({consumer_route})
        if required and consumer_route is not None
        else frozenset()
    )
    if not _same_exact_artifact_contract(
        actual=descriptor.required_for,
        expected=expected_required_for,
    ):
        raise ValueError(
            "ArtifactDescriptor.required_for differs from model authority."
        )
    if type(applicable) is not bool or type(required) is not bool:
        raise TypeError("Artifact applicability and requiredness must be exact bools.")
    if descriptor.required is not required:
        raise ValueError("Artifact descriptive requiredness differs from authority.")
    _validate_axes_and_leaves(axes=axes, leaves=tuple(leaves.values()))


def _artifact_authority_from_template_snapshot(
    *,
    descriptor: ArtifactDescriptor,
    payload_runtime_type: type[object],
    template_snapshot: _CanonicalArtifactTemplate | None,
    container_runtime_types: Mapping[TreePath, type[object]] = MappingProxyType({}),
    leaves: Mapping[TreePath, LeafAuthority] = MappingProxyType({}),
    axes: tuple[AxisAuthority, ...] = (),
    state_roles: tuple[str, ...] = (),
    action_roles: tuple[str, ...] = (),
    categorical_domains: Mapping[str, CategoryDomain] = MappingProxyType({}),
    consumer_route: ReplayRouteIdentity | None = None,
    applicable: bool = True,
    required: bool = False,
) -> ArtifactAuthority:
    """Build a trusted authority from an already observed template declaration."""
    containers = dict(container_runtime_types)
    copied_leaves = dict(leaves)
    copied_axes = tuple(axes)
    copied_state_roles = tuple(state_roles)
    copied_action_roles = tuple(action_roles)
    categories = dict(categorical_domains)
    _validate_artifact_authority_declarations(
        descriptor=descriptor,
        payload_runtime_type=payload_runtime_type,
        containers=containers,
        leaves=copied_leaves,
        axes=copied_axes,
        state_roles=copied_state_roles,
        action_roles=copied_action_roles,
        categories=categories,
        consumer_route=consumer_route,
        applicable=applicable,
        required=required,
    )
    public_template_leaves: tuple[jax.Array, ...] = ()
    if template_snapshot is None:
        if containers or copied_leaves:
            raise ValueError(
                "An authority without a materialization template cannot declare "
                "containers or leaves."
            )
        canonical_snapshot = None
        template = None
    else:
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Artifact authority requires an exact template snapshot.")
        canonical_snapshot = _rebuild_cached_artifact_template(
            template=template_snapshot.payload,
            snapshot=template_snapshot,
            payload_runtime_type=payload_runtime_type,
            containers=containers,
            leaves=copied_leaves,
        )
        template = canonical_snapshot.payload
        public_template_leaves = canonical_snapshot.leaves
        canonical_snapshot = _rebuild_cached_artifact_template(
            template=template,
            snapshot=canonical_snapshot,
            payload_runtime_type=payload_runtime_type,
            containers=containers,
            leaves=copied_leaves,
        )

    authority = object.__new__(ArtifactAuthority)
    object.__setattr__(authority, "descriptor", descriptor)
    object.__setattr__(authority, "payload_runtime_type", payload_runtime_type)
    object.__setattr__(authority, "template", template)
    object.__setattr__(
        authority,
        "container_runtime_types",
        MappingProxyType(containers),
    )
    object.__setattr__(authority, "leaves", MappingProxyType(copied_leaves))
    object.__setattr__(authority, "axes", copied_axes)
    object.__setattr__(authority, "state_roles", copied_state_roles)
    object.__setattr__(authority, "action_roles", copied_action_roles)
    object.__setattr__(
        authority,
        "categorical_domains",
        MappingProxyType(categories),
    )
    object.__setattr__(authority, "consumer_route", consumer_route)
    object.__setattr__(authority, "applicable", applicable)
    object.__setattr__(authority, "required", required)
    _bind_artifact_authority_template(
        authority=authority,
        snapshot=canonical_snapshot,
        public_template_leaves=public_template_leaves,
    )
    return authority


def _copy_artifact_authority_without_binding(
    *, authority: ArtifactAuthority
) -> ArtifactAuthority:
    """Copy public fields without granting the new identity reconstruction authority."""
    if type(authority) is not ArtifactAuthority:
        raise TypeError("Only an exact ArtifactAuthority can be copied.")
    copied = object.__new__(ArtifactAuthority)
    for name in (
        "descriptor",
        "payload_runtime_type",
        "template",
        "container_runtime_types",
        "leaves",
        "axes",
        "state_roles",
        "action_roles",
        "categorical_domains",
        "consumer_route",
        "applicable",
        "required",
    ):
        object.__setattr__(
            copied,
            name,
            object.__getattribute__(authority, name),
        )
    return copied


def _artifact_authority_pickle_state(
    *, authority: ArtifactAuthority
) -> _ArtifactAuthorityPickleState:
    """Capture a validated authority and its sealed callback-free declaration."""
    if type(authority) is not ArtifactAuthority:
        raise TypeError("Only an exact ArtifactAuthority can be transported.")
    template_snapshot = _artifact_authority_template_snapshot(authority)
    canonical = _artifact_authority_from_template_snapshot(
        descriptor=authority.descriptor,
        payload_runtime_type=authority.payload_runtime_type,
        template_snapshot=template_snapshot,
        container_runtime_types=authority.container_runtime_types,
        leaves=authority.leaves,
        axes=authority.axes,
        state_roles=authority.state_roles,
        action_roles=authority.action_roles,
        categorical_domains=authority.categorical_domains,
        consumer_route=authority.consumer_route,
        applicable=authority.applicable,
        required=authority.required,
    )
    return _ArtifactAuthorityPickleState(
        descriptor=canonical.descriptor,
        payload_runtime_type=canonical.payload_runtime_type,
        template_snapshot=_artifact_authority_template_snapshot(canonical),
        container_runtime_types=canonical.container_runtime_types,
        leaves=canonical.leaves,
        axes=canonical.axes,
        state_roles=canonical.state_roles,
        action_roles=canonical.action_roles,
        categorical_domains=canonical.categorical_domains,
        consumer_route=canonical.consumer_route,
        applicable=canonical.applicable,
        required=canonical.required,
    )


def _restore_artifact_authority_from_pickle(state: object) -> ArtifactAuthority:
    """Rebuild one transported authority through the validated private constructor."""
    if type(state) is not _ArtifactAuthorityPickleState:
        raise TypeError("Artifact authority pickle state must be exact.")
    owned = state
    return _artifact_authority_from_template_snapshot(
        descriptor=owned.descriptor,
        payload_runtime_type=owned.payload_runtime_type,
        template_snapshot=owned.template_snapshot,
        container_runtime_types=owned.container_runtime_types,
        leaves=owned.leaves,
        axes=owned.axes,
        state_roles=owned.state_roles,
        action_roles=owned.action_roles,
        categorical_domains=owned.categorical_domains,
        consumer_route=owned.consumer_route,
        applicable=owned.applicable,
        required=owned.required,
    )


def _canonicalize_declared_template(
    *,
    template: object,
    payload_runtime_type: type[object],
    containers: Mapping[TreePath, type[object]],
    leaves: Mapping[TreePath, LeafAuthority],
) -> object:
    """Validate and detach a template for callback-free lazy reconstruction."""
    return _canonicalize_declared_template_snapshot(
        template=template,
        payload_runtime_type=payload_runtime_type,
        containers=containers,
        leaves=leaves,
    ).payload


def _canonicalize_artifact_payload_snapshot(  # noqa: C901
    *, payload: object, authority: ArtifactAuthority
) -> _CanonicalArtifactPayload:
    """Canonicalize one payload once and retain the exact validated leaves."""
    expected_type = authority.payload_runtime_type
    if not _payload_has_runtime_type(payload=payload, expected=expected_type):
        raise TypeError(
            f"payload type is {type(payload).__name__!r}; expected exact "
            f"{expected_type.__name__!r}"
        )
    template_snapshot = _artifact_authority_template_snapshot(authority)
    if template_snapshot is None:
        raise TypeError("model authority supplies no canonical PyTree template")

    (
        _template_tree,
        template_paths,
        template_leaves,
        construction_plan,
    ) = _validate_cached_artifact_template(
        template=template_snapshot.payload,
        snapshot=template_snapshot,
        payload_runtime_type=authority.payload_runtime_type,
        containers=authority.container_runtime_types,
        leaves=authority.leaves,
    )
    supplied_leaves = _artifact_leaf_values_from_plan(
        payload=payload,
        plan=construction_plan,
        leaf_count=len(template_leaves),
        validate_static=False,
    )
    supplied_paths = template_paths

    canonical_leaves: list[jax.Array] = []
    for index, (path, supplied, template) in enumerate(
        zip(supplied_paths, supplied_leaves, template_leaves, strict=True)
    ):
        if not (
            isinstance(supplied, jax.Array | np.ndarray | np.generic)
            or any(type(supplied) is allowed for allowed in (bool, int, float))
        ):
            raise TypeError(f"leaf {index} is not a supported numerical leaf")
        leaf_authority = authority.leaves[path]
        array = (
            np.asarray(supplied) if not isinstance(supplied, jax.Array) else supplied
        )
        if not (
            np.issubdtype(np.dtype(array.dtype), np.number)
            or np.issubdtype(np.dtype(array.dtype), np.bool_)
        ):
            raise TypeError(f"leaf {index} is not numerical or Boolean")
        if tuple(array.shape) != leaf_authority.shape:
            raise ValueError(
                f"leaf {index} has shape {tuple(array.shape)!r}; "
                f"expected {leaf_authority.shape!r}"
            )
        if np.dtype(array.dtype) != np.dtype(leaf_authority.dtype):
            raise ValueError(
                f"leaf {index} has dtype {array.dtype!s}; expected "
                f"{leaf_authority.dtype!s}"
            )
        if tuple(getattr(template, "shape", ())) != leaf_authority.shape or np.dtype(
            getattr(template, "dtype", None)
        ) != np.dtype(leaf_authority.dtype):
            raise TypeError("model authority template differs from leaf authority")
        canonical_leaf = jax.numpy.asarray(array)
        if tuple(canonical_leaf.shape) != leaf_authority.shape or np.dtype(
            canonical_leaf.dtype
        ) != np.dtype(leaf_authority.dtype):
            raise ValueError(
                f"the active JAX profile cannot preserve leaf {index} exactly"
            )
        canonical_leaves.append(
            _copy_artifact_array_leaf(
                leaf=canonical_leaf,
                label=f"Artifact payload leaf {path!r}",
            )
        )

    _check_approved_artifact_containers(
        payload_runtime_type=authority.payload_runtime_type,
        container_runtime_types=authority.container_runtime_types,
    )
    owned_leaves = tuple(canonical_leaves)
    canonical = _reconstruct_artifact_from_plan(
        plan=construction_plan,
        leaves=owned_leaves,
    )
    if not _payload_has_runtime_type(payload=canonical, expected=expected_type):
        raise TypeError("Artifact construction plan returned a different payload type.")
    return _CanonicalArtifactPayload(
        payload=canonical,
        leaf_paths=supplied_paths,
        leaves=owned_leaves,
        payload_kind=(
            "array"
            if len(supplied_leaves) == 1 and supplied_paths == ((),)
            else "pytree"
        ),
    )


def _canonicalize_artifact_payload(
    *, payload: object, authority: ArtifactAuthority
) -> object:
    """Copy one payload into its model-built exact PyTree and leaf representation."""
    return _canonicalize_artifact_payload_snapshot(
        payload=payload,
        authority=authority,
    ).payload


@dataclass(frozen=True, slots=True, kw_only=True)
class _CanonicalArtifactEntry(_LazyEntry):
    """Private artifact state that returns a fresh detached graph on every read."""

    plan_snapshot: _CanonicalArtifactTemplate
    leaves: tuple[jax.Array, ...]

    @property
    def load_state(self) -> LoadState:
        """An owned eager entry is already loaded."""
        return LoadState.LOADED

    def _fresh(
        self,
        *,
        template_snapshot: _CanonicalArtifactTemplate,
    ) -> object:
        """Copy private buffers and reconstruct without a plugin callback."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Owned artifact reconstruction requires an exact snapshot.")
        if template_snapshot.leaf_paths != self.plan_snapshot.leaf_paths:
            raise TypeError("Owned artifact TreePaths differ from model authority.")
        if len(self.leaves) != len(template_snapshot.leaves):
            raise TypeError("Owned artifact leaf count differs from model authority.")
        copied: list[jax.Array] = []
        for index, (leaf, expected) in enumerate(
            zip(self.leaves, template_snapshot.leaves, strict=True)
        ):
            if not isinstance(leaf, jax.Array) or leaf.is_deleted():
                raise TypeError("Owned artifact private leaf was deleted.")
            if tuple(leaf.shape) != tuple(expected.shape) or np.dtype(
                leaf.dtype
            ) != np.dtype(expected.dtype):
                raise TypeError("Owned artifact leaf differs from model authority.")
            copied.append(
                _copy_artifact_array_leaf(
                    leaf=leaf,
                    label=f"Owned artifact leaf {index}",
                )
            )
        return _reconstruct_artifact_from_template_snapshot(
            template_snapshot=template_snapshot,
            leaves=tuple(copied),
        )

    def materialize(self, *, template: object | None = None) -> object:
        """Return a fresh graph; eager-entry compatibility ignores raw templates."""
        del template
        return self._fresh(template_snapshot=self.plan_snapshot)

    def materialize_from_template_snapshot(
        self,
        *,
        template_snapshot: object,
    ) -> object:
        """Return a fresh graph through the current model-authoritative plan."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Owned artifact reconstruction requires an exact snapshot.")
        return self._fresh(template_snapshot=template_snapshot)


def _canonical_artifact_entry_from_authority(
    *,
    payload: object,
    authority: ArtifactAuthority,
) -> _CanonicalArtifactEntry:
    """Detach one eager artifact before any other result callback may run."""
    canonical = _canonicalize_artifact_payload_snapshot(
        payload=payload,
        authority=authority,
    )
    plan_snapshot = _artifact_authority_template_snapshot(authority)
    if plan_snapshot is None:
        raise TypeError("Model authority supplies no artifact reconstruction plan.")
    private_leaves = tuple(
        _copy_artifact_array_leaf(
            leaf=leaf,
            label=f"Owned artifact private leaf {index}",
        )
        for index, leaf in enumerate(canonical.leaves)
    )
    return _CanonicalArtifactEntry(
        plan_snapshot=plan_snapshot,
        leaves=private_leaves,
    )


def _copy_solution_value(*, value: object, label: str) -> object:
    """Copy one numerical value without changing its concrete representation."""
    if isinstance(value, jax.Array):
        return _copy_artifact_array_leaf(leaf=value, label=label)
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True, order="K", subok=False)
        if not (
            np.issubdtype(copied.dtype, np.number)
            or np.issubdtype(copied.dtype, np.bool_)
        ):
            raise TypeError(f"{label} is not numerical or Boolean.")
        copied.flags.writeable = False
        return copied
    if isinstance(value, np.generic):
        if not (
            np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.bool_)
        ):
            raise TypeError(f"{label} is not numerical or Boolean.")
        return np.array(value, copy=True)[()]
    if any(type(value) is allowed for allowed in (bool, int, float, complex)):
        return value
    raise TypeError(f"{label} is not a supported numerical value.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CanonicalValueEntry(_LazyEntry):
    """Private numerical state that returns a fresh value on every read."""

    value: object

    @property
    def load_state(self) -> LoadState:
        """An owned eager value is already loaded."""
        return LoadState.LOADED

    def materialize(self, *, template: object | None = None) -> object:
        """Return an independent numerical value."""
        del template
        return _copy_solution_value(value=self.value, label="Owned solution value")


def _canonical_value_entry(*, value: object) -> _CanonicalValueEntry:
    """Detach one eager value before any lazy result callback may run."""
    source = value.materialize() if type(value) is _CanonicalValueEntry else value
    private = _copy_solution_value(value=source, label="Solution value")
    return _CanonicalValueEntry(value=private)


@dataclass(frozen=True, kw_only=True)
class ReplayRouteSnapshot:
    """One immutable, preflighted cell passed unchanged to a replay route."""

    artifacts: Mapping[ArtifactKey, object]
    authorities: Mapping[ArtifactKey, ArtifactAuthority]
    metadata: SolutionMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        object.__setattr__(
            self, "authorities", MappingProxyType(dict(self.authorities))
        )


@dataclass(frozen=True, kw_only=True)
class ReplayModelContext:
    """Narrow solve-grid view from which a route declares its dependencies.

    The names and node mappings describe the canonical, period-specific solution
    state/action space on which replay artifacts are defined. Simulation-only carried
    states are deliberately absent because they are not solution axes.
    """

    regime_name: RegimeName
    period: int
    state_names: tuple[str, ...]
    """Solution-state names in canonical product-map order."""

    action_names: tuple[str, ...]
    """Solution-action names in canonical product-map order."""

    state_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``state_names``."""

    action_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``action_names``."""

    def __post_init__(self) -> None:
        if type(self.regime_name) is not str or not self.regime_name:
            raise TypeError("ReplayModelContext.regime_name must be a nonempty str.")
        if type(self.period) is not int or self.period < 0:
            raise TypeError("ReplayModelContext.period must be a nonnegative int.")
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "action_names", tuple(self.action_names))
        object.__setattr__(
            self, "state_nodes", MappingProxyType(dict(self.state_nodes))
        )
        object.__setattr__(
            self, "action_nodes", MappingProxyType(dict(self.action_nodes))
        )


@dataclass(frozen=True, kw_only=True)
class ReplayRouteRequirements:
    """Exact artifact keys one external route requires in every active cell."""

    required_artifacts: frozenset[ArtifactKey]

    def __post_init__(self) -> None:
        artifacts = frozenset(self.required_artifacts)
        if any(type(key) is not ArtifactKey for key in artifacts):
            raise TypeError(
                "ReplayRouteRequirements.required_artifacts must contain exact "
                "ArtifactKeys."
            )
        object.__setattr__(self, "required_artifacts", artifacts)


@dataclass(frozen=True, kw_only=True)
class SimulationBuildContext:
    """Public, model-owned solve-grid facts for constructing one replay reader.

    This carries the same state/action view used to declare route requirements.
    Simulation-only carried states are supplied later in ``ReplayReader.states`` but
    are not artifact axes and therefore do not appear here.
    """

    period: int
    regime_name: RegimeName
    state_names: tuple[str, ...]
    """Solution-state names in canonical product-map order."""

    action_names: tuple[str, ...]
    """Solution-action names in canonical product-map order."""

    state_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``state_names``."""

    action_nodes: Mapping[str, FloatND | IntND]
    """Period-specific grid nodes keyed exactly by ``action_names``."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_names", tuple(self.state_names))
        object.__setattr__(self, "action_names", tuple(self.action_names))
        object.__setattr__(
            self, "state_nodes", MappingProxyType(dict(self.state_nodes))
        )
        object.__setattr__(
            self, "action_nodes", MappingProxyType(dict(self.action_nodes))
        )


@dataclass(frozen=True, kw_only=True)
class ActionOutput:
    """Named action arrays returned by an external replay reader."""

    actions: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))


@runtime_checkable
class ReplayReader(Protocol):
    """JAX-transformable reader built from a validated replay snapshot."""

    def __call__(
        self, *, states: Mapping[str, object], fallback_actions: Mapping[str, object]
    ) -> ActionOutput:
        """Return each named action as a scalar or per-subject-broadcastable array.

        ``states`` contains the regime's full per-subject simulation state, including
        any carried-only states omitted from ``SimulationBuildContext``.
        """
        ...


# Built-in identities live on the public transport boundary. The engine imports
# these same singleton objects rather than defining private lookalikes, so an
# installed solver and pylcm always address the same schema.
EGM_CONTINUATION = ArtifactKey(type_id="pylcm.egm.continuation", schema_version=1)
SIMULATION_POLICY = ArtifactKey(type_id="pylcm.simulation.policy", schema_version=1)
DISSOLUTION_FLAG = ArtifactKey(
    type_id="pylcm.collective.dissolution_flag", schema_version=1
)
SOLVER_DIAGNOSTICS = ArtifactKey(type_id="pylcm.solver.diagnostics", schema_version=1)


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


@runtime_checkable
class ReplayRoute(Protocol):
    """How one regime's simulated decision is obtained, declared by the model.

    Forward simulation and the pre-simulation payload check both dispatch on
    this object rather than on the concrete payload class, so a regime that
    retains no replay payload and one that retains an unfamiliar payload are
    described in the same vocabulary.
    """

    @property
    def replay_mode(self) -> ReplayMode:
        """How the decision is obtained under this route."""
        ...

    @property
    def payload_type(self) -> type[object] | None:
        """Exact class of the retained payload, `None` when none is kept."""
        ...

    @property
    def policy_applicable(self) -> bool:
        """Whether this route structurally publishes a replay payload."""
        ...

    @property
    def policy_required(self) -> bool:
        """Whether every successful solve must retain that payload."""
        ...

    @property
    def consumer_route(self) -> str | None:
        """Name of the reader consuming the payload, `None` without one."""
        ...


def _replay_route_identity(route: ReplayRoute) -> ReplayRouteIdentity:
    """Return the durable identity of a trusted built-in or executable route."""
    declared_identity = getattr(route, "identity", None)
    if type(declared_identity) is ReplayRouteIdentity:
        return declared_identity
    route_ids: dict[tuple[ReplayMode, str | None], str] = {
        (ReplayMode.VALID_RECOMPUTATION, None): "pylcm.grid_recomputation",
        (ReplayMode.EXACT_REPLAY, "egm_off_grid"): "pylcm.egm_off_grid",
        (ReplayMode.EXACT_REPLAY, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.EXACT_REPLAY, "nnbegm_nested"): "pylcm.nnbegm_nested",
        (ReplayMode.VALID_RECOMPUTATION, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.VALID_RECOMPUTATION, "nnbegm_nested"): "pylcm.nnbegm_nested",
        (ReplayMode.UNSUPPORTED, "nnbegm_finite"): "pylcm.nnbegm_finite",
        (ReplayMode.UNSUPPORTED, "nnbegm_nested"): "pylcm.nnbegm_nested",
    }
    try:
        route_id = route_ids[(route.replay_mode, route.consumer_route)]
    except KeyError as error:
        raise TypeError(
            "A replay route has no durable built-in or plugin identity."
        ) from error
    return ReplayRouteIdentity(route_id=route_id, route_version=1)


@runtime_checkable
class ExecutableReplayRoute(ReplayRoute, Protocol):
    """Installed plugin route that validates artifacts and builds its own reader."""

    @property
    def identity(self) -> ReplayRouteIdentity:
        """Return the route's durable identity and exact compatibility version."""
        ...

    @property
    def plugin_identity(self) -> SolverIdentity:
        """Return the installed plugin identity implementing this route."""
        ...

    def requirements(self, *, context: ReplayModelContext) -> ReplayRouteRequirements:
        """Declare the exact artifact dependencies for this model view."""
        ...

    def validate(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> None:
        """Check solver-specific mathematical invariants before simulation."""
        ...

    def build_reader(
        self,
        *,
        snapshot: ReplayRouteSnapshot,
        context: SimulationBuildContext,
    ) -> ReplayReader:
        """Build a JAX-transformable reader from the validated snapshot."""
        ...


@runtime_checkable
class ContinuationArtifact(Protocol):
    """A payload a period kernel publishes for the previous period's kernels.

    The engine stores and rolls it opaquely under its `artifact_key`; only a
    parent solver that declares the same key reads its fields.
    """

    @property
    def artifact_key(self) -> ArtifactKey:
        """Versioned identity of the payload's schema."""
        ...


@dataclass(frozen=True, kw_only=True)
class KernelOutput:
    """One solver kernel's value and explicitly typed artifact channels.

    This is the dependency-safe producer envelope for solver extensions. Artifact
    identity is carried by :class:`ArtifactKey`, while the engine decides which
    declared artifacts it understands and consumes. The mappings are copied at
    construction and exposed as immutable views so a producer cannot mutate a
    published kernel result after returning it.

    Numerical diagnostics are kept outside this producer envelope. In-tree solvers
    that publish ``SolverDiagnostics`` use the engine-private result representation
    until that payload is represented as a public artifact.
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
            entries = dict(getattr(self, field_name))
            if not all(type(key) is ArtifactKey for key in entries):
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
            object.__setattr__(self, field_name, MappingProxyType(entries))


@dataclass(frozen=True, order=True, kw_only=True)
class ArtifactRef:
    """Address of one artifact in a regime-period solution cell."""

    period: int
    regime: RegimeName
    key: ArtifactKey

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


if TYPE_CHECKING:
    _FloatValueBoundary: TypeAlias = FloatND  # noqa: UP040
    _RegimeValuesBoundary: TypeAlias = Mapping[RegimeName, FloatND]  # noqa: UP040
    _MaterializedValuesBoundary: TypeAlias = MappingProxyType[  # noqa: UP040
        int, MappingProxyType[str, FloatND]
    ]
    _ValueStoreBoundary: TypeAlias = "ValueStore"  # noqa: UP040
    _ArtifactStoreBoundary: TypeAlias = "ArtifactStore"  # noqa: UP040
    _ValuePeriodBoundary: TypeAlias = int  # noqa: UP040
    _RegimeNameBoundary: TypeAlias = RegimeName  # noqa: UP040
    _ArtifactRefBoundary: TypeAlias = ArtifactRef  # noqa: UP040
    _ArtifactKeyBoundary: TypeAlias = ArtifactKey  # noqa: UP040
else:
    # Lazy stores own their validation and materialization boundaries. Runtime
    # annotation traversal would load them before those explicit checks run.
    _FloatValueBoundary = object
    _RegimeValuesBoundary = object
    _MaterializedValuesBoundary = object
    _ValueStoreBoundary = object
    _ArtifactStoreBoundary = object
    _ValuePeriodBoundary = object
    _RegimeNameBoundary = object
    _ArtifactRefBoundary = object
    _ArtifactKeyBoundary = object


def _traverse_public_mapping_items(
    *, mapping: object, label: str
) -> list[tuple[object, object]]:
    """Consume one item traversal of a public mapping into an owned list of pairs.

    A public mapping may be any `Mapping` implementation, so its key view, item
    view, and length can disagree or execute backing code. Store constructors read
    this one item traversal and nothing else, and check every raw key exactly before
    it is hashed into a store, so an equality alias (`True` for `1`) or a repeated
    address cannot contract away unseen.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        items = list(mapping.items())
    except Exception as error:
        raise TypeError(f"{label} cannot be traversed as mapping items.") from error
    for item in items:
        if type(item) is not tuple or len(item) != 2:  # noqa: PLR2004
            raise TypeError(f"{label} items must be exact key-value pairs.")
    return items


def _require_exact_value_period(period: object) -> int:
    if type(period) is not int:
        raise TypeError("Value periods must be exact ints.")
    if period < 0:
        raise ValueError("Value periods must be nonnegative.")
    return period


def _require_exact_regime_name(regime: object) -> RegimeName:
    if type(regime) is not str:
        raise TypeError("Value regime names must be exact strs.")
    if not regime:
        raise ValueError("Value regime names must not be empty.")
    return regime


def _require_exact_artifact_key(key: object) -> ArtifactKey:
    if type(key) is not ArtifactKey:
        raise TypeError("Artifact keys must be exact ArtifactKey objects.")
    if type(key.type_id) is not str or not key.type_id:
        raise TypeError("Artifact key type_id must be a nonempty exact str.")
    if type(key.schema_version) is not int or key.schema_version < 1:
        raise TypeError("Artifact key schema_version must be a positive exact int.")
    return key


def _require_exact_artifact_ref(ref: object) -> ArtifactRef:
    if type(ref) is not ArtifactRef:
        raise TypeError("Artifact addresses must be exact ArtifactRef objects.")
    _require_exact_value_period(ref.period)
    _require_exact_regime_name(ref.regime)
    _require_exact_artifact_key(ref.key)
    return ref


@dataclass(frozen=True, eq=False)
class _ValuePeriodView(Mapping[RegimeName, FloatND]):
    """Read-through view of one period in a :class:`ValueStore`."""

    store: _ValueStoreBoundary
    period: int

    def __getitem__(self, regime: _RegimeNameBoundary) -> _FloatValueBoundary:
        return self.store._load(period=self.period, regime=regime)  # noqa: SLF001

    def __iter__(self) -> Iterator[RegimeName]:
        return iter(self.store._regimes_by_period[self.period])  # noqa: SLF001

    def __len__(self) -> int:
        return len(self.store._regimes_by_period[self.period])  # noqa: SLF001

    def __contains__(self, regime: object) -> bool:
        """Check one regime coordinate without materializing its value."""
        if type(regime) is not str or not regime:
            return False
        return regime in self.store._regimes_by_period[self.period]  # noqa: SLF001


def _admit_value_entry(
    *,
    period: object,
    regime: object,
    value: object,
    entries: dict[tuple[int, RegimeName], object],
    regimes_by_period: dict[int, list[RegimeName]],
) -> None:
    """Check one raw coordinate exactly and for uniqueness, then own its value."""
    coordinate = (
        _require_exact_value_period(period),
        _require_exact_regime_name(regime),
    )
    if coordinate in entries:
        raise ValueError(f"ValueStore coordinate {coordinate!r} appears twice.")
    entries[coordinate] = (
        value
        if isinstance(value, _LazyEntry) and type(value) is not _CanonicalValueEntry
        else _canonical_value_entry(value=value)
    )
    regimes_by_period.setdefault(coordinate[0], []).append(coordinate[1])


@dataclass(frozen=True, eq=False)
class ValueStore(Mapping[int, Mapping[RegimeName, FloatND]]):
    """Immutable, independently materializable value-function store.

    Eager solves and restored archives expose the same mapping interface.  The
    latter keep a lazy entry per ``(period, regime)``; inspecting coordinates or
    load state never reads a numerical payload.
    """

    _entries: Mapping[object, object] = field(default_factory=dict, repr=False)
    _regimes_by_period: Mapping[int, tuple[RegimeName, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # The mapping is read through exactly one item traversal. Its form — flat
        # ``(period, regime)`` tuples or ``period -> regime -> value`` — is decided
        # from those same items, and every raw coordinate is checked exactly and for
        # uniqueness before it is inserted, so ``True`` cannot overwrite ``1`` and a
        # repeated address cannot be contracted into one.
        items = _traverse_public_mapping_items(
            mapping=self._entries, label="ValueStore entries"
        )
        is_flat = [type(key) is tuple for key, _ in items]
        if any(is_flat) and not all(is_flat):
            raise TypeError(
                "ValueStore entries must be keyed either by (period, regime) tuples "
                "or by periods, not by both."
            )

        entries: dict[tuple[int, RegimeName], object] = {}
        regimes_by_period: dict[int, list[RegimeName]] = {}

        if all(is_flat):
            for coordinate, value in items:
                typed_coordinate = cast("tuple[object, ...]", coordinate)
                if len(typed_coordinate) != 2:  # noqa: PLR2004
                    raise ValueError("A ValueStore coordinate must have two entries.")
                _admit_value_entry(
                    entries=entries,
                    regimes_by_period=regimes_by_period,
                    period=typed_coordinate[0],
                    regime=typed_coordinate[1],
                    value=value,
                )
        else:
            for period, regime_to_value in items:
                exact_period = _require_exact_value_period(period)
                if not isinstance(regime_to_value, Mapping):
                    raise TypeError(
                        f"ValueStore period {exact_period} must map to a mapping of "
                        "regime names to values."
                    )
                for regime, value in _traverse_public_mapping_items(
                    mapping=regime_to_value,
                    label=f"ValueStore period {exact_period} entries",
                ):
                    _admit_value_entry(
                        entries=entries,
                        regimes_by_period=regimes_by_period,
                        period=exact_period,
                        regime=regime,
                        value=value,
                    )
        object.__setattr__(self, "_entries", MappingProxyType(entries))
        object.__setattr__(
            self,
            "_regimes_by_period",
            MappingProxyType(
                {
                    period: tuple(regimes)
                    for period, regimes in regimes_by_period.items()
                }
            ),
        )

    def __getitem__(self, period: _ValuePeriodBoundary) -> _RegimeValuesBoundary:
        period = _require_exact_value_period(period)
        if period not in self._regimes_by_period:
            raise KeyError(period)
        return _ValuePeriodView(store=self, period=period)

    def __iter__(self) -> Iterator[int]:
        return iter(self._regimes_by_period)

    def __len__(self) -> int:
        return len(self._regimes_by_period)

    def __contains__(self, period: object) -> bool:
        """Check one period coordinate without materializing a value."""
        if type(period) is not int or period < 0:
            return False
        return period in self._regimes_by_period

    def _load(
        self,
        *,
        period: _ValuePeriodBoundary,
        regime: _RegimeNameBoundary,
    ) -> _FloatValueBoundary:
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        entry = self._entries[(period, regime)]
        value = _materialize_entry(entry=entry)
        if type(entry) is not _CanonicalValueEntry:
            value = _copy_solution_value(
                value=value,
                label=f"Solution value at period={period}, regime={regime!r}",
            )
        return cast("FloatND", value)

    def _raw(
        self, *, period: _ValuePeriodBoundary, regime: _RegimeNameBoundary
    ) -> object:
        """Return one eager value or lazy handle without materializing it."""
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        return self._entries[(period, regime)]

    def load_state(
        self, *, period: _ValuePeriodBoundary, regime: _RegimeNameBoundary
    ) -> LoadState:
        """Return one value entry's state without materializing it."""
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        entry = self._entries[(period, regime)]
        return entry.load_state if isinstance(entry, _LazyEntry) else LoadState.LOADED

    def materialize(self) -> _MaterializedValuesBoundary:
        """Return an exact immutable built-in snapshot of every value entry."""
        return MappingProxyType(
            {
                period: MappingProxyType(
                    {
                        regime: self._load(period=period, regime=regime)
                        for regime in view
                    }
                )
                for period, view in self.items()
            }
        )


@dataclass(frozen=True, eq=False)
class ArtifactStore(Mapping[ArtifactRef, object]):
    """Immutable store of explicitly addressed solution artifacts.

    The mapping interface keeps artifacts solver-extensible. ``project`` gives engine
    consumers the nested ``period -> regime -> payload`` view for one known artifact
    key.
    """

    # Typed with ``Any`` keys so the runtime annotation check does not traverse the
    # key view; ``__post_init__`` is the one admission boundary.
    _entries: Mapping[Any, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # One item traversal; each raw address is an exact ``ArtifactRef`` and unique
        # before it is inserted, so no equality alias or repeat can contract.
        entries: dict[ArtifactRef, object] = {}
        for raw_ref, payload in _traverse_public_mapping_items(
            mapping=self._entries, label="ArtifactStore entries"
        ):
            ref = _require_exact_artifact_ref(raw_ref)
            if ref in entries:
                raise ValueError(f"ArtifactStore address {ref!r} appears twice.")
            entries[ref] = payload
        object.__setattr__(self, "_entries", MappingProxyType(entries))

    def __getitem__(self, ref: _ArtifactRefBoundary) -> object:
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(entry=self._entries[ref])

    def __iter__(self) -> Iterator[ArtifactRef]:
        return iter(cast("Mapping[ArtifactRef, object]", self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ref: object) -> bool:
        """Check one artifact address without materializing its payload."""
        try:
            ref = _require_exact_artifact_ref(ref)
        except TypeError, ValueError:
            return False
        return ref in self._entries

    def project(
        self, key: _ArtifactKeyBoundary
    ) -> Mapping[int, Mapping[RegimeName, object]]:
        """Project one artifact schema to an immutable nested period mapping."""
        key = _require_exact_artifact_key(key)
        projected: dict[int, dict[RegimeName, object]] = {}
        for ref in self._entries:
            _require_exact_artifact_ref(ref)
            if _same_exact_artifact_contract(actual=ref.key, expected=key):
                projected.setdefault(ref.period, {})[ref.regime] = self[ref]
        return MappingProxyType(
            {
                period: MappingProxyType(regime_to_payload)
                for period, regime_to_payload in sorted(projected.items())
            }
        )

    def _raw(self, ref: _ArtifactRefBoundary) -> object:
        """Return one eager payload or lazy handle without materializing it."""
        ref = _require_exact_artifact_ref(ref)
        return self._entries[ref]

    def load_state(self, ref: _ArtifactRefBoundary) -> LoadState:
        """Return one artifact entry's state without materializing it."""
        ref = _require_exact_artifact_ref(ref)
        entry = self._entries[ref]
        return entry.load_state if isinstance(entry, _LazyEntry) else LoadState.LOADED

    # keyword-only-exempt: primary-argument=ref
    def materialize(
        self, ref: _ArtifactRefBoundary, *, template: object | None = None
    ) -> object:
        """Load one entry and optionally rebuild its declared PyTree shape."""
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(
            entry=self._entries[ref],
            template=template,
        )

    # keyword-only-exempt: primary-argument=ref
    def _materialize_from_template_snapshot(
        self,
        ref: _ArtifactRefBoundary,
        *,
        template_snapshot: object,
    ) -> object:
        """Load one entry through an engine-owned cached PyTree declaration."""
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(
            entry=self._entries[ref],
            template_snapshot=template_snapshot,
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

    ``model_fingerprint`` is the durable semantic identity used for restored
    results. ``model_instance_id`` remains a separate same-instance guard for
    in-memory results. ``params_fingerprint`` binds the result to the canonical
    solve parameters used by that solve.
    """

    retention: ResultRetention
    n_periods: int
    regime_names: tuple[RegimeName, ...]
    solver_types: Mapping[RegimeName, str]
    model_instance_id: str
    params_fingerprint: str
    value_schemas: Mapping[tuple[int, RegimeName], ValueArraySchema]
    model_fingerprint: str = "0" * _SHA256_HEX_LENGTH
    solver_identities: Mapping[RegimeName, SolverIdentity] = field(default_factory=dict)
    replay_routes: Mapping[RegimeName, ReplayRouteIdentity | None] = field(
        default_factory=dict
    )
    artifact_descriptors: Mapping[ArtifactRef, ArtifactDescriptor] = field(
        default_factory=dict
    )
    source: SolutionSource = SolutionSource.IN_MEMORY
    pylcm_version: str = PYLCM_VERSION
    solver_api_version: int = SOLVER_API_VERSION
    solution_schema_version: int = SOLUTION_SCHEMA_VERSION

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
    metadata: SolutionMetadata
    retained_continuations: _ArtifactStoreBoundary = field(
        default_factory=ArtifactStore
    )
    replay_artifacts: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    auxiliary_artifacts: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    omissions: _SolutionOmissionsInput = field(default_factory=dict)
    diagnostics: _ArtifactStoreBoundary = field(default_factory=ArtifactStore)
    _artifact_authority: Mapping[ArtifactRef, ArtifactAuthority] = field(
        default_factory=lambda: MappingProxyType({}),
        init=False,
        repr=False,
        compare=False,
    )

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


__all__ = [
    "DISSOLUTION_FLAG",
    "EGM_CONTINUATION",
    "PYLCM_VERSION",
    "SIMULATION_POLICY",
    "SOLUTION_FORMAT_VERSION",
    "SOLUTION_SCHEMA_VERSION",
    "SOLVER_API_VERSION",
    "SOLVER_DIAGNOSTICS",
    "ActionOutput",
    "ArtifactAuthority",
    "ArtifactChannel",
    "ArtifactDescriptor",
    "ArtifactKey",
    "ArtifactRef",
    "ArtifactStore",
    "AxisAuthority",
    "AxisDescriptor",
    "AxisRole",
    "CategoryDomain",
    "ContinuationArtifact",
    "ExecutableReplayRoute",
    "KernelOutput",
    "LeafAuthority",
    "LeafDescriptor",
    "LoadState",
    "OmissionReason",
    "PersistencePolicy",
    "ReplayMode",
    "ReplayModelContext",
    "ReplayReader",
    "ReplayRoute",
    "ReplayRouteIdentity",
    "ReplayRouteRequirements",
    "ReplayRouteSnapshot",
    "ResultRetention",
    "SimulationBuildContext",
    "SolutionMetadata",
    "SolutionResult",
    "SolutionSource",
    "SolverIdentity",
    "TreePath",
    "ValueArraySchema",
    "ValueStore",
]
