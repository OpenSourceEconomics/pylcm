"""Transport-safe artifact descriptors and their exact constructor validation."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    cast,
)

from lcm._solver_api.identity import (
    ArtifactChannel,
    ArtifactKey,
    AxisDescriptor,
    AxisRole,
    CategoryDomain,
    LeafAuthority,
    LeafDescriptor,
    PersistencePolicy,
    ReplayRouteIdentity,
    TreePath,
    _CategoricalDomainsBoundary,
)


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
    """Versioned schema identity of the payload."""
    channel: ArtifactChannel
    """Semantic channel the payload is stored on."""
    persistence: PersistencePolicy
    """Whether the payload may be written to a solution archive."""
    payload_type_id: str
    """Qualified name of the payload's runtime type; `"jax.Array"` for a bare array."""
    payload_version: int = 1
    """Version of the payload's container layout."""
    leaf_descriptors: tuple[LeafDescriptor, ...] = ()
    """Schema of every numerical leaf, in PyTree flattening order."""
    named_axes: tuple[AxisDescriptor, ...] = ()
    """Every axis a leaf dimension refers to by name."""
    state_roles: tuple[str, ...] = ()
    """Names of the model states the leading axes index, in order."""
    action_roles: tuple[str, ...] = ()
    """Names of the model actions some axes index, in order."""
    categorical_domains: _CategoricalDomainsBoundary = field(default_factory=dict)
    """Exact label domain of each categorical state or action role."""
    required_for: frozenset[ReplayRouteIdentity] = frozenset()
    """Replay routes that cannot run without this payload."""
    required: bool = False
    """Whether every successful solve must retain the payload."""

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


def _check_role_names(*, names: tuple[str, ...], label: str) -> None:
    """Reject ambiguous or non-built-in state/action role declarations."""
    if any(type(name) is not str or not name for name in names):
        raise TypeError(f"Artifact {label} roles must be nonempty exact strs.")
    if len(set(names)) != len(names):
        raise ValueError(f"Artifact {label} roles must be unique.")
