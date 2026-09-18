"""Model-owned artifact authority: construction plans, template binding, payload
canonicalization, and lazy canonical entries."""

import dataclasses
import functools
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import GetSetDescriptorType, MappingProxyType, MemberDescriptorType
from typing import (
    Any,
    Protocol,
    SupportsIndex,
    cast,
    runtime_checkable,
)

import jax
import numpy as np

from lcm._solver_api.contract import (
    _artifact_static_metadata_field_names,
    _same_exact_artifact_contract,
    _same_inert_pytree_metadata,
    _snapshot_inert_pytree_metadata,
)
from lcm._solver_api.descriptors import (
    ArtifactDescriptor,
    _capture_artifact_tree_path,
    _capture_category_domain,
    _capture_container_runtime_type,
    _capture_leaf_authority,
    _capture_mapping_item_stream_once,
    _capture_nonempty_mapping_name,
)
from lcm._solver_api.identity import (
    AxisAuthority,
    CategoryDomain,
    LeafAuthority,
    ReplayRouteIdentity,
    TreePath,
    _CategoricalDomainsBoundary,
    _ContainerRuntimeTypesBoundary,
    _LeafAuthoritiesBoundary,
)
from lcm.exceptions import ExecutionPlanningError


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
    """Transport-safe description of the same artifact."""
    payload_runtime_type: type[object]
    """Exact runtime class of the payload; `jax.Array` for a bare array."""
    template: object | None
    """Engine-built payload of the declared layout, or `None` for an artifact the model
    declares but never publishes in this cell.
    """
    container_runtime_types: _ContainerRuntimeTypesBoundary = field(
        default_factory=dict
    )
    """Exact container class at each container path of the PyTree."""
    leaves: _LeafAuthoritiesBoundary = field(default_factory=dict)
    """Authority of every numerical leaf, keyed by its path."""
    axes: tuple[AxisAuthority, ...] = ()
    """Every axis a leaf dimension refers to, with its canonical coordinates."""
    state_roles: tuple[str, ...] = ()
    """Names of the model states the leading axes index, in order."""
    action_roles: tuple[str, ...] = ()
    """Names of the model actions some axes index, in order."""
    categorical_domains: _CategoricalDomainsBoundary = field(default_factory=dict)
    """Exact label domain of each categorical state or action role."""
    consumer_route: ReplayRouteIdentity | None = None
    """The replay route that reads the payload, or `None` when none does."""
    applicable: bool = True
    """Whether the model structurally publishes this artifact in this cell."""
    required: bool = False
    """Whether every successful solve must retain the payload."""

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


@runtime_checkable
class _ArrayCopier(Protocol):
    """Call-local physical copy dependency; never retained by a public store."""

    def __call__(self, *, leaf: jax.Array, label: str) -> jax.Array:
        """Return one independent array preserving exact dtype, shape and layout."""
        ...


def _copy_artifact_array_leaf(
    *, leaf: object, label: str, array_copier: _ArrayCopier | None = None
) -> jax.Array:
    """Make one independent exact JAX buffer for an owned artifact graph."""
    if not isinstance(leaf, jax.Array):
        raise TypeError(f"{label} is not a JAX array.")
    if leaf.is_deleted():
        raise TypeError(f"{label} has been deleted or donated.")
    try:
        source_shape = tuple(leaf.shape)
        source_dtype = np.dtype(leaf.dtype)
        source_sharding = leaf.sharding
        copied = (
            jax.numpy.array(leaf, copy=True)
            if array_copier is None
            else array_copier(leaf=leaf, label=label)
        )
    except TypeError, ExecutionPlanningError:
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
    *, payload: object, authority: ArtifactAuthority, borrow: bool = False
) -> _CanonicalArtifactPayload:
    """Canonicalize one payload once and retain the exact validated leaves.

    The leaves are copied into private buffers unless `borrow` is set, which an
    engine caller uses for buffers it allocated itself and nobody else holds.
    """
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
            canonical_leaf
            if borrow
            else _copy_artifact_array_leaf(
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
