"""Explicit immutable snapshots for caller-visible solution envelopes.

The public result dataclasses are frozen, but a hostile lazy decoder can still retain
one of them and mutate it with :func:`object.__setattr__`.  Mapping copies alone are
therefore not a trust boundary: every dataclass wrapper inspected after decoding must
be reconstructed while preserving only the deliberately trusted runtime types and
artifact templates.
"""

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from lcm.solver_api import (
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    AxisAuthority,
    AxisDescriptor,
    AxisRole,
    CategoryDomain,
    LeafAuthority,
    LeafDescriptor,
    OmissionReason,
    PersistencePolicy,
    ReplayRouteIdentity,
    ResultRetention,
    SolutionMetadata,
    SolutionSource,
    SolverIdentity,
    TreePath,
    ValueArraySchema,
    ValueStore,
    _artifact_authority_from_template_snapshot,
    _artifact_authority_template_snapshot,
    _canonical_artifact_entry_from_authority,
    _CanonicalArtifactEntry,
    _CanonicalArtifactTemplate,
    _LazyEntry,
    _same_exact_artifact_contract,
    _validate_axes_and_leaves,
)

if TYPE_CHECKING:
    _ArtifactStoreBoundary: TypeAlias = ArtifactStore  # noqa: UP040
    _ValueStoreBoundary: TypeAlias = ValueStore  # noqa: UP040
    _OmissionsInput: TypeAlias = Mapping[  # noqa: UP040
        ArtifactRef, OmissionReason
    ]
    _OmissionsSnapshot: TypeAlias = MappingProxyType[  # noqa: UP040
        ArtifactRef, OmissionReason
    ]
    _AuthoritiesInput: TypeAlias = Mapping[  # noqa: UP040
        ArtifactRef, ArtifactAuthority
    ]
    _AuthoritiesSnapshot: TypeAlias = MappingProxyType[  # noqa: UP040
        ArtifactRef, ArtifactAuthority
    ]
else:
    # Explicit body checks own these hostile-input boundaries; runtime annotation
    # traversal must not inspect their contents first.
    _ArtifactStoreBoundary = object
    _ValueStoreBoundary = object
    _OmissionsInput = object
    _OmissionsSnapshot = object
    _AuthoritiesInput = object
    _AuthoritiesSnapshot = object


# keyword-only-exempt: primary-argument=mapping
def capture_exact_mapping[Key, Value](
    mapping: object,
    *,
    label: str,
    snapshot_key: Callable[[Any], Key],
    snapshot_value: Callable[[Any], Value],
) -> dict[Key, Value]:
    """Own one exact mapping through one canonicalizing item traversal.

    ``MappingProxyType`` can proxy an arbitrary ``Mapping`` implementation. Its
    ``items()``, ``values()``, and ``len()`` operations may therefore execute backing
    code or expose different views. Consume exactly one item iterator, normalize
    traversal failures, and only hash keys after their canonical snapshot exists.
    """
    if type(mapping) is not MappingProxyType:
        raise TypeError(f"{label} must be an immutable exact mapping.")
    source = mapping
    try:
        iterator = iter(source.items())
    except Exception as error:
        raise TypeError(
            f"{label} cannot be traversed as exact mapping items."
        ) from error

    copied: dict[Key, Value] = {}
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            raise TypeError(
                f"{label} cannot be traversed as exact mapping items."
            ) from error
        if type(item) is not tuple or len(item) != 2:  # noqa: PLR2004
            raise TypeError(f"{label} items must be exact key-value pairs.")
        raw_key, raw_value = item
        key = snapshot_key(raw_key)
        value = snapshot_value(raw_value)
        if key in copied:
            raise ValueError(f"{label} keys collide after exact reconstruction.")
        copied[key] = value
    return copied


def snapshot_artifact_key(key: ArtifactKey) -> ArtifactKey:
    """Reconstruct one exact public artifact identity."""
    if type(key) is not ArtifactKey:
        raise TypeError("Artifact keys must be exact ArtifactKey objects.")
    _require_nonempty_exact_str(key.type_id, label="ArtifactKey.type_id")
    _require_positive_exact_int(
        key.schema_version,
        label="ArtifactKey.schema_version",
    )
    return ArtifactKey(type_id=key.type_id, schema_version=key.schema_version)


def snapshot_artifact_ref(ref: ArtifactRef) -> ArtifactRef:
    """Reconstruct one exact public artifact address and its nested key."""
    if type(ref) is not ArtifactRef:
        raise TypeError("Artifact addresses must be exact ArtifactRef objects.")
    _require_nonnegative_exact_int(ref.period, label="ArtifactRef.period")
    _require_nonempty_exact_str(ref.regime, label="ArtifactRef.regime")
    return ArtifactRef(
        period=ref.period,
        regime=ref.regime,
        key=snapshot_artifact_key(ref.key),
    )


def snapshot_artifact_store(
    *,
    store: _ArtifactStoreBoundary,
    authorities: _AuthoritiesInput | None = None,
) -> _ArtifactStoreBoundary:
    """Copy addresses and detach every authority-backed eager payload."""
    if type(store) is not ArtifactStore:
        raise TypeError("Artifact stores must be exact ArtifactStore objects.")
    entries = capture_exact_mapping(
        store._entries,  # noqa: SLF001
        label="Artifact store entries",
        snapshot_key=snapshot_artifact_ref,
        snapshot_value=_keep_payload,
    )
    if authorities is not None:
        for ref, payload in entries.items():
            authority = authorities.get(ref)
            if authority is None:
                continue
            if type(payload) is _CanonicalArtifactEntry:
                owned_payload = payload.materialize()
            elif isinstance(payload, _LazyEntry):
                continue
            else:
                owned_payload = payload
            entries[ref] = _canonical_artifact_entry_from_authority(
                payload=owned_payload,
                authority=authority,
            )
    return ArtifactStore(entries)


def snapshot_value_store(store: _ValueStoreBoundary) -> _ValueStoreBoundary:
    """Copy value coordinates without materializing any payload."""
    if type(store) is not ValueStore:
        raise TypeError("Solution values must be an exact ValueStore.")
    entries = capture_exact_mapping(
        store._entries,  # noqa: SLF001
        label="Solution value entries",
        snapshot_key=_snapshot_value_coordinate,
        snapshot_value=_keep_payload,
    )
    return ValueStore(cast("Mapping[object, object]", entries))


def snapshot_omissions(
    omissions: _OmissionsInput,
) -> _OmissionsSnapshot:
    """Copy omission addresses while retaining exact enum values for later checks."""
    copied = capture_exact_mapping(
        omissions,
        label="Solution omissions",
        snapshot_key=snapshot_artifact_ref,
        snapshot_value=_snapshot_omission_reason,
    )
    return MappingProxyType(copied)


def snapshot_solution_metadata(metadata: SolutionMetadata) -> SolutionMetadata:
    """Reconstruct the complete public metadata graph without shared wrappers."""
    if type(metadata) is not SolutionMetadata:
        raise TypeError("Solution metadata must be exact SolutionMetadata.")
    _validate_solution_metadata_fields(metadata)
    retention = metadata.retention
    n_periods = metadata.n_periods
    regime_names = tuple(metadata.regime_names)
    model_instance_id = metadata.model_instance_id
    params_fingerprint = metadata.params_fingerprint
    model_fingerprint = metadata.model_fingerprint
    source = metadata.source
    pylcm_version = metadata.pylcm_version
    solver_api_version = metadata.solver_api_version
    solution_schema_version = metadata.solution_schema_version
    solver_types_source = metadata.solver_types
    value_schemas_source = metadata.value_schemas
    solver_identities_source = metadata.solver_identities
    replay_routes_source = metadata.replay_routes
    artifact_descriptors_source = metadata.artifact_descriptors
    solver_types = capture_exact_mapping(
        solver_types_source,
        label="solver_types",
        snapshot_key=partial(_snapshot_nonempty_exact_str, label="solver_types regime"),
        snapshot_value=partial(
            _snapshot_nonempty_exact_str, label="solver_types value"
        ),
    )
    value_schemas = capture_exact_mapping(
        value_schemas_source,
        label="value_schemas",
        snapshot_key=_snapshot_value_coordinate,
        snapshot_value=_snapshot_value_array_schema,
    )
    solver_identities = capture_exact_mapping(
        solver_identities_source,
        label="solver_identities",
        snapshot_key=partial(
            _snapshot_nonempty_exact_str, label="solver_identities regime"
        ),
        snapshot_value=_snapshot_solver_identity,
    )
    replay_routes = capture_exact_mapping(
        replay_routes_source,
        label="replay_routes",
        snapshot_key=partial(
            _snapshot_nonempty_exact_str, label="replay_routes regime"
        ),
        snapshot_value=_snapshot_optional_replay_route_identity,
    )
    artifact_descriptors = capture_exact_mapping(
        artifact_descriptors_source,
        label="artifact_descriptors",
        snapshot_key=snapshot_artifact_ref,
        snapshot_value=snapshot_artifact_descriptor,
    )
    return SolutionMetadata(
        retention=retention,
        n_periods=n_periods,
        regime_names=regime_names,
        solver_types=solver_types,
        model_instance_id=model_instance_id,
        params_fingerprint=params_fingerprint,
        value_schemas=value_schemas,
        model_fingerprint=model_fingerprint,
        solver_identities=solver_identities,
        replay_routes=replay_routes,
        artifact_descriptors=artifact_descriptors,
        source=source,
        pylcm_version=pylcm_version,
        solver_api_version=solver_api_version,
        solution_schema_version=solution_schema_version,
    )


def snapshot_artifact_descriptor(
    descriptor: ArtifactDescriptor,
) -> ArtifactDescriptor:
    """Reconstruct a public artifact schema and every nested wrapper."""
    if type(descriptor) is not ArtifactDescriptor:
        raise TypeError(
            "Artifact descriptors must be exact ArtifactDescriptor objects."
        )
    _validate_artifact_descriptor_fields(descriptor)
    key = snapshot_artifact_key(descriptor.key)
    channel = descriptor.channel
    persistence = descriptor.persistence
    payload_type_id = descriptor.payload_type_id
    payload_version = descriptor.payload_version
    leaf_descriptors = tuple(
        _snapshot_leaf_descriptor(leaf) for leaf in descriptor.leaf_descriptors
    )
    named_axes = tuple(
        _snapshot_axis_descriptor(axis) for axis in descriptor.named_axes
    )
    state_roles = tuple(descriptor.state_roles)
    action_roles = tuple(descriptor.action_roles)
    required_for = frozenset(
        _snapshot_replay_route_identity(identity)
        for identity in descriptor.required_for
    )
    required = descriptor.required
    categorical_domains = capture_exact_mapping(
        descriptor.categorical_domains,
        label="categorical_domains",
        snapshot_key=partial(
            _snapshot_nonempty_exact_str, label="categorical domain name"
        ),
        snapshot_value=_snapshot_category_domain,
    )
    return ArtifactDescriptor(
        key=key,
        channel=channel,
        persistence=persistence,
        payload_type_id=payload_type_id,
        payload_version=payload_version,
        leaf_descriptors=leaf_descriptors,
        named_axes=named_axes,
        state_roles=state_roles,
        action_roles=action_roles,
        categorical_domains=categorical_domains,
        required_for=required_for,
        required=required,
    )


def snapshot_artifact_authorities(
    authorities: _AuthoritiesInput,
) -> _AuthoritiesSnapshot:
    """Copy private authority wrappers while preserving trusted templates/types."""
    copied = capture_exact_mapping(
        authorities,
        label="Artifact authority",
        snapshot_key=snapshot_artifact_ref,
        snapshot_value=_snapshot_artifact_authority,
    )
    return MappingProxyType(copied)


def snapshot_artifact_template_declaration(
    authority: ArtifactAuthority,
) -> _CanonicalArtifactTemplate | None:
    """Return an owned cached declaration without rerunning its flatten callback."""
    try:
        return _artifact_authority_template_snapshot(authority)
    except TypeError, ValueError:
        raise
    except Exception as error:
        raise TypeError("Artifact template cannot be snapshotted exactly.") from error


def snapshot_artifact_template(authority: ArtifactAuthority) -> object | None:
    """Return a callback-only template copy detached from private authority."""
    snapshot = snapshot_artifact_template_declaration(authority)
    return None if snapshot is None else snapshot.payload


def _snapshot_artifact_authority(authority: ArtifactAuthority) -> ArtifactAuthority:
    """Rebuild one authority wrapper from its cached PyTree declaration."""
    if type(authority) is not ArtifactAuthority:
        raise TypeError("Artifact authorities must be exact ArtifactAuthority objects.")
    try:
        template_snapshot = _artifact_authority_template_snapshot(authority)
    except TypeError, ValueError:
        raise
    except Exception as error:
        raise TypeError("Artifact template cannot be snapshotted exactly.") from error
    _validate_artifact_authority_fields(authority)
    descriptor_source = authority.descriptor
    payload_runtime_type = authority.payload_runtime_type
    containers_source = authority.container_runtime_types
    leaves_source = authority.leaves
    axes_source = authority.axes
    state_roles_source = authority.state_roles
    action_roles_source = authority.action_roles
    categories_source = authority.categorical_domains
    consumer_route_source = authority.consumer_route
    applicable = authority.applicable
    required = authority.required
    descriptor = snapshot_artifact_descriptor(descriptor_source)
    axes = tuple(_snapshot_axis_authority(axis) for axis in axes_source)
    state_roles = tuple(state_roles_source)
    action_roles = tuple(action_roles_source)
    consumer_route = (
        None
        if consumer_route_source is None
        else _snapshot_replay_route_identity(consumer_route_source)
    )
    containers = capture_exact_mapping(
        containers_source,
        label="authority container_runtime_types",
        snapshot_key=_snapshot_tree_path,
        snapshot_value=_snapshot_runtime_type,
    )
    leaves = capture_exact_mapping(
        leaves_source,
        label="authority leaves",
        snapshot_key=_snapshot_tree_path,
        snapshot_value=_snapshot_leaf_authority,
    )
    categories = capture_exact_mapping(
        categories_source,
        label="authority categorical_domains",
        snapshot_key=partial(
            _snapshot_nonempty_exact_str, label="authority categorical domain name"
        ),
        snapshot_value=_snapshot_category_domain,
    )
    _validate_authority_copy(
        descriptor=descriptor,
        leaves=leaves,
        axes=axes,
        state_roles=state_roles,
        action_roles=action_roles,
        categories=categories,
        consumer_route=consumer_route,
        applicable=applicable,
        required=required,
    )

    return _artifact_authority_from_template_snapshot(
        descriptor=descriptor,
        payload_runtime_type=payload_runtime_type,
        template_snapshot=template_snapshot,
        container_runtime_types=containers,
        leaves=leaves,
        axes=axes,
        state_roles=state_roles,
        action_roles=action_roles,
        categorical_domains=categories,
        consumer_route=consumer_route,
        applicable=applicable,
        required=required,
    )


def _validate_authority_copy(
    *,
    descriptor: ArtifactDescriptor,
    leaves: Mapping[TreePath, LeafAuthority],
    axes: tuple[AxisAuthority, ...],
    state_roles: tuple[str, ...],
    action_roles: tuple[str, ...],
    categories: Mapping[str, CategoryDomain],
    consumer_route: ReplayRouteIdentity | None,
    applicable: bool,
    required: bool,
) -> None:
    """Validate copied wrapper relations without touching the executable template."""
    if type(applicable) is not bool or type(required) is not bool:
        raise TypeError("Artifact applicability and requiredness must be exact bools.")
    if not _same_exact_artifact_contract(
        actual=descriptor.leaf_descriptors,
        expected=tuple(leaf.descriptor for leaf in leaves.values()),
    ):
        raise ValueError("Artifact descriptive leaves differ from model authority.")
    if not _same_exact_artifact_contract(
        actual=descriptor.named_axes,
        expected=tuple(axis.descriptor for axis in axes),
    ):
        raise ValueError("Artifact descriptive axes differ from model authority.")
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
    expected_required_for = (
        frozenset({consumer_route})
        if required and consumer_route is not None
        else frozenset()
    )
    if not _same_exact_artifact_contract(
        actual=descriptor.required_for,
        expected=expected_required_for,
    ):
        raise ValueError("Artifact descriptor routes differ from model authority.")
    if type(applicable) is not bool or type(required) is not bool:
        raise TypeError("Artifact applicability and requiredness must be exact bools.")
    if descriptor.required is not required:
        raise ValueError("Artifact descriptive requiredness differs from authority.")
    _validate_axes_and_leaves(axes=axes, leaves=tuple(leaves.values()))


def _snapshot_value_array_schema(schema: ValueArraySchema) -> ValueArraySchema:
    if type(schema) is not ValueArraySchema:
        raise TypeError("Value schemas must be exact ValueArraySchema objects.")
    _require_exact_shape(schema.shape, label="value schema shape")
    _require_nonempty_exact_str(schema.dtype, label="value schema dtype")
    _require_exact_names(schema.axis_names, label="value schema axis_names")
    if len(schema.axis_names) != len(schema.shape):
        raise ValueError("Value schema axis_names must name every dimension.")
    return ValueArraySchema(
        shape=tuple(schema.shape),
        dtype=schema.dtype,
        axis_names=tuple(schema.axis_names),
    )


def _snapshot_value_coordinate(coordinate: object) -> tuple[int, str]:
    if type(coordinate) is not tuple or len(coordinate) != 2:  # noqa: PLR2004
        raise TypeError("Solution value coordinates must be exact pairs.")
    period, regime = coordinate
    _require_nonnegative_exact_int(period, label="solution value period")
    _require_nonempty_exact_str(regime, label="solution value regime")
    return cast("int", period), cast("str", regime)


def _snapshot_omission_reason(reason: object) -> OmissionReason:
    if type(reason) is not OmissionReason:
        raise TypeError(
            "Solution omission reasons must be exact OmissionReason values."
        )
    return reason


def _keep_payload(payload: object) -> object:
    """Return a store payload as is; only its address is reconstructed."""
    return payload


# keyword-only-exempt: primary-argument=value
def _snapshot_nonempty_exact_str(value: object, *, label: str) -> str:
    _require_nonempty_exact_str(value, label=label)
    return cast("str", value)


def _snapshot_runtime_type(value: object) -> type[object]:
    if not isinstance(value, type):
        raise TypeError("Artifact container runtime declarations must be types.")
    return value


def _snapshot_solver_identity(identity: SolverIdentity) -> SolverIdentity:
    if type(identity) is not SolverIdentity:
        raise TypeError("Solver identities must be exact SolverIdentity objects.")
    _require_nonempty_exact_str(identity.plugin_id, label="solver plugin_id")
    _require_nonempty_exact_str(identity.plugin_version, label="solver plugin_version")
    _require_positive_exact_int(
        identity.solver_api_version,
        label="solver API version",
    )
    return SolverIdentity(
        plugin_id=identity.plugin_id,
        plugin_version=identity.plugin_version,
        solver_api_version=identity.solver_api_version,
    )


def _snapshot_replay_route_identity(
    identity: ReplayRouteIdentity,
) -> ReplayRouteIdentity:
    if type(identity) is not ReplayRouteIdentity:
        raise TypeError("Replay identities must be exact ReplayRouteIdentity objects.")
    _require_nonempty_exact_str(identity.route_id, label="replay route_id")
    _require_positive_exact_int(identity.route_version, label="replay route_version")
    return ReplayRouteIdentity(
        route_id=identity.route_id,
        route_version=identity.route_version,
    )


def _snapshot_optional_replay_route_identity(
    identity: object,
) -> ReplayRouteIdentity | None:
    if identity is None:
        return None
    if type(identity) is not ReplayRouteIdentity:
        raise TypeError("Replay identities must be exact ReplayRouteIdentity objects.")
    return _snapshot_replay_route_identity(identity)


def _snapshot_category_domain(domain: CategoryDomain) -> CategoryDomain:
    if type(domain) is not CategoryDomain:
        raise TypeError("Categorical domains must be exact CategoryDomain objects.")
    _require_exact_names(domain.labels, label="category labels")
    _require_exact_tuple(domain.codes, label="category codes")
    if any(type(code) is not int for code in domain.codes):
        raise TypeError("Category codes must be exact ints.")
    if type(domain.ordered) is not bool:
        raise TypeError("Category ordering must be an exact bool.")
    return CategoryDomain(
        labels=tuple(domain.labels),
        codes=tuple(domain.codes),
        ordered=domain.ordered,
    )


def _snapshot_axis_descriptor(axis: AxisDescriptor) -> AxisDescriptor:
    if type(axis) is not AxisDescriptor:
        raise TypeError("Axis descriptors must be exact AxisDescriptor objects.")
    _validate_axis_fields(
        name=axis.name,
        length=axis.length,
        role=axis.role,
        coordinates=axis.coordinates,
        label="axis descriptor",
    )
    return AxisDescriptor(
        name=axis.name,
        length=axis.length,
        role=axis.role,
        coordinates=tuple(axis.coordinates),
    )


def _snapshot_axis_authority(axis: AxisAuthority) -> AxisAuthority:
    if type(axis) is not AxisAuthority:
        raise TypeError("Axis authorities must be exact AxisAuthority objects.")
    _validate_axis_fields(
        name=axis.name,
        length=axis.length,
        role=axis.role,
        coordinates=axis.coordinates,
        label="axis authority",
    )
    return AxisAuthority(
        name=axis.name,
        length=axis.length,
        role=axis.role,
        coordinates=tuple(axis.coordinates),
    )


def _snapshot_leaf_descriptor(leaf: LeafDescriptor) -> LeafDescriptor:
    if type(leaf) is not LeafDescriptor:
        raise TypeError("Leaf descriptors must be exact LeafDescriptor objects.")
    _validate_leaf_fields(
        path=leaf.path,
        shape=leaf.shape,
        dtype=leaf.dtype,
        axis_names=leaf.axis_names,
        label="leaf descriptor",
    )
    return LeafDescriptor(
        path=_snapshot_tree_path(leaf.path),
        shape=tuple(leaf.shape),
        dtype=leaf.dtype,
        axis_names=tuple(leaf.axis_names),
    )


def _snapshot_leaf_authority(leaf: LeafAuthority) -> LeafAuthority:
    if type(leaf) is not LeafAuthority:
        raise TypeError("Leaf authorities must be exact LeafAuthority objects.")
    if not isinstance(leaf.runtime_type, type):
        raise TypeError("Leaf authority runtime_type must be a type.")
    _validate_leaf_fields(
        path=leaf.path,
        shape=leaf.shape,
        dtype=leaf.dtype,
        axis_names=leaf.axis_names,
        label="leaf authority",
    )
    return LeafAuthority(
        path=_snapshot_tree_path(leaf.path),
        runtime_type=leaf.runtime_type,
        shape=tuple(leaf.shape),
        dtype=leaf.dtype,
        axis_names=tuple(leaf.axis_names),
    )


def _snapshot_tree_path(path: object) -> TreePath:
    _require_exact_tuple(path, label="tree path")
    components = cast("tuple[object, ...]", path)
    if any(type(component) is not str or not component for component in components):
        raise TypeError("Tree paths must contain nonempty exact strs.")
    return cast("TreePath", tuple(components))


def _validate_solution_metadata_fields(
    metadata: SolutionMetadata,
) -> None:
    """Reject noncanonical metadata before any value is compared or hashed."""
    if type(metadata.retention) is not ResultRetention:
        raise TypeError("Solution metadata retention must be exact ResultRetention.")
    _require_positive_exact_int(metadata.n_periods, label="solution n_periods")
    _require_exact_names(metadata.regime_names, label="solution regime_names")
    for field_name in (
        "model_instance_id",
        "params_fingerprint",
        "model_fingerprint",
        "pylcm_version",
    ):
        _require_exact_str(getattr(metadata, field_name), label=field_name)
    if type(metadata.source) is not SolutionSource:
        raise TypeError("Solution metadata source must be exact SolutionSource.")
    _require_positive_exact_int(
        metadata.solver_api_version,
        label="solution solver_api_version",
    )
    _require_positive_exact_int(
        metadata.solution_schema_version,
        label="solution solution_schema_version",
    )

    for label, mapping in (
        ("solver_types", metadata.solver_types),
        ("value_schemas", metadata.value_schemas),
        ("solver_identities", metadata.solver_identities),
        ("replay_routes", metadata.replay_routes),
        ("artifact_descriptors", metadata.artifact_descriptors),
    ):
        _require_exact_mapping(mapping, label=label)


def _validate_artifact_descriptor_fields(
    descriptor: ArtifactDescriptor,
) -> None:
    """Validate every descriptor field before its public constructor hashes it."""
    if type(descriptor.key) is not ArtifactKey:
        raise TypeError("Artifact descriptor key must be exact ArtifactKey.")
    if type(descriptor.channel) is not ArtifactChannel:
        raise TypeError("Artifact descriptor channel must be exact ArtifactChannel.")
    if type(descriptor.persistence) is not PersistencePolicy:
        raise TypeError(
            "Artifact descriptor persistence must be exact PersistencePolicy."
        )
    _require_nonempty_exact_str(
        descriptor.payload_type_id,
        label="artifact payload_type_id",
    )
    _require_positive_exact_int(
        descriptor.payload_version,
        label="artifact payload_version",
    )
    if type(descriptor.required) is not bool:
        raise TypeError("Artifact descriptor required must be an exact bool.")

    _require_exact_tuple(descriptor.leaf_descriptors, label="leaf_descriptors")
    if any(type(leaf) is not LeafDescriptor for leaf in descriptor.leaf_descriptors):
        raise TypeError("Artifact descriptor leaves must be exact LeafDescriptor.")
    _require_exact_tuple(descriptor.named_axes, label="named_axes")
    if any(type(axis) is not AxisDescriptor for axis in descriptor.named_axes):
        raise TypeError("Artifact descriptor axes must be exact AxisDescriptor.")
    _require_exact_names(descriptor.state_roles, label="state_roles")
    _require_exact_names(descriptor.action_roles, label="action_roles")

    _require_exact_mapping(
        descriptor.categorical_domains,
        label="categorical_domains",
    )

    if type(descriptor.required_for) is not frozenset:
        raise TypeError("Artifact descriptor required_for must be an exact frozenset.")
    if any(
        type(identity) is not ReplayRouteIdentity
        for identity in descriptor.required_for
    ):
        raise TypeError("Artifact descriptor routes must be exact replay identities.")


def _validate_artifact_authority_fields(
    authority: ArtifactAuthority,
) -> None:
    """Validate authority scalars and outer containers before copying its template."""
    if type(authority.descriptor) is not ArtifactDescriptor:
        raise TypeError("Artifact authority descriptor must be exact.")
    if not isinstance(authority.payload_runtime_type, type):
        raise TypeError("Artifact authority payload_runtime_type must be a type.")
    if type(authority.applicable) is not bool or type(authority.required) is not bool:
        raise TypeError("Artifact applicability and requiredness must be exact bools.")
    if (
        authority.consumer_route is not None
        and type(authority.consumer_route) is not ReplayRouteIdentity
    ):
        raise TypeError("Artifact consumer route must be an exact replay identity.")

    _require_exact_mapping(
        authority.container_runtime_types,
        label="authority container_runtime_types",
    )

    _require_exact_mapping(authority.leaves, label="authority leaves")

    _require_exact_tuple(authority.axes, label="authority axes")
    if any(type(axis) is not AxisAuthority for axis in authority.axes):
        raise TypeError("Artifact authority axes must be exact AxisAuthority.")
    _require_exact_names(authority.state_roles, label="authority state_roles")
    _require_exact_names(authority.action_roles, label="authority action_roles")

    _require_exact_mapping(
        authority.categorical_domains,
        label="authority categorical_domains",
    )


def _validate_axis_fields(
    *,
    name: object,
    length: object,
    role: object,
    coordinates: object,
    label: str,
) -> None:
    _require_nonempty_exact_str(name, label=f"{label} name")
    _require_nonnegative_exact_int(length, label=f"{label} length")
    if type(role) is not AxisRole:
        raise TypeError(f"{label} role must be exact AxisRole.")
    _require_exact_json_scalars(coordinates, label=f"{label} coordinates")


def _validate_leaf_fields(
    *,
    path: object,
    shape: object,
    dtype: object,
    axis_names: object,
    label: str,
) -> None:
    _snapshot_tree_path(path)
    _require_exact_shape(shape, label=f"{label} shape")
    _require_nonempty_exact_str(dtype, label=f"{label} dtype")
    _require_exact_names(axis_names, label=f"{label} axis_names")
    typed_shape = cast("tuple[object, ...]", shape)
    typed_axis_names = cast("tuple[object, ...]", axis_names)
    if len(typed_axis_names) != len(typed_shape):
        raise ValueError(f"{label} axis_names must name every dimension.")


# keyword-only-exempt: primary-argument=value
def _require_exact_mapping(value: object, *, label: str) -> None:
    if type(value) is not MappingProxyType:
        raise TypeError(f"{label} must be an immutable exact mapping.")


# keyword-only-exempt: primary-argument=value
def _require_exact_tuple(value: object, *, label: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple.")


# keyword-only-exempt: primary-argument=value
def _require_exact_str(value: object, *, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact str.")


# keyword-only-exempt: primary-argument=value
def _require_nonempty_exact_str(value: object, *, label: str) -> None:
    _require_exact_str(value, label=label)
    if not value:
        raise ValueError(f"{label} must not be empty.")


# keyword-only-exempt: primary-argument=value
def _require_nonnegative_exact_int(value: object, *, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int.")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative.")


# keyword-only-exempt: primary-argument=value
def _require_positive_exact_int(value: object, *, label: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact int.")
    if value < 1:
        raise ValueError(f"{label} must be positive.")


# keyword-only-exempt: primary-argument=value
def _require_exact_shape(value: object, *, label: str) -> None:
    _require_exact_tuple(value, label=label)
    shape = cast("tuple[object, ...]", value)
    if any(type(size) is not int for size in shape):
        raise TypeError(f"{label} must contain exact ints.")
    if any(cast("int", size) < 0 for size in shape):
        raise ValueError(f"{label} must contain nonnegative sizes.")


# keyword-only-exempt: primary-argument=value
def _require_exact_names(value: object, *, label: str) -> None:
    _require_exact_tuple(value, label=label)
    names = cast("tuple[object, ...]", value)
    if any(type(name) is not str or not name for name in names):
        raise TypeError(f"{label} must contain nonempty exact strs.")


# keyword-only-exempt: primary-argument=value
def _require_exact_json_scalars(value: object, *, label: str) -> None:
    _require_exact_tuple(value, label=label)
    items = cast("tuple[object, ...]", value)
    allowed = (bool, int, float, str)
    if any(not any(type(item) is cls for cls in allowed) for item in items):
        raise TypeError(f"{label} must contain exact JSON scalars.")
