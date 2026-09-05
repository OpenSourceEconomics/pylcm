"""Single-traversal ownership of caller-visible solution mappings."""

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, NamedTuple, Never, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.solver_api as solver_api_module
from _lcm.engine import StateActionSpace
from _lcm.persistence import solution as solution_persistence
from _lcm.solution import model_authority as model_authority_module
from _lcm.solution.result_snapshot import (
    snapshot_artifact_authorities,
    snapshot_artifact_store,
    snapshot_artifact_template_declaration,
    snapshot_solution_metadata,
)
from lcm import LinSpacedGrid, Model
from lcm.exceptions import (
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    SolutionIntegrityError,
)
from lcm.persistence import save_solution
from lcm.solver_api import (
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    CategoryDomain,
    LeafAuthority,
    LeafDescriptor,
    LoadState,
    PersistencePolicy,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    ValueStore,
)
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


class _OneShotMapping(Mapping[object, object]):
    """Expose one item stream while refusing independent size/value observations."""

    def __init__(self, entries: Mapping[object, object]) -> None:
        self._entries = dict(entries)
        self.traversals = 0

    def __getitem__(self, key: object) -> object:
        return self._entries[key]

    def __iter__(self) -> Iterator[object]:
        self.traversals += 1
        if self.traversals > 1:
            raise RuntimeError("mapping was traversed more than once")
        return iter(self._entries)

    def __len__(self) -> int:
        raise RuntimeError("mapping length was observed independently")

    def values(self) -> Never:
        raise RuntimeError("mapping values were traversed independently")


class _SingleItemMapping(Mapping[object, object]):
    """Yield one raw pair without hashing its caller-owned key."""

    def __init__(self, *, key: object, value: object) -> None:
        self._key = key
        self._value = value

    def __getitem__(self, key: object) -> object:
        if key is not self._key:
            raise KeyError(key)
        return self._value

    def __iter__(self) -> Iterator[object]:
        return iter((self._key,))

    def __len__(self) -> Never:
        raise RuntimeError("single-item mapping length must not be observed")


class _RaisingPathComponent:
    """Expose any hash/equality use before exact TreePath validation."""

    def __hash__(self) -> Never:
        raise RuntimeError("invalid TreePath component was hashed")

    def __eq__(self, _other: object) -> Never:
        raise RuntimeError("invalid TreePath component was compared")


class _RaisingMapping(_OneShotMapping):
    """Fail while the sole item stream is being advanced."""

    def __iter__(self) -> Iterator[object]:
        self.traversals += 1
        raise RuntimeError("mapping item traversal failed")


class _CountingLazyEntry(solver_api_module._LazyEntry):
    """Record any payload materialization past structural preflight."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        self.materialization_count += 1
        return self.value


class _DeletingLazyEntry(solver_api_module._LazyEntry):
    """Delete retained arrays when materialized to exercise cross-entry isolation."""

    def __init__(self, *, targets: tuple[jax.Array, ...], value: jax.Array) -> None:
        self.targets = targets
        self.value = value
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:
        del template
        self.materialization_count += 1
        for target in self.targets:
            target.delete()
        return self.value


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _CountingTree:
    """Frozen test PyTree whose flatten callback records every invocation."""

    value: object
    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Expose one numerical leaf and count this callback."""
        type(self).flatten_count += 1
        return (self.value,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _CountingTree:
        """Rebuild the test tree from its sole numerical leaf."""
        cls.unflatten_count += 1
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ZeroLeafCustomTree:
    """Frozen custom zero-node whose unflatten could return shared state."""

    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0
    singleton: ClassVar[_ZeroLeafCustomTree | None] = None

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Expose a custom zero-leaf node and count the sole observation."""
        type(self).flatten_count += 1
        return (), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        _children: tuple[object, ...],
    ) -> _ZeroLeafCustomTree:
        """Expose whether rejection happened before a shared object escaped."""
        cls.unflatten_count += 1
        if cls.singleton is None:
            cls.singleton = cls()
        return cls.singleton


class _EmptyTupleSubclass(NamedTuple):
    """Exact tuple subclass excluded from the safe zero-node allowlist."""


class _TupleSpoofMeta(type):
    """Metaclass whose equality and hash impersonate the built-in tuple class."""

    def __hash__(cls) -> int:
        return hash(tuple)

    def __eq__(cls, other: object) -> bool:
        return other is tuple


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _TupleSpoofZeroLeaf(metaclass=_TupleSpoofMeta):
    """Custom zero-node that weak class-set membership would accept as tuple."""

    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        type(self).flatten_count += 1
        return (), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        _children: tuple[object, ...],
    ) -> _TupleSpoofZeroLeaf:
        cls.unflatten_count += 1
        return cls()


class _IntegerSpoofMeta(type):
    """Metaclass whose equality and hash impersonate the built-in int class."""

    def __hash__(cls) -> int:
        return hash(int)

    def __eq__(cls, other: object) -> bool:
        return other is int


class _IntegerSpoof(metaclass=_IntegerSpoofMeta):
    """Non-scalar instance weak exact-type set membership would accept."""


class _StringSpoofMeta(type):
    """Metaclass whose equality and hash impersonate the built-in str class."""

    def __hash__(cls) -> int:
        return hash(str)

    def __eq__(cls, other: object) -> bool:
        return other is str


def _repr_as_exact_string(_value: object) -> str:
    """Collide with the representation of the exact string ``"x"``."""
    return repr("x")


_StringSpoof = _StringSpoofMeta(
    "str",
    (),
    {"__repr__": _repr_as_exact_string},
)


class _FloatSpoofMeta(type):
    """Metaclass whose equality and hash impersonate the built-in float class."""

    def __hash__(cls) -> int:
        return hash(float)

    def __eq__(cls, other: object) -> bool:
        return other is float


class _FloatSpoof(metaclass=_FloatSpoofMeta):
    """Array-convertible object weak scalar-leaf membership would accept."""

    conversion_count: ClassVar[int] = 0

    # keyword-only-exempt: library-callback=numpy.asarray
    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,  # noqa: FBT001
    ) -> np.ndarray:
        del copy
        type(self).conversion_count += 1
        return np.asarray(1.0, dtype=dtype)


class _GetAttrKeySubclass(jax.tree_util.GetAttrKey):
    """Subclass of a JAX attribute path segment."""


class _SequenceKeySubclass(jax.tree_util.SequenceKey):
    """Subclass of a JAX sequence path segment."""


class _FlattenedIndexKeySubclass(jax.tree_util.FlattenedIndexKey):
    """Subclass of a JAX flattened-index path segment."""


class _DictKeySubclass(jax.tree_util.DictKey):
    """Subclass of a JAX mapping path segment."""


@dataclass(frozen=True)
class _PlanBox:
    """Nested callback result used to witness repeated-container aliasing."""

    value: object


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _AdversarialPlanTree:
    """Two-leaf PyTree whose unflatten behavior selects an attack class."""

    left: object
    right: object
    mode: ClassVar[str] = "normal"
    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0
    singleton: ClassVar[_AdversarialPlanTree | None] = None

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        type(self).flatten_count += 1
        return (self.left, self.right), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _AdversarialPlanTree:
        cls.unflatten_count += 1
        if cls.mode == "singleton":
            if cls.singleton is None:
                cls.singleton = cls(0.0, 1.0)
            return cls.singleton
        if cls.mode == "drop":
            return cls(children[0], 0.0)
        if cls.mode == "duplicate":
            return cls(children[0], children[0])
        if cls.mode == "transform":
            return cls(0.0, children[1])
        if cls.mode == "alias":
            shared = _PlanBox(children[0])
            return cls(shared, shared)
        return cls(children[0], children[1])


@jax.tree_util.register_pytree_node_class
class _NonEmptyTupleSubclass(tuple):
    """A tuple subclass remains unsupported even when it has numerical leaves."""

    __slots__ = ()

    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0

    def __new__(cls, value: object) -> Self:
        return cast("Self", tuple.__new__(cls, (value,)))

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        type(self).flatten_count += 1
        return (self[0],), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _NonEmptyTupleSubclass:
        cls.unflatten_count += 1
        return cls(children[0])


@dataclass(frozen=True)
class _FakeMarkerDonor:
    """Source of genuine dataclass marker objects for a forged class."""

    value: object


@jax.tree_util.register_pytree_node_class
class _FakeMarkerTree:
    """Mutable marker copy with hidden storage outside its declared field set."""

    __dataclass_fields__ = _FakeMarkerDonor.__dataclass_fields__
    __dataclass_params__ = _FakeMarkerDonor.__dataclass_params__
    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0

    def __init__(self, value: object) -> None:
        self.value = value
        self.hidden = []

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        type(self).flatten_count += 1
        return (self.value,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _FakeMarkerTree:
        cls.unflatten_count += 1
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ReentrantTree:
    """Template whose sole flatten attempts nested authority initialization."""

    value: object
    target: ClassVar[ArtifactAuthority | None] = None
    flatten_count: ClassVar[int] = 0
    reentry_error: ClassVar[BaseException | None] = None

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        cls = type(self)
        cls.flatten_count += 1
        if cls.target is not None:
            try:
                cls.target.__post_init__()
            except BaseException as error:  # noqa: BLE001
                cls.reentry_error = error
        return (self.value,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _ReentrantTree:
        return cls(children[0])


def _custom_tree_authority(
    *,
    template: object,
    runtime_type: type[object],
    leaf_count: int,
) -> ArtifactAuthority:
    """Declare one scalar-leaf custom tree for plan-contract regressions."""
    leaf_authorities = tuple(
        LeafAuthority(
            path=(f"flattened:{index}",),
            runtime_type=jax.Array,
            shape=(),
            dtype="float32",
            axis_names=(),
        )
        for index in range(leaf_count)
    )
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id=f"tests.plan.{runtime_type.__name__}"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id=f"tests.{runtime_type.__name__}",
        leaf_descriptors=tuple(leaf.descriptor for leaf in leaf_authorities),
    )
    return ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=runtime_type,
        template=template,
        container_runtime_types={(): runtime_type},
        leaves={leaf.path: leaf for leaf in leaf_authorities},
    )


def _counting_tree_authority(*, value: float = 0.0) -> ArtifactAuthority:
    """Declare one scalar custom PyTree authority for cache-boundary tests."""
    leaf_path = ("flattened:0",)
    leaf = LeafAuthority(
        path=leaf_path,
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.counting-tree"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="tests.CountingTree",
        leaf_descriptors=(leaf.descriptor,),
    )
    return ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=_CountingTree,
        template=_CountingTree(jnp.asarray(value, dtype=jnp.float32)),
        container_runtime_types={(): _CountingTree},
        leaves={leaf_path: leaf},
    )


def _tuple_tree_authority(*, template: tuple[object, ...]) -> ArtifactAuthority:
    """Declare a tuple with one scalar leaf and optional zero-leaf children."""
    leaf_path = ("sequence:0",)
    leaf = LeafAuthority(
        path=leaf_path,
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.tuple-tree"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="builtins.tuple",
        leaf_descriptors=(leaf.descriptor,),
    )
    return ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=tuple,
        template=template,
        container_runtime_types={(): tuple},
        leaves={leaf_path: leaf},
    )


def _array_authority(*, value: float = 0.0) -> ArtifactAuthority:
    """Declare one bare scalar JAX array authority."""
    leaf = LeafAuthority(
        path=(),
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.root-array"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="jax.Array",
        leaf_descriptors=(leaf.descriptor,),
    )
    return ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=jax.Array,
        template=jnp.asarray(value, dtype=jnp.float32),
        leaves={(): leaf},
    )


def _minimal_metadata() -> SolutionMetadata:
    return SolutionMetadata(
        retention=ResultRetention.VALUES,
        n_periods=1,
        regime_names=("working",),
        solver_types={"working": "example.GridSearch"},
        model_instance_id="model-1",
        params_fingerprint="0" * 64,
        value_schemas={
            (0, "working"): ValueArraySchema(
                shape=(1,),
                dtype="float32",
                axis_names=("wealth",),
            )
        },
    )


def _minimal_lazy_result() -> tuple[SolutionResult, _CountingLazyEntry]:
    lazy = _CountingLazyEntry(jnp.asarray([1.0], dtype=jnp.float32))
    result = SolutionResult(
        values=ValueStore({(0, "working"): lazy}),
        metadata=_minimal_metadata(),
    )
    return result, lazy


def _replace_first_value_with_counter(
    solution: SolutionResult,
) -> tuple[SolutionResult, _CountingLazyEntry]:
    values = cast("ValueStore", solution.values)
    entries = dict(values._entries)
    coordinate = cast("tuple[int, str]", next(iter(entries)))
    lazy = _CountingLazyEntry(values[coordinate[0]][coordinate[1]])
    entries[coordinate] = lazy
    return replace(solution, values=ValueStore(entries)), lazy


def test_metadata_snapshot_consumes_one_exact_item_stream() -> None:
    metadata = _minimal_metadata()
    backing = _OneShotMapping({"working": "example.GridSearch"})
    object.__setattr__(metadata, "solver_types", MappingProxyType(backing))

    copied = snapshot_solution_metadata(metadata)

    assert copied.solver_types == {"working": "example.GridSearch"}
    assert backing.traversals == 1


def test_result_envelope_owns_each_mapping_once_without_loading() -> None:
    result, lazy = _minimal_lazy_result()
    metadata_backing = _OneShotMapping({"working": "example.GridSearch"})
    object.__setattr__(
        result.metadata,
        "solver_types",
        MappingProxyType(metadata_backing),
    )
    values = cast("ValueStore", result.values)
    value_backing = _OneShotMapping({(0, "working"): lazy})
    object.__setattr__(values, "_entries", MappingProxyType(value_backing))

    store_backings: list[_OneShotMapping] = []
    for field_name in (
        "retained_continuations",
        "replay_artifacts",
        "auxiliary_artifacts",
        "diagnostics",
    ):
        store = cast("ArtifactStore", getattr(result, field_name))
        backing = _OneShotMapping({})
        object.__setattr__(store, "_entries", MappingProxyType(backing))
        store_backings.append(backing)
    omission_backing = _OneShotMapping({})
    object.__setattr__(result, "omissions", MappingProxyType(omission_backing))

    copied = Model._snapshot_solution_envelope(solution=result)

    assert copied.metadata.solver_types == {"working": "example.GridSearch"}
    assert lazy.materialization_count == 0
    assert metadata_backing.traversals == 1
    assert value_backing.traversals == 1
    assert omission_backing.traversals == 1
    assert all(backing.traversals == 1 for backing in store_backings)


def test_descriptor_and_authority_mappings_are_each_captured_once() -> None:
    key = ArtifactKey(type_id="example.payload")
    ref = ArtifactRef(period=0, regime="working", key=key)
    domain = CategoryDomain(labels=("no", "yes"), codes=(0, 1), ordered=False)
    leaf_path = ("flattened:0",)
    leaf_descriptor = LeafDescriptor(
        path=leaf_path,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    leaf_authority = LeafAuthority(
        path=leaf_path,
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    descriptor_categories = _OneShotMapping({"status": domain})
    descriptor = ArtifactDescriptor(
        key=key,
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="example.CountingTree",
        leaf_descriptors=(leaf_descriptor,),
        state_roles=("status",),
        categorical_domains=cast(
            "Mapping[str, CategoryDomain]",
            MappingProxyType(descriptor_categories),
        ),
    )
    containers = _OneShotMapping({(): _CountingTree})
    leaves = _OneShotMapping({leaf_path: leaf_authority})
    authority_categories = _OneShotMapping({"status": domain})
    authority = ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=_CountingTree,
        template=_CountingTree(jnp.asarray(1.0, dtype=jnp.float32)),
        container_runtime_types=cast(
            "Mapping[tuple[str, ...], type[object]]",
            MappingProxyType(containers),
        ),
        leaves=cast(
            "Mapping[tuple[str, ...], LeafAuthority]",
            MappingProxyType(leaves),
        ),
        state_roles=("status",),
        categorical_domains=cast(
            "Mapping[str, CategoryDomain]",
            MappingProxyType(authority_categories),
        ),
    )
    outer = _OneShotMapping({ref: authority})
    _CountingTree.flatten_count = 0

    supplied = cast(
        "Mapping[ArtifactRef, ArtifactAuthority]",
        MappingProxyType(outer),
    )
    copied = snapshot_artifact_authorities(supplied)

    assert copied[ref].descriptor.categorical_domains == {"status": domain}
    assert outer.traversals == 1
    assert descriptor_categories.traversals == 1
    assert containers.traversals == 1
    assert leaves.traversals == 1
    assert authority_categories.traversals == 1
    assert _CountingTree.flatten_count == 0


@pytest.mark.parametrize("spoof_location", ["mapping_key", "leaf_path"])
def test_authority_tree_paths_are_validated_before_hashing(
    spoof_location: str,
) -> None:
    """Caller objects cannot execute hash/equality before exact path validation."""
    leaf_path = ("flattened:0",)
    leaf_descriptor = LeafDescriptor(
        path=leaf_path,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    leaf_authority = LeafAuthority(
        path=leaf_path,
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    invalid_component = _RaisingPathComponent()
    mapping_path: object = leaf_path
    if spoof_location == "mapping_key":
        mapping_path = (invalid_component,)
    else:
        object.__setattr__(leaf_authority, "path", (invalid_component,))

    leaves = cast(
        "Mapping[tuple[str, ...], LeafAuthority]",
        _SingleItemMapping(key=mapping_path, value=leaf_authority),
    )
    with pytest.raises(TypeError, match="TreePath"):
        ArtifactAuthority(
            descriptor=ArtifactDescriptor(
                key=ArtifactKey(type_id="example.prehash"),
                channel=ArtifactChannel.AUXILIARY,
                persistence=PersistencePolicy.NOT_PERSISTED,
                payload_type_id="example.CountingTree",
                leaf_descriptors=(leaf_descriptor,),
            ),
            payload_runtime_type=_CountingTree,
            template=_CountingTree(jnp.asarray(1.0, dtype=jnp.float32)),
            leaves=leaves,
        )


def test_public_and_engine_authorities_observe_each_template_exactly_once() -> None:
    """Declaration, model binding, and envelope copying share one observation."""
    _CountingTree.flatten_count = 0
    public = _counting_tree_authority()
    assert _CountingTree.flatten_count == 1

    _CountingTree.flatten_count = 0
    engine = model_authority_module._authority_from_template(
        key=ArtifactKey(type_id="tests.engine-counting-tree"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_runtime_type=_CountingTree,
        template=_CountingTree(jnp.asarray(0.0, dtype=jnp.float32)),
        applicable=True,
        required=False,
    )
    assert _CountingTree.flatten_count == 1

    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
    )
    regime = next(iter(model._regimes.values()))
    state_action_space = StateActionSpace(
        states=MappingProxyType({}),
        discrete_actions=MappingProxyType({}),
        continuous_actions=MappingProxyType({}),
        state_and_discrete_action_names=(),
    )
    bound = model_authority_module._bind_model_owned_artifact_facts(
        authority=engine,
        regime=regime,
        state_action_space=state_action_space,
    )
    assert _CountingTree.flatten_count == 1

    public_ref = ArtifactRef(
        period=0,
        regime="working",
        key=public.descriptor.key,
    )
    copied = snapshot_artifact_authorities(MappingProxyType({public_ref: public}))
    assert copied[public_ref].template is not public.template
    assert snapshot_artifact_template_declaration(bound) is not None
    assert _CountingTree.flatten_count == 1


def test_authority_cache_forgery_cannot_replace_reconstruction_semantics() -> None:
    """Coherent exposed-field replacement cannot replace the private binding."""
    _CountingTree.flatten_count = 0
    authority = _counting_tree_authority(value=1.0)
    declaration = snapshot_artifact_template_declaration(authority)
    assert declaration is not None
    forged_template = _CountingTree(jnp.asarray(99.0, dtype=jnp.float32))
    forged_snapshot = solver_api_module._CanonicalArtifactTemplate(
        payload=forged_template,
        tree=declaration.tree,
        leaf_paths=declaration.leaf_paths,
        leaves=(jnp.asarray(99.0, dtype=jnp.float32),),
        construction_plan=declaration.construction_plan,
    )
    object.__setattr__(authority, "template", forged_template)
    object.__setattr__(authority, "_template_snapshot", forged_snapshot)
    _CountingTree.flatten_count = 0

    with pytest.raises(TypeError, match="reconstruction fields differ"):
        snapshot_artifact_template_declaration(authority)

    assert _CountingTree.flatten_count == 0


def test_detached_template_snapshot_mutation_does_not_touch_binding() -> None:
    """The cache accessor never returns the registry's own declaration object."""
    authority = _counting_tree_authority(value=2.0)
    first = snapshot_artifact_template_declaration(authority)
    assert first is not None
    object.__setattr__(first, "leaf_paths", (("forged",),))

    second = snapshot_artifact_template_declaration(authority)

    assert second is not None
    assert second is not first
    assert second.leaf_paths == (("flattened:0",),)


def test_copied_authority_identity_has_no_template_binding() -> None:
    """A generic object copy cannot inherit private reconstruction authority."""
    authority = _counting_tree_authority()
    _CountingTree.flatten_count = 0
    copied = copy.copy(authority)
    assert copied is not authority

    with pytest.raises(TypeError, match="no identity-bound template"):
        snapshot_artifact_template_declaration(copied)

    assert _CountingTree.flatten_count == 0


def test_authority_post_init_reentry_fails_before_mapping_or_template_callback() -> (
    None
):
    """A live identity cannot reseal hostile replacement fields."""
    authority = _counting_tree_authority()
    hostile = _RaisingMapping({(): _CountingTree})
    object.__setattr__(
        authority,
        "container_runtime_types",
        MappingProxyType(hostile),
    )
    _CountingTree.flatten_count = 0

    with pytest.raises(TypeError, match="write-once"):
        authority.__post_init__()

    assert hostile.traversals == 0
    assert _CountingTree.flatten_count == 0


def test_safe_nested_zero_leaf_nodes_roundtrip_without_public_container_entries() -> (
    None
):
    """Exact tuple and None nodes are owned while numerical paths stay unchanged."""
    supplied = (jnp.asarray(3.0, dtype=jnp.float32), (), (None, ()))
    authority = _tuple_tree_authority(template=supplied)

    canonical = solver_api_module._canonicalize_artifact_payload(
        payload=supplied,
        authority=authority,
    )

    assert type(canonical) is tuple
    assert canonical[1:] == ((), (None, ()))
    assert authority.container_runtime_types == {(): tuple}
    assert tuple(authority.leaves) == (("sequence:0",),)


@pytest.mark.parametrize(
    ("zero_node", "path_pattern", "type_name"),
    [
        ([], r"pytree-child:1", "list"),
        ({}, r"pytree-child:1", "dict"),
        (_EmptyTupleSubclass(), r"pytree-child:1", "_EmptyTupleSubclass"),
        (((), []), r"pytree-child:1.*pytree-child:1", "list"),
    ],
)
def test_mutable_zero_leaf_nodes_are_rejected_at_unique_structural_paths(
    *,
    zero_node: object,
    path_pattern: str,
    type_name: str,
) -> None:
    """Every hidden mutable zero-node is named and rejected before unflatten."""
    template = (jnp.asarray(1.0, dtype=jnp.float32), zero_node)

    with pytest.raises(
        TypeError,
        match=rf"{path_pattern}.*{type_name}",
    ):
        _tuple_tree_authority(template=template)


def test_custom_zero_leaf_node_is_rejected_before_shared_unflatten() -> None:
    """Frozen/custom zero-nodes cannot smuggle a shared mutable reconstruction."""
    _ZeroLeafCustomTree.flatten_count = 0
    _ZeroLeafCustomTree.unflatten_count = 0
    _ZeroLeafCustomTree.singleton = _ZeroLeafCustomTree()
    template = (
        jnp.asarray(1.0, dtype=jnp.float32),
        _ZeroLeafCustomTree.singleton,
    )

    with pytest.raises(
        TypeError,
        match=r"pytree-child:1.*_ZeroLeafCustomTree",
    ):
        _tuple_tree_authority(template=template)

    assert _ZeroLeafCustomTree.flatten_count == 1
    assert _ZeroLeafCustomTree.unflatten_count == 0


def test_zero_leaf_allowlist_uses_runtime_type_identity() -> None:
    """A metaclass cannot impersonate the exact safe tuple zero-node."""
    _TupleSpoofZeroLeaf.flatten_count = 0
    _TupleSpoofZeroLeaf.unflatten_count = 0

    with pytest.raises(
        TypeError,
        match=r"pytree-child:1.*_TupleSpoofZeroLeaf",
    ):
        _tuple_tree_authority(
            template=(
                jnp.asarray(1.0, dtype=jnp.float32),
                _TupleSpoofZeroLeaf(),
            )
        )

    assert _TupleSpoofZeroLeaf.flatten_count == 1
    assert _TupleSpoofZeroLeaf.unflatten_count == 0


def test_mapping_key_scalar_grammar_uses_runtime_type_identity() -> None:
    """A class named like ``str`` cannot collide with an exact string path."""
    exact_path = solver_api_module._normalize_jax_tree_path(
        (jax.tree_util.DictKey("x"),)
    )
    spoof = _StringSpoof()

    assert exact_path == ("mapping:str:'x'",)
    assert type(spoof).__name__ == "str"
    assert repr(spoof) == repr("x")
    with pytest.raises(TypeError, match="exact JSON scalars"):
        solver_api_module._normalize_jax_tree_path((jax.tree_util.DictKey(spoof),))


def test_persistence_json_scalar_grammar_uses_runtime_type_identity() -> None:
    """A spoofing metaclass cannot enter the archive's scalar grammar."""
    with pytest.raises(SolutionIntegrityError, match="exact JSON scalar"):
        solution_persistence._require_exact_json_scalar(
            value=_IntegerSpoof(),
            label="spoof",
        )


def test_template_leaf_scalar_grammar_uses_runtime_type_identity() -> None:
    """A metaclass cannot trigger conversion by impersonating exact float."""
    _FloatSpoof.conversion_count = 0

    with pytest.raises(TypeError, match="not numerical"):
        solver_api_module._snapshot_artifact_template_once(
            template=(_FloatSpoof(),),
            payload_runtime_type=tuple,
        )

    assert _FloatSpoof.conversion_count == 0


@pytest.mark.parametrize(
    "component",
    [
        _GetAttrKeySubclass("field"),
        _SequenceKeySubclass(0),
        _FlattenedIndexKeySubclass(0),
        _DictKeySubclass("key"),
    ],
)
def test_jax_tree_path_key_subclasses_are_rejected(*, component: object) -> None:
    """Only exact JAX path-entry classes participate in stable normalization."""
    with pytest.raises(TypeError, match="Unsupported artifact TreePath component"):
        solver_api_module._normalize_jax_tree_path((component,))


@pytest.mark.parametrize("key", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_mapping_tree_path_keys_are_rejected(*, key: float) -> None:
    """Nonfinite mapping keys cannot collapse distinct paths to one spelling."""
    with pytest.raises(ValueError, match="must be finite"):
        solver_api_module._normalize_jax_tree_path((jax.tree_util.DictKey(key),))


def test_whole_payload_zero_leaf_nodes_have_intentional_boundary_semantics() -> None:
    """Empty tuple is not an artifact; None remains the no-template sentinel."""
    for payload in ((), None):
        with_paths, tree = jax.tree_util.tree_flatten_with_path(payload)
        paths = tuple(
            solver_api_module._normalize_jax_tree_path(path)
            for path, _leaf in with_paths
        )
        assert (
            solver_api_module._container_types_from_tree(
                tree=tree,
                leaf_paths=paths,
            )
            == {}
        )

    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.empty-tuple"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="builtins.tuple",
    )
    with pytest.raises(TypeError, match=r"non-array artifact payload.*root container"):
        ArtifactAuthority(
            descriptor=descriptor,
            payload_runtime_type=tuple,
            template=(),
        )


def test_root_array_authority_owns_public_private_and_fresh_buffers() -> None:
    """Deleting one exposed root array cannot poison the private declaration."""
    authority = _array_authority(value=4.0)
    public = cast("jax.Array", authority.template)
    first = snapshot_artifact_template_declaration(authority)
    assert first is not None
    assert first.payload is not public
    first_array = cast("jax.Array", first.payload)
    first_array.delete()

    second = snapshot_artifact_template_declaration(authority)

    assert second is not None
    second_array = cast("jax.Array", second.payload)
    assert second_array is not public
    assert second_array is not first_array
    assert float(second_array) == pytest.approx(4.0)

    public.delete()
    with pytest.raises(TypeError, match=r"deleted|differs from its binding"):
        snapshot_artifact_template_declaration(authority)


def test_owned_artifact_store_returns_fresh_callback_free_graphs() -> None:
    """Direct public reads never expose retained containers or array buffers."""
    _CountingTree.flatten_count = 0
    _CountingTree.unflatten_count = 0
    authority = _counting_tree_authority(value=5.0)
    ref = ArtifactRef(period=0, regime="working", key=authority.descriptor.key)
    store = snapshot_artifact_store(
        store=ArtifactStore({ref: _CountingTree(jnp.asarray(5.0, dtype=jnp.float32))}),
        authorities={ref: authority},
    )
    declaration_counts = (
        _CountingTree.flatten_count,
        _CountingTree.unflatten_count,
    )

    first = cast("_CountingTree", store[ref])
    second = cast("_CountingTree", store[ref])

    assert first is not second
    assert first.value is not second.value
    first_leaf = cast("jax.Array", first.value)
    first_leaf.delete()
    object.__setattr__(first, "value", jnp.asarray(99.0, dtype=jnp.float32))
    third = cast("_CountingTree", store[ref])
    assert float(cast("jax.Array", third.value)) == pytest.approx(5.0)
    assert (
        _CountingTree.flatten_count,
        _CountingTree.unflatten_count,
    ) == declaration_counts


def test_nested_public_template_mutation_is_detected_without_callbacks() -> None:
    """Direct field replacement cannot alter the registry's reconstruction plan."""
    _CountingTree.flatten_count = 0
    _CountingTree.unflatten_count = 0
    authority = _counting_tree_authority(value=6.0)
    declaration_counts = (
        _CountingTree.flatten_count,
        _CountingTree.unflatten_count,
    )
    public = cast("_CountingTree", authority.template)
    object.__setattr__(public, "value", jnp.asarray(7.0, dtype=jnp.float32))

    with pytest.raises(TypeError, match="numerical fields differ"):
        snapshot_artifact_template_declaration(authority)

    assert (
        _CountingTree.flatten_count,
        _CountingTree.unflatten_count,
    ) == declaration_counts


@pytest.mark.parametrize(
    "mode",
    ["singleton", "drop", "duplicate", "transform", "alias"],
)
def test_unflatten_plan_rejects_every_leaf_fidelity_attack(*, mode: str) -> None:
    """One token observation rejects shared, dropped, duplicated, or aliased state."""
    _AdversarialPlanTree.mode = mode
    _AdversarialPlanTree.flatten_count = 0
    _AdversarialPlanTree.unflatten_count = 0
    _AdversarialPlanTree.singleton = None
    template = _AdversarialPlanTree(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
    )

    with pytest.raises(TypeError, match=r"leaf token|alias"):
        _custom_tree_authority(
            template=template,
            runtime_type=_AdversarialPlanTree,
            leaf_count=2,
        )

    assert _AdversarialPlanTree.flatten_count == 1
    assert _AdversarialPlanTree.unflatten_count == 1


def test_nonempty_tuple_subclass_is_rejected_before_unflatten() -> None:
    """Tuple safety is exact-class, including for nodes with numerical children."""
    _NonEmptyTupleSubclass.flatten_count = 0
    _NonEmptyTupleSubclass.unflatten_count = 0

    with pytest.raises(TypeError, match=r"exact tuple|closed dataclass"):
        _custom_tree_authority(
            template=_NonEmptyTupleSubclass(jnp.asarray(1.0, dtype=jnp.float32)),
            runtime_type=_NonEmptyTupleSubclass,
            leaf_count=1,
        )

    assert _NonEmptyTupleSubclass.flatten_count == 1
    assert _NonEmptyTupleSubclass.unflatten_count == 0


def test_forged_dataclass_markers_cannot_hide_mutable_instance_state() -> None:
    """Copied dataclass markers do not authorize undeclared callback storage."""
    _FakeMarkerTree.flatten_count = 0
    _FakeMarkerTree.unflatten_count = 0

    with pytest.raises(TypeError, match="hidden instance dictionary state"):
        _custom_tree_authority(
            template=_FakeMarkerTree(jnp.asarray(1.0, dtype=jnp.float32)),
            runtime_type=_FakeMarkerTree,
            leaf_count=1,
        )

    assert _FakeMarkerTree.flatten_count == 1
    assert _FakeMarkerTree.unflatten_count == 1


def test_authority_initialization_reentry_is_atomic() -> None:
    """A flatten callback cannot bind the same preallocated authority recursively."""
    leaf_path = ("flattened:0",)
    leaf = LeafAuthority(
        path=leaf_path,
        runtime_type=jax.Array,
        shape=(),
        dtype="float32",
        axis_names=(),
    )
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.reentrant-tree"),
        channel=ArtifactChannel.AUXILIARY,
        persistence=PersistencePolicy.NOT_PERSISTED,
        payload_type_id="tests.ReentrantTree",
        leaf_descriptors=(leaf.descriptor,),
    )
    authority = object.__new__(ArtifactAuthority)
    object.__setattr__(authority, "descriptor", descriptor)
    object.__setattr__(authority, "payload_runtime_type", _ReentrantTree)
    object.__setattr__(
        authority,
        "template",
        _ReentrantTree(jnp.asarray(1.0, dtype=jnp.float32)),
    )
    object.__setattr__(authority, "container_runtime_types", {(): _ReentrantTree})
    object.__setattr__(authority, "leaves", {leaf_path: leaf})
    object.__setattr__(authority, "axes", ())
    object.__setattr__(authority, "state_roles", ())
    object.__setattr__(authority, "action_roles", ())
    object.__setattr__(authority, "categorical_domains", {})
    object.__setattr__(authority, "consumer_route", None)
    object.__setattr__(authority, "applicable", True)
    object.__setattr__(authority, "required", False)
    _ReentrantTree.target = authority
    _ReentrantTree.flatten_count = 0
    _ReentrantTree.reentry_error = None
    try:
        authority.__post_init__()
    finally:
        _ReentrantTree.target = None

    assert isinstance(_ReentrantTree.reentry_error, TypeError)
    assert "re-entrant" in str(_ReentrantTree.reentry_error)
    assert _ReentrantTree.flatten_count == 1
    assert snapshot_artifact_template_declaration(authority) is not None


def test_value_store_returns_fresh_owned_arrays() -> None:
    """Public value reads cannot delete the store's retained numerical buffer."""
    source = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    store = ValueStore({(0, "working"): source})
    source.delete()

    first = store[0]["working"]
    first.delete()
    second = store[0]["working"]

    assert second is not source
    assert second is not first
    np.testing.assert_array_equal(np.asarray(second), np.asarray([1.0, 2.0]))


def test_value_materialization_isolates_cross_entry_lazy_deletion() -> None:
    """Each lazy result is copied before the next callback can delete its source."""
    first_source = jnp.asarray([1.0], dtype=jnp.float32)
    eager_source = jnp.asarray([3.0], dtype=jnp.float32)
    first_lazy = _CountingLazyEntry(first_source)
    deleting_lazy = _DeletingLazyEntry(
        targets=(first_source, eager_source),
        value=jnp.asarray([2.0], dtype=jnp.float32),
    )
    store = ValueStore(
        {
            (0, "first"): first_lazy,
            (0, "deleting"): deleting_lazy,
            (0, "eager"): eager_source,
        }
    )

    materialized = store.materialize()

    np.testing.assert_array_equal(np.asarray(materialized[0]["first"]), [1.0])
    np.testing.assert_array_equal(np.asarray(materialized[0]["deleting"]), [2.0])
    np.testing.assert_array_equal(np.asarray(materialized[0]["eager"]), [3.0])
    assert first_lazy.materialization_count == 1
    assert deleting_lazy.materialization_count == 1


def test_simulate_normalizes_mapping_failure_before_lazy_load() -> None:
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
    )
    params = get_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    malformed, lazy = _replace_first_value_with_counter(solution)
    metadata = replace(solution.metadata)
    backing = _RaisingMapping({})
    object.__setattr__(metadata, "solver_types", MappingProxyType(backing))
    object.__setattr__(malformed, "metadata", metadata)

    with pytest.raises(InvalidSimulationInputError, match="cannot be snapshotted"):
        model.simulate(
            params=params,
            initial_conditions={
                "wealth": jnp.asarray([2.0]),
                "age": jnp.asarray([18.0]),
                "regime_id": jnp.asarray(
                    [RegimeId.working_life],
                    dtype=jnp.int32,
                ),
            },
            solution=malformed,
            log_level="off",
        )

    assert backing.traversals == 1
    assert lazy.materialization_count == 0


def test_save_normalizes_mapping_failure_before_lazy_load(tmp_path: Path) -> None:
    result, lazy = _minimal_lazy_result()
    backing = _RaisingMapping({})
    object.__setattr__(
        result.metadata,
        "solver_types",
        MappingProxyType(backing),
    )
    destination = tmp_path / "solution.h5"

    with pytest.raises(IncompatibleSolutionError, match="cannot be copied"):
        save_solution(solution=result, path=destination)

    assert backing.traversals == 1
    assert lazy.materialization_count == 0
    assert not destination.exists()
