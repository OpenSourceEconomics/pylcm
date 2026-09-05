"""Persistence and independent lazy loading of complete solution results."""

import dataclasses
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, cast

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.solver_api as solver_api_module
from _lcm.persistence import solution as solution_persistence
from _lcm.solution.result_snapshot import (
    snapshot_artifact_template_declaration,
)
from lcm.exceptions import IncompatibleSolutionError, SolutionIntegrityError
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import (
    EGM_CONTINUATION,
    SOLUTION_SCHEMA_VERSION,
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
    LoadState,
    OmissionReason,
    PersistencePolicy,
    ReplayRouteIdentity,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    ValueStore,
)

_REGIME = "working"
_REPLAY_KEY = ArtifactKey(type_id="example.static_policy", schema_version=1)
_VALUE_COORDINATES = ((0, _REGIME), (1, _REGIME), (2, _REGIME))
_REPLAY_REFS = (
    ArtifactRef(period=0, regime=_REGIME, key=_REPLAY_KEY),
    ArtifactRef(period=1, regime=_REGIME, key=_REPLAY_KEY),
)
_REPLAY_ROUTE = ReplayRouteIdentity(route_id="example.static", route_version=3)
_EMPLOYMENT_DOMAIN = CategoryDomain(
    labels=("employed", "unemployed"),
    codes=(1, 0),
    ordered=False,
)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _StatefulPersistenceTree:
    """Custom PyTree that exposes every flatten and can fail either callback."""

    value: object
    source: str
    flatten_sources: ClassVar[list[str]] = []
    emitted_leaves: ClassVar[list[jax.Array]] = []
    unflatten_count: ClassVar[int] = 0
    fail_flatten_source: ClassVar[str | None] = None
    fail_unflatten: ClassVar[bool] = False

    @classmethod
    def reset(cls) -> None:
        """Reset callback observations and failures."""
        cls.flatten_sources.clear()
        cls.emitted_leaves.clear()
        cls.unflatten_count = 0
        cls.fail_flatten_source = None
        cls.fail_unflatten = False

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Emit a call-dependent leaf so repeated observations are visible."""
        cls = type(self)
        cls.flatten_sources.append(self.source)
        if cls.fail_flatten_source == self.source:
            raise RuntimeError("flatten callback failed")
        leaf = jnp.asarray(self.value) + jnp.asarray(
            len(cls.flatten_sources),
            dtype=jnp.float32,
        )
        cls.emitted_leaves.append(leaf)
        return (leaf,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _StatefulPersistenceTree:
        """Rebuild the tree while making later flatten observations identifiable."""
        cls.unflatten_count += 1
        if cls.fail_unflatten:
            raise RuntimeError("unflatten callback failed")
        return cls(children[0], source="canonical-template")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _SharedZeroLeafTemplate:
    """Custom zero-node whose callback would expose one shared object."""

    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0
    singleton: ClassVar[_SharedZeroLeafTemplate | None] = None

    @classmethod
    def reset(cls) -> None:
        """Reset callback observations and shared state."""
        cls.flatten_count = 0
        cls.unflatten_count = 0
        cls.singleton = cls()

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        """Expose no numerical children."""
        type(self).flatten_count += 1
        return (), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        _children: tuple[object, ...],
    ) -> _SharedZeroLeafTemplate:
        """Return shared state if an unsafe materializer reaches this callback."""
        cls.unflatten_count += 1
        if cls.singleton is None:
            cls.singleton = cls()
        return cls.singleton


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _RawPersistenceTree:
    """Compatible raw template whose plan has distinguishable static state."""

    value: object
    marker: str
    flatten_count: ClassVar[int] = 0
    unflatten_count: ClassVar[int] = 0

    def tree_flatten(self) -> tuple[tuple[object, ...], None]:
        type(self).flatten_count += 1
        return (self.value,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    @classmethod
    def tree_unflatten(
        cls,
        _metadata: None,
        children: tuple[object, ...],
    ) -> _RawPersistenceTree:
        cls.unflatten_count += 1
        return cls(children[0], marker="raw-plan")


class _ArtifactMutatingLazyEntry(solver_api_module._LazyEntry):
    """Mutate a retained eager artifact while one value is loading."""

    def __init__(self, *, target: _StatefulPersistenceTree, value: object) -> None:
        self.target = target
        self.value = value
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:
        del template
        self.materialization_count += 1
        target_leaf = cast("jax.Array", self.target.value)
        target_leaf.delete()
        object.__setattr__(
            self.target,
            "value",
            jnp.asarray([99.0, 99.0], dtype=jnp.float32),
        )
        return self.value


class _ObjectDtypeLazyEntry(solver_api_module._LazyEntry):
    """Return an unsupported value only after save preflight starts."""

    @property
    def load_state(self) -> LoadState:
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:
        del template
        return np.asarray([object()], dtype=object)


class _SaveEnvelopeMutatingLazyEntry(solver_api_module._LazyEntry):
    """Mutate caller envelope fields and nested wrappers during one lazy read."""

    def __init__(self, *, value: object) -> None:
        self._value = value
        self.solution: SolutionResult | None = None
        self.materialization_count = 0

    @property
    def load_state(self) -> LoadState:
        """Report an unloaded adversarial value."""
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:  # noqa: ARG002
        """Replace caller fields after the save envelope should be owned."""
        if self.solution is None:
            raise AssertionError("The adversarial entry has no owning solution.")
        self.materialization_count += 1
        solution = self.solution
        schema = solution.metadata.value_schemas[_VALUE_COORDINATES[0]]
        descriptor = solution.metadata.artifact_descriptors[_REPLAY_REFS[0]]
        authority = solution._artifact_authority[_REPLAY_REFS[0]]
        object.__setattr__(schema, "shape", (1,))
        object.__setattr__(descriptor.leaf_descriptors[0], "shape", (1,))
        object.__setattr__(authority.leaves[()], "shape", (1,))
        object.__setattr__(
            solution.metadata,
            "retention",
            ResultRetention.VALUES,
        )
        value_store = cast("ValueStore", solution.values)
        object.__setattr__(value_store, "_entries", MappingProxyType({}))
        object.__setattr__(value_store, "_regimes_by_period", MappingProxyType({}))
        object.__setattr__(solution, "replay_artifacts", ArtifactStore())
        object.__setattr__(
            solution,
            "omissions",
            MappingProxyType(dict.fromkeys(_REPLAY_REFS, OmissionReason.NOT_REQUESTED)),
        )
        object.__setattr__(solution, "_artifact_authority", MappingProxyType({}))
        return self._value


def _read_manifest(archive: h5py.File) -> dict[str, object]:
    """Read the test archive's JSON manifest."""
    return cast("dict[str, object]", json.loads(bytes(archive["manifest"][()])))


def _replace_manifest(*, archive: h5py.File, manifest: dict[str, object]) -> None:
    """Replace a test manifest together with its outer checksum."""
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    del archive["manifest"]
    dataset = archive.create_dataset(
        "manifest",
        data=np.frombuffer(manifest_bytes, dtype=np.uint8),
    )
    dataset.attrs["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()


def _replace_manifest_bytes(*, archive: h5py.File, manifest_bytes: bytes) -> None:
    """Replace a test manifest with already encoded bytes and a valid checksum."""
    del archive["manifest"]
    dataset = archive.create_dataset(
        "manifest",
        data=np.frombuffer(manifest_bytes, dtype=np.uint8),
    )
    dataset.attrs["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()


def _make_solution() -> SolutionResult:
    """Build a small result with independently persistable array entries."""
    values = {
        period: {regime: jnp.asarray([period + 1.0, period + 2.0], dtype=jnp.float32)}
        for period, regime in _VALUE_COORDINATES
    }
    replay_entries = {
        ref: jnp.asarray([ref.period, ref.period + 0.5], dtype=jnp.float32)
        for ref in _REPLAY_REFS
    }
    replay_axis = AxisAuthority(
        name="candidate",
        length=2,
        role=AxisRole.CANDIDATE,
        coordinates=(0, 1),
    )
    employment_axis = AxisAuthority(
        name="employment",
        length=2,
        role=AxisRole.STATE,
        coordinates=_EMPLOYMENT_DOMAIN.labels,
    )
    choice_axis = AxisAuthority(
        name="choice",
        length=2,
        role=AxisRole.ACTION,
        coordinates=(False, True),
    )
    replay_leaf = LeafAuthority(
        path=(),
        runtime_type=jax.Array,
        shape=(2,),
        dtype="float32",
        axis_names=("candidate",),
    )
    replay_descriptor = ArtifactDescriptor(
        key=_REPLAY_KEY,
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="jax.Array",
        payload_version=2,
        leaf_descriptors=(
            LeafDescriptor(
                path=(),
                shape=(2,),
                dtype="float32",
                axis_names=("candidate",),
            ),
        ),
        named_axes=(
            AxisDescriptor(
                name="candidate",
                length=2,
                role=AxisRole.CANDIDATE,
                coordinates=(0, 1),
            ),
            employment_axis.descriptor,
            choice_axis.descriptor,
        ),
        state_roles=("employment",),
        action_roles=("choice",),
        categorical_domains={"employment": _EMPLOYMENT_DOMAIN},
        required_for=frozenset({_REPLAY_ROUTE}),
        required=True,
    )
    metadata = SolutionMetadata(
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        n_periods=3,
        regime_names=(_REGIME,),
        solver_types={_REGIME: "example.StaticArraySolver"},
        model_instance_id="test-model-instance",
        params_fingerprint="0" * 64,
        value_schemas={
            coordinate: ValueArraySchema(
                shape=(2,),
                dtype="float32",
                axis_names=("wealth",),
            )
            for coordinate in _VALUE_COORDINATES
        },
        artifact_descriptors=dict.fromkeys(_REPLAY_REFS, replay_descriptor),
        replay_routes={_REGIME: _REPLAY_ROUTE},
    )
    result = SolutionResult(
        values=values,
        metadata=metadata,
        replay_artifacts=ArtifactStore(replay_entries),
    )
    object.__setattr__(
        result,
        "_artifact_authority",
        MappingProxyType(
            {
                ref: ArtifactAuthority(
                    descriptor=result.metadata.artifact_descriptors[ref],
                    payload_runtime_type=jax.Array,
                    template=jnp.zeros((2,), dtype=jnp.float32),
                    leaves={(): replay_leaf},
                    axes=(replay_axis, employment_axis, choice_axis),
                    state_roles=("employment",),
                    action_roles=("choice",),
                    categorical_domains={"employment": _EMPLOYMENT_DOMAIN},
                    consumer_route=_REPLAY_ROUTE,
                    required=True,
                )
                for ref in _REPLAY_REFS
            }
        ),
    )
    return result


def _make_stateful_pytree_solution() -> tuple[SolutionResult, ArtifactRef]:
    """Replace one persistable array with a stateful custom PyTree."""
    _StatefulPersistenceTree.reset()
    source = _make_solution()
    ref = _REPLAY_REFS[0]
    source_descriptor = source.metadata.artifact_descriptors[ref]
    source_authority = source._artifact_authority[ref]
    leaf_path = ("flattened:0",)
    leaf_descriptor = dataclasses.replace(
        source_descriptor.leaf_descriptors[0],
        path=leaf_path,
    )
    descriptor = dataclasses.replace(
        source_descriptor,
        payload_type_id="tests.StatefulPersistenceTree",
        leaf_descriptors=(leaf_descriptor,),
    )
    descriptors = dict(source.metadata.artifact_descriptors)
    descriptors[ref] = descriptor
    metadata = dataclasses.replace(
        source.metadata,
        artifact_descriptors=descriptors,
    )

    replay_entries = dict(source.replay_artifacts)
    replay_entries[ref] = _StatefulPersistenceTree(
        value=jnp.asarray([10.0, 20.0], dtype=jnp.float32),
        source="supplied-payload",
    )
    result = dataclasses.replace(
        source,
        metadata=metadata,
        replay_artifacts=ArtifactStore(replay_entries),
    )
    source_leaf = next(iter(source_authority.leaves.values()))
    leaf_authority = dataclasses.replace(source_leaf, path=leaf_path)
    custom_authority = ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=_StatefulPersistenceTree,
        template=_StatefulPersistenceTree(
            value=jnp.zeros((2,), dtype=jnp.float32),
            source="declared-template",
        ),
        container_runtime_types={(): _StatefulPersistenceTree},
        leaves={leaf_path: leaf_authority},
        axes=source_authority.axes,
        state_roles=source_authority.state_roles,
        action_roles=source_authority.action_roles,
        categorical_domains=source_authority.categorical_domains,
        consumer_route=source_authority.consumer_route,
        applicable=source_authority.applicable,
        required=source_authority.required,
    )
    authorities = dict(source._artifact_authority)
    authorities[ref] = custom_authority
    object.__setattr__(
        result,
        "_artifact_authority",
        MappingProxyType(authorities),
    )

    # Exclude the sole declaration-time flatten from save-time observations.
    _StatefulPersistenceTree.reset()
    return result, ref


def _make_optional_omission_solution(
    *,
    reason: OmissionReason,
    applicable: bool = True,
) -> SolutionResult:
    """Build one selected optional omission with explicit model applicability."""
    source = _make_solution()
    optional_ref = _REPLAY_REFS[1]
    optional_descriptor = dataclasses.replace(
        source.metadata.artifact_descriptors[optional_ref],
        required_for=frozenset(),
        required=False,
    )
    descriptors = dict(source.metadata.artifact_descriptors)
    descriptors[optional_ref] = optional_descriptor
    metadata = dataclasses.replace(
        source.metadata,
        artifact_descriptors=descriptors,
    )
    replay_entries = dict(source.replay_artifacts)
    del replay_entries[optional_ref]
    result = dataclasses.replace(
        source,
        metadata=metadata,
        replay_artifacts=ArtifactStore(replay_entries),
        omissions={optional_ref: reason},
    )
    authorities = dict(source._artifact_authority)
    authorities[optional_ref] = dataclasses.replace(
        authorities[optional_ref],
        descriptor=optional_descriptor,
        applicable=applicable,
        required=False,
    )
    object.__setattr__(
        result,
        "_artifact_authority",
        MappingProxyType(authorities),
    )
    return result


def _make_values_only_solution(*, value: object, dtype: str) -> SolutionResult:
    """Build one authority-free result for value transport edge cases."""
    array = jnp.asarray(value, dtype=dtype)
    return SolutionResult(
        values={0: {_REGIME: array}},
        metadata=SolutionMetadata(
            retention=ResultRetention.VALUES,
            n_periods=1,
            regime_names=(_REGIME,),
            solver_types={_REGIME: "example.ValueOnlySolver"},
            model_instance_id="value-only-model",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, _REGIME): ValueArraySchema(
                    shape=tuple(array.shape),
                    dtype=np.dtype(array.dtype).name,
                    axis_names=tuple(f"axis_{index}" for index in range(array.ndim)),
                )
            },
        ),
    )


def _assert_all_unloaded(solution: SolutionResult) -> None:
    """Assert that no independently addressed payload has materialized."""
    assert isinstance(solution.values, ValueStore)
    for period, regime in _VALUE_COORDINATES:
        assert (
            solution.values.load_state(period=period, regime=regime)
            is LoadState.UNLOADED
        )
    for ref in _REPLAY_REFS:
        assert solution.replay_artifacts.load_state(ref) is LoadState.UNLOADED


def test_solution_metadata_rejects_value_schema_at_the_horizon() -> None:
    """Value schemas may address periods only inside ``range(n_periods)``."""
    source = _make_solution().metadata
    schema = source.value_schemas[_VALUE_COORDINATES[0]]

    with pytest.raises(ValueError, match="invalid coordinate"):
        dataclasses.replace(
            source,
            value_schemas={(source.n_periods, _REGIME): schema},
        )


def test_complete_solution_roundtrip_materializes_only_the_requested_entries(
    tmp_path: Path,
) -> None:
    """Loading one value or replay entry leaves every sibling entry unloaded."""
    solution = _make_solution()
    assert isinstance(solution.values, ValueStore)
    for period, regime in _VALUE_COORDINATES:
        assert (
            solution.values.load_state(period=period, regime=regime) is LoadState.LOADED
        )
    for ref in _REPLAY_REFS:
        assert solution.replay_artifacts.load_state(ref) is LoadState.LOADED

    path = tmp_path / "solution.lcm"
    assert save_solution(solution=solution, path=path) == path

    restored = load_solution(path=path)

    assert isinstance(restored, SolutionResult)
    assert isinstance(restored.values, ValueStore)
    assert restored.metadata.n_periods == 3
    assert restored.metadata.artifact_descriptors[_REPLAY_REFS[0]].persistence is (
        PersistencePolicy.MODEL_VERIFIABLE
    )
    assert (
        restored.metadata.artifact_descriptors == solution.metadata.artifact_descriptors
    )
    assert not restored.omissions
    _assert_all_unloaded(restored)

    np.testing.assert_array_equal(
        restored.value(period=0, regime=_REGIME),
        np.asarray([1.0, 2.0], dtype=np.float32),
    )

    assert restored.values.load_state(period=0, regime=_REGIME) is LoadState.LOADED
    for period in (1, 2):
        assert (
            restored.values.load_state(period=period, regime=_REGIME)
            is LoadState.UNLOADED
        )
    for ref in _REPLAY_REFS:
        assert restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED

    np.testing.assert_array_equal(
        restored.replay_artifacts[_REPLAY_REFS[0]],
        np.asarray([0.0, 0.5], dtype=np.float32),
    )

    assert restored.replay_artifacts.load_state(_REPLAY_REFS[0]) is LoadState.LOADED
    assert restored.replay_artifacts.load_state(_REPLAY_REFS[1]) is LoadState.UNLOADED
    for period in (1, 2):
        assert (
            restored.values.load_state(period=period, regime=_REGIME)
            is LoadState.UNLOADED
        )


def test_save_extracts_custom_pytree_from_sealed_plan_without_callbacks(
    tmp_path: Path,
) -> None:
    """Write represented fields directly and canonicalize callback-injected state."""
    solution, ref = _make_stateful_pytree_solution()
    path = tmp_path / "stateful-pytree.lcm"
    try:
        save_solution(solution=solution, path=path)

        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.emitted_leaves == []
        assert _StatefulPersistenceTree.unflatten_count == 0

        with h5py.File(path, "r") as archive:
            manifest = _read_manifest(archive)
            entries = cast("list[dict[str, object]]", manifest["artifacts"])
            entry = next(
                candidate
                for candidate in entries
                if candidate["period"] == ref.period
                and candidate["regime"] == ref.regime
                and candidate["type_id"] == ref.key.type_id
                and candidate["schema_version"] == ref.key.schema_version
            )
            leaves = cast("list[dict[str, object]]", entry["leaves"])
            written = np.asarray(archive[str(leaves[0]["dataset"])][()])

        np.testing.assert_array_equal(
            written,
            np.asarray([10.0, 20.0], dtype=np.float32),
        )
    finally:
        _StatefulPersistenceTree.reset()


@pytest.mark.parametrize("callback", ["flatten", "unflatten"])
def test_save_uses_sealed_plan_after_callbacks_are_armed(
    *,
    tmp_path: Path,
    callback: str,
) -> None:
    """Later plugin callbacks are outside the save reconstruction path."""
    solution, _ref = _make_stateful_pytree_solution()
    path = tmp_path / "stateful-pytree.lcm"
    if callback == "flatten":
        _StatefulPersistenceTree.fail_flatten_source = "supplied-payload"
    else:
        _StatefulPersistenceTree.fail_unflatten = True

    try:
        save_solution(solution=solution, path=path)

        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.unflatten_count == 0
        assert path.exists()
        assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
    finally:
        _StatefulPersistenceTree.reset()


def test_lazy_custom_pytree_restore_is_fresh_and_callback_free(
    tmp_path: Path,
) -> None:
    """Rebuild from private leaves and the sealed model plan on every read."""
    solution, ref = _make_stateful_pytree_solution()
    path = save_solution(
        solution=solution,
        path=tmp_path / "stateful-pytree.lcm",
    )
    restored = load_solution(path=path)
    authority = solution._artifact_authority[ref]
    _StatefulPersistenceTree.reset()

    try:
        declaration = snapshot_artifact_template_declaration(authority)
        assert declaration is not None
        first = cast(
            "_StatefulPersistenceTree",
            restored.replay_artifacts._materialize_from_template_snapshot(
                ref,
                template_snapshot=declaration,
            ),
        )
        first_leaf = cast("jax.Array", first.value)
        first_leaf.delete()
        object.__setattr__(
            first,
            "value",
            jnp.asarray([99.0, 99.0], dtype=jnp.float32),
        )
        materialized = cast(
            "_StatefulPersistenceTree",
            restored.replay_artifacts._materialize_from_template_snapshot(
                ref,
                template_snapshot=declaration,
            ),
        )
        canonical = solver_api_module._canonicalize_artifact_payload(
            payload=materialized,
            authority=authority,
        )

        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.emitted_leaves == []
        assert _StatefulPersistenceTree.unflatten_count == 0
        assert isinstance(canonical, _StatefulPersistenceTree)
        assert materialized is not first
        assert materialized.value is not first_leaf
        assert materialized.value is not canonical.value
        assert materialized.source == "canonical-template"
        assert canonical.source == "canonical-template"
        np.testing.assert_array_equal(
            canonical.value,
            np.asarray([10.0, 20.0], dtype=np.float32),
        )
    finally:
        _StatefulPersistenceTree.reset()


def test_authority_plan_overrides_a_previously_cached_raw_plan(tmp_path: Path) -> None:
    """A compatibility read cannot choose later model-authoritative semantics."""
    solution, ref = _make_stateful_pytree_solution()
    restored = load_solution(
        path=save_solution(
            solution=solution,
            path=tmp_path / "raw-plan-first.lcm",
        )
    )
    authority = solution._artifact_authority[ref]
    _StatefulPersistenceTree.reset()
    _RawPersistenceTree.flatten_count = 0
    _RawPersistenceTree.unflatten_count = 0

    try:
        raw = restored.replay_artifacts.materialize(
            ref,
            template=_RawPersistenceTree(
                jnp.zeros((2,), dtype=jnp.float32),
                marker="caller-template",
            ),
        )
        assert isinstance(raw, _RawPersistenceTree)
        assert raw.marker == "raw-plan"
        raw_leaf = cast("jax.Array", raw.value)
        raw_leaf.delete()

        declaration = snapshot_artifact_template_declaration(authority)
        assert declaration is not None
        authoritative = restored.replay_artifacts._materialize_from_template_snapshot(
            ref,
            template_snapshot=declaration,
        )

        assert isinstance(authoritative, _StatefulPersistenceTree)
        assert authoritative.source == "canonical-template"
        np.testing.assert_array_equal(authoritative.value, [10.0, 20.0])
        assert authoritative.value is not raw_leaf
        assert _RawPersistenceTree.flatten_count == 1
        assert _RawPersistenceTree.unflatten_count == 1
        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.unflatten_count == 0
    finally:
        _StatefulPersistenceTree.reset()


def test_lazy_root_arrays_return_fresh_buffers_after_delete(tmp_path: Path) -> None:
    """Lazy value and artifact caches never expose their retained array buffers."""
    restored = load_solution(
        path=save_solution(
            solution=_make_solution(),
            path=tmp_path / "lazy-root-arrays.lcm",
        )
    )

    first_value = restored.value(period=0, regime=_REGIME)
    first_artifact = cast("jax.Array", restored.replay_artifacts[_REPLAY_REFS[0]])
    first_value.delete()
    first_artifact.delete()
    second_value = restored.value(period=0, regime=_REGIME)
    second_artifact = cast("jax.Array", restored.replay_artifacts[_REPLAY_REFS[0]])

    np.testing.assert_array_equal(second_value, [1.0, 2.0])
    np.testing.assert_array_equal(second_artifact, [0.0, 0.5])
    assert second_value is not first_value
    assert second_artifact is not first_artifact


def test_safe_zero_leaf_nodes_survive_lazy_persistence_roundtrip(
    tmp_path: Path,
) -> None:
    """Archive reconstruction preserves exact nested tuple and None zero-nodes."""
    source = _make_solution()
    ref = _REPLAY_REFS[0]
    source_descriptor = source.metadata.artifact_descriptors[ref]
    source_authority = source._artifact_authority[ref]
    leaf_path = ("sequence:0",)
    leaf_descriptor = dataclasses.replace(
        source_descriptor.leaf_descriptors[0],
        path=leaf_path,
    )
    descriptor = dataclasses.replace(
        source_descriptor,
        payload_type_id="builtins.tuple",
        leaf_descriptors=(leaf_descriptor,),
    )
    descriptors = dict(source.metadata.artifact_descriptors)
    descriptors[ref] = descriptor
    metadata = dataclasses.replace(
        source.metadata,
        artifact_descriptors=descriptors,
    )

    numerical = jnp.asarray([7.0, 8.0], dtype=jnp.float32)
    payload = (numerical, (), (None, ()))
    replay_entries = dict(source.replay_artifacts)
    replay_entries[ref] = payload
    result = dataclasses.replace(
        source,
        metadata=metadata,
        replay_artifacts=ArtifactStore(replay_entries),
    )
    source_leaf = next(iter(source_authority.leaves.values()))
    leaf_authority = dataclasses.replace(source_leaf, path=leaf_path)
    tuple_authority = ArtifactAuthority(
        descriptor=descriptor,
        payload_runtime_type=tuple,
        template=(jnp.zeros((2,), dtype=jnp.float32), (), (None, ())),
        container_runtime_types={(): tuple},
        leaves={leaf_path: leaf_authority},
        axes=source_authority.axes,
        state_roles=source_authority.state_roles,
        action_roles=source_authority.action_roles,
        categorical_domains=source_authority.categorical_domains,
        consumer_route=source_authority.consumer_route,
        applicable=source_authority.applicable,
        required=source_authority.required,
    )
    authorities = dict(source._artifact_authority)
    authorities[ref] = tuple_authority
    object.__setattr__(
        result,
        "_artifact_authority",
        MappingProxyType(authorities),
    )

    restored = load_solution(
        path=save_solution(
            solution=result,
            path=tmp_path / "zero-leaf-safe-nodes.lcm",
        )
    )
    declaration = snapshot_artifact_template_declaration(tuple_authority)
    assert declaration is not None
    materialized = restored.replay_artifacts._materialize_from_template_snapshot(
        ref,
        template_snapshot=declaration,
    )

    assert type(materialized) is tuple
    np.testing.assert_array_equal(materialized[0], numerical)
    assert materialized[1:] == ((), (None, ()))


@pytest.mark.parametrize("zero_node", [[], {}], ids=["list", "dict"])
def test_public_raw_materialize_rejects_mutable_zero_leaf_templates(
    *, tmp_path: Path, zero_node: object
) -> None:
    """The public compatibility route validates hidden zero-node containers."""
    solution, ref = _make_stateful_pytree_solution()
    restored = load_solution(
        path=save_solution(
            solution=solution,
            path=tmp_path / "raw-mutable-zero.lcm",
        )
    )

    with pytest.raises(TypeError, match=r"pytree-child:1.*(?:list|dict)"):
        restored.replay_artifacts.materialize(
            ref,
            template=(jnp.zeros((2,), dtype=jnp.float32), zero_node),
        )

    assert restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED


def test_public_raw_materialize_rejects_shared_custom_zero_template(
    tmp_path: Path,
) -> None:
    """Raw compatibility materialization rejects before custom unflatten."""
    solution, ref = _make_stateful_pytree_solution()
    restored = load_solution(
        path=save_solution(
            solution=solution,
            path=tmp_path / "raw-shared-zero.lcm",
        )
    )
    _SharedZeroLeafTemplate.reset()
    assert _SharedZeroLeafTemplate.singleton is not None

    try:
        with pytest.raises(
            TypeError,
            match=r"pytree-child:1.*_SharedZeroLeafTemplate",
        ):
            restored.replay_artifacts.materialize(
                ref,
                template=(
                    jnp.zeros((2,), dtype=jnp.float32),
                    _SharedZeroLeafTemplate.singleton,
                ),
            )

        assert _SharedZeroLeafTemplate.flatten_count == 1
        assert _SharedZeroLeafTemplate.unflatten_count == 0
        assert restored.replay_artifacts.load_state(ref) is LoadState.UNLOADED
    finally:
        _SharedZeroLeafTemplate.reset()


def test_lazy_entries_survive_a_cwd_change_after_relative_archive_load(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy payloads reopen the archive independently of the caller's later cwd."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    save_solution(
        solution=_make_solution(),
        path=archive_dir / "solution.lcm",
    )
    monkeypatch.chdir(tmp_path)
    restored = load_solution(path=Path("archives/solution.lcm"))
    _assert_all_unloaded(restored)

    later_cwd = tmp_path / "later"
    later_cwd.mkdir()
    monkeypatch.chdir(later_cwd)

    np.testing.assert_array_equal(
        restored.value(period=0, regime=_REGIME),
        np.asarray([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        restored.replay_artifacts[_REPLAY_REFS[0]],
        np.asarray([0.0, 0.5], dtype=np.float32),
    )


def test_lazy_store_membership_inspects_coordinates_without_loading(
    tmp_path: Path,
) -> None:
    """Membership tests leave independently addressed payloads unloaded."""
    restored = load_solution(
        path=save_solution(
            solution=_make_solution(),
            path=tmp_path / "solution.lcm",
        )
    )

    assert 0 in restored.values
    assert _REGIME in restored.values[0]
    assert _REPLAY_REFS[0] in restored.replay_artifacts
    absent_ref = ArtifactRef(
        period=0,
        regime=_REGIME,
        key=ArtifactKey(type_id="example.absent", schema_version=1),
    )
    assert absent_ref not in restored.replay_artifacts
    _assert_all_unloaded(restored)


def test_checksum_verification_preserves_unloaded_state(tmp_path: Path) -> None:
    """Archive verification hashes payloads without materializing their values."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )

    restored = load_solution(path=path, verify_checksums=True)

    _assert_all_unloaded(restored)


def test_manifest_checksum_mismatch_is_rejected_before_metadata_load(
    tmp_path: Path,
) -> None:
    """Reject a manifest whose bytes do not match its recorded checksum."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        archive["manifest"].attrs["sha256"] = "0" * 64

    with pytest.raises(SolutionIntegrityError, match="manifest checksum"):
        load_solution(path=path)


def test_deep_manifest_json_is_normalized_to_an_integrity_error(
    tmp_path: Path,
) -> None:
    """Normalize parser recursion failure for a checksum-consistent manifest."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    manifest_bytes = b"[" * 100_000 + b"0" + b"]" * 100_000
    with h5py.File(path, "r+") as archive:
        _replace_manifest_bytes(archive=archive, manifest_bytes=manifest_bytes)

    with pytest.raises(SolutionIntegrityError, match=r"valid JSON|JSON object"):
        load_solution(path=path)


def test_load_rejects_noncanonical_single_leaf_pytree_kind(tmp_path: Path) -> None:
    """A one-root-leaf payload must use the canonical array representation."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        value_entries = cast("list[dict[str, object]]", manifest["values"])
        value_entries[0]["payload_kind"] = "pytree"
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="invalid metadata"):
        load_solution(path=path)


def test_payload_checksum_mismatch_is_rejected_on_materialization(
    tmp_path: Path,
) -> None:
    """Keep a restored payload lazy until its checksum is checked on access."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = cast("dict[str, object]", json.loads(bytes(archive["manifest"][()])))
        value_entries = cast("list[dict[str, object]]", manifest["values"])
        leaves = cast("list[dict[str, object]]", value_entries[0]["leaves"])
        dataset = archive[str(leaves[0]["dataset"])]
        dataset[...] = np.asarray(dataset[()]) + 1

    restored = load_solution(path=path)

    assert isinstance(restored.values, ValueStore)
    assert restored.values.load_state(period=0, regime=_REGIME) is LoadState.UNLOADED
    with pytest.raises(SolutionIntegrityError, match="checksum"):
        restored.value(period=0, regime=_REGIME)


@pytest.mark.parametrize(
    "version_field",
    [
        "format_version",
        "pylcm_version",
        "solver_api_version",
        "solution_schema_version",
    ],
)
def test_incompatible_archive_version_is_rejected(
    *, tmp_path: Path, version_field: str
) -> None:
    """Require an exact match for every archive compatibility version."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        current_version = manifest[version_field]
        manifest[version_field] = (
            current_version + 1
            if type(current_version) is int
            else f"{current_version}.incompatible"
        )
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(IncompatibleSolutionError, match=version_field):
        load_solution(path=path)


def test_incompatible_metadata_pylcm_version_is_rejected(tmp_path: Path) -> None:
    """Require the descriptive package identity to match the archive envelope."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        metadata["pylcm_version"] = f"{metadata['pylcm_version']}.incompatible"
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(IncompatibleSolutionError, match="pylcm_version"):
        load_solution(path=path)


def test_load_rejects_value_schema_at_the_horizon(tmp_path: Path) -> None:
    """Reject an archive whose descriptive value address exceeds its horizon."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        value_schemas = cast("list[dict[str, object]]", metadata["value_schemas"])
        value_schemas[0]["period"] = metadata["n_periods"]
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="descriptive metadata"):
        load_solution(path=path)


def test_load_rejects_value_payload_at_the_horizon(tmp_path: Path) -> None:
    """Reject an archive payload address outside ``range(n_periods)``."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        value_entries = cast("list[dict[str, object]]", manifest["values"])
        value_entries[0]["period"] = metadata["n_periods"]
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="invalid period or regime"):
        load_solution(path=path)


def test_failed_save_leaves_existing_archive_unchanged(tmp_path: Path) -> None:
    """Publish neither a partial replacement nor a leftover temporary archive."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    original_bytes = path.read_bytes()
    source = _make_solution()
    value_entries: dict[object, object] = {
        coordinate: source.value(period=coordinate[0], regime=coordinate[1])
        for coordinate in _VALUE_COORDINATES
    }
    value_entries[_VALUE_COORDINATES[0]] = _ObjectDtypeLazyEntry()
    invalid_solution = dataclasses.replace(
        source,
        values=ValueStore(value_entries),
    )
    object.__setattr__(
        invalid_solution,
        "_artifact_authority",
        source._artifact_authority,
    )

    with pytest.raises(TypeError, match="not numerical or Boolean"):
        save_solution(solution=invalid_solution, path=path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_normalizes_mutated_metadata_before_writing(tmp_path: Path) -> None:
    """Translate a beartype metadata failure before creating archive state."""
    source = _make_solution()
    metadata = dataclasses.replace(source.metadata)
    object.__setattr__(metadata, "source", "in_memory")
    malformed = dataclasses.replace(source, metadata=metadata)
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError):
        save_solution(solution=malformed, path=path)

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


class _CompatibilityVersionSubclass(int):
    """Equality-compatible version token with the wrong exact runtime type."""


class _ArmedHashString(str):  # noqa: SLOT000
    """Raise if a snapshot hashes this noncanonical metadata key."""

    armed = False

    def __hash__(self) -> int:
        if self.armed:
            raise RuntimeError("hostile hash escaped")
        return super().__hash__()


class _HostileComparisonInt(int):
    """Raise if a boundary compares this noncanonical integer."""

    def __lt__(self, other: object) -> bool:
        raise RuntimeError("hostile comparison escaped")


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("solver_api_version", True),
        (
            "solution_schema_version",
            _CompatibilityVersionSubclass(SOLUTION_SCHEMA_VERSION),
        ),
    ],
)
def test_save_rejects_weakly_typed_compatibility_versions(
    *, tmp_path: Path, field_name: str, invalid: object
) -> None:
    """Reject equality-compatible versions before decoding or creating state."""
    source = _make_solution()
    metadata = dataclasses.replace(source.metadata)
    current = getattr(metadata, field_name)
    assert invalid == current
    assert type(invalid) is not int
    object.__setattr__(metadata, field_name, invalid)
    value_entries: dict[object, object] = {
        (period, regime): source.value(period=period, regime=regime)
        for period, regime in _VALUE_COORDINATES
    }
    lazy_value = _SaveEnvelopeMutatingLazyEntry(
        value=value_entries[_VALUE_COORDINATES[0]]
    )
    value_entries[_VALUE_COORDINATES[0]] = lazy_value
    malformed = dataclasses.replace(
        source,
        metadata=metadata,
        values=ValueStore(value_entries),
    )
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    lazy_value.solution = malformed
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError):
        save_solution(solution=malformed, path=path)

    assert lazy_value.materialization_count == 0
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_rejects_incompatible_version_before_pytree_callbacks(
    tmp_path: Path,
) -> None:
    """Check compatibility before snapshotting any plugin-owned PyTree."""

    @dataclasses.dataclass(frozen=True)
    class _CountingPyTree:
        value: object

    callback_counts = {"flatten": 0, "unflatten": 0}

    def flatten(payload: _CountingPyTree) -> tuple[tuple[object, ...], None]:
        callback_counts["flatten"] += 1
        return (payload.value,), None

    # keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
    def unflatten(_aux: None, children: tuple[object, ...]) -> _CountingPyTree:
        callback_counts["unflatten"] += 1
        return _CountingPyTree(children[0])

    jax.tree_util.register_pytree_node(_CountingPyTree, flatten, unflatten)

    source = _make_solution()
    metadata = dataclasses.replace(source.metadata)
    object.__setattr__(metadata, "solver_api_version", True)
    ref = next(iter(source._artifact_authority))
    object.__setattr__(
        source._artifact_authority[ref],
        "template",
        _CountingPyTree(jnp.zeros((2,), dtype=jnp.float32)),
    )
    malformed = dataclasses.replace(source, metadata=metadata)
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError, match="solver_api_version"):
        save_solution(solution=malformed, path=path)

    assert callback_counts == {"flatten": 0, "unflatten": 0}
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_rejects_armed_metadata_key_before_hashing(tmp_path: Path) -> None:
    """Reject noncanonical mapping keys before copying or decoding values."""
    source = _make_solution()
    hostile_key = _ArmedHashString(_REGIME)
    solver_types = {hostile_key: source.metadata.solver_types[_REGIME]}
    hostile_key.armed = True
    metadata = dataclasses.replace(source.metadata)
    object.__setattr__(metadata, "solver_types", MappingProxyType(solver_types))
    value_entries: dict[object, object] = {
        coordinate: source.value(period=coordinate[0], regime=coordinate[1])
        for coordinate in _VALUE_COORDINATES
    }
    lazy_value = _SaveEnvelopeMutatingLazyEntry(
        value=value_entries[_VALUE_COORDINATES[0]]
    )
    value_entries[_VALUE_COORDINATES[0]] = lazy_value
    malformed = dataclasses.replace(
        source,
        metadata=metadata,
        values=ValueStore(value_entries),
    )
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    lazy_value.solution = malformed
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError):
        save_solution(solution=malformed, path=path)

    assert lazy_value.materialization_count == 0
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_rejects_nonexact_value_schema_shape_before_comparison(
    tmp_path: Path,
) -> None:
    """Reject a hostile shape integer before its comparison or value decoding."""
    source = _make_solution()
    coordinate = _VALUE_COORDINATES[0]
    schema = dataclasses.replace(source.metadata.value_schemas[coordinate])
    object.__setattr__(schema, "shape", (_HostileComparisonInt(2),))
    schemas = dict(source.metadata.value_schemas)
    schemas[coordinate] = schema
    metadata = dataclasses.replace(source.metadata)
    object.__setattr__(metadata, "value_schemas", MappingProxyType(schemas))
    value_entries: dict[object, object] = {
        item: source.value(period=item[0], regime=item[1])
        for item in _VALUE_COORDINATES
    }
    lazy_value = _SaveEnvelopeMutatingLazyEntry(value=value_entries[coordinate])
    value_entries[coordinate] = lazy_value
    malformed = dataclasses.replace(
        source,
        metadata=metadata,
        values=ValueStore(value_entries),
    )
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    lazy_value.solution = malformed
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError):
        save_solution(solution=malformed, path=path)

    assert lazy_value.materialization_count == 0
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_rejects_nonexact_value_coordinate_before_hashing(
    tmp_path: Path,
) -> None:
    """Reject a hostile flat coordinate before ValueStore reconstruction."""
    source = _make_solution()
    coordinate = _VALUE_COORDINATES[0]
    lazy_value = _SaveEnvelopeMutatingLazyEntry(
        value=source.value(period=coordinate[0], regime=coordinate[1])
    )
    hostile_regime = _ArmedHashString(_REGIME)
    entries: dict[object, object] = {
        (
            (period, hostile_regime)
            if (period, regime) == coordinate
            else (period, regime)
        ): (
            lazy_value
            if (period, regime) == coordinate
            else source.value(period=period, regime=regime)
        )
        for period, regime in _VALUE_COORDINATES
    }
    hostile_regime.armed = True
    values = ValueStore()
    object.__setattr__(values, "_entries", MappingProxyType(entries))
    malformed = dataclasses.replace(source, values=values)
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)
    lazy_value.solution = malformed
    path = tmp_path / "solution.lcm"

    with pytest.raises(IncompatibleSolutionError):
        save_solution(solution=malformed, path=path)

    assert lazy_value.materialization_count == 0
    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_succeeds_without_a_platform_directory_open_flag(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic publication remains portable when ``O_DIRECTORY`` is unavailable."""
    monkeypatch.delattr(solution_persistence.os, "O_DIRECTORY", raising=False)

    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )

    assert path.is_file()


def test_save_preflight_rejects_incomplete_descriptor_coverage_before_writing(
    tmp_path: Path,
) -> None:
    """Require every declared artifact to be either present or explicitly omitted."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    original_bytes = path.read_bytes()
    source = _make_solution()
    incomplete = dataclasses.replace(
        source,
        replay_artifacts=ArtifactStore(
            {_REPLAY_REFS[0]: source.replay_artifacts[_REPLAY_REFS[0]]}
        ),
    )
    object.__setattr__(
        incomplete,
        "_artifact_authority",
        source._artifact_authority,
    )

    with pytest.raises(IncompatibleSolutionError, match="present payload or omission"):
        save_solution(solution=incomplete, path=path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_rejects_value_payload_at_the_horizon(tmp_path: Path) -> None:
    """Check value addresses against the horizon before schema coverage."""
    source = _make_solution()
    values = {
        period: dict(regime_to_value)
        for period, regime_to_value in source.values.items()
    }
    values[source.metadata.n_periods] = {
        _REGIME: jnp.asarray([4.0, 5.0], dtype=jnp.float32)
    }
    malformed = dataclasses.replace(source, values=values)
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)

    with pytest.raises(IncompatibleSolutionError, match="invalid period or regime"):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_present_replay_artifact_under_values_retention(
    tmp_path: Path,
) -> None:
    """A present payload must be selected by the result's retention contract."""
    source = _make_solution()
    malformed = dataclasses.replace(
        source,
        metadata=dataclasses.replace(
            source.metadata,
            retention=ResultRetention.VALUES,
        ),
    )
    object.__setattr__(malformed, "_artifact_authority", source._artifact_authority)

    with pytest.raises(IncompatibleSolutionError, match="does not select"):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_present_inapplicable_custom_artifact(tmp_path: Path) -> None:
    """A present custom payload contradicts model authority if inapplicable."""
    source = _make_solution()
    ref = _REPLAY_REFS[1]
    descriptor = dataclasses.replace(
        source.metadata.artifact_descriptors[ref],
        required_for=frozenset(),
        required=False,
    )
    descriptors = dict(source.metadata.artifact_descriptors)
    descriptors[ref] = descriptor
    malformed = dataclasses.replace(
        source,
        metadata=dataclasses.replace(
            source.metadata,
            artifact_descriptors=descriptors,
        ),
    )
    authorities = dict(source._artifact_authority)
    authorities[ref] = dataclasses.replace(
        authorities[ref],
        descriptor=descriptor,
        applicable=False,
        required=False,
    )
    object.__setattr__(
        malformed,
        "_artifact_authority",
        MappingProxyType(authorities),
    )

    with pytest.raises(IncompatibleSolutionError, match="not applicable"):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")

    assert list(tmp_path.iterdir()) == []


def test_save_owns_the_complete_envelope_before_lazy_materialization(
    tmp_path: Path,
) -> None:
    """A lazy value cannot mix caller mutations into the written ledger."""
    source = _make_solution()
    value_entries: dict[object, object] = {
        (period, regime): source.values[period][regime]
        for period, regime in _VALUE_COORDINATES
    }
    coordinate = _VALUE_COORDINATES[0]
    adversarial = _SaveEnvelopeMutatingLazyEntry(value=value_entries[coordinate])
    value_entries[coordinate] = adversarial
    stateful = dataclasses.replace(source, values=ValueStore(value_entries))
    object.__setattr__(
        stateful,
        "_artifact_authority",
        source._artifact_authority,
    )
    adversarial.solution = stateful
    caller_authority = stateful._artifact_authority[_REPLAY_REFS[0]]

    restored = load_solution(
        path=save_solution(solution=stateful, path=tmp_path / "solution.lcm")
    )

    assert adversarial.materialization_count == 1
    assert stateful.metadata.retention is ResultRetention.VALUES
    assert stateful.metadata.value_schemas[coordinate].shape == (1,)
    assert stateful.metadata.artifact_descriptors[_REPLAY_REFS[0]].leaf_descriptors[
        0
    ].shape == (1,)
    assert caller_authority.leaves[()].shape == (1,)
    assert not stateful.replay_artifacts
    assert not stateful._artifact_authority
    assert restored.metadata.retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
    assert restored.metadata.value_schemas[coordinate].shape == (2,)
    assert restored.metadata.artifact_descriptors[_REPLAY_REFS[0]].leaf_descriptors[
        0
    ].shape == (2,)
    assert set(restored.replay_artifacts) == set(_REPLAY_REFS)
    assert not restored.omissions


def test_save_detaches_eager_artifact_before_lazy_value_mutates_it(
    tmp_path: Path,
) -> None:
    """The archive writes the pre-callback custom artifact snapshot."""
    source, ref = _make_stateful_pytree_solution()
    target = cast(
        "_StatefulPersistenceTree",
        source.replay_artifacts._raw(ref),
    )
    value_entries: dict[object, object] = {
        (period, regime): source.values[period][regime]
        for period, regime in _VALUE_COORDINATES
    }
    coordinate = _VALUE_COORDINATES[0]
    adversarial = _ArtifactMutatingLazyEntry(
        target=target,
        value=value_entries[coordinate],
    )
    value_entries[coordinate] = adversarial
    stateful = dataclasses.replace(source, values=ValueStore(value_entries))
    object.__setattr__(
        stateful,
        "_artifact_authority",
        source._artifact_authority,
    )

    try:
        restored = load_solution(
            path=save_solution(
                solution=stateful,
                path=tmp_path / "mutated-eager-artifact.lcm",
            )
        )
        declaration = snapshot_artifact_template_declaration(
            source._artifact_authority[ref]
        )
        assert declaration is not None
        materialized = cast(
            "_StatefulPersistenceTree",
            restored.replay_artifacts._materialize_from_template_snapshot(
                ref,
                template_snapshot=declaration,
            ),
        )

        assert adversarial.materialization_count == 1
        np.testing.assert_array_equal(target.value, [99.0, 99.0])
        np.testing.assert_array_equal(materialized.value, [10.0, 20.0])
        assert materialized.source == "canonical-template"
        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.unflatten_count == 0
    finally:
        _StatefulPersistenceTree.reset()


def test_result_construction_rejects_noncanonical_omission_reason() -> None:
    """A user string never enters the omissions as if it were an omission enum."""
    source = _make_solution()

    with pytest.raises(TypeError, match="exact OmissionReason"):
        dataclasses.replace(
            source,
            replay_artifacts=ArtifactStore(
                {_REPLAY_REFS[0]: source.replay_artifacts[_REPLAY_REFS[0]]}
            ),
            omissions={_REPLAY_REFS[1]: cast("object", "not_requested")},
        )


def test_save_preflight_rejects_noncanonical_omission_reason(tmp_path: Path) -> None:
    """Never coerce a user string into an omission enum during serialization, even
    when the constructor boundary was bypassed."""
    source = _make_solution()
    malformed = dataclasses.replace(
        source,
        replay_artifacts=ArtifactStore(
            {_REPLAY_REFS[0]: source.replay_artifacts[_REPLAY_REFS[0]]}
        ),
        omissions={_REPLAY_REFS[1]: OmissionReason.NOT_REQUESTED},
    )
    object.__setattr__(
        malformed,
        "omissions",
        MappingProxyType({_REPLAY_REFS[1]: cast("object", "not_requested")}),
    )
    object.__setattr__(
        malformed,
        "_artifact_authority",
        source._artifact_authority,
    )

    with pytest.raises(IncompatibleSolutionError, match="cannot be copied"):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")

    assert list(tmp_path.iterdir()) == []


def test_save_rejects_not_applicable_for_an_applicable_selected_optional_artifact(
    tmp_path: Path,
) -> None:
    """An absent selected optional capability is unsupported, not inapplicable."""
    malformed = _make_optional_omission_solution(reason=OmissionReason.NOT_APPLICABLE)

    with pytest.raises(IncompatibleSolutionError, match="expected 'unsupported'"):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "reason", [OmissionReason.NOT_REQUESTED, OmissionReason.NOT_PERSISTED]
)
def test_save_rejects_undescribed_continuation_omission(
    *, tmp_path: Path, reason: OmissionReason
) -> None:
    """Every continuation in the complete solution schema needs a descriptor."""
    ref = ArtifactRef(period=0, regime=_REGIME, key=EGM_CONTINUATION)
    malformed = dataclasses.replace(
        _make_values_only_solution(value=[1.0], dtype="float32"),
        omissions={ref: reason},
    )

    with pytest.raises(
        IncompatibleSolutionError, match="without an artifact descriptor"
    ):
        save_solution(solution=malformed, path=tmp_path / "solution.lcm")


def test_load_rejects_undescribed_continuation_omission(tmp_path: Path) -> None:
    """An archive cannot smuggle a standard continuation outside its schema."""
    path = save_solution(
        solution=_make_values_only_solution(value=[1.0], dtype="float32"),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        omissions = cast("list[dict[str, object]]", manifest["omissions"])
        omissions.append(
            {
                "period": 0,
                "regime": _REGIME,
                "type_id": EGM_CONTINUATION.type_id,
                "schema_version": EGM_CONTINUATION.schema_version,
                "reason": OmissionReason.NOT_PERSISTED.value,
            }
        )
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="no artifact descriptor"):
        load_solution(path=path)


def test_load_rejects_descriptor_without_payload_or_omission(tmp_path: Path) -> None:
    """Treat artifact descriptor coverage as an exact archive invariant."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        artifact_entries = cast("list[dict[str, object]]", manifest["artifacts"])
        artifact_entries.pop()
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="exactly cover"):
        load_solution(path=path)


def test_load_rejects_present_replay_artifact_under_values_retention(
    tmp_path: Path,
) -> None:
    """The persisted presence ledger must agree with descriptive retention."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        metadata["retention"] = ResultRetention.VALUES.value
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="does not select"):
        load_solution(path=path)


def test_load_rejects_unsupported_omission_for_required_artifact(
    tmp_path: Path,
) -> None:
    """A selected required model-verifiable payload cannot be unsupported."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        artifact_entries = cast("list[dict[str, object]]", manifest["artifacts"])
        removed = artifact_entries.pop()
        cast("list[object]", manifest["omissions"]).append(
            {
                "period": removed["period"],
                "regime": removed["regime"],
                "type_id": removed["type_id"],
                "schema_version": removed["schema_version"],
                "reason": OmissionReason.UNSUPPORTED.value,
            }
        )
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="Required artifact"):
        load_solution(path=path)


def test_inapplicable_optional_model_verifiable_omission_roundtrips(
    tmp_path: Path,
) -> None:
    """Descriptive persistence policy does not imply model applicability."""
    path = save_solution(
        solution=_make_optional_omission_solution(
            reason=OmissionReason.NOT_APPLICABLE,
            applicable=False,
        ),
        path=tmp_path / "solution.lcm",
    )

    restored = load_solution(path=path)

    assert restored.omissions[_REPLAY_REFS[1]] is OmissionReason.NOT_APPLICABLE


def test_load_rejects_artifact_leaf_path_mismatch(tmp_path: Path) -> None:
    """Bind each physical leaf to its declared stable PyTree path."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        descriptors = cast("list[dict[str, object]]", metadata["artifact_descriptors"])
        leaves = cast("list[dict[str, object]]", descriptors[0]["leaf_descriptors"])
        leaves[0]["path"] = ["attribute:wrong"]
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match="leaves do not match"):
        load_solution(path=path)


@pytest.mark.parametrize(
    "field",
    [
        "payload_version",
        "leaf_shape",
        "axis_coordinates",
        "state_roles",
        "action_roles",
        "categorical_labels",
        "categorical_codes",
        "categorical_ordered",
        "required_for",
    ],
)
def test_load_rejects_malformed_rich_artifact_descriptor(
    *, tmp_path: Path, field: str
) -> None:
    """Strictly parse every nested descriptive artifact-schema field."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        metadata = cast("dict[str, object]", manifest["metadata"])
        descriptors = cast("list[dict[str, object]]", metadata["artifact_descriptors"])
        descriptor = descriptors[0]
        if field == "payload_version":
            descriptor["payload_version"] = True
        elif field == "leaf_shape":
            leaves = cast("list[dict[str, object]]", descriptor["leaf_descriptors"])
            leaves[0]["shape"] = [3]
        elif field == "axis_coordinates":
            axes = cast("list[dict[str, object]]", descriptor["named_axes"])
            axes[0]["coordinates"] = [0]
        elif field in {"state_roles", "action_roles"}:
            descriptor[field] = ["candidate"]
        elif field.startswith("categorical_"):
            domains = cast(
                "dict[str, dict[str, object]]", descriptor["categorical_domains"]
            )
            domain = domains["employment"]
            if field == "categorical_labels":
                domain["labels"] = ["same", "same"]
            elif field == "categorical_codes":
                domain["codes"] = [1, True]
            else:
                domain["ordered"] = 0
        else:
            routes = cast("list[dict[str, object]]", descriptor["required_for"])
            routes.append(dict(routes[0]))
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError):
        load_solution(path=path)


@pytest.mark.parametrize("mutation", ["extra_field", "invalid_reason"])
def test_load_rejects_malformed_omission_entry(
    *, tmp_path: Path, mutation: str
) -> None:
    """Parse omission records as exact, versioned metadata objects."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        artifact_entries = cast("list[dict[str, object]]", manifest["artifacts"])
        removed = artifact_entries.pop()
        omission = {
            "period": removed["period"],
            "regime": removed["regime"],
            "type_id": removed["type_id"],
            "schema_version": removed["schema_version"],
            "reason": "not_requested",
        }
        if mutation == "extra_field":
            omission["unexpected"] = True
        else:
            omission["reason"] = "not-a-reason"
        cast("list[object]", manifest["omissions"]).append(omission)
        _replace_manifest(archive=archive, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match=r"[Oo]mission"):
        load_solution(path=path)


@pytest.mark.parametrize("link_kind", ["soft", "hard_alias"])
def test_load_rejects_nonindependent_payload_group_links(
    *, tmp_path: Path, link_kind: str
) -> None:
    """Validate each payload-group ancestor instead of only the leaf link."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        payloads = archive["payloads"]
        del payloads["00000001"]
        if link_kind == "soft":
            payloads["00000001"] = h5py.SoftLink("/payloads/00000000")
        else:
            payloads["00000001"] = payloads["00000000"]

    with pytest.raises(SolutionIntegrityError, match="HDF5 structure"):
        load_solution(path=path)


def test_load_rejects_undeclared_hdf5_members(tmp_path: Path) -> None:
    """Reject hidden archive objects outside the exact manifest membership."""
    path = save_solution(
        solution=_make_solution(),
        path=tmp_path / "solution.lcm",
    )
    with h5py.File(path, "r+") as archive:
        archive.create_group("unexpected")

    with pytest.raises(SolutionIntegrityError, match="manifest"):
        load_solution(path=path)


def test_scalar_value_roundtrip_preserves_zero_dimensional_shape(
    tmp_path: Path,
) -> None:
    """Do not promote scalar leaves to length-one arrays while copying them."""
    path = save_solution(
        solution=_make_values_only_solution(value=3.0, dtype="float32"),
        path=tmp_path / "scalar.lcm",
    )

    restored = load_solution(path=path)

    assert restored.value(period=0, regime=_REGIME).shape == ()


def test_materialization_refuses_jax_dtype_narrowing(tmp_path: Path) -> None:
    """A float64 archive must not silently become float32 under JAX defaults."""
    with jax.enable_x64(new_val=True):
        path = save_solution(
            solution=_make_values_only_solution(value=[1.0, 2.0], dtype="float64"),
            path=tmp_path / "float64.lcm",
        )
    restored = load_solution(path=path)

    with (
        jax.enable_x64(new_val=False),
        pytest.raises(IncompatibleSolutionError, match="would materialize it as"),
    ):
        restored.value(period=0, regime=_REGIME)


def test_unsupported_numpy_dtype_is_normalized_and_stays_unloaded(
    tmp_path: Path,
) -> None:
    """Normalize a JAX conversion failure without caching a partial value."""
    array = np.asarray([1.0, 2.0], dtype=np.longdouble)
    solution = SolutionResult(
        values=ValueStore({(0, _REGIME): array}),
        metadata=SolutionMetadata(
            retention=ResultRetention.VALUES,
            n_periods=1,
            regime_names=(_REGIME,),
            solver_types={_REGIME: "example.ValueOnlySolver"},
            model_instance_id="value-only-model",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, _REGIME): ValueArraySchema(
                    shape=tuple(array.shape),
                    dtype=np.dtype(array.dtype).name,
                    axis_names=("axis_0",),
                )
            },
        ),
    )
    path = save_solution(
        solution=solution,
        path=tmp_path / "longdouble.lcm",
    )
    restored = load_solution(path=path)

    values = cast("ValueStore", restored.values)
    assert values.load_state(period=0, regime=_REGIME) is LoadState.UNLOADED
    with pytest.raises(IncompatibleSolutionError, match="cannot materialize"):
        restored.value(period=0, regime=_REGIME)
    assert values.load_state(period=0, regime=_REGIME) is LoadState.UNLOADED
