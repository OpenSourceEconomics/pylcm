"""The transfer catalogue is a total function over the layouts the engine plans.

Every stored/required sharding pair the planner can produce names exactly one
operator, and the one pair that cannot be served — two meshes that overlap
without one containing the other — is refused while planning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.value_transfer import (
    CoordinateSelection,
    ResolvedValueTransfer,
    TransferOperationClass,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    ValueViewDescriptor,
    ValueViewLeaf,
    apply_value_transfer,
    classify_value_transfer,
)
from _lcm.execution.value_views import lower_value_view_selection
from lcm.exceptions import ExecutionPlanningError

# Run these tests on a four-CPU-device topology. The pin only applies in a
# process whose JAX backends are not yet initialized (a serial run importing
# this module early); otherwise the tests skip. The device-count update is
# attempted FIRST because it is the one that raises after initialization —
# this keeps the pin atomic. The reverse order would flip the default
# platform to CPU (that update succeeds at any time) and then skip, leaving
# every later model build in the process compiled for CPU while arrays from
# earlier accelerator computations stay committed to their device.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest parallel"
)


def _mesh(*, devices: list[jax.Device], axis: str) -> jax.sharding.Mesh:
    """A one-axis mesh over `devices`."""
    return jax.sharding.Mesh(devices, axis_names=(axis,))


def _sharded(*, mesh: jax.sharding.Mesh) -> jax.Array:
    """Eight elements sharded four ways on `mesh`."""
    stored = jax.device_put(
        jnp.arange(8, dtype=jnp.float32), jax.NamedSharding(mesh, jax.P("d"))
    )
    assert jnp.isfinite(stored).all()
    return stored


def _resolved(
    *,
    stored: jax.Array,
    required: jax.sharding.Sharding,
    kind: ValueTransferKind,
) -> ResolvedValueTransfer:
    """One resolved next-period regime-value transfer of `stored`."""
    return ResolvedValueTransfer(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="retired"
        ),
        source=ValueConsumerAddress(
            source_period=3,
            source_regime="working",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("retired",),
        ),
        kind=kind,
        stored_sharding=stored.sharding,
        source_sharding=required,
        expected_shape=stored.shape,
        expected_dtype=stored.dtype,
    )


@_skip_pytest_parallel
@pytest.mark.parametrize(
    ("stored_spec", "required_spec", "expected"),
    [
        (jax.P("d"), jax.P("d"), ValueTransferKind.ALIGNED_LOCAL),
        (jax.P("d"), jax.P(), ValueTransferKind.ALL_GATHER),
        (jax.P(), jax.P("d"), ValueTransferKind.LOCAL_SLICE),
    ],
)
def test_one_mesh_pairs_name_their_operator(
    *,
    stored_spec: jax.sharding.PartitionSpec,
    required_spec: jax.sharding.PartitionSpec,
    expected: ValueTransferKind,
) -> None:
    """A pair of shardings on one mesh names exactly one catalogue operator."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")

    assert (
        classify_value_transfer(
            stored_sharding=jax.NamedSharding(mesh, stored_spec),
            required_sharding=jax.NamedSharding(mesh, required_spec),
        )
        is expected
    )


@_skip_pytest_parallel
def test_a_change_of_sharded_axis_is_a_reshard() -> None:
    """Moving a value from one named axis to another is a reshard."""
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), axis_names=("a", "b")
    )

    assert (
        classify_value_transfer(
            stored_sharding=jax.NamedSharding(mesh, jax.P("a", None)),
            required_sharding=jax.NamedSharding(mesh, jax.P(None, "b")),
        )
        is ValueTransferKind.RESHARD
    )


@_skip_pytest_parallel
def test_a_nested_submesh_target_is_a_cross_mesh_copy() -> None:
    """A node placed on a submesh reads a value stored on the full mesh."""
    full = _mesh(devices=jax.devices()[:4], axis="d")
    sub = _mesh(devices=jax.devices()[:2], axis="d")

    assert (
        classify_value_transfer(
            stored_sharding=jax.NamedSharding(full, jax.P("d")),
            required_sharding=jax.NamedSharding(sub, jax.P("d")),
        )
        is ValueTransferKind.CROSS_MESH_COPY
    )


@_skip_pytest_parallel
def test_overlapping_but_unequal_meshes_are_refused() -> None:
    """Two meshes sharing some devices, neither inside the other, fail closed."""
    left = _mesh(devices=jax.devices()[:3], axis="d")
    right = _mesh(devices=jax.devices()[1:4], axis="d")

    with pytest.raises(ExecutionPlanningError, match="Overlapping"):
        classify_value_transfer(
            stored_sharding=jax.NamedSharding(left, jax.P("d")),
            required_sharding=jax.NamedSharding(right, jax.P("d")),
        )


@_skip_pytest_parallel
def test_an_all_gather_delivers_the_stored_values_unchanged() -> None:
    """A gather changes the layout of a value, never the value."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = jax.device_put(
        jnp.arange(8, dtype=jnp.float32), jax.NamedSharding(mesh, jax.P("d"))
    )
    assert jnp.isfinite(stored).all()
    transfer = _resolved(
        stored=stored,
        required=jax.NamedSharding(mesh, jax.P()),
        kind=ValueTransferKind.ALL_GATHER,
    )

    gathered = apply_value_transfer(value=stored, transfer=transfer)

    assert gathered.tobytes() == stored.tobytes()


@_skip_pytest_parallel
def test_a_local_slice_delivers_the_stored_values_unchanged() -> None:
    """Slicing a replicated value onto a mesh axis changes the layout, not the value."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = jax.device_put(
        jnp.arange(8, dtype=jnp.float32), jax.NamedSharding(mesh, jax.P())
    )
    assert jnp.isfinite(stored).all()
    transfer = _resolved(
        stored=stored,
        required=jax.NamedSharding(mesh, jax.P("d")),
        kind=ValueTransferKind.LOCAL_SLICE,
    )

    sliced = apply_value_transfer(value=stored, transfer=transfer)

    assert sliced.tobytes() == stored.tobytes()


@_skip_pytest_parallel
def test_a_reshard_delivers_the_stored_values_unchanged() -> None:
    """Moving a value between mesh axes changes the layout, never the value."""
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), axis_names=("a", "b")
    )
    stored = jax.device_put(
        jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        jax.NamedSharding(mesh, jax.P("a", None)),
    )
    assert jnp.isfinite(stored).all()
    transfer = _resolved(
        stored=stored,
        required=jax.NamedSharding(mesh, jax.P(None, "b")),
        kind=ValueTransferKind.RESHARD,
    )

    resharded = apply_value_transfer(value=stored, transfer=transfer)

    assert resharded.tobytes() == stored.tobytes()


@_skip_pytest_parallel
def test_a_copy_onto_a_nested_submesh_delivers_the_stored_values_unchanged() -> None:
    """A value copied onto a submesh of its own mesh keeps every element."""
    full = _mesh(devices=jax.devices()[:4], axis="d")
    sub = _mesh(devices=jax.devices()[:2], axis="d")
    stored = jax.device_put(
        jnp.arange(8, dtype=jnp.float32), jax.NamedSharding(full, jax.P("d"))
    )
    assert jnp.isfinite(stored).all()
    transfer = _resolved(
        stored=stored,
        required=jax.NamedSharding(sub, jax.P("d")),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    copied = apply_value_transfer(value=stored, transfer=transfer)

    assert copied.tobytes() == stored.tobytes()


@_skip_pytest_parallel
def test_a_copy_onto_a_disjoint_mesh_delivers_the_stored_values_unchanged() -> None:
    """A value copied onto a mesh sharing no device with its own keeps every element."""
    left = _mesh(devices=jax.devices()[:2], axis="d")
    right = _mesh(devices=jax.devices()[2:4], axis="d")
    stored = jax.device_put(
        jnp.arange(8, dtype=jnp.float32), jax.NamedSharding(left, jax.P("d"))
    )
    assert jnp.isfinite(stored).all()
    transfer = _resolved(
        stored=stored,
        required=jax.NamedSharding(right, jax.P("d")),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    copied = apply_value_transfer(value=stored, transfer=transfer)

    assert copied.tobytes() == stored.tobytes()


@_skip_pytest_parallel
def test_a_partition_entry_naming_no_placement_is_refused() -> None:
    """A spec entry that is not an axis name, a tuple of names, or None fails closed."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    unconstrained = jax.sharding.PartitionSpec(jax.sharding.PartitionSpec.UNCONSTRAINED)

    with pytest.raises(ExecutionPlanningError, match="must be a mesh-axis name"):
        classify_value_transfer(
            stored_sharding=jax.NamedSharding(mesh, unconstrained),
            required_sharding=jax.NamedSharding(mesh, jax.P()),
        )


@_skip_pytest_parallel
def test_an_aligned_transfer_is_a_local_operation() -> None:
    """A value already in its required layout moves nothing."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = _sharded(mesh=mesh)
    cost = _resolved(
        stored=stored,
        required=stored.sharding,
        kind=ValueTransferKind.ALIGNED_LOCAL,
    ).cost

    assert cost.operation_class is TransferOperationClass.LOCAL


@_skip_pytest_parallel
@pytest.mark.parametrize(
    ("attribute", "elements"),
    [("logical_bytes", 8), ("per_device_bytes", 2), ("temporary_bytes", 0)],
)
def test_an_aligned_transfer_holds_only_its_own_shard(
    *, attribute: str, elements: int
) -> None:
    """An aligned transfer occupies one shard per device and no temporary."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = _sharded(mesh=mesh)
    cost = _resolved(
        stored=stored,
        required=stored.sharding,
        kind=ValueTransferKind.ALIGNED_LOCAL,
    ).cost

    assert getattr(cost, attribute) == elements * stored.dtype.itemsize


@_skip_pytest_parallel
def test_an_aligned_transfer_touches_every_mesh_device() -> None:
    """The recorded device set is the placement the operator runs on."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = _sharded(mesh=mesh)
    cost = _resolved(
        stored=stored,
        required=stored.sharding,
        kind=ValueTransferKind.ALIGNED_LOCAL,
    ).cost

    assert cost.devices == tuple(sorted(device.id for device in jax.devices()[:4]))


@_skip_pytest_parallel
def test_a_gather_is_a_collective() -> None:
    """Gathering a sharded value onto every device is a collective."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    cost = _resolved(
        stored=_sharded(mesh=mesh),
        required=jax.NamedSharding(mesh, jax.P()),
        kind=ValueTransferKind.ALL_GATHER,
    ).cost

    assert cost.operation_class is TransferOperationClass.COLLECTIVE


@_skip_pytest_parallel
@pytest.mark.parametrize(
    ("attribute", "elements"), [("per_device_bytes", 8), ("temporary_bytes", 8)]
)
def test_a_gather_holds_the_whole_value_per_device(
    *, attribute: str, elements: int
) -> None:
    """A gather lands the full array on every device and holds it once more."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = _sharded(mesh=mesh)
    cost = _resolved(
        stored=stored,
        required=jax.NamedSharding(mesh, jax.P()),
        kind=ValueTransferKind.ALL_GATHER,
    ).cost

    assert getattr(cost, attribute) == elements * stored.dtype.itemsize


_TYPE_VALUES = np.arange(3 * 8 * 2, dtype=np.float32).reshape(3, 8, 2) * 0.5 - 3.0
_COLLECTIVE_OPS = ("all-gather", "all-reduce", "all-to-all", "collective-permute")


def _typed_view(*, required: jax.sharding.Sharding, code: int) -> ValueViewDescriptor:
    """The type-`code` block of a `(pref_type, assets, health)` value."""
    return ValueViewDescriptor(
        artifact=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="retired"
        ),
        leaf=ValueViewLeaf.SELECTED,
        stored_axis_names=("pref_type", "assets", "health"),
        stored_shape=_TYPE_VALUES.shape,
        dtype=jnp.float32,
        weak_type=False,
        consumer_shape=_TYPE_VALUES.shape[1:],
        required_sharding=required,
        selections=(
            CoordinateSelection(
                state_name="pref_type", start=code, width=1, codes=(code,)
            ),
        ),
    )


def _typed_transfer(
    *, stored: jax.Array, required: jax.sharding.Sharding, kind: ValueTransferKind
) -> ResolvedValueTransfer:
    """A type-2 view read of `stored` onto `required`."""
    view = _typed_view(required=required, code=2)
    return ResolvedValueTransfer(
        target=view.artifact,
        source=ValueConsumerAddress(
            source_period=3,
            source_regime="working",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("retired",),
        ),
        kind=kind,
        stored_sharding=stored.sharding,
        source_sharding=required,
        expected_shape=stored.shape,
        expected_dtype=stored.dtype,
        view=view,
    )


def _asset_sharded_types() -> jax.Array:
    """Every type's values, partitioned along assets over four devices."""
    mesh = _mesh(devices=jax.devices()[:4], axis="d")
    stored = jax.device_put(
        _TYPE_VALUES, jax.NamedSharding(mesh, jax.P(None, "d", None))
    )
    assert np.array_equal(np.asarray(stored), _TYPE_VALUES)
    return stored


def _group_layout() -> jax.sharding.Sharding:
    """A replicated layout on the type group holding devices 2 and 3."""
    return jax.NamedSharding(_mesh(devices=jax.devices()[2:4], axis="d"), jax.P())


@_skip_pytest_parallel
def test_a_selected_block_reaches_another_device_group_bit_for_bit() -> None:
    """Type 2 is selected on the asset partitions and copied to its group whole."""
    stored = _asset_sharded_types()
    transfer = _typed_transfer(
        stored=stored,
        required=_group_layout(),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    delivered = apply_value_transfer(value=stored, transfer=transfer)

    assert delivered.tobytes() == _TYPE_VALUES[2].tobytes()


@_skip_pytest_parallel
def test_a_selected_block_lands_only_on_its_device_group() -> None:
    """The consumer group's devices, and no others, hold the copied block."""
    stored = _asset_sharded_types()
    transfer = _typed_transfer(
        stored=stored,
        required=_group_layout(),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    delivered = apply_value_transfer(value=stored, transfer=transfer)

    assert {device.id for device in delivered.devices()} == {
        device.id for device in jax.devices()[2:4]
    }


@_skip_pytest_parallel
def test_a_cross_group_block_holds_one_type_per_group_device() -> None:
    """Each group device holds one type's values, a third of every type's."""
    stored = _asset_sharded_types()

    cost = _typed_transfer(
        stored=stored,
        required=_group_layout(),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    ).cost

    assert cost.per_device_bytes == _TYPE_VALUES[2].nbytes


@_skip_pytest_parallel
def test_the_compiled_selection_communicates_nothing() -> None:
    """Selecting on the asset partitions lowers to no collective operation."""
    stored = _asset_sharded_types()
    transfer = _typed_transfer(
        stored=stored,
        required=_group_layout(),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    text = str(lower_value_view_selection(transfer=transfer).compile().as_text())

    assert [op for op in _COLLECTIVE_OPS if op in text] == []


@_skip_pytest_parallel
def test_the_collective_probe_fires_on_a_full_type_gather() -> None:
    """The probe above detects the replication of every type it rules out."""
    stored = _asset_sharded_types()
    replicated = jax.NamedSharding(_mesh(devices=jax.devices()[:4], axis="d"), jax.P())

    text = str(
        jax.jit(lambda value: value, out_shardings=replicated)
        .lower(stored)
        .compile()
        .as_text()
    )

    assert "all-gather" in text


@_skip_pytest_parallel
def test_the_compiled_selection_outputs_only_the_local_block() -> None:
    """Each device's selection output is its asset shard of one type."""
    stored = _asset_sharded_types()
    transfer = _typed_transfer(
        stored=stored,
        required=_group_layout(),
        kind=ValueTransferKind.CROSS_MESH_COPY,
    )

    stats = lower_value_view_selection(transfer=transfer).compile().memory_analysis()

    assert stats is not None
    assert stats.output_size_in_bytes == _TYPE_VALUES[2].nbytes // 4


@_skip_pytest_parallel
def test_a_selection_along_a_partitioned_axis_is_refused() -> None:
    """Types partitioned across devices are not selected by this route."""
    mesh = _mesh(devices=jax.devices()[:2], axis="d")
    values = np.zeros((2, 8, 2), dtype=np.float32)
    stored = jax.device_put(values, jax.NamedSharding(mesh, jax.P("d", None, None)))
    required = jax.NamedSharding(mesh, jax.P())
    view = ValueViewDescriptor(
        artifact=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=4, regime="retired"
        ),
        leaf=ValueViewLeaf.SELECTED,
        stored_axis_names=("pref_type", "assets", "health"),
        stored_shape=values.shape,
        dtype=jnp.float32,
        weak_type=False,
        consumer_shape=values.shape[1:],
        required_sharding=required,
        selections=(
            CoordinateSelection(state_name="pref_type", start=1, width=1, codes=(1,)),
        ),
    )

    with pytest.raises(ExecutionPlanningError, match="partitioned"):
        ResolvedValueTransfer(
            target=view.artifact,
            source=ValueConsumerAddress(
                source_period=3,
                source_regime="working",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                path=("retired",),
            ),
            kind=ValueTransferKind.ALL_GATHER,
            stored_sharding=stored.sharding,
            source_sharding=required,
            expected_shape=stored.shape,
            expected_dtype=stored.dtype,
            view=view,
        )
