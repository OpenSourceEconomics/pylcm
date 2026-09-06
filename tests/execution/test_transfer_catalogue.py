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
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    apply_value_transfer,
    classify_value_transfer,
)
from lcm.exceptions import ExecutionPlanningError

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
