"""Unbudgeted simulation operations accept grid points beside sharded subject rows."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.memory import run_simulation_operation


def _subject_sharding() -> jax.NamedSharding:
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Select this test with at least two visible devices.")
    mesh = jax.make_mesh((2,), ("subjects",), devices=devices[:2])
    return jax.NamedSharding(mesh, jax.P("subjects"))


def _count_below_last_point(
    *, values: jax.Array, points: jax.Array, subjects: jax.Array
) -> jax.Array:
    return jnp.sum(subjects & (values < points[-1]))


def test_run_simulation_operation_reads_points_committed_to_one_device() -> None:
    """Grid points committed to a single device meet subject rows sharded over
    two devices, and the operation reads them as one replicated operand."""
    sharding = _subject_sharding()
    values = jax.device_put(jnp.array([0.5, 2.0, 3.5, 1.0, 4.0, 2.5]), sharding)
    subjects = jax.device_put(
        jnp.array([True, True, False, True, True, True]), sharding
    )
    points = jax.device_put(jnp.array([0.0, 1.5, 3.0]), jax.devices()[0])
    assert len(values.sharding.device_set) == 2
    assert points.sharding.device_set == {jax.devices()[0]}
    count = run_simulation_operation(
        memory=None,
        function=_count_below_last_point,
        arguments={"values": values, "points": points, "subjects": subjects},
    )
    assert int(count) == 4


def test_run_simulation_operation_keeps_sharded_outputs_sharded() -> None:
    """Aligning a single-device operand leaves the subject rows' sharding intact."""
    sharding = _subject_sharding()
    values = jax.device_put(jnp.arange(6.0), sharding)
    offset = jax.device_put(jnp.asarray(10.0), jax.devices()[1])
    shifted = run_simulation_operation(
        memory=None,
        function=lambda *, values, offset: values + offset,
        arguments={"values": values, "offset": offset},
    )
    np.testing.assert_array_equal(np.asarray(shifted), np.arange(6.0) + 10.0)
    assert shifted.sharding.is_equivalent_to(sharding, ndim=1)


def _return_points(*, values: jax.Array, points: jax.Array) -> jax.Array:  # noqa: ARG001
    return points


def test_run_simulation_operation_reuses_the_replica_of_a_live_operand() -> None:
    """Repeated operations on one single-device operand read the same replica."""
    sharding = _subject_sharding()
    values = jax.device_put(jnp.arange(6.0), sharding)
    points = jax.device_put(jnp.array([0.0, 1.5, 3.0]), jax.devices()[0])
    first, second = (
        run_simulation_operation(
            memory=None,
            function=_return_points,
            arguments={"values": values, "points": points},
        )
        for _ in range(2)
    )
    assert first is second


def test_run_simulation_operation_replica_does_not_keep_its_operand_alive() -> None:
    """A single-device operand is freed once its last outside reference goes."""
    sharding = _subject_sharding()
    values = jax.device_put(jnp.arange(6.0), sharding)
    points = jax.device_put(jnp.array([0.0, 1.5, 3.0]), jax.devices()[0])
    run_simulation_operation(
        memory=None,
        function=_return_points,
        arguments={"values": values, "points": points},
    )
    operand = weakref.ref(points)
    del points
    gc.collect()
    assert operand() is None
