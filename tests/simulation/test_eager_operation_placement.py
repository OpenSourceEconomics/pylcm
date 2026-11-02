"""Unbudgeted simulation operations accept grid points beside sharded subject rows."""

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
