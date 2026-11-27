"""Unbudgeted simulation operations accept grid points beside sharded agent rows."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.memory import run_simulation_operation, run_simulation_producer

# Pin before backend initialization; ordinary-suite workers must skip atomically.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

pytestmark = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Requires a fresh four-CPU-device process"
)


def _agent_sharding() -> jax.NamedSharding:
    mesh = jax.make_mesh((2,), ("agents",), devices=jax.devices()[:2])
    return jax.NamedSharding(mesh, jax.P("agents"))


def _count_below_last_point(
    *, values: jax.Array, points: jax.Array, agents: jax.Array
) -> jax.Array:
    return jnp.sum(agents & (values < points[-1]))


def test_run_simulation_operation_reads_points_committed_to_one_device() -> None:
    """Grid points committed to a single device meet agent rows sharded over
    two devices, and the operation reads them as one replicated operand."""
    sharding = _agent_sharding()
    values = jax.device_put(jnp.array([0.5, 2.0, 3.5, 1.0, 4.0, 2.5]), sharding)
    agents = jax.device_put(jnp.array([True, True, False, True, True, True]), sharding)
    points = jax.device_put(jnp.array([0.0, 1.5, 3.0]), jax.devices()[0])
    assert len(values.sharding.device_set) == 2
    assert points.sharding.device_set == {jax.devices()[0]}
    count = run_simulation_operation(
        memory=None,
        function=_count_below_last_point,
        arguments={"values": values, "points": points, "agents": agents},
    )
    assert int(count) == 4


def test_run_simulation_producer_reads_points_committed_to_one_device() -> None:
    """An unbudgeted producer reads single-device grid points beside agent rows
    sharded over two devices, as an operation does."""
    sharding = _agent_sharding()
    values = jax.device_put(jnp.array([0.5, 2.0, 3.5, 1.0, 4.0, 2.5]), sharding)
    agents = jax.device_put(jnp.array([True, True, False, True, True, True]), sharding)
    points = jax.device_put(jnp.array([0.0, 1.5, 3.0]), jax.devices()[0])
    count = run_simulation_producer(
        memory=None,
        function=_count_below_last_point,
        arguments={"values": values, "points": points, "agents": agents},
    )
    assert int(count) == 4


def test_run_simulation_operation_keeps_sharded_outputs_sharded() -> None:
    """Aligning a single-device operand leaves the agent rows' sharding intact."""
    sharding = _agent_sharding()
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
    sharding = _agent_sharding()
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
    sharding = _agent_sharding()
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


def test_run_simulation_operation_passes_uncommitted_operands_through() -> None:
    """An uncommitted operand reaches the operation as is; JAX places it itself."""
    values = jax.device_put(jnp.arange(6.0), _agent_sharding())
    points = jnp.array([0.0, 1.5, 3.0])
    assert not points.committed
    returned = run_simulation_operation(
        memory=None,
        function=_return_points,
        arguments={"values": values, "points": points},
    )
    assert returned is points
