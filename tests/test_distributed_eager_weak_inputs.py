"""Weak eager inputs bind on their actual ordered, restricted device layouts."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.eager_core import make_eager_core
from _lcm.execution.runtime_sharding import runtime_shardings_match
from lcm.typing import ValueND
from tests.execution.test_eager_core import eager_program

# Pin before backend initialization; ordinary-suite workers must skip atomically.
try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Requires a fresh four-CPU-device process"
)


@_skip_pytest_parallel
@pytest.mark.parametrize("device_ids", [(3, 1, 2), (2, 3, 1)])
@pytest.mark.parametrize("shape", [(), (12,)])
def test_strong_binding_preserves_actual_ordered_physical_layout(
    *, device_ids: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    devices = tuple(jax.devices()[index] for index in device_ids)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("subject",))
    target = jax.NamedSharding(mesh, jax.P("subject") if shape else jax.P())
    source = jax.device_put(jnp.full(shape, 2.0), target)
    assert source.weak_type
    assert source.committed
    assert isinstance(source.sharding, jax.NamedSharding)
    assert (
        tuple(device.id for device in source.sharding.mesh.devices.flat) == device_ids
    )
    assert 0 not in {device.id for device in source.devices()}
    returned: list[object] = []

    def body(*, value: ValueND) -> object:
        assert not value.weak_type
        assert isinstance(value.sharding, jax.NamedSharding)
        assert (
            tuple(device.id for device in value.sharding.mesh.devices.flat)
            == device_ids
        )
        assert value.sharding.memory_kind == target.memory_kind
        assert runtime_shardings_match(
            actual=value.sharding, expected=target, ndim=value.ndim
        )
        assert value.sharding.devices_indices_map(shape) == target.devices_indices_map(
            shape
        )
        assert source.weak_type
        assert not source.is_deleted()
        output = {"value": (value, None)}
        returned.append(output)
        return output

    descriptor = jax.ShapeDtypeStruct(shape, source.dtype, sharding=target)
    adapter = make_eager_core(
        program=eager_program(function=body, arguments={"value": descriptor}),
        execution_sharding=target,
    )
    output = adapter(value=source)
    assert output is returned.pop()
    value = cast("dict[str, tuple[ValueND, None]]", output)["value"][0]
    np.testing.assert_array_equal(value, np.full(shape, 2.0))
    np.testing.assert_array_equal(source, np.full(shape, 2.0))
    before = {
        shard.device: shard.data.unsafe_buffer_pointer()
        for shard in source.addressable_shards
    }
    after = {
        shard.device: shard.data.unsafe_buffer_pointer()
        for shard in value.addressable_shards
    }
    assert set(before) == set(after) == set(devices)
    assert all(before[device] != after[device] for device in devices)
