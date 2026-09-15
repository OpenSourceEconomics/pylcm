"""Subject-local loops preserve global order, keys and compiler-visible placement."""

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.programs import _SubjectTiled
from _lcm.simulation.subject_parallel import shard_subject_function


def _devices(*, count: int, reverse: bool = False) -> tuple[jax.Device, ...]:
    available = tuple(jax.devices())
    if len(available) < count:
        pytest.skip(f"Select this test with {count} visible devices.")
    selected = available[:count]
    return selected[::-1] if reverse else selected


def _mesh(*, devices: tuple[jax.Device, ...]) -> jax.sharding.Mesh:
    return jax.make_mesh(
        (len(devices),), ("X",), (jax.sharding.AxisType.Auto,), devices=devices
    )


def _cell(*, state: jax.Array, key: jax.Array, grid: jax.Array) -> object:
    value = jnp.sum((state + grid) ** 2)
    return {
        "value": value,
        "random": jax.random.bits(key, (), dtype=jnp.uint32),
        "tie": jnp.argmax(jnp.array([value, value, value - 1])),
    }


@pytest.mark.parametrize("device_count", [1, 2, 8])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [1, 3, 2048])
def test_local_tiles_match_vmap_and_keep_shards(
    *, device_count: int, reverse: bool, width: int
) -> None:
    devices = _devices(count=device_count, reverse=reverse)
    mesh = _mesh(devices=devices)
    n_subjects = 7 * device_count
    states = np.linspace(0, 1, n_subjects, dtype=np.float32)
    keys = jax.random.split(jax.random.key(31), n_subjects)
    # Same extent as the population is NOT a declaration of subject dependence.
    grid = np.linspace(1, 2, n_subjects, dtype=np.float32)
    arguments = {
        "state": jax.device_put(states, jax.NamedSharding(mesh, jax.P("X"))),
        "key": jax.device_put(keys, jax.NamedSharding(mesh, jax.P("X"))),
        "grid": jax.device_put(grid, jax.NamedSharding(mesh, jax.P())),
    }
    tiled = _SubjectTiled(func=_cell, subject_arg_names=("state", "key"))
    adapted = shard_subject_function(
        function=tiled,
        subject_arg_names=tiled.subject_shard_arg_names,
        arguments=arguments,
        static_kwargs={SUBJECT_WIDTH_KEYWORD: width},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    abstract = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=leaf.sharding
        ),
        arguments,
    )
    compiled = jax.jit(adapted).lower(**abstract).compile()
    result = compiled(**arguments)
    expected = jax.vmap(
        lambda state, key: _cell(state=state, key=key, grid=jnp.asarray(grid))
    )(states, keys)
    for actual, reference in zip(
        jax.tree.leaves(result), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(actual.dtype, jnp.inexact):
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)
        else:
            np.testing.assert_array_equal(actual, reference)
        assert tuple(actual.sharding.mesh.devices.flat) == devices
        assert len(actual.addressable_shards) == device_count
        for index, shard in enumerate(actual.addressable_shards):
            assert shard.data.shape[0] == 7
            assert shard.index[0].indices(n_subjects) == (7 * index, 7 * (index + 1), 1)
    for leaf in jax.tree.leaves(arguments):
        assert not leaf.is_deleted()


def _increment(*, state: jax.Array) -> jax.Array:
    return state + 1


def test_wrapper_does_not_capture_call_owned_arrays() -> None:
    devices = _devices(count=1)
    array = jax.device_put(np.arange(16, dtype=np.float32), devices[0])
    reference = weakref.ref(array)
    tiled = _SubjectTiled(func=_increment, subject_arg_names=("state",))
    wrapper = shard_subject_function(
        function=tiled,
        subject_arg_names=("state",),
        arguments={"state": array},
        static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    del array
    gc.collect()
    assert reference() is None
    assert callable(wrapper)


@pytest.mark.parametrize("n_subjects", [1, 13, 17])
def test_global_keys_survive_padding_and_device_partition(*, n_subjects: int) -> None:
    devices = _devices(count=8)
    mesh = _mesh(devices=devices)
    keys = jax.random.split(jax.random.key(17), n_subjects + 1)[1:]
    padded = -(-n_subjects // 8) * 8
    all_keys = jnp.concatenate(
        [keys, jnp.repeat(keys[-1:], padded - n_subjects, axis=0)]
    )
    placed = jax.device_put(all_keys, jax.NamedSharding(mesh, jax.P("X")))

    def draw(*, key: jax.Array) -> jax.Array:
        return jax.random.bits(key, (), dtype=jnp.uint32)

    tiled = _SubjectTiled(func=draw, subject_arg_names=("key",))
    adapted = shard_subject_function(
        function=tiled,
        subject_arg_names=("key",),
        arguments={"key": placed},
        static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    actual = jax.jit(adapted)(key=placed)
    np.testing.assert_array_equal(actual[:n_subjects], jax.vmap(draw)(key=keys))


@pytest.mark.parametrize("bad_shape", [(), (0,), (15,)])
def test_refuses_unaligned_or_missing_subject_axis(
    *, bad_shape: tuple[int, ...]
) -> None:
    devices = _devices(count=8)
    tiled = _SubjectTiled(func=_increment, subject_arg_names=("state",))
    with pytest.raises(ValueError, match=r"(?i)subject"):
        shard_subject_function(
            function=tiled,
            subject_arg_names=("state",),
            arguments={"state": jax.ShapeDtypeStruct(bad_shape, jnp.float32)},
            static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
            devices=devices,
            subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
        )


def test_refuses_output_that_drops_the_subject_axis() -> None:
    devices = _devices(count=8)
    mesh = _mesh(devices=devices)
    array = jax.device_put(
        np.arange(16, dtype=np.float32), jax.NamedSharding(mesh, jax.P("X"))
    )

    def invalid(**arguments: jax.Array) -> jax.Array:
        return jnp.sum(arguments["state"])

    adapted = shard_subject_function(
        function=invalid,
        subject_arg_names=("state",),
        arguments={"state": array},
        static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    with pytest.raises(ValueError, match="preserve its leading extent"):
        jax.jit(adapted).lower(state=array)


@pytest.mark.parametrize("constant", [False, True])
def test_foreign_replica_mesh_and_eliminated_inputs(*, constant: bool) -> None:
    devices = _devices(count=8)
    mesh = _mesh(devices=devices)
    solve_mesh = jax.make_mesh(
        (8,), ("solve_assets",), (jax.sharding.AxisType.Auto,), devices=devices
    )
    arguments = {
        "state": jax.device_put(
            np.arange(56, dtype=np.float32), jax.NamedSharding(mesh, jax.P("X"))
        ),
        "grid": jax.device_put(
            np.arange(8, dtype=np.float32), jax.NamedSharding(solve_mesh, jax.P())
        ),
    }

    def cell(*, state: jax.Array, grid: jax.Array) -> jax.Array:
        return jnp.float32(4) if constant else jnp.sum(state + grid)

    tiled = _SubjectTiled(func=cell, subject_arg_names=("state",))
    adapted = shard_subject_function(
        function=tiled,
        subject_arg_names=("state",),
        arguments=arguments,
        static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    with jax.set_mesh(mesh):
        compiled = jax.jit(adapted).lower(**arguments).compile()
    output = compiled(**arguments)
    expected = np.full(56, 4) if constant else np.arange(56) * 8 + 28
    np.testing.assert_array_equal(output, expected)
    assert [shard.data.shape for shard in output.addressable_shards] == [(7,)] * 8
