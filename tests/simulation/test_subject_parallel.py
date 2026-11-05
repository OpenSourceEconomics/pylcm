"""Subject-local loops preserve global order, keys and compiler-visible placement."""

import gc
import inspect
import weakref
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import TypedDict

from _lcm.execution.output_layout import StateAxesLeading
from _lcm.execution.placement import ACTION_PARTITION_AXIS
from _lcm.regime_building.max_Q_over_a import get_action_partitioned_max_Q_over_a
from _lcm.regime_building.Q_and_F import get_Q_and_F_terminal
from _lcm.simulation.program_types import SUBJECT_WIDTH_KEYWORD
from _lcm.simulation.programs import _SubjectTiled
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.subject_parallel import shard_subject_function
from _lcm.solution.action_reduction import HardMaxAccumulator
from _lcm.solution.action_streaming import build_partitioned_streaming_max_Q_over_a
from _lcm.solution.feasibility import BoundedResult, summarize_bound_inputs
from _lcm.typing import ArgumentTree, ConstraintFunction, EconFunction, PytreeValue
from lcm import ExecutionConfig
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import BoolND, FloatND, RegimeName, ScalarInt
from tests.constraints.test_dynamic_lower_bound import TemporalRegimeId, _temporal_model
from tests.solution.test_dynamic_bound_authority import _bound


class _PartitionArguments(TypedDict, closed=True):
    next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND]
    period: ScalarInt
    age: FloatND
    income: FloatND
    consumption: FloatND


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_action_partition_replicates_active_owner_integrity(
    *,
    invalid: bool,
    reverse: bool,
) -> None:
    """Empty devices contribute neutral flags; the active summary is replicated."""
    devices = _devices(count=2, reverse=reverse)
    mesh = jax.make_mesh(
        (2,), (ACTION_PARTITION_AXIS,), (jax.sharding.AxisType.Auto,), devices=devices
    )

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    kernel = get_Q_and_F_terminal(
        flat_param_names=frozenset(),
        functions=MappingProxyType(
            {
                "utility": cast("EconFunction", utility),
                "floor": cast("EconFunction", floor),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    arguments: _PartitionArguments = {
        "next_regime_to_V_arr": MappingProxyType({}),
        "period": jnp.int32(0),
        "age": jnp.asarray(0.0),
        "income": jnp.asarray(jnp.nan if invalid else -3.0),
        "consumption": jnp.asarray([2.0]),
    }
    partitioned = build_partitioned_streaming_max_Q_over_a(
        Q_and_F=kernel,
        action_names=("consumption",),
        block_width=1,
        n_partitions=2,
        axis_name=ACTION_PARTITION_AXIS,
    )

    def local_flags():
        local = partitioned.local(
            partition=jax.lax.axis_index(ACTION_PARTITION_AXIS).astype(jnp.int32),
            **arguments,
        )
        assert type(local) is BoundedResult
        return jnp.stack(local.bound_status[0])[None, :], cast(
            "HardMaxAccumulator", local.output
        ).any_feasible[None]

    flags, feasible = jax.jit(
        jax.shard_map(
            local_flags,
            mesh=mesh,
            in_specs=(),
            out_specs=(jax.P(ACTION_PARTITION_AXIS), jax.P(ACTION_PARTITION_AXIS)),
            axis_names=frozenset({ACTION_PARTITION_AXIS}),
            check_vma=False,
        )
    )()
    np.testing.assert_array_equal(flags, [[False, False], [invalid, False]])
    np.testing.assert_array_equal(feasible, [False, not invalid])
    mapped = get_action_partitioned_max_Q_over_a(
        Q_and_F=kernel,
        batch_sizes={},
        action_names=("consumption",),
        state_names=(),
        mesh=mesh,
        n_partitions=2,
        return_feasibility=True,
    )
    result = jax.jit(lambda: mapped(**arguments, _lcm_action_block_width=1))()
    assert type(result) is BoundedResult
    np.testing.assert_array_equal(result.output[0], -np.inf if invalid else 2.0)
    np.testing.assert_array_equal(result.output[1], invalid)
    np.testing.assert_array_equal(jnp.stack(result.bound_status[0]), [invalid, False])
    assert all(leaf.sharding.spec == jax.P() for leaf in jax.tree.leaves(result))


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_public_subject_sharding_keeps_callable_floor_integrity(
    *,
    runtime_checks: bool,
    reverse: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded public cohorts retain global validity at their later decision."""
    devices = _devices(count=2, reverse=reverse)

    def floor(*, income: FloatND, period: ScalarInt) -> FloatND:
        return jnp.where((period == 1) & (income == 0.5), jnp.nan, 0.0)

    model = _temporal_model(
        lower_function=floor,
        execution_config=ExecutionConfig(
            devices=tuple(device.id for device in devices),
            sharded_states=(),
            simulation_sharding="subjects",
            device_memory_bytes=1 << 28,
            axis_widths={"subject": 1},
        ),
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(solution.values[1]["saving"], [[1.0, 1.0], [2.0, 2.0]])
    observed = []
    dispatch = SimulationRuntime.dispatch

    def observe(self: SimulationRuntime, **kwargs):
        result = dispatch(self, **kwargs)
        if self.execution.simulation_sharding == "subjects":
            scalar_flags = tuple(
                leaf
                for leaf in jax.tree.leaves(result)
                if isinstance(leaf, jax.Array)
                and leaf.dtype == jnp.bool_
                and leaf.shape == ()
            )
            if scalar_flags:
                assert len(scalar_flags) == 2
                assert all(leaf.sharding.spec == jax.P() for leaf in scalar_flags)
                assert tuple(device.id for device in self.subject_devices) == tuple(
                    sorted(device.id for device in devices)
                )
                observed.append(jnp.stack(scalar_flags))
        return result

    monkeypatch.setattr(SimulationRuntime, "dispatch", observe)
    initial = {
        "wealth": jnp.asarray([2.0, 2.0, 2.0]),
        "income": jnp.asarray([1.0, 1.0, 1.0]),
        "age": jnp.asarray([0, 0, 0]),
        "regime_id": jnp.asarray([TemporalRegimeId.saving] * 3, dtype=jnp.int32),
    }
    positive = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="off",
        runtime_checks=runtime_checks,
    )
    np.testing.assert_array_equal(
        positive.to_dataframe()["consumption"].dropna(), [0.25, 1.75] * 3
    )
    assert observed
    np.testing.assert_array_equal(
        jnp.stack(observed), jnp.zeros((len(observed), 2), dtype=bool)
    )
    observed.clear()
    with pytest.raises(InvalidValueFunctionError, match="floor"):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={**initial, "income": jnp.asarray([1.0, 1.0, 0.5])},
            log_level="off",
            runtime_checks=runtime_checks,
        )
    assert observed
    np.testing.assert_array_equal(observed[-1], [True, False])


def _bounded_cell(
    *, floor: FloatND, active: BoolND
) -> BoundedResult[tuple[FloatND, BoolND]]:
    return BoundedResult(
        output=(floor * floor, active),
        bound_status=(summarize_bound_inputs(floor_values=floor, active=active),),
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("active_last", [False, True])
def test_subject_partitions_or_scalar_integrity_across_devices(
    *, reverse: bool, width: int, active_last: bool
) -> None:
    """Ordinary rows keep their shards; active validity is globally replicated."""
    code = inspect.unwrap(shard_subject_function).__code__
    assert (
        "output_roles" in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    ), "The subject sharding owner needs explicit mixed scalar output roles."
    devices = _devices(count=2, reverse=reverse)
    mesh = _mesh(devices=devices)
    arguments = {
        "floor": jax.device_put(
            jnp.asarray([1.0, jnp.nan, 2.0, jnp.nan]),
            jax.NamedSharding(mesh, jax.P("X")),
        ),
        "active": jax.device_put(
            jnp.asarray([True, False, True, active_last]),
            jax.NamedSharding(mesh, jax.P("X")),
        ),
    }
    tiled = _SubjectTiled(
        func=_bounded_cell,
        subject_arg_names=("floor", "active"),
        n_bound_records=1,
    )
    adapted = shard_subject_function(
        function=tiled,
        subject_arg_names=tiled.subject_shard_arg_names,
        arguments=arguments,
        static_kwargs={SUBJECT_WIDTH_KEYWORD: width},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
        output_roles=(
            "value",
            "active",
            *(
                StateAxesLeading(state_names=(), dtype=jnp.bool_, shape=())
                for _ in range(2)
            ),
        ),
    )
    result = jax.jit(adapted)(**arguments)
    np.testing.assert_array_equal(result[0], [1.0, jnp.nan, 4.0, jnp.nan])
    np.testing.assert_array_equal(result[1], [True, False, True, active_last])
    assert result[0].sharding.spec == jax.P("X")
    assert result[2].shape == result[3].shape == ()
    assert result[2].sharding.spec == result[3].sharding.spec == jax.P()
    np.testing.assert_array_equal(jnp.stack(result[2:]), [active_last, False])


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


def _cell(*, state: jax.Array, key: jax.Array, grid: jax.Array) -> dict[str, jax.Array]:
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


def _increment_with_optional(*, state: jax.Array, optional: PytreeValue) -> jax.Array:
    del optional
    return state + 1


@pytest.mark.parametrize(
    "empty",
    [{}, MappingProxyType({}), (), {"nested": ({}, [], None)}, None],
    ids=["dict", "mappingproxy", "tuple", "nested", "none"],
)
def test_empty_subject_subtrees_keep_real_array_extent(*, empty: PytreeValue) -> None:
    """Leafless optional inputs survive device mapping and local remainder tiles."""
    devices = _devices(count=2)
    mesh = _mesh(devices=devices)
    state = jax.device_put(
        np.arange(16, dtype=np.float32), jax.NamedSharding(mesh, jax.P("X"))
    )
    tiled = _SubjectTiled(
        func=_increment_with_optional, subject_arg_names=("state", "optional")
    )
    adapted = shard_subject_function(
        function=tiled,
        subject_arg_names=tiled.subject_shard_arg_names,
        arguments={"state": state, "optional": empty},
        static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
        devices=devices,
        subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
    )
    result = jax.jit(adapted)(state=state, optional=empty)
    np.testing.assert_array_equal(result, np.arange(16, dtype=np.float32) + 1)
    assert result.sharding.spec == jax.P("X")
    assert tuple(result.sharding.mesh.devices.flat) == devices
    assert [shard.data.shape for shard in result.addressable_shards] == [(8,), (8,)]


def test_empty_subject_subtrees_cannot_borrow_shared_array_extent() -> None:
    """A shared array cannot supply the missing subject dimension."""
    devices = _devices(count=2)
    with pytest.raises(ValueError, match="same leading extent"):
        shard_subject_function(
            function=_increment_with_optional,
            subject_arg_names=("optional",),
            arguments={
                "optional": {"nested": ({}, [], None)},
                "state": jax.ShapeDtypeStruct((16,), jnp.float32),
            },
            static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
            devices=devices,
            subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
        )


def test_empty_subject_subtrees_do_not_supply_missing_argument() -> None:
    devices = _devices(count=2)
    with pytest.raises(ValueError, match="'missing' is missing"):
        shard_subject_function(
            function=_increment_with_optional,
            subject_arg_names=("state", "optional", "missing"),
            arguments={
                "state": jax.ShapeDtypeStruct((16,), jnp.float32),
                "optional": {},
            },
            static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
            devices=devices,
            subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
        )


@pytest.mark.parametrize(
    "bad_leaf",
    [
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((0,), jnp.float32),
        jax.ShapeDtypeStruct((15,), jnp.float32),
        0.0,
    ],
    ids=["scalar-array", "zero-axis", "different-extent", "nonarray-scalar"],
)
def test_empty_subject_subtrees_do_not_hide_invalid_leaf(
    *, bad_leaf: ArgumentTree
) -> None:
    devices = _devices(count=2)
    with pytest.raises(ValueError, match=r"(?i)subject"):
        shard_subject_function(
            function=_increment_with_optional,
            subject_arg_names=("state", "optional"),
            arguments={
                "state": jax.ShapeDtypeStruct((16,), jnp.float32),
                "optional": {"empty": {}, "invalid": bad_leaf},
            },
            static_kwargs={SUBJECT_WIDTH_KEYWORD: 3},
            devices=devices,
            subject_width_keyword=SUBJECT_WIDTH_KEYWORD,
        )
