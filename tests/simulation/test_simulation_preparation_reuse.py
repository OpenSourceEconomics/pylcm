"""Preparation reads existing metadata; runtime ownership and admission stay fresh."""

import dataclasses
import gc
import hashlib
import weakref
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation import chunk_profile_inventory, operand_placement, process_grids
from _lcm.simulation.residency import DeviceBufferFootprint, measure_buffer_footprint
from lcm import Model
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import UserInitialConditions, UserParams
from tests.simulation.test_normal_process_grid_admission import _inputs


@pytest.fixture(scope="module")
def supplied_case() -> tuple[Model, UserParams, UserInitialConditions, Any]:
    """Use real multi-period/regime programs and a supplied, budgeted solution."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    return model, params, initial, solution


def _snapshot(result: Any) -> dict[tuple[str, int, str, int], tuple[object, ...]]:
    """Hash every published record field, not a selected economic summary."""
    snapshot = {}
    for regime, periods in result.raw_results.items():
        for period, record in periods.items():
            for field in dataclasses.fields(record):
                leaves = jax.tree.leaves(getattr(record, field.name))
                for index, leaf in enumerate(leaves):
                    inspected_leaf = leaf
                    if isinstance(inspected_leaf, jax.Array) and jax.dtypes.issubdtype(
                        inspected_leaf.dtype, jax.dtypes.prng_key
                    ):
                        inspected_leaf = jax.random.key_data(inspected_leaf)
                    array = np.asarray(inspected_leaf)
                    snapshot[regime, period, field.name, index] = (
                        array.shape,
                        array.dtype.str,
                        hashlib.sha256(array.tobytes()).hexdigest(),
                    )
    assert snapshot, "The oracle must inspect nonempty public output records."
    return snapshot


@contextmanager
def _observe_preparation(*, monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """Count structural calls while preserving their actual implementations."""
    counts: Counter[str] = Counter()
    eval_shape = jax.eval_shape
    make_jaxpr = jax.make_jaxpr
    payload = chunk_profile_inventory.payload_bytes
    operand = operand_placement._required_operand_bytes
    trace_recipe = process_grids._trace_process_jaxpr

    def shape(function: Any, *args: Any, **kwargs: Any) -> Any:
        if function is jax.random.key_data:
            counts["key_shape_traces"] += 1
        elif function is jnp.asarray:
            counts["operand_shape_traces"] += 1
        return eval_shape(function, *args, **kwargs)

    def graph(function: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            getattr(function, "__name__", None) == "_process_value_identity"
            and getattr(function, "__module__", None) == process_grids.__name__
        ):
            counts["attached_identity_traces"] += 1
        return make_jaxpr(function, *args, **kwargs)

    def declared_payload(**kwargs: Any) -> Any:
        counts["payload_reads"] += 1
        return payload(**kwargs)

    def required_operand(**kwargs: Any) -> int:
        counts["operand_reads"] += 1
        return operand(**kwargs)

    def recipe(**kwargs: Any) -> Any:
        counts["producer_graphs"] += 1
        return trace_recipe(**kwargs)

    with monkeypatch.context() as observe:
        observe.setattr(jax, "eval_shape", shape)
        observe.setattr(jax, "make_jaxpr", graph)
        observe.setattr(chunk_profile_inventory, "payload_bytes", declared_payload)
        observe.setattr(operand_placement, "_required_operand_bytes", required_operand)
        observe.setattr(process_grids, "_trace_process_jaxpr", recipe)
        yield counts


def test_public_repeat_preserves_outputs(supplied_case: Any) -> None:
    """Positive readiness case: genuine public simulation and complete output parity."""
    model, params, initial, solution = supplied_case
    first = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=11,
    )
    second = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=11,
    )
    assert _snapshot(first) == _snapshot(second)


def test_operand_low_budget_refuses_before_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative readiness case: a real oversized host operand cannot allocate."""
    device = jax.devices()[0]
    attempts = []

    def forbidden_placement(*, leaf: object, sharding: jax.sharding.Sharding) -> object:
        del sharding
        attempts.append(leaf)
        raise AssertionError("Operand placement ran before admission.")

    monkeypatch.setattr(operand_placement, "_place_operand_leaf", forbidden_placement)
    with pytest.raises(ExecutionPlanningError, match="before allocation"):
        operand_placement.place_simulation_arguments(
            arguments={"state": np.ones(8, dtype=np.float64)},
            subject_arg_names=("state",),
            value_reads=(),
            devices=(device,),
            budget_bytes=1,
            live_footprint=DeviceBufferFootprint(spans={}),
            budget_devices=(device,),
        )
    assert attempts == []


def test_public_supplied_simulation_does_not_retrace_metadata(
    *,
    supplied_case: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm public call uses metadata, not repeatedly retraced size/identity code."""
    model, params, initial, solution = supplied_case
    expected = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=19,
    )
    with _observe_preparation(monkeypatch=monkeypatch) as counts:
        actual = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            log_level="off",
            seed=19,
        )
    assert _snapshot(actual) == _snapshot(expected)
    assert counts["payload_reads"] > 1
    assert counts["operand_reads"] > 1
    assert counts["producer_graphs"] == 0  # This fixture uses explicit normal stages.
    assert {
        name: counts[name]
        for name in (
            "key_shape_traces",
            "operand_shape_traces",
            "attached_identity_traces",
        )
    } == {
        "key_shape_traces": 0,
        "operand_shape_traces": 0,
        "attached_identity_traces": 0,
    }, f"Redundant abstract preparation in one supplied simulation: {dict(counts)}"


def test_attached_value_contract_accepts_then_rejects() -> None:
    """Check real graph binders before running the structural counterexample."""
    value = np.ones(3, dtype=np.float64)
    variable = jax.make_jaxpr(lambda array: array)(value).jaxpr.invars[0]
    process_grids._validate_attached_process_value(
        value=value,
        variable=variable,
        host_constant=True,
    )
    with pytest.raises(ExecutionPlanningError, match="traced input contract"):
        process_grids._validate_attached_process_value(
            value=np.ones(4, dtype=np.float64),
            variable=variable,
            host_constant=True,
        )


def test_attached_values_are_checked_without_identity_retracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh comparisons remain necessary; constructing identity graphs does not."""
    value = np.ones(3, dtype=np.float64)
    variable = jax.make_jaxpr(lambda array: array)(value).jaxpr.invars[0]
    with _observe_preparation(monkeypatch=monkeypatch) as counts:
        for updated in (value, value + 1):
            process_grids._validate_attached_process_value(
                value=updated,
                variable=variable,
                host_constant=True,
            )
        with pytest.raises(ExecutionPlanningError, match="traced input contract"):
            process_grids._validate_attached_process_value(
                value=np.ones(4, dtype=np.float64),
                variable=variable,
                host_constant=True,
            )
    assert counts["attached_identity_traces"] == 0


@pytest.mark.parametrize("shape", [(), (0,), (1,), (3,), (2, 3), (0, 4)])
@pytest.mark.parametrize(
    "kind", ["float64", "int32", "bool", "threefry2x32", "rbg", "unsafe_rbg"]
)
def test_payload_storage_matches_materialized_shards(
    *,
    shape: tuple[int, ...],
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw storage independently sizes keys; duplicate future owners stay charged."""
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    if kind in {"threefry2x32", "rbg", "unsafe_rbg"}:
        value = jax.random.split(jax.random.key(71, impl=kind), shape)
        before = np.asarray(jax.random.key_data(value)).tobytes()
    else:
        value = jnp.zeros(shape, dtype=kind)
        before = np.asarray(value).tobytes()
    value = jax.device_put(value, sharding)
    metadata = jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)
    expected = {}
    for shard in value.addressable_shards:
        raw = (
            jax.random.key_data(shard.data)
            if jax.dtypes.issubdtype(shard.data.dtype, jax.dtypes.prng_key)
            else shard.data
        )
        expected[shard.device] = 3 * np.asarray(raw).nbytes
    with _observe_preparation(monkeypatch=monkeypatch) as counts:
        actual = chunk_profile_inventory.payload_bytes(
            tree={"first": metadata, "alias": (metadata, [metadata])}
        )
    assert actual == expected
    assert counts["key_shape_traces"] == 0
    after = (
        jax.random.key_data(value)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        else value
    )
    assert np.asarray(after).tobytes() == before


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "bool",
        "int",
        "float",
        "complex",
        "numpy-scalar",
        "numpy-vector",
        "noncontiguous",
        "empty",
        "device",
    ],
)
def test_operand_bytes_match_actual_canonical_conversion(
    *,
    x64: bool,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialization, not the optimized abstraction algorithm, is the oracle."""
    with jax.enable_x64(x64):
        values = {
            "bool": True,
            "int": 9,
            "float": -0.0,
            "complex": complex(1, -2),
            "numpy-scalar": np.float64(3),
            "numpy-vector": np.arange(5, dtype=np.int64),
            "noncontiguous": np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2],
            "empty": np.empty((2, 0), dtype=np.float64),
            "device": jnp.asarray([1.0, 2.0]),
        }
        value = values[kind]
        canonical = jnp.asarray(value)
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        with _observe_preparation(monkeypatch=monkeypatch) as counts:
            actual = operand_placement._required_operand_bytes(
                leaf=value,
                sharding=sharding,
            )
        assert actual == np.asarray(canonical).nbytes
        assert counts["operand_shape_traces"] == 0


@pytest.mark.parametrize("change", ["shape", "dtype", "weak", "host-owner"])
def test_attached_contract_checks_each_new_value(change: str) -> None:
    """A previous valid attachment never authorizes a mismatched next one."""
    value = np.asarray(1.0, dtype=np.float64)
    variable = jax.make_jaxpr(lambda array: array)(value).jaxpr.invars[0]
    process_grids._validate_attached_process_value(
        value=value,
        variable=variable,
        host_constant=True,
    )
    changed = {
        "shape": np.asarray([1.0], dtype=np.float64),
        "dtype": np.asarray(1.0, dtype=np.float32),
        "weak": 1.0,
        "host-owner": jnp.asarray(value),
    }[change]
    with pytest.raises(ExecutionPlanningError):
        process_grids._validate_attached_process_value(
            value=changed,
            variable=variable,
            host_constant=True,
        )


@pytest.mark.parametrize("x64", [False, True])
def test_attached_metadata_matches_independently_traced_binders(*, x64: bool) -> None:
    """Bounded promotion/value/shape family includes zeros and adjacent floats."""
    with jax.enable_x64(x64):
        values = (
            False,
            0,
            1,
            -1,
            0.0,
            -0.0,
            np.nextafter(1.0, 2.0),
            np.float32(0),
            np.float64(3),
            np.int64(7),
            complex(2, -1),
            np.asarray([0.0, -0.0, 1.0]),
            np.empty((0, 2)),
            jnp.asarray(1.0),
            jnp.asarray([1.0, 2.0]),
        )
        for value in values:
            variable = jax.make_jaxpr(lambda array: array)(value).jaxpr.invars[0]
            process_grids._validate_attached_process_value(
                value=value,
                variable=variable,
                host_constant=not isinstance(value, jax.Array),
            )


def test_operand_admission_rechecks_growing_live_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same argument metadata is admissible first, but not with a new live owner."""
    device = jax.devices()[0]
    value = np.arange(8, dtype=np.float64)
    original = operand_placement._place_operand_leaf
    attempts = []

    def observed(*, leaf: object, sharding: jax.sharding.Sharding) -> object:
        attempts.append(leaf)
        return original(leaf=leaf, sharding=sharding)

    monkeypatch.setattr(operand_placement, "_place_operand_leaf", observed)
    kwargs = {
        "arguments": {"state": value},
        "subject_arg_names": ("state",),
        "value_reads": (),
        "devices": (device,),
        "budget_bytes": 256,
        "budget_devices": (device,),
    }
    first = operand_placement.place_simulation_arguments(
        **kwargs,
        live_footprint=DeviceBufferFootprint(spans={}),
    )
    np.testing.assert_array_equal(first["state"], value)
    assert len(attempts) == 1
    blocking = jax.device_put(np.ones(128, dtype=np.float64), device)
    current = measure_buffer_footprint(tree={"new-owner": blocking})
    with pytest.raises(ExecutionPlanningError, match="before allocation"):
        operand_placement.place_simulation_arguments(**kwargs, live_footprint=current)
    assert len(attempts) == 1


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_operand_admission_exact_boundary_and_occurrences(delta: int) -> None:
    """Destination plus scratch charges both future occurrences of a host leaf."""
    device = jax.devices()[0]
    value = np.arange(3, dtype=np.float64)
    required = 4 * value.nbytes  # two copies, each destination plus scratch
    kwargs = {
        "arguments": {"a": value, "b": value},
        "subject_arg_names": ("a", "b"),
        "value_reads": (),
        "devices": (device,),
        "budget_bytes": required + delta,
        "budget_devices": (device,),
        "live_footprint": DeviceBufferFootprint(spans={}),
    }
    if delta < 0:
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            operand_placement.place_simulation_arguments(**kwargs)
    else:
        placed = operand_placement.place_simulation_arguments(**kwargs)
        np.testing.assert_array_equal(placed["a"], value)
        np.testing.assert_array_equal(placed["b"], value)


def test_metadata_paths_under_jit_vmap_and_scan() -> None:
    """Static metadata costs follow the transformed logical axes, not stored values."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def costs(value: jax.Array) -> jax.Array:
        abstract = jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)
        return jnp.asarray(
            [
                chunk_profile_inventory.payload_bytes(tree=abstract)[jax.devices()[0]],
                operand_placement._required_operand_bytes(
                    leaf=value, sharding=sharding
                ),
            ]
        )

    value = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    total = np.asarray(value).nbytes
    row = np.asarray(value[0]).nbytes
    np.testing.assert_array_equal(costs(value), [total, total])
    np.testing.assert_array_equal(jax.jit(costs)(value), [total, total])
    expected = np.tile([row, row], (3, 1))
    np.testing.assert_array_equal(jax.jit(jax.vmap(costs))(value), expected)
    scan_costs = jax.jit(
        lambda x: jax.lax.scan(lambda carry, row: (carry, costs(row)), None, x)
    )
    _, scanned = scan_costs(value)
    np.testing.assert_array_equal(scanned, expected)


@pytest.mark.parametrize("subjects", [1, 3, 7])
@pytest.mark.parametrize("seed", [11, 19])
def test_public_shape_seed_and_state_mutations(
    *,
    supplied_case: Any,
    subjects: int,
    seed: int,
) -> None:
    """A scalar Bellman oracle fixes choices/values independently of preparation."""
    model, params, initial, solution = supplied_case
    changed = {name: jnp.repeat(value, subjects) for name, value in initial.items()}
    changed["income"] = jnp.linspace(1.75, 2.25, subjects)
    first = model.simulate(
        params=params,
        initial_conditions=changed,
        solution=solution,
        seed=seed,
        log_level="off",
    )
    second = model.simulate(
        params=params,
        initial_conditions=changed,
        solution=solution,
        seed=seed,
        log_level="off",
    )
    assert _snapshot(first) == _snapshot(second)
    current = first.raw_results["alive"][0]
    assert set(current.states) == {"income"}
    assert set(current.actions) == {"saving"}
    expected = np.asarray([float(income) + 1.0 for income in changed["income"]])
    np.testing.assert_array_equal(current.V_arr, expected)
    np.testing.assert_array_equal(current.actions["saving"], np.ones(subjects))
    np.testing.assert_array_equal(current.in_regime, np.ones(subjects, dtype=bool))
    np.testing.assert_array_equal(
        first.raw_results["done"][1].in_regime, np.ones(subjects, dtype=bool)
    )
    for regime, periods in first.raw_results.items():
        for period, record in periods.items():
            for field in dataclasses.fields(record):
                left = getattr(record, field.name)
                right = getattr(second.raw_results[regime][period], field.name)
                assert jax.tree.structure(left) == jax.tree.structure(right)


def test_metadata_reads_do_not_keep_concrete_owners() -> None:
    """A call-local abstraction must not extend a host or device array's lifetime."""

    def visit() -> tuple[weakref.ReferenceType[Any], weakref.ReferenceType[Any]]:
        host = np.arange(7, dtype=np.float64)
        device = jax.device_put(host)
        sharding = device.sharding
        for leaf in (host, device):
            operand_placement._required_operand_bytes(leaf=leaf, sharding=sharding)
            variable = jax.make_jaxpr(lambda x: x)(leaf).jaxpr.invars[0]
            process_grids._validate_attached_process_value(
                value=leaf,
                variable=variable,
                host_constant=leaf is host,
            )
        return weakref.ref(host), weakref.ref(device)

    references = visit()
    gc.collect()
    assert all(reference() is None for reference in references)
