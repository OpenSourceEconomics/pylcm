"""Independent outer cohorts preserve rows, streams and profiled program shapes.

Run in a fresh eight-CPU-device process to cover actual one/three/eight-device
subsets. This module never changes JAX topology or imports topology-pinning tests.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import _lcm.simulation.runtime as simulation_runtime
import _lcm.simulation.simulate as simulation
from _lcm.simulation import chunk_admission
from _lcm.simulation.random import _generate_windowed_simulation_keys
from _lcm.simulation.runtime import SimulationRuntime
from _lcm.simulation.taste_stream import prepare_decision_taste_keys
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ScalarInt
from tests.test_models import n_nbegm_toy as toy


@categorical(ordered=False)
class _Kind:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


def _model(*, devices: tuple[int, ...], width: int | None) -> Model:
    return Model(
        regimes={
            "working": Regime(
                active=lambda age: age < 2,
                transition=lambda age: jnp.where(
                    age < 1, _RegimeId.working, _RegimeId.retired
                ),
                states={"wealth": LinSpacedGrid(start=1, stop=40, n_points=4)},
                actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
                functions={
                    "utility": lambda wealth, consumption, kind: (
                        -((consumption - (kind + 1)) ** 2) + wealth * 0.125
                    )
                },
                state_transitions={
                    "wealth": lambda wealth, consumption: wealth - consumption
                },
            ),
            "retired": Regime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=1, stop=40, n_points=4)},
                functions={"utility": lambda wealth, kind: wealth * (kind + 1)},
            ),
        },
        states={"kind": DiscreteGrid(_Kind)},
        state_transitions={"kind": fixed_transition("kind")},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=("kind",),
            axis_widths={} if width is None else {"subject": width},
            device_memory_bytes=2**30,
        ),
    )


def _shape_tree(tree: object) -> object:
    return jax.tree.map(lambda leaf: (leaf.shape, str(leaf.dtype)), tree)


def _assert_raw_equal(*, actual: Any, expected: Any) -> None:
    """Compare original raw row order, masks, values and every published field."""
    assert jax.tree.structure(actual.raw_results) == jax.tree.structure(
        expected.raw_results
    )
    for got, want in zip(
        jax.tree.leaves(actual.raw_results),
        jax.tree.leaves(expected.raw_results),
        strict=True,
    ):
        got_array, want_array = np.asarray(got), np.asarray(want)
        if np.issubdtype(want_array.dtype, np.inexact):
            eps = 8 * np.finfo(want_array.dtype).eps
            np.testing.assert_allclose(got_array, want_array, rtol=eps, atol=eps)
        else:
            np.testing.assert_array_equal(got_array, want_array)


def _install_profile_observers(
    *,
    monkeypatch: pytest.MonkeyPatch,
    profiled_programs: set,
    profile_shapes: dict,
    profile_layouts: dict,
    profile_widths: list,
) -> None:
    """Record every declared body of the chunk profile each call selects.

    A budgeted call that reuses an identical abstract chunk profile does not
    recompile its declared bodies, so `prepare_abstract` fires only on the call
    that builds a given profile. Keep each profile's own build records and
    replay exactly those on every later call that selects it, warm or cold: the
    guarded contract is the profile a call dispatches against, never whether
    that profile happened to be compiled during this call.
    """
    prepare = SimulationRuntime.prepare_abstract
    profile_chunk = chunk_admission._ChunkProfiler.profile_widths
    build_records: list[dict[str, Any]] = []
    building: list[bool] = []
    profile_builds: dict[int, tuple[Any, tuple[dict[str, Any], ...]]] = {}

    def apply_records(records: tuple[dict[str, Any], ...]) -> None:
        for record in records:
            profiled_programs.add(record["declaration"])
            profile_shapes[record["executable"]] = record["shapes"]
            profile_layouts[record["executable"]] = record["layouts"]
            profile_widths.append(record["widths"])

    def record_profile(self: SimulationRuntime, **call: Any) -> Any:
        prepared = prepare(self, **call)
        executable = prepared.executable
        assert isinstance(executable, jax.stages.Compiled)
        record = {
            "declaration": (
                id(self),
                id(call["program"].function),
                call["period"],
                call["n_subjects"],
            ),
            "executable": id(executable),
            "shapes": _shape_tree(executable.out_info),
            "layouts": tuple(
                leaf.sharding for leaf in jax.tree.leaves(executable.out_info)
            ),
            "widths": dict(call["widths"]),
        }
        if building:
            build_records.append(record)
        else:
            apply_records((record,))
        return prepared

    def record_chunk_profile(self: Any, **call: Any) -> Any:
        build_records.clear()
        building.append(True)
        try:
            profile = profile_chunk(self, **call)
        finally:
            building.pop()
        # Hold the profile itself, so its identity cannot be recycled while its
        # records are still replayable.
        _, records = profile_builds.setdefault(
            id(profile), (profile, tuple(build_records))
        )
        build_records.clear()
        apply_records(records)
        return profile

    monkeypatch.setattr(SimulationRuntime, "prepare_abstract", record_profile)
    monkeypatch.setattr(
        chunk_admission._ChunkProfiler, "profile_widths", record_chunk_profile
    )


def _observe_execution(
    *, monkeypatch: pytest.MonkeyPatch, device_count: int
) -> tuple[list[dict[str, int]], list[tuple[slice, int, int]]]:
    """Check required subject inputs and compiler-profiled output layouts."""
    profile_shapes: dict[int, object] = {}
    profiled_programs = set()
    profile_layouts = {}
    profile_widths = []
    chunks = []
    placed_subjects = []
    dispatch = SimulationRuntime.dispatch
    compiled_call = simulation_runtime.CompiledSimulationProgram.__call__
    place = simulation_runtime.place_simulation_arguments
    run_chunk = simulation._simulate_subject_chunk
    _install_profile_observers(
        monkeypatch=monkeypatch,
        profiled_programs=profiled_programs,
        profile_shapes=profile_shapes,
        profile_layouts=profile_layouts,
        profile_widths=profile_widths,
    )

    def record_dispatch(self: SimulationRuntime, **call: Any) -> Any:
        declaration = (
            id(self),
            id(call["program"].function),
            call["period"],
            call["n_subjects"],
        )
        assert declaration in profiled_programs, (
            "Dispatch preceded its declared-body profile"
        )
        return dispatch(self, **call)

    def record_compiled(
        self: simulation_runtime.CompiledSimulationProgram, **call: Any
    ) -> Any:
        # Generic program names repeat across regimes. The exact selected
        # executable also identifies its width/layout specialization.
        key = id(self.executable)
        assert key in profile_shapes, "Dispatch selected an unprofiled executable"
        result = compiled_call(self, **call)
        assert _shape_tree(result) == profile_shapes[key]
        _assert_output_shardings(result=result, expected=profile_layouts[key])
        return result

    def record_placement(**call: Any) -> Any:
        placed = place(**call)
        for name in call["subject_arg_names"]:
            for value in jax.tree.leaves(placed.get(name, ())):
                if isinstance(value, jax.Array) and value.ndim:
                    _assert_subject_rows(value=value, device_count=device_count)
                    placed_subjects.append(value.shape[0])
        return placed

    def record_chunk(**call: Any) -> Any:
        chunks.append(
            (call["subject_slice"], call["n_subjects"], call["original_n_subjects"])
        )
        assert (
            len(call["initial_regime_ids"])
            == call["subject_slice"].stop - call["subject_slice"].start
        )
        before = len(placed_subjects)
        result = run_chunk(**call)
        assert len(placed_subjects) > before
        if call["subject_slice"].stop - call["subject_slice"].start >= 2048:
            # Preserve the observed real-alignment witness separately from the
            # general contract, which does not prescribe every V output layout.
            _assert_subject_rows(
                value=result["working"][0].V_arr, device_count=device_count
            )
        return result

    monkeypatch.setattr(SimulationRuntime, "dispatch", record_dispatch)
    monkeypatch.setattr(
        simulation_runtime.CompiledSimulationProgram, "__call__", record_compiled
    )
    monkeypatch.setattr(
        simulation_runtime, "place_simulation_arguments", record_placement
    )
    monkeypatch.setattr(simulation, "_simulate_subject_chunk", record_chunk)
    return profile_widths, chunks


def _assert_output_shardings(*, result: object, expected: tuple) -> None:
    for value, sharding in zip(jax.tree.leaves(result), expected, strict=True):
        assert sharding is not None
        assert value.sharding.is_equivalent_to(sharding, value.ndim)


def _assert_subject_rows(*, value: jax.Array, device_count: int) -> None:
    """Require actual nonempty, disjoint shards covering every global row."""
    assert len(value.addressable_shards) == device_count
    assert {shard.device.id for shard in value.addressable_shards} == set(
        range(device_count)
    )
    rows = []
    for shard in value.addressable_shards:
        indices = np.arange(value.shape[0])[shard.index[0]]
        assert len(indices) > 0
        rows.extend(indices.tolist())
    assert sorted(rows) == list(range(value.shape[0]))


@pytest.mark.parametrize(
    ("device_count", "count", "width"),
    [
        (1, 6, 2),
        (3, 7, 2),
        (8, 17, 5),
        (1, 1, 2),
        (3, 1, 2),
        (8, 1, 5),
        (3, 4100, 2048),
    ],
)
def test_independent_outer_cohorts_preserve_profiled_shapes_and_real_rows(
    *, device_count: int, count: int, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if jax.default_backend() != "cpu" or jax.device_count() < device_count:
        pytest.skip("Requires the selected number of actual CPU devices")
    model = _model(devices=tuple(range(device_count)), width=width)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    initial = {
        # Deliberately unsorted, positive rows: zero-padding is not equivalent.
        "wealth": 20.0 + (np.arange(count)[::-1] % 19),
        "kind": np.arange(count, dtype=np.int32) % 3,
        "age": np.zeros(count),
        "regime_id": np.zeros(count, dtype=np.int32),
    }
    frontier = chunk_admission._independent_outer_candidates
    profile_widths, chunks = _observe_execution(
        monkeypatch=monkeypatch, device_count=device_count
    )
    results = []
    maps = []
    for arm in (0, -1):
        chunks.clear()
        profile_widths.clear()

        def force_candidate(*, index: int = arm, **call: Any) -> tuple[int, ...]:
            return (frontier(**call)[index],)

        with monkeypatch.context() as forced:
            forced.setattr(
                chunk_admission, "_independent_outer_candidates", force_candidate
            )
            result = model.simulate(
                params=params,
                solution=solution,
                initial_conditions=initial,
                seed=17,
                log_level="off",
            )
        results.append(result)
        extent = chunks[0][0].stop
        entry_count = -(-count // device_count) * device_count
        candidate = min(width, entry_count) if arm == 0 else entry_count
        expected_extent = -(-candidate // device_count) * device_count
        assert extent == expected_extent
        padded = -(-count // extent) * extent
        assert chunks == [
            (slice(start, start + extent), padded, count)
            for start in range(0, padded, extent)
        ]
        assert profile_widths
        assert all(
            item.get("subject", min(width, extent)) == min(width, extent)
            for item in profile_widths
        )
        maps.append({tuple(sorted(item.items())) for item in profile_widths})
        assert result.n_subjects == count
        np.testing.assert_array_equal(
            np.asarray(result.raw_results["working"][0].states["wealth"]),
            initial["wealth"],
        )
        assert not np.asarray(result.raw_results["retired"][0].in_regime).any()
    assert maps[0] == maps[1]
    _assert_raw_equal(actual=results[1], expected=results[0])
    for targets in (None, ["utility"]):
        left = results[0].to_dataframe(use_labels=False, additional_targets=targets)
        right = results[1].to_dataframe(use_labels=False, additional_targets=targets)
        eps = (
            8 * np.finfo(np.asarray(next(iter(solution.values[0].values()))).dtype).eps
        )
        pd.testing.assert_frame_equal(left, right, rtol=eps, atol=eps)


def test_scalar_anchor_preserves_subject_pin_for_larger_outer_candidate(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scalar anchor must retain the pin needed by the largest candidate."""
    if jax.default_backend() != "cpu":
        pytest.skip("Requires an actual CPU device")
    model = _model(devices=(0,), width=1)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    profiles, chunks = _observe_execution(monkeypatch=monkeypatch, device_count=1)
    receipts = []
    original_plan = chunk_admission._plan_independent_chunks

    def record_plan(**call: Any) -> Any:
        plan = original_plan(**call)
        receipts.append(plan.receipt)
        return plan

    monkeypatch.setattr(chunk_admission, "_plan_independent_chunks", record_plan)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": np.array([20.0, 21.0, 22.0]),
            "kind": np.array([0, 1, 2], dtype=np.int32),
            "age": np.zeros(3),
            "regime_id": np.zeros(3, dtype=np.int32),
        },
        seed=17,
        log_level="off",
    )
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.candidates == (1, 2, 3)
    assert receipt.selected_subjects == 3
    assert dict(receipt.axis_widths)["subject"] == 1
    assert [attempt.n_subjects for attempt in receipt.attempts] == [3]
    assert all(
        dict(attempt.axis_widths)["subject"] == 1 for attempt in receipt.attempts
    )
    assert profiles
    assert any("subject" in widths for widths in profiles)
    assert all(widths["subject"] == 1 for widths in profiles if "subject" in widths)
    assert chunks == [(slice(0, 3), 3, 3)]
    assert result.n_subjects == 3
    np.testing.assert_array_equal(
        result.raw_results["working"][0].states["wealth"], [20.0, 21.0, 22.0]
    )


def test_independent_frontier_preserves_2048_inner_width_on_three_devices() -> None:
    assert chunk_admission._independent_outer_candidates(
        population=226848, alignment=3, subject_width=2048
    ) == (2049, 4098, 8193, 16386, 32769, 65538, 131073, 226848)


@pytest.mark.parametrize("partitionable", [False, True])
@pytest.mark.parametrize("impl", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_changed_padded_population_preserves_actual_stochastic_and_taste_keys(
    *, impl: str, partitionable: bool
) -> None:
    """Compare key bits per original global row, including duplicate-last tails."""
    original_count = 7
    key = jax.random.key(19, impl=impl)
    taste_key = jax.random.key(31, impl="threefry2x32")
    ordinary = jax.jit(
        _generate_windowed_simulation_keys,
        static_argnames=(
            "names",
            "n_initial_states",
            "original_n_subjects",
            "partitionable",
            "width",
        ),
    )
    outputs = []
    with jax.threefry_partitionable(partitionable):
        for extent in (3, 6):
            padded = -(-original_count // extent) * extent
            rows = {
                name: []
                for name in (
                    "key_state",
                    "key_next_regime",
                    "ordinary_taste",
                    "independent_taste",
                )
            }
            carries = []
            for start in range(0, padded, extent):
                carry, named = ordinary(
                    key=key,
                    start=np.int32(start),
                    names=("state", "next_regime"),
                    n_initial_states=padded,
                    original_n_subjects=original_count,
                    partitionable=partitionable,
                    width=extent,
                )
                carries.append(np.asarray(jax.random.key_data(carry)))
                for name, value in named.items():
                    rows[name].append(np.asarray(jax.random.key_data(value)))
                for label, root in (
                    ("ordinary_taste", None),
                    ("independent_taste", taste_key),
                ):
                    taste_carry, taste = prepare_decision_taste_keys(
                        key=carry,
                        taste_key=root,
                        taste_address=tuple(range(8)),
                        n_subjects=padded,
                        subject_slice=slice(start, start + extent),
                        original_n_subjects=original_count,
                        memory=None,
                    )
                    rows[label].append(np.asarray(jax.random.key_data(taste)))
                    carries.append(np.asarray(jax.random.key_data(taste_carry)))
            bank = {name: np.concatenate(parts) for name, parts in rows.items()}
            for value in bank.values():
                np.testing.assert_array_equal(
                    value[original_count:],
                    np.repeat(
                        value[original_count - 1 : original_count],
                        padded - original_count,
                        axis=0,
                    ),
                )
            outputs.append((bank, carries[:3]))
    for name in outputs[0][0]:
        np.testing.assert_array_equal(
            outputs[0][0][name][:original_count], outputs[1][0][name][:original_count]
        )
    for left, right in zip(outputs[0][1], outputs[1][1], strict=True):
        np.testing.assert_array_equal(left, right)


def test_finite_replay_outer_extent_matches_profile_without_changing_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual finite prepare/rank banks scale with C while inner width stays one."""
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        execution_config=ExecutionConfig(
            devices=(0,),
            axis_widths={"subject": 1},
            device_memory_bytes=2**30,
        ),
    )
    params = {"discount_factor": 0.95}
    solution = model.solve(params=params, log_level="off")
    initial = {
        "wealth": np.array([1.0467, 4.3, 2.5]),
        "illiquid": np.array([2.04, 1.37, 1.5]),
        "age": np.full(3, 20.0),
        "regime_id": np.zeros(3, dtype=np.int32),
    }
    profiles = {}
    observed = set()
    prepare = SimulationRuntime.prepare_abstract
    dispatch = SimulationRuntime.dispatch

    def record_profile(self: SimulationRuntime, **call: Any) -> Any:
        result = prepare(self, **call)
        assert isinstance(result.executable, jax.stages.Compiled)
        profiles[(id(call["program"].function), call["period"], call["n_subjects"])] = (
            _shape_tree(result.executable.out_info)
        )
        return result

    def record_dispatch(self: SimulationRuntime, **call: Any) -> Any:
        result = dispatch(self, **call)
        name = call["program"].name
        if name in {"simulate_policy_prepare", "simulate_policy_rank"}:
            assert call["residency"].axis_widths["subject"] == 1
            assert (
                _shape_tree(result)
                == profiles[
                    (id(call["program"].function), call["period"], call["n_subjects"])
                ]
            )
            observed.add((name, call["n_subjects"]))
        return result

    monkeypatch.setattr(SimulationRuntime, "prepare_abstract", record_profile)
    monkeypatch.setattr(SimulationRuntime, "dispatch", record_dispatch)
    frontier = chunk_admission._independent_outer_candidates
    results = []
    for arm in (0, -1):

        def force_candidate(*, index: int = arm, **call: Any) -> tuple[int, ...]:
            return (frontier(**call)[index],)

        with monkeypatch.context() as forced:
            forced.setattr(
                chunk_admission, "_independent_outer_candidates", force_candidate
            )
            results.append(
                model.simulate(
                    params=params,
                    solution=solution,
                    initial_conditions=initial,
                    seed=17,
                    log_level="off",
                )
            )
    assert observed == {
        (name, extent)
        for name in ("simulate_policy_prepare", "simulate_policy_rank")
        for extent in (1, 3)
    }
    _assert_raw_equal(actual=results[1], expected=results[0])


def test_auto_anchor_admits_a_budgeted_call_without_an_explicit_subject_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget alone admits an outer cohort; no `simulation_chunk_policy` exists."""
    model = _model(devices=(0,), width=None)
    receipts = []
    original_plan = chunk_admission._plan_independent_chunks

    def record_plan(**call: Any) -> Any:
        plan = original_plan(**call)
        receipts.append(plan.receipt)
        return plan

    monkeypatch.setattr(chunk_admission, "_plan_independent_chunks", record_plan)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.array([20.0, 21.0, 22.0]),
            "kind": jnp.array([0, 1, 2], dtype=jnp.int32),
            "age": jnp.zeros(3),
            "regime_id": jnp.zeros(3, dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    )
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.selected_subjects > 0
    assert receipt.selected_subjects % receipt.alignment == 0 or (
        receipt.selected_subjects == 3
    )
    assert dict(receipt.axis_widths)["subject"] > 0
    assert result.n_subjects == 3
