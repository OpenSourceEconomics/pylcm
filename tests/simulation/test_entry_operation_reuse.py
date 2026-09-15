"""Entry code survives calls while owners, data and budget permission do not."""

import gc
import json
import sys
import weakref
from types import MappingProxyType
from typing import Any

import cloudpickle
import jax
import numpy as np
import pytest

from _lcm.simulation import chunk_admission
from _lcm.simulation.entry_allocations import (
    SimulationEntryAllocations,
    _pad_initial_leaf,
)
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from benchmarks.asv._compile_counters import count_compile_requests
from lcm.exceptions import ExecutionPlanningError
from tests.simulation.test_independent_outer_cohorts import _model


def _initial(*, count: int, shift: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "wealth": 20.0 + np.arange(count)[::-1] + shift,
        "kind": np.arange(count, dtype=np.int32) % 3,
        "age": np.zeros(count),
        "regime_id": np.zeros(count, dtype=np.int32),
    }


def test_padded_public_calls_reuse_entry_code_and_accept_new_shapes(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    if jax.default_backend() != "cpu" or jax.device_count() < 3:
        pytest.skip("Requires three actual CPU devices")
    model = _model(devices=(0, 1, 2), width=2)
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    frontier = chunk_admission._independent_outer_candidates

    def force_anchor(**call: Any) -> tuple[int, ...]:
        return (frontier(**call)[0],)

    monkeypatch.setattr(chunk_admission, "_independent_outer_candidates", force_anchor)
    compile_candidate = ProfiledSimulationOperations.compile_candidate
    misses = []

    def observe_compile(self: ProfiledSimulationOperations, **call: Any) -> Any:
        if call["function"] is _pad_initial_leaf and call["key"] not in self.cache:
            misses.append(
                {
                    "function": "_pad_initial_leaf",
                    "arguments": [
                        {"shape": tuple(leaf.shape), "dtype": str(leaf.dtype)}
                        for leaf in jax.tree.leaves(call["arguments"])
                    ],
                    "static": dict(call["static_arguments"]),
                }
            )
        return compile_candidate(self, **call)

    monkeypatch.setattr(
        ProfiledSimulationOperations, "compile_candidate", observe_compile
    )
    cache_sizes = []
    for count, cold_expected in ((13, True), (16, True), (13, False)):
        initial = _initial(count=count)
        with count_compile_requests() as first_counts:
            first = model.simulate(
                params=params,
                solution=solution,
                initial_conditions=initial,
                log_level="off",
            )
            jax.block_until_ready(first.raw_results)
        assert (first_counts.compile_requests > 0) is cold_expected
        misses.clear()
        changed = _initial(count=count, shift=0.5)
        with count_compile_requests() as warm_counts:
            warm = model.simulate(
                params=params,
                solution=solution,
                initial_conditions=changed,
                log_level="off",
            )
            jax.block_until_ready(warm.raw_results)
        diagnostic = {
            "population": count,
            "entry_cache_misses": misses.copy(),
            "backend_requests": warm_counts.compile_requests,
            "lowering_requests": warm_counts.lowering_requests,
        }
        sys.stdout.write(json.dumps(diagnostic) + "\n")
        sys.stdout.flush()
        # Baseline must fail on actual repeated entry compilation, before the
        # new private cache field is inspected.
        assert warm_counts.compile_requests == 0, diagnostic
        assert misses == [], diagnostic
        operations = model._simulate_entry_operations
        cache_sizes.append(len(operations.cache))
        np.testing.assert_array_equal(
            np.asarray(first.raw_results["working"][0].states["wealth"]),
            initial["wealth"],
        )
        np.testing.assert_array_equal(
            np.asarray(warm.raw_results["working"][0].states["wealth"]),
            changed["wealth"],
        )
    assert 0 < cache_sizes[0] < cache_sizes[1] == cache_sizes[2]


def test_shared_entry_code_releases_owners_and_rechecks_each_budget() -> None:
    model = _model(devices=(0,), width=2)
    operations = model._simulate_entry_operations
    source = jax.device_put(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    initial = MappingProxyType({"wealth": source})
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=(jax.devices()[0],),
        budget_bytes=2**20,
        operations=operations,
    )
    padded, _ = owner.pad(initial_conditions=initial, multiple=4)
    profile = next(iter(operations.cache.values()))
    references = (
        weakref.ref(source),
        weakref.ref(padded["wealth"]),
        weakref.ref(owner),
    )
    owner.close()
    del source, initial, padded, owner
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(operations.cache) == 1

    changed = jax.device_put(np.array([7.0, 8.0, 9.0], dtype=np.float32))
    fresh = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(changed,)),
        solution=None,
        model_roots=(),
        devices=(jax.devices()[0],),
        budget_bytes=profile.reservation_bytes - 1,
        operations=operations,
    )
    current = MappingProxyType({"wealth": changed})
    with count_compile_requests() as counts:
        with pytest.raises(ExecutionPlanningError, match="workspace"):
            fresh.pad(initial_conditions=current, multiple=4)
        fresh.budget_bytes = 2**20
        recovered, _ = fresh.pad(initial_conditions=current, multiple=4)
    assert counts.compile_requests == 0
    np.testing.assert_array_equal(recovered["wealth"], [7.0, 8.0, 9.0, 9.0])
    assert len(operations.cache) == 1
    fresh.close()


def test_model_pickle_resets_entry_executables() -> None:
    if jax.default_backend() != "cpu" or jax.device_count() < 3:
        pytest.skip("Requires three actual CPU devices")
    model = _model(devices=(0, 1, 2), width=2)
    params = {"discount_factor": 0.5}
    initial = _initial(count=13)
    model.simulate(params=params, initial_conditions=initial, log_level="off")
    assert model._simulate_entry_operations.cache
    assert "_simulate_entry_operations" not in model.__getstate__()
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert restored._simulate_entry_operations is not model._simulate_entry_operations
    assert restored._simulate_entry_operations.cache == {}
    assert restored._simulate_runtime_regimes == {}
    with count_compile_requests() as counts:
        result = restored.simulate(
            params=params, initial_conditions=initial, log_level="off"
        )
        jax.block_until_ready(result.raw_results)
    assert counts.compile_requests > 0
    assert restored._simulate_entry_operations.cache
    np.testing.assert_array_equal(
        np.asarray(result.raw_results["working"][0].states["wealth"]), initial["wealth"]
    )
