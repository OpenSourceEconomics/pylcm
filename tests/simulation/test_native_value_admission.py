"""Native value archives retain lazy ownership under an explicit device budget."""

import gc
import threading
import weakref
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partialmethod
from pathlib import Path
from typing import Any, cast

import h5py
import jax
import jax.stages
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
from _lcm.persistence import solution as archive
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.residency import measure_buffer_footprint, resident_bytes_by_device
from _lcm.solution.native_values import NativeValueMaterializer
from lcm import ExecutionConfig
from lcm.exceptions import (
    ExecutionPlanningError,
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    SolutionIntegrityError,
)
from lcm.persistence import load_solution
from lcm.solver_api import LoadState, SolutionResult, ValueStore
from tests.simulation.test_foreign_result_allocation import _capture_owner
from tests.solution.test_solution_persistence import _make_values_only_solution
from tests.solution.test_solution_result import _small_grid_search_inputs


@pytest.mark.parametrize("preloaded", [False, True])
def test_budgeted_native_value_archive_replays_and_reuses_its_resolution(
    *, tmp_path: Path, preloaded: bool
) -> None:
    """The archive and the original owned result replay the same small model."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    expected = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=original,
        log_level="off",
        seed=17,
    ).to_dataframe()
    path = original.save(path=tmp_path / "values.h5")
    loaded = load_solution(path=path)
    assert isinstance(loaded.values, ValueStore)
    coordinates = [
        (period, regime) for period in loaded.values for regime in loaded.values[period]
    ]
    assert coordinates
    assert all(
        loaded.values.load_state(period=period, regime=regime) is LoadState.UNLOADED
        for period, regime in coordinates
    )
    if preloaded:
        loaded.values.materialize()
    actual = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=loaded,
        log_level="off",
        seed=17,
    ).to_dataframe()
    assert_frame_equal(actual, expected)
    assert all(
        loaded.values.load_state(period=period, regime=regime) is LoadState.LOADED
        for period, regime in coordinates
    )
    # The accepted consumed view must survive independent deletion of a fresh
    # public read: neither its archive cache nor the replay owns that wrapper.
    period, regime = coordinates[0]
    public_read = loaded.values[period][regime]
    public_read.delete()
    repeated = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=loaded,
        log_level="off",
        seed=17,
    ).to_dataframe()
    assert_frame_equal(repeated, expected)


class _UnadmittedAllocationError(AssertionError):
    """A real allocation reached the sentinel after headroom was exhausted."""


@pytest.mark.parametrize("failed_upload", [False, True])
def test_native_upload_is_admitted_after_verification_before_device_put(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_upload: bool
) -> None:
    """A valid host read cannot upload when existing live payload fills the budget."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    loaded = load_solution(path=original.save(path=tmp_path / "values.h5"))
    owners = _capture_owner(monkeypatch)
    verified: list[np.ndarray] = []
    detections: list[bool] = []
    original_read = archive._read_and_verify_leaves
    original_put = jax.device_put

    def guard(value: object, *args: Any, **kwargs: Any) -> object:
        if any(value is array for array in verified):
            detections.append(True)
            raise _UnadmittedAllocationError(
                "Verified archive leaf uploaded without headroom"
            )
        return original_put(value, *args, **kwargs)

    def read(**kwargs: Any) -> tuple[np.ndarray, ...]:
        arrays = original_read(**kwargs)
        verified.extend(arrays)
        live = owners[-1].snapshot()
        if not failed_upload:
            owners[-1].budget_bytes = max(
                sum(stop - start for start, stop in spans)
                for spans in live.spans.values()
            )
        assert owners[-1].budget_bytes > 1
        with pytest.raises(_UnadmittedAllocationError):
            jax.device_put(arrays[0])
        return arrays

    monkeypatch.setattr(archive, "_read_and_verify_leaves", read)
    monkeypatch.setattr(jax, "device_put", guard)
    with pytest.raises(
        _UnadmittedAllocationError if failed_upload else ExecutionPlanningError
    ):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=loaded,
            seed=17,
            log_level="off",
        )
    assert len(verified) == 1, "The native guard must not replace upload admission"
    assert detections == ([True, True] if failed_upload else [True])
    assert all(
        entry.load_state is LoadState.UNLOADED
        for entry in _native_entries(loaded).values()
    )


# keyword-only-exempt: library-callback=functools.partialmethod
def _limit_after_first_copy(
    self: SimulationEntryAllocations,
    *,
    leaf: jax.Array,
    label: str,
    original: Any,
    completed: list[weakref.ReferenceType[jax.Array]],
    requested: list[bool],
    snapshots: list[bool],
) -> jax.Array:
    requested.append(True)
    result = original(self, leaf=leaf, label=label)
    completed.append(weakref.ref(result))
    live = self.snapshot()
    expected = measure_buffer_footprint(tree=(leaf, result))
    missing = resident_bytes_by_device(
        live=expected,
        arguments=live,
        devices=tuple(expected.spans),
    )
    assert not any(missing.values()), (
        "Cache upload and detached copy must remain charged"
    )
    snapshots.append(True)
    self.budget_bytes = max(
        sum(stop - start for start, stop in spans) for spans in live.spans.values()
    )
    return result


def test_native_detached_copy_bank_is_charged_before_the_next_dispatch(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After one copy fills the budget, the next copy must refuse before execution."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    loaded = load_solution(path=original.save(path=tmp_path / "values.h5"))
    owners = _capture_owner(monkeypatch)
    completed: list[weakref.ReferenceType[jax.Array]] = []
    requested: list[bool] = []
    snapshots: list[bool] = []
    original_dispatch = jax.stages.Compiled.__call__

    def dispatch(self: jax.stages.Compiled, *args: Any, **kwargs: Any) -> object:
        if completed:
            raise _UnadmittedAllocationError(
                "Second detached copy dispatched without headroom"
            )
        return original_dispatch(self, *args, **kwargs)

    monkeypatch.setattr(
        SimulationEntryAllocations,
        "copy_solution_leaf",
        partialmethod(
            _limit_after_first_copy,
            original=SimulationEntryAllocations.copy_solution_leaf,
            completed=completed,
            requested=requested,
            snapshots=snapshots,
        ),
    )
    monkeypatch.setattr(jax.stages.Compiled, "__call__", dispatch)
    with pytest.raises(ExecutionPlanningError):
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=loaded,
            seed=17,
            log_level="off",
        )
    assert requested == [True, True], "Refusal must follow a successful first copy"
    assert snapshots == [True]
    assert len(completed) == 1
    assert not owners[-1]._foreign_copies
    cached = [
        entry
        for entry in _native_entries(loaded).values()
        if entry.load_state is LoadState.LOADED
    ]
    assert len(cached) == 1
    assert np.asarray(_native_cache(cached[0]).leaves[0]).size > 0


@pytest.mark.parametrize("malformed", ["model", "coverage", "schema"])
def test_invalid_native_envelope_is_rejected_before_archive_read(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str
) -> None:
    """Whole-result metadata and coordinate checks precede the trusted loader."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    loaded = load_solution(path=original.save(path=tmp_path / "values.h5"))
    if malformed == "model":
        loaded = replace(
            loaded, metadata=replace(loaded.metadata, model_fingerprint="0" * 64)
        )
    elif malformed == "coverage":
        coordinates = dict(_native_entries(loaded))
        coordinates.pop(next(reversed(coordinates)))
        loaded = replace(
            loaded, values=ValueStore(cast("Mapping[object, object]", coordinates))
        )
    else:
        schemas = dict(loaded.metadata.value_schemas)
        schemas.pop(next(reversed(schemas)))
        loaded = replace(
            loaded, metadata=replace(loaded.metadata, value_schemas=schemas)
        )

    def forbidden(**kwargs: Any) -> object:
        del kwargs
        raise AssertionError("Invalid envelope reached a native payload read")

    monkeypatch.setattr(archive, "_read_and_verify_leaves", forbidden)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=params, initial_conditions=initial, solution=loaded, log_level="off"
        )
    assert all(
        entry.load_state is LoadState.UNLOADED
        for entry in _native_entries(loaded).values()
    )


def test_native_checksum_failure_leaves_no_uploaded_cache(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real HDF5 payload corruption must fail before the writer is invoked."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    path = original.save(path=tmp_path / "values.h5")
    loaded = load_solution(path=path)
    first = next(iter(_native_entries(loaded).values()))
    with h5py.File(path, "r+") as handle:
        dataset = handle[first.leaves[0]["dataset"]]
        dataset[...] = np.asarray(dataset[()]) + 1
    original_write = SimulationEntryAllocations.__call__

    # keyword-only-exempt: library-callback=SimulationEntryAllocations.__call__
    def guard(
        self: SimulationEntryAllocations, *, name: str, **kwargs: Any
    ) -> jax.Array:
        assert not name.startswith("native_value:"), (
            "Corrupt native value reached upload"
        )
        return original_write(self, name=name, **kwargs)

    monkeypatch.setattr(SimulationEntryAllocations, "__call__", guard)
    with pytest.raises(SolutionIntegrityError, match="checksum"):
        model.simulate(
            params=params, initial_conditions=initial, solution=loaded, log_level="off"
        )
    assert all(
        entry.load_state is LoadState.UNLOADED
        for entry in _native_entries(loaded).values()
    )


@pytest.mark.parametrize("same_entry", [False, True])
def test_concurrent_native_materialization_observes_unlocked_cache_bank(
    *, tmp_path: Path, same_entry: bool
) -> None:
    """Two concurrent loaders can snapshot the bank and publish each entry only once."""
    model, params, _initial = _small_grid_search_inputs()
    original = model.solve(params=params, log_level="off")
    loaded = load_solution(path=original.save(path=tmp_path / "values.h5"))
    entries = tuple(_native_entries(loaded).values())
    assert len(entries) >= 2
    selected = (entries[0], entries[0] if same_entry else entries[1])
    starts = threading.Barrier(2)
    uploads = threading.Barrier(2) if not same_entry else None
    counts: list[str] = []
    owners = tuple(
        SimulationEntryAllocations(
            original_inputs=SimulationEntryInputs(arrays=()),
            solution=loaded,
            model_roots=(),
            devices=(jax.devices()[0],),
            budget_bytes=2**20,
        )
        for _ in selected
    )

    def run(index: int) -> object:
        owner = owners[index]

        def write(
            *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
        ) -> jax.Array:
            counts.append(name)
            if uploads is not None:
                uploads.wait(timeout=10)
            # This detects unsafe held locks before a real snapshot could deadlock.
            # Both entries have reached their writer when checking the two-entry case.
            cache = selected[index]._cache
            acquired = cache.lock.acquire(blocking=False)
            assert acquired, "Materializer holds a cache lock across admission"
            cache.lock.release()
            if uploads is not None:
                uploads.wait(timeout=10)
            owner.snapshot()
            return owner(value=value, dtype=dtype, name=name)

        materializer = NativeValueMaterializer(
            array_writer=write, array_copier=owner.copy_solution_leaf
        )
        starts.wait(timeout=10)
        return materializer(entry=selected[index])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(run, index) for index in range(2))
        results = tuple(future.result(timeout=20) for future in futures)
    assert len(counts) == (1 if same_entry else 2)
    for result, entry in zip(results, selected, strict=True):
        assert isinstance(result, jax.Array)
        assert result is not _native_cache(entry).leaves[0]
        np.testing.assert_array_equal(result, _native_cache(entry).leaves[0])
    for owner in owners:
        owner.close()


@pytest.mark.parametrize("fail_copy", [False, True])
def test_native_dependencies_and_temporary_copies_do_not_escape_the_call(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_copy: bool
) -> None:
    """Published banks survive while transient dependencies leave no memo roots."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    original = model.solve(params=params, log_level="off")
    loaded = load_solution(path=original.save(path=tmp_path / "values.h5"))
    owners: list[weakref.ReferenceType[SimulationEntryAllocations]] = []
    loaders: list[weakref.ReferenceType[NativeValueMaterializer]] = []
    temporaries: list[weakref.ReferenceType[jax.Array]] = []

    def create_owner(**kwargs: Any) -> SimulationEntryAllocations:
        owner = SimulationEntryAllocations(**kwargs)
        owners.append(weakref.ref(owner))
        return owner

    def create_loader(**kwargs: Any) -> NativeValueMaterializer:
        loader = NativeValueMaterializer(**kwargs)
        loaders.append(weakref.ref(loader))
        return loader

    original_copy = SimulationEntryAllocations.copy_solution_leaf

    # keyword-only-exempt: library-callback=copy_solution_leaf
    def copy(
        self: SimulationEntryAllocations, *, leaf: jax.Array, label: str
    ) -> jax.Array:
        result = original_copy(self, leaf=leaf, label=label)
        temporaries.append(weakref.ref(result))
        if fail_copy:
            raise RuntimeError("injected later-copy failure")
        return result

    monkeypatch.setattr(model_module, "SimulationEntryAllocations", create_owner)
    monkeypatch.setattr(model_module, "NativeValueMaterializer", create_loader)
    monkeypatch.setattr(SimulationEntryAllocations, "copy_solution_leaf", copy)
    if fail_copy:
        # The copy validation boundary normalizes unexpected copier errors.
        with pytest.raises(InvalidSimulationInputError, match="copied safely"):
            model.simulate(
                params=params,
                initial_conditions=initial,
                solution=loaded,
                log_level="off",
            )
    else:
        model.simulate(
            params=params, initial_conditions=initial, solution=loaded, log_level="off"
        )
    gc.collect()
    assert owners
    assert loaders
    assert temporaries
    assert all(ref() is None for ref in owners + loaders)
    if fail_copy:
        assert all(ref() is None for ref in temporaries)
    cache_arrays = tuple(
        _native_cache(entry).leaves[0]
        for entry in _native_entries(loaded).values()
        if entry.load_state is LoadState.LOADED
    )
    assert cache_arrays
    assert all(not value.is_deleted() for value in cache_arrays)


def test_admitted_native_upload_refuses_dtype_narrowing_before_writer(
    *, tmp_path: Path
) -> None:
    """A verified float64 payload cannot enter a float32 upload operation."""

    with jax.enable_x64(new_val=True):
        original = _make_values_only_solution(value=[1.0, 2.0], dtype="float64")
        path = original.save(path=tmp_path / "float64.h5")
    loaded = load_solution(path=path)
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=()),
        solution=loaded,
        model_roots=(),
        devices=(jax.devices()[0],),
        budget_bytes=2**20,
    )

    def forbidden(
        *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
    ) -> jax.Array:
        del value, dtype, name
        raise AssertionError("Narrowed native dtype reached its upload writer")

    loader = NativeValueMaterializer(
        array_writer=forbidden, array_copier=owner.copy_solution_leaf
    )
    entry = next(iter(_native_entries(loaded).values()))
    with (
        jax.enable_x64(new_val=False),
        pytest.raises(IncompatibleSolutionError, match="would materialize it as"),
    ):
        loader(entry=entry)
    assert entry.load_state is LoadState.UNLOADED
    owner.close()


def _native_entries(
    solution: SolutionResult,
) -> Mapping[tuple[int, str], archive._LazyHdf5Entry]:
    """Verify that a native fixture really carries trusted archive handles."""
    assert isinstance(solution.values, ValueStore)
    entries = solution.values._entries
    assert all(type(entry) is archive._LazyHdf5Entry for entry in entries.values())
    return cast("Mapping[tuple[int, str], archive._LazyHdf5Entry]", entries)


def _native_cache(entry: archive._LazyHdf5Entry) -> archive._LoadedEntryPayload:
    """Observe a real published native leaf bank without public detached reads."""
    cached = entry._cache.value
    assert isinstance(cached, archive._LoadedEntryPayload)
    return cached
