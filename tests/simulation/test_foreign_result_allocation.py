"""Foreign eager snapshots must admit and retain each private copy explicitly."""

import gc
import weakref
from collections.abc import Callable
from dataclasses import replace
from functools import partial, partialmethod
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
from _lcm.persistence import solution as archive
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.simulation.solution_copies import copy_solution_leaf
from _lcm.solution.result_snapshot import snapshot_artifact_store
from lcm import AgeGrid, ExecutionConfig, Model
from lcm._solver_api import authority as authority_module
from lcm._solver_api import entries
from lcm.exceptions import ExecutionPlanningError, InvalidSimulationInputError
from lcm.persistence import load_solution
from lcm.solver_api import ArtifactRef, ArtifactStore, ValueStore
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    DissolutionRegimeId,
    _make_dissolution_regimes,
)
from tests.solution.test_solution_result import _small_grid_search_inputs
from tests.solution.test_solution_result_snapshot import (
    _array_authority,
    _replace_first_value_with_counter,
)


class _UnadmittedForeignCopyError(AssertionError):
    """A concrete copy reached the eager allocator inside foreign resolution."""


def _capture_owner(monkeypatch: pytest.MonkeyPatch) -> list[SimulationEntryAllocations]:
    owners: list[SimulationEntryAllocations] = []

    def create(**arguments: Any) -> SimulationEntryAllocations:
        owner = SimulationEntryAllocations(**arguments)
        owners.append(owner)
        return owner

    monkeypatch.setattr(model_module, "SimulationEntryAllocations", create)
    return owners


# keyword-only-exempt: library-callback=functools.partialmethod
def _guard_foreign_resolution(
    self: Model,
    *,
    original: Callable[..., object],
    monkeypatch: pytest.MonkeyPatch,
    owners: list[SimulationEntryAllocations],
    lower_budget: bool,
    reached: list[bool],
    **arguments: Any,
) -> object:
    owner = owners[-1]
    reached.append(True)
    live = owner.snapshot()
    if lower_budget:
        owner.budget_bytes = max(
            sum(stop - start for start, stop in spans) for spans in live.spans.values()
        )
        assert owner.budget_bytes > 1
    array = jnp.array

    def guard(value: object, *args: Any, **kwargs: Any) -> jax.Array:
        if (
            kwargs.get("copy") is True
            and isinstance(value, jax.Array)
            and not isinstance(value, jax.core.Tracer)
        ):
            raise _UnadmittedForeignCopyError(
                "Foreign snapshot allocated an eager copy before profiled admission"
            )
        return array(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jnp, "array", guard)
        source = next(
            entry.value
            for entry in arguments["solution"].values._entries.values()
            if type(entry) is entries._CanonicalValueEntry
        )
        with pytest.raises(_UnadmittedForeignCopyError):
            jnp.array(source, copy=True)
        return original(self, **arguments)


@pytest.mark.parametrize("lower_budget", [False, True])
def test_public_foreign_snapshot_admits_before_private_copy(
    *, monkeypatch: pytest.MonkeyPatch, lower_budget: bool
) -> None:
    """Start from an accepted same-model foreign envelope, without a solve confound."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    owned = model.solve(params=params, log_level="off")
    foreign = replace(owned)
    assert foreign._engine_view is None
    owners = _capture_owner(monkeypatch)
    reached: list[bool] = []
    monkeypatch.setattr(
        Model,
        "_consume_foreign_solution",
        partialmethod(
            _guard_foreign_resolution,
            original=Model._consume_foreign_solution,
            monkeypatch=monkeypatch,
            owners=owners,
            lower_budget=lower_budget,
            reached=reached,
        ),
    )
    run = partial(
        model.simulate,
        params=params,
        initial_conditions=initial,
        solution=foreign,
        seed=17,
        log_level="off",
    )
    if lower_budget:
        with pytest.raises(ExecutionPlanningError):
            run()
    else:
        run()
    assert reached == [True]


def test_foreign_copy_bank_is_live_before_the_next_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weak observations detect private leaves held outside the entry inventory."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    foreign = replace(model.solve(params=params, log_level="off"))
    owners = _capture_owner(monkeypatch)
    copied: list[weakref.ReferenceType[jax.Array]] = []
    inspected: list[bool] = []
    original = entries._copy_solution_value

    def observe(*, value: object, label: str, **kwargs: Any) -> object:
        still_live = tuple(leaf for ref in copied if (leaf := ref()) is not None)
        if still_live:
            inspected.append(True)
            expected = measure_buffer_footprint(tree=still_live)
            missing = resident_bytes_by_device(
                live=expected,
                arguments=owners[-1].snapshot(),
                devices=tuple(expected.spans),
            )
            assert not any(missing.values()), (
                f"Live foreign copies missing before the next copy: {dict(missing)}"
            )
        result = original(value=value, label=label, **kwargs)
        if isinstance(result, jax.Array):
            result.block_until_ready()
            copied.append(weakref.ref(result))
        return result

    monkeypatch.setattr(entries, "_copy_solution_value", observe)
    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=foreign,
        seed=17,
        log_level="off",
    )
    assert inspected


def test_generous_budget_foreign_copy_is_valid_and_preserves_originals() -> None:
    """The guarded envelope is already supported and numerically ordinary."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    owned = model.solve(params=params, log_level="off")
    foreign = replace(owned)
    store = foreign.values
    assert isinstance(store, ValueStore)
    originals = tuple(
        entry.value
        for entry in store._entries.values()
        if isinstance(entry, entries._CanonicalValueEntry)
    )
    original_levels = tuple(np.asarray(value).copy() for value in originals)
    expected = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=owned,
        seed=17,
        log_level="off",
    )
    actual = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=foreign,
        seed=17,
        log_level="off",
    )
    assert_frame_equal(
        actual.to_dataframe(use_labels=False),
        expected.to_dataframe(use_labels=False),
        check_exact=True,
    )
    assert len(foreign._consumed_views) == 1
    resolved = next(iter(foreign._consumed_views.values()))
    assert isinstance(resolved, tuple)
    values = resolved[0]
    copied = measure_buffer_footprint(tree=values)
    source = measure_buffer_footprint(tree=originals)
    exclusive = resident_bytes_by_device(
        live=copied, arguments=source, devices=tuple(copied.spans)
    )
    complete = resident_bytes_by_device(
        live=copied,
        arguments=DeviceBufferFootprint(spans={}),
        devices=tuple(copied.spans),
    )
    assert dict(exclusive) == dict(complete)
    for array, levels in zip(originals, original_levels, strict=True):
        np.testing.assert_array_equal(array, levels)


def test_unknown_lazy_value_is_refused_before_its_callback() -> None:
    """The existing explicit budget envelope does not trust arbitrary lazy code."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    solution = model.solve(params=params, log_level="off")
    foreign, lazy = _replace_first_value_with_counter(solution)
    with pytest.raises(ExecutionPlanningError):
        model.simulate(
            params=params, initial_conditions=initial, solution=foreign, log_level="off"
        )
    assert lazy.materialization_count == 0


@pytest.mark.parametrize("fail_validation", [False, True])
def test_foreign_copy_owners_release_and_never_enter_the_compiler_cache(
    *, monkeypatch: pytest.MonkeyPatch, fail_validation: bool
) -> None:
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    foreign = replace(model.solve(params=params, log_level="off"))
    owners = _capture_owner(monkeypatch)
    copied: list[weakref.ReferenceType[jax.Array]] = []
    original = SimulationEntryAllocations.copy_solution_leaf

    # keyword-only-exempt: library-callback=functools.partialmethod
    def observe(
        self: SimulationEntryAllocations, *, leaf: jax.Array, label: str
    ) -> jax.Array:
        result = original(self, leaf=leaf, label=label)
        copied.append(weakref.ref(result))
        return result

    monkeypatch.setattr(SimulationEntryAllocations, "copy_solution_leaf", observe)
    run = partial(
        model.simulate,
        params=params,
        initial_conditions=initial,
        solution=foreign,
        seed=17,
        log_level="off",
    )
    if fail_validation:

        def refuse(**arguments: Any) -> None:
            del arguments
            raise InvalidSimulationInputError("Seeded late schema rejection")

        monkeypatch.setattr(
            Model, "_check_solution_value_schemas", staticmethod(refuse)
        )
        with pytest.raises(InvalidSimulationInputError, match="Seeded late schema"):
            run()
    else:
        run()
    assert copied
    assert owners[-1]._foreign_copies == []
    cache = owners[-1].operations
    assert cache.cache
    owner_ref = weakref.ref(owners[-1])
    owners.clear()
    gc.collect()
    assert owner_ref() is None
    if fail_validation:
        assert not foreign._consumed_views
        assert all(ref() is None for ref in copied)
    else:
        resolved = next(iter(foreign._consumed_views.values()))
        retained = {
            id(leaf)
            for leaf in jax.tree.leaves(resolved)
            if isinstance(leaf, jax.Array)
        }
        assert all(ref() is None or id(ref()) in retained for ref in copied)
        before = len(copied)
        run()
        assert len(copied) == before
        assert next(iter(foreign._consumed_views.values())) is resolved
    assert not cache.in_flight


def test_growing_copy_bank_rechecks_the_same_compiled_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = jnp.arange(257, dtype=jnp.float32)
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=tuple(source.devices()),
        budget_bytes=2**20,
    )
    owner.copy_solution_leaf(leaf=source, label="first")
    compiled = next(iter(owner.operations.cache.values()))
    external = resident_bytes_by_device(
        live=owner.snapshot(),
        arguments=measure_buffer_footprint(tree=source),
        devices=owner.devices,
    )
    owner.budget_bytes = max(external.values()) + compiled.peak_bytes
    owner.copy_solution_leaf(leaf=source, label="second")
    assert next(iter(owner.operations.cache.values())) is compiled

    def forbidden(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("Rejected copy reached concrete executable dispatch")

    monkeypatch.setattr(jax.stages.Compiled, "__call__", forbidden)
    with pytest.raises(ExecutionPlanningError, match="budget"):
        owner.copy_solution_leaf(leaf=source, label="third")
    assert len(owner._foreign_copies) == 2
    assert len(owner.operations.cache) == 1
    owner.close()


@pytest.mark.parametrize("preloaded", [False, True])
def test_budgeted_native_archive_uses_admitted_loader_without_public_callback(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preloaded: bool
) -> None:
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    solution = model.solve(params=params, log_level="off")
    path = solution.save(path=tmp_path / "solution.h5")
    foreign = load_solution(path=path)
    if preloaded:
        assert isinstance(foreign.values, ValueStore)
        foreign.values.materialize()

    def forbidden(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("Unprofiled native archive callback was invoked")

    monkeypatch.setattr(archive._LazyHdf5Entry, "materialize", forbidden)
    model.simulate(
        params=params, initial_conditions=initial, solution=foreign, log_level="off"
    )


@pytest.mark.parametrize(
    "channel",
    [
        "retained_continuations",
        "replay_artifacts",
        "auxiliary_artifacts",
        "diagnostics",
    ],
)
def test_unexpected_supplied_artifact_is_rejected_before_unprofiled_copy(
    *, monkeypatch: pytest.MonkeyPatch, channel: str
) -> None:
    """Reject unexpected artifact addresses before reading their payloads."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    solution = model.solve(params=params, log_level="off")
    artifact_authority = _array_authority(value=4.0)
    ref = ArtifactRef(
        period=0, regime="working_life", key=artifact_authority.descriptor.key
    )
    store = snapshot_artifact_store(
        store=ArtifactStore({ref: jnp.asarray(4.0, dtype=jnp.float32)}),
        authorities={ref: artifact_authority},
    )
    foreign = replace(solution, **{channel: store})
    object.__setattr__(foreign, "_artifact_authority", {ref: artifact_authority})
    original = authority_module._copy_artifact_array_leaf

    def guard(*, leaf: object, label: str, **kwargs: Any) -> jax.Array:
        if kwargs.get("array_copier") is None:
            raise _UnadmittedForeignCopyError(
                "Unexpected artifact reached unprofiled copy"
            )
        return original(leaf=leaf, label=label, **kwargs)

    monkeypatch.setattr(entries, "_copy_artifact_array_leaf", guard)
    monkeypatch.setattr(authority_module, "_copy_artifact_array_leaf", guard)
    with pytest.raises(_UnadmittedForeignCopyError):
        store[ref]
    with pytest.raises(InvalidSimulationInputError, match="artifact"):
        model.simulate(
            params=params, initial_conditions=initial, solution=foreign, log_level="off"
        )


def test_mixed_backend_foreign_copy_refuses_before_allocation() -> None:
    """A CPU payload cannot silently inherit a declared GPU allocator ceiling.

    Only execution-device metadata is synthetic; the source is a real CPU array.
    This exercises refusal, and makes no claim about GPU execution or copies.
    """
    source = jnp.arange(8, dtype=jnp.float32)
    assert all(device.platform == "cpu" for device in source.devices())
    selected_gpu = cast("jax.Device", Mock(spec=jax.Device, platform="gpu"))

    def forbidden_live_read() -> DeviceBufferFootprint:
        raise AssertionError("Mixed-backend route reached copy admission")

    with pytest.raises(ExecutionPlanningError, match="same backend"):
        copy_solution_leaf(
            leaf=source,
            operations=ProfiledSimulationOperations(),
            live_footprint=forbidden_live_read,
            budget_devices=(selected_gpu,),
            budget_bytes=2**20,
        )
    np.testing.assert_array_equal(source, np.arange(8, dtype=np.float32))


def test_artifact_route_refuses_before_the_foreign_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid artifact-bearing result cannot start unprofiled authority copies."""
    model = Model(
        regimes=_make_dissolution_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        execution_config=ExecutionConfig(device_memory_bytes=2**28),
    )
    owned = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    foreign = replace(owned)
    assert foreign.replay_artifacts
    initial = {
        "wage": np.array([1.0, 2.0, 3.0]),
        "age": np.zeros(3),
        "regime_id": np.full(3, int(model.regime_names_to_ids["married"])),
        "own_stakeholder": np.full(3, model.stakeholder_names_to_ids["f"]),
    }

    def forbidden(**kwargs: object) -> object:
        del kwargs
        raise AssertionError("Unprofiled foreign artifact snapshot started")

    monkeypatch.setattr(Model, "_snapshot_solution_envelope", staticmethod(forbidden))
    with pytest.raises(ExecutionPlanningError, match="unprofiled artifact authority"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions=initial,
            solution=foreign,
            log_level="off",
        )


def test_host_value_payload_is_rejected_before_its_unprofiled_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical store wrapper cannot authorize a non-JAX model value upload."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    owned = model.solve(params=params, log_level="off")
    assert isinstance(owned.values, ValueStore)
    values = dict(owned.values._entries)
    coordinate = next(iter(values))
    original_entry = values[coordinate]
    assert type(original_entry) is entries._CanonicalValueEntry
    host = np.asarray(original_entry.value)
    values[coordinate] = host
    foreign = replace(owned, values=ValueStore(values))
    original_copy = entries._copy_solution_value

    def guarded_copy(**kwargs: Any) -> object:
        if isinstance(kwargs["value"], np.ndarray):
            raise _UnadmittedForeignCopyError(
                "Invalid host model value reached its copy allocator"
            )
        return original_copy(**kwargs)

    monkeypatch.setattr(entries, "_copy_solution_value", guarded_copy)
    with pytest.raises(AssertionError, match="host model value"):
        entries._copy_solution_value(value=host, label="positive control")
    with pytest.raises(InvalidSimulationInputError, match="JAX"):
        model.simulate(
            params=params, initial_conditions=initial, solution=foreign, log_level="off"
        )
