"""Numeric entry staging uses the first selected device before subject sharding.

Run alone in a fresh four-CPU-device process.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax
import numpy as np
import pytest

from _lcm.dtypes import safe_to_int_dtype
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.initial_conditions import canonicalize_initial_conditions
from _lcm.simulation.operand_placement import place_simulation_arguments
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import SolutionResult

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _TOPOLOGY_AVAILABLE = True
except RuntimeError:
    _TOPOLOGY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TOPOLOGY_AVAILABLE, reason="Requires a fresh four-CPU-device process"
)


@pytest.mark.parametrize("order", [(3, 1, 2), (2, 3, 1)])
def test_numeric_entry_stages_and_pads_on_the_first_actual_device(
    order: tuple[int, ...],
) -> None:
    """A seven-row source remains readable while staged padding creates nine rows."""
    devices = tuple(jax.devices()[index] for index in order)
    source = jax.device_put(np.arange(7, dtype=np.int32), device=jax.devices()[0])
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=devices,
        budget_bytes=2**20,
    )
    initial = canonicalize_initial_conditions(
        initial_conditions={"regime_id": source},
        regimes=MappingProxyType({}),
        array_writer=owner,
    )
    owner.publish(stage="initial", tree=initial)
    assert initial["regime_id"].devices() == {devices[0]}
    padded, count = owner.pad(initial_conditions=initial, multiple=3)
    assert count == 7
    assert padded["regime_id"].devices() == {devices[0]}
    np.testing.assert_array_equal(padded["regime_id"], [0, 1, 2, 3, 4, 5, 6, 6, 6])
    owner.publish(stage="initial", tree=padded)
    placed = place_simulation_arguments(
        arguments=padded,
        subject_arg_names=("regime_id",),
        value_reads=(),
        devices=devices,
        budget_bytes=owner.budget_bytes,
        budget_devices=tuple(jax.devices()),
        live_footprint=owner.snapshot(),
    )["regime_id"]
    assert isinstance(placed, jax.Array)
    assert isinstance(placed.sharding, jax.NamedSharding)
    assert tuple(placed.sharding.mesh.devices.flat) == devices
    assert placed.sharding.spec == jax.P("X")
    assert all(shard.data.shape == (3,) for shard in placed.addressable_shards)
    np.testing.assert_array_equal(placed, [0, 1, 2, 3, 4, 5, 6, 6, 6])
    np.testing.assert_array_equal(source, np.arange(7, dtype=np.int32))


def test_retained_source_on_an_excluded_device_still_consumes_its_budget() -> None:
    """A small destination cannot hide a larger retained bank on source device zero."""
    source = jax.device_put(np.arange(16, dtype=np.int32), device=jax.devices()[0])
    assert source.nbytes == 64
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=tuple(jax.devices()[1:]),
        budget_bytes=63,
    )
    with pytest.raises(ExecutionPlanningError, match="budget"):
        safe_to_int_dtype(
            value=np.array([1], dtype=np.int64), name="small", array_writer=owner
        )
    np.testing.assert_array_equal(source, np.arange(16, dtype=np.int32))


@pytest.mark.parametrize("order", [(3, 1, 2), (2, 3, 1)])
def test_foreign_value_copies_preserve_ordered_source_mesh(
    order: tuple[int, ...],
) -> None:
    """Private copies keep the solve layout even when simulation selects a subset."""
    devices = tuple(jax.devices()[index] for index in order)
    sharding = jax.NamedSharding(
        jax.make_mesh((3,), ("source",), devices=devices), jax.P("source")
    )
    levels = np.arange(12, dtype=np.float64 if jax.config.x64_enabled else np.float32)
    source = jax.device_put(levels, sharding)
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=tuple(jax.devices()[1:3]),
        budget_bytes=2**20,
    )
    first = owner.copy_solution_leaf(leaf=source, label="first")
    second = owner.copy_solution_leaf(leaf=source, label="second")
    assert len(owner.operations.cache) == 1
    for array in (first, second):
        assert array.sharding == sharding
        assert isinstance(array.sharding, jax.NamedSharding)
        assert tuple(array.sharding.mesh.devices.flat) == devices
        np.testing.assert_array_equal(array, levels)
    owner.release_foreign_copies()
    first.delete()
    np.testing.assert_array_equal(source, levels)
    np.testing.assert_array_equal(second, levels)
    owner.close()


def test_foreign_copy_peak_is_charged_on_excluded_source_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty simulation-device budget cannot hide a copy executing on CPU0."""
    source = jax.device_put(np.arange(257, dtype=np.float32), jax.devices()[0])
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=(source,)),
        solution=None,
        model_roots=(),
        devices=tuple(jax.devices()[1:]),
        budget_bytes=2**20,
    )
    copied = owner.copy_solution_leaf(leaf=source, label="profile")
    assert copied.devices() == {jax.devices()[0]}
    compiled = next(iter(owner.operations.cache.values()))
    owner.release_foreign_copies()
    copied.delete()
    owner.budget_bytes = compiled.peak_bytes - 1
    assert owner.budget_bytes >= source.nbytes

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("Unadmitted copy reached its excluded source device")

    monkeypatch.setattr(jax.stages.Compiled, "__call__", forbidden)
    with pytest.raises(ExecutionPlanningError, match="budget"):
        owner.copy_solution_leaf(leaf=source, label="rejected")
    np.testing.assert_array_equal(source, np.arange(257, dtype=np.float32))


def test_foreign_copy_cache_distinguishes_source_device_order() -> None:
    """Equal shapes and device sets with different ordering need separate profiles."""
    owner = SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=()),
        solution=None,
        model_roots=(),
        devices=tuple(jax.devices()[1:]),
        budget_bytes=2**20,
    )
    for order in ((3, 1, 2), (2, 3, 1)):
        devices = tuple(jax.devices()[index] for index in order)
        layout = jax.NamedSharding(
            jax.make_mesh((3,), ("source",), devices=devices), jax.P("source")
        )
        source = jax.device_put(np.arange(12, dtype=np.float32), layout)
        copy = owner.copy_solution_leaf(leaf=source, label="ordered")
        assert copy.sharding == layout
        np.testing.assert_array_equal(copy, np.arange(12, dtype=np.float32))
    assert len(owner.operations.cache) == 2
    owner.close()


@pytest.mark.parametrize("order", [(3, 1, 2), (2, 3, 1)])
@pytest.mark.parametrize("preloaded", [False, True])
def test_native_archive_upload_and_copy_keep_actual_device_ownership(
    *, tmp_path: Path, order: tuple[int, ...], preloaded: bool
) -> None:
    """Native uploads use selected staging; preloaded banks keep their ordered mesh."""
    from pandas.testing import assert_frame_equal  # noqa: PLC0415

    from lcm import ExecutionConfig  # noqa: PLC0415
    from lcm.persistence import load_solution  # noqa: PLC0415
    from tests.simulation.test_native_value_admission import (  # noqa: PLC0415
        _native_cache,
        _native_entries,
    )
    from tests.solution.test_solution_result import (  # noqa: PLC0415
        _small_grid_search_inputs,
    )

    devices = tuple(jax.devices()[index] for index in order)
    execution = ExecutionConfig(devices=(order[1], order[2]), device_memory_bytes=2**28)
    model, params, initial = _small_grid_search_inputs(execution_config=execution)
    solution = model.solve(params=params, log_level="off")
    expected = model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="off"
    ).to_dataframe()
    loaded = load_solution(path=solution.save(path=tmp_path / "values.h5"))
    entries = tuple(_native_entries(loaded).values())
    source_mesh = jax.make_mesh((3,), ("native_source",), devices=devices)
    source_layout = jax.NamedSharding(source_mesh, jax.P("native_source"))
    scalar_layout = jax.NamedSharding(source_mesh, jax.P())
    if preloaded:
        # This explicitly constructs an already-loaded caller cache on its source
        # mesh before the budgeted public call; it is not a budgeted upload claim.
        def preload(
            *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
        ) -> jax.Array:
            del name
            assert value.dtype == dtype
            layout = scalar_layout if value.ndim == 0 else source_layout
            return jax.device_put(value, layout)

        for entry in entries:
            entry._materialize(
                template=None, template_snapshot=None, array_writer=preload
            )
    actual = model.simulate(
        params=params, initial_conditions=initial, solution=loaded, log_level="off"
    ).to_dataframe()
    assert_frame_equal(actual, expected)
    cache_arrays = tuple(_native_cache(entry).leaves[0] for entry in entries)
    assert {value.ndim for value in cache_arrays} == {0, 1}
    if preloaded:
        for value in cache_arrays:
            expected_layout = scalar_layout if value.ndim == 0 else source_layout
            assert value.sharding == expected_layout
            assert isinstance(value.sharding, jax.NamedSharding)
            assert tuple(value.sharding.mesh.devices.flat) == devices
            if value.ndim == 1:
                shards = value.addressable_shards
                assert len(shards) == 3
                assert {shard.device for shard in shards} == set(devices)
                assert len({repr(shard.index) for shard in shards}) == 3
                assert all(shard.data.shape == (1,) for shard in shards)
                assert not value.sharding.is_fully_replicated
    else:
        # The public model resolves selected IDs in ascending order; staging uses
        # that actual order, while the caller's preloaded source mesh stays ordered.
        first_selected = jax.devices()[model.execution_devices[0]]
        assert all(value.devices() == {first_selected} for value in cache_arrays)
    # Resolved private leaves preserve cache layout even when its source mesh
    # includes a same-platform device excluded from the simulation execution mesh.
    resolved = cast(
        "tuple[Mapping[int, Mapping[str, jax.Array]], object, object, object]",
        next(iter(loaded._consumed_views.values())),
    )[0]
    for period, by_regime in resolved.items():
        for regime, value in by_regime.items():
            cached = _native_cache(_native_entries(loaded)[(period, regime)]).leaves[0]
            assert value.sharding == cached.sharding
            assert value is not cached
            np.testing.assert_array_equal(value, cached)
    public = loaded.values[next(iter(loaded.values))][
        next(iter(loaded.values[next(iter(loaded.values))]))
    ]
    public.delete()
    assert all(not array.is_deleted() for array in cache_arrays)


def test_native_source_budget_includes_retained_bank_and_future_copy_scratch(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real excluded-source bank leaves no room for its profiled transfer scratch."""
    from typing import Any  # noqa: PLC0415

    import lcm.model as model_module  # noqa: PLC0415
    from _lcm.simulation import chunk_admission  # noqa: PLC0415
    from _lcm.simulation.residency import (  # noqa: PLC0415
        DeviceBufferFootprint,
        measure_buffer_footprint,
        resident_bytes_by_device,
        union_buffer_footprints,
    )
    from lcm import ExecutionConfig  # noqa: PLC0415
    from lcm.persistence import load_solution  # noqa: PLC0415
    from tests.simulation.test_native_value_admission import (  # noqa: PLC0415
        _native_entries,
    )
    from tests.solution.test_solution_result import (  # noqa: PLC0415
        _small_grid_search_inputs,
    )

    budget = 4096
    source_device = jax.devices()[3]
    selected = (jax.devices()[1], jax.devices()[2])
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(devices=(1, 2), device_memory_bytes=budget)
    )
    solution = model.solve(params=params, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "source-budget.h5"))
    entries = tuple(_native_entries(loaded).values())
    source_mesh = jax.make_mesh(
        (3,), ("native_source",), devices=(source_device, *selected)
    )

    def preload(
        *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
    ) -> jax.Array:
        del name
        assert value.dtype == dtype
        spec = jax.P() if value.ndim == 0 else jax.P("native_source")
        return jax.device_put(value, jax.NamedSharding(source_mesh, spec))

    for entry in entries:
        entry._materialize(template=None, template_snapshot=None, array_writer=preload)
    original_prepare = model_module.prepare_simulation_chunks
    original_required = chunk_admission._required_bytes
    banks: list[jax.Array] = []
    requirements: list[tuple[dict[jax.Device, int], int]] = []

    def prepare(*, retained_footprint: DeviceBufferFootprint, **kwargs: Any) -> object:
        existing = resident_bytes_by_device(
            live=retained_footprint,
            arguments=DeviceBufferFootprint(spans={}),
            devices=(source_device,),
        )[source_device]
        assert 0 < existing < budget
        # A test-owned, actual array fills exactly the remaining source capacity.
        # This is fixture allocation, not a claim of library admission for it.
        bank = jax.device_put(
            np.zeros(budget - existing, dtype=np.uint8), source_device
        ).block_until_ready()
        banks.append(bank)
        live = union_buffer_footprints(
            footprints=(retained_footprint, measure_buffer_footprint(tree=bank))
        )
        assert (
            resident_bytes_by_device(
                live=live,
                arguments=DeviceBufferFootprint(spans={}),
                devices=(source_device,),
            )[source_device]
            == budget
        )
        return original_prepare(retained_footprint=live, **kwargs)

    def required(**kwargs: Any) -> Mapping[jax.Device, int]:
        result = original_required(**kwargs)
        profile = kwargs["profile"]
        scratch = profile.fixed_reservation[source_device]
        assert scratch > 0
        assert kwargs["resident"][source_device] == budget
        assert all(result[device] <= budget for device in selected)
        requirements.append((dict(result), scratch))
        return result

    monkeypatch.setattr(model_module, "prepare_simulation_chunks", prepare)
    monkeypatch.setattr(chunk_admission, "_required_bytes", required)
    monkeypatch.setattr(model_module, "simulate", _forbid_source_overflow_dispatch)
    with pytest.raises(
        ExecutionPlanningError, match="No declared simulation chunk fits"
    ):
        model.simulate(
            params=params, initial_conditions=initial, solution=loaded, log_level="off"
        )
    assert requirements, "The real compiler profile must establish source headroom"
    assert all(
        result[source_device] == budget + scratch for result, scratch in requirements
    )
    _assert_native_bank_and_caches_readable(
        banks=banks, solution=loaded, devices=(source_device, *selected)
    )


def _assert_native_bank_and_caches_readable(
    *, banks: list[jax.Array], solution: SolutionResult, devices: tuple[jax.Device, ...]
) -> None:
    """Real test-owned storage and every published cache survive the refusal."""
    from tests.simulation.test_native_value_admission import (  # noqa: PLC0415
        _native_cache,
        _native_entries,
    )

    assert len(banks) == 1
    np.testing.assert_array_equal(banks[0], np.zeros(banks[0].shape, dtype=np.uint8))
    for (period, regime), entry in _native_entries(solution).items():
        value = _native_cache(entry).leaves[0]
        assert not value.is_deleted()
        assert isinstance(value.sharding, jax.NamedSharding)
        assert tuple(value.sharding.mesh.devices.flat) == devices
        np.testing.assert_array_equal(value, solution.values[period][regime])


def _forbid_source_overflow_dispatch(**kwargs: object) -> object:
    """Detect any public execution after the deliberately overflowing source bank."""
    del kwargs
    raise AssertionError("An overflowing source reservation reached simulation")
