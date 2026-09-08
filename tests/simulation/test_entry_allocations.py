"""Numeric entry writes must be admitted before a device payload is allocated."""

import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, partialmethod
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm.model as model_module
from _lcm.dtypes import safe_to_float_dtype, safe_to_int_dtype
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.params.processing import cast_params_to_canonical_dtypes
from _lcm.params.sequence_leaf import SequenceLeaf, UserSequenceLeaf
from _lcm.simulation import entry_allocations as entry_allocations_module
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import (
    SimulationEntryInputs,
    capture_simulation_entry_inputs,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.typing import FlatParams
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError, InvalidParamsError
from tests.solution.test_solution_result import _small_grid_search_inputs


@dataclass(frozen=True, kw_only=True)
class _ForbiddenConversion:
    source: np.ndarray
    original: Callable[..., jax.Array]

    def __call__(self, value: object, *args: Any, **kwargs: Any) -> jax.Array:
        if value is self.source:
            raise AssertionError("Numeric device allocation preceded admission")
        return self.original(value, *args, **kwargs)


@dataclass(frozen=True, kw_only=True)
class _RecordingWriter:
    owner: SimulationEntryAllocations
    names: list[str]

    def __call__(
        self, *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
    ) -> jax.Array:
        self.names.append(name)
        return self.owner(value=value, dtype=dtype, name=name)


@dataclass(kw_only=True)
class _ResolvedCopies:
    values: object | None = None


@dataclass(frozen=True, kw_only=True)
class _ModelUploadInventory:
    expected: DeviceBufferFootprint
    original: Callable[..., object]
    calls: list[bool]

    def __call__(self, **arguments: Any) -> object:
        live = arguments["live_footprint"]
        assert isinstance(live, DeviceBufferFootprint)
        missing = resident_bytes_by_device(
            live=self.expected, arguments=live, devices=tuple(self.expected.spans)
        )
        assert not any(missing.values()), "Model-owned inputs absent before upload"
        self.calls.append(True)
        return self.original(**arguments)


def test_model_owned_inputs_are_charged_before_the_first_upload_with_checks_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing fixed parameters, grids, IDs and ages are live before user uploads."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    params = jax.tree.map(np.asarray, params)
    initial = jax.tree.map(np.asarray, initial)
    roots = (
        model.ages.values,
        model.regime_names_to_ids,
        tuple(
            (
                regime.resolved_fixed_params,
                regime.solution.resolved_fixed_params,
                regime.solution._base_state_action_space.states,
                regime.solution._base_state_action_space.actions,
            )
            for regime in model._regimes.values()
        ),
    )
    expected = measure_buffer_footprint(tree=roots)
    assert any(expected.spans.values())
    calls: list[bool] = []
    monkeypatch.setattr(
        entry_allocations_module,
        "place_simulation_arguments",
        _ModelUploadInventory(
            expected=expected,
            original=entry_allocations_module.place_simulation_arguments,
            calls=calls,
        ),
    )
    model.simulate(params=params, initial_conditions=initial, log_level="off")
    assert calls


# keyword-only-exempt: library-callback=functools.partialmethod
def _copy_resolved_values(
    self: Model,
    *,
    observed: _ResolvedCopies,
    original: Callable[..., tuple[object, object, object, object]],
    **arguments: Any,
) -> tuple[object, object, object, object]:
    values, policies, flags, readers = original(self, **arguments)
    copied = jax.tree.map(jnp.copy, values)
    jax.block_until_ready(copied)
    original_spans = measure_buffer_footprint(tree=values)
    copied_spans = measure_buffer_footprint(tree=copied)
    assert any(
        resident_bytes_by_device(
            live=copied_spans,
            arguments=original_spans,
            devices=tuple(copied_spans.spans),
        ).values()
    )
    observed.values = copied
    return copied, policies, flags, readers


# keyword-only-exempt: library-callback=functools.partialmethod
def _check_resolved_snapshot(
    self: SimulationEntryAllocations,
    *,
    observed: _ResolvedCopies,
    original: Callable[..., DeviceBufferFootprint],
) -> DeviceBufferFootprint:
    snapshot = original(self)
    if observed.values is not None:
        spans = measure_buffer_footprint(tree=observed.values)
        missing = resident_bytes_by_device(
            live=spans, arguments=snapshot, devices=tuple(spans.spans)
        )
        assert not any(missing.values()), "Resolved value copies are absent at entry"
    return snapshot


def _check_preflight_inventory(
    *,
    observed: _ResolvedCopies,
    original: Callable[..., object],
    retained_footprint: DeviceBufferFootprint | None,
    **arguments: Any,
) -> object:
    assert retained_footprint is not None
    spans = measure_buffer_footprint(tree=observed.values)
    missing = resident_bytes_by_device(
        live=spans, arguments=retained_footprint, devices=tuple(spans.spans)
    )
    assert not any(missing.values()), "Resolved values are absent from preflight"
    return original(retained_footprint=retained_footprint, **arguments)


def test_resolved_value_copies_remain_charged_before_initial_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's result and separately validated value views are both live."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    solution = model.solve(params=params, log_level="off")
    observed = _ResolvedCopies()
    monkeypatch.setattr(
        Model,
        "_resolve_solution_result",
        partialmethod(
            _copy_resolved_values,
            observed=observed,
            original=Model._resolve_solution_result,
        ),
    )
    monkeypatch.setattr(
        SimulationEntryAllocations,
        "snapshot",
        partialmethod(
            _check_resolved_snapshot,
            observed=observed,
            original=SimulationEntryAllocations.snapshot,
        ),
    )
    monkeypatch.setattr(
        model_module,
        "validate_simulation_inputs",
        partial(
            _check_preflight_inventory,
            observed=observed,
            original=model_module.validate_simulation_inputs,
        ),
    )
    model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="debug"
    )
    assert observed.values is not None


# keyword-only-exempt: library-callback=functools.partialmethod
def _record_padding_dispatch(
    self: jax.stages.Compiled,
    *,
    calls: list[jax.stages.Compiled],
    original: Callable[..., object],
    **arguments: Any,
) -> object:
    calls.append(self)
    return original(self, **arguments)


@pytest.mark.parametrize("stage", ["params", "initial"])
def test_host_numeric_payload_refuses_before_its_first_device_conversion(
    *, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing residency fits; the canonical destination itself cannot fit."""
    model, params, initial = _small_grid_search_inputs(
        execution_config=ExecutionConfig(
            device_memory_bytes=1 if stage == "params" else 128
        )
    )
    params = jax.tree.map(np.asarray, params)
    initial = {
        name: np.repeat(np.asarray(value), 256) for name, value in initial.items()
    }
    source = params["discount_factor"] if stage == "params" else initial["wealth"]
    assert isinstance(source, np.ndarray)
    assert source.dtype.kind in "fiu"
    entry = capture_simulation_entry_inputs(
        execution=model._execution,
        params=params,
        initial_conditions=initial,
        solution=None,
    )
    assert entry is not None
    assert not entry.arrays
    monkeypatch.setattr(
        jnp, "asarray", _ForbiddenConversion(source=source, original=jnp.asarray)
    )
    with pytest.raises(ExecutionPlanningError, match="budget"):
        model.simulate(params=params, initial_conditions=initial, log_level="off")


def _owner(
    *, budget: int, arrays: tuple[jax.Array, ...] = ()
) -> SimulationEntryAllocations:
    return SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=arrays),
        solution=None,
        model_roots=(),
        devices=(jax.devices()[0],),
        budget_bytes=budget,
    )


def test_completed_leaf_is_owned_and_charged_before_the_next_upload() -> None:
    """A completed first leaf cannot disappear from the second leaf's admission."""
    owner = _owner(budget=64)
    first = owner(
        value=np.arange(4, dtype=np.int32), dtype=np.dtype(np.int32), name="first"
    )
    reference = weakref.ref(first)
    first_footprint = measure_buffer_footprint(tree=first)
    del first
    gc.collect()
    assert reference() is not None
    missing = resident_bytes_by_device(
        live=first_footprint, arguments=owner.snapshot(), devices=owner.devices
    )
    assert all(count == 0 for count in missing.values())
    # Existing first16 fits; second destination28 + transfer scratch28 does not.
    with pytest.raises(ExecutionPlanningError, match="budget"):
        owner(
            value=np.arange(7, dtype=np.int32), dtype=np.dtype(np.int32), name="second"
        )
    owner.close()
    gc.collect()
    assert reference() is None


def test_profiled_padding_refuses_before_dispatch_and_rechecks_retained_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real padding profile is reused, while each call admits current owners."""
    source = jax.device_put(np.array([1.0, 2.0, -0.0], dtype=np.float32))
    owner = _owner(budget=2**20, arrays=(source,))
    initial = MappingProxyType({"wealth": source})
    padded, count = owner.pad(initial_conditions=initial, multiple=4)
    assert count == 3
    np.testing.assert_array_equal(
        padded["wealth"], np.array([1.0, 2.0, -0.0, -0.0], dtype=np.float32)
    )
    assert np.signbit(np.asarray(padded["wealth"])[-1])
    profile = next(iter(owner.operations.cache.values()))
    assert profile.peak_bytes >= source.nbytes + padded["wealth"].nbytes
    owner.publish(stage="initial", tree=initial)
    del padded
    calls: list[jax.stages.Compiled] = []
    monkeypatch.setattr(
        jax.stages.Compiled,
        "__call__",
        partialmethod(
            _record_padding_dispatch, calls=calls, original=jax.stages.Compiled.__call__
        ),
    )
    owner.budget_bytes = profile.peak_bytes - 1
    with pytest.raises(ExecutionPlanningError, match="workspace"):
        owner.pad(initial_conditions=initial, multiple=4)
    assert calls == []
    owner.budget_bytes = profile.peak_bytes
    accepted, _ = owner.pad(initial_conditions=initial, multiple=4)
    assert calls == [profile.executable]
    with pytest.raises(ExecutionPlanningError, match="workspace"):
        owner.pad(initial_conditions=initial, multiple=4)
    assert calls == [profile.executable]
    np.testing.assert_array_equal(
        accepted["wealth"], np.array([1.0, 2.0, -0.0, -0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(source, np.array([1.0, 2.0, -0.0], dtype=np.float32))


def test_parameter_memo_and_wrapper_occurrences_keep_their_original_contract() -> None:
    """Top-level sharing survives, while separate nested wrappers still recurse."""
    value = np.array([1, 2, 3], dtype=np.int64)
    first_wrapper = UserMappingLeaf({"numbers": value})
    second_wrapper = UserSequenceLeaf([value])
    flat = cast(
        "FlatParams",
        MappingProxyType(
            {
                "alive": MappingProxyType(
                    {
                        "utility__first": value,
                        "utility__shared": value,
                        "utility__mapping": first_wrapper,
                        "utility__sequence": second_wrapper,
                    }
                )
            }
        ),
    )
    owner = _owner(budget=2**20)
    writer = _RecordingWriter(owner=owner, names=[])
    result = cast_params_to_canonical_dtypes(flat, array_writer=writer)
    assert len(writer.names) == 3
    leaves = result["alive"]
    assert leaves["utility__first"] is leaves["utility__shared"]
    assert isinstance(leaves["utility__mapping"], MappingLeaf)
    assert isinstance(leaves["utility__sequence"], SequenceLeaf)
    for leaf in jax.tree.leaves(result):
        assert isinstance(leaf, jax.Array)
        assert leaf.dtype == np.dtype(np.int32)
        np.testing.assert_array_equal(leaf, [1, 2, 3])


def test_no_padding_and_aligned_boolean_inputs_keep_identity() -> None:
    """No-op contracts do not allocate a replacement mapping or Boolean leaf."""
    source = jax.device_put(np.array([True, False, True]))
    initial = MappingProxyType({"regime_id": jnp.array([0, 0, 0], dtype=jnp.int32)})
    owner = _owner(budget=2**20, arrays=(source,))
    result = owner(value=source, dtype=np.dtype(np.bool_), name="flag")
    assert result is source
    padded, count = owner.pad(initial_conditions=initial, multiple=3)
    assert padded is initial
    assert count == 3
    assert not owner.operations.cache


def test_original_dtype_refusals_happen_before_the_writer_runs() -> None:
    """Range and unsupported-type errors retain their original qualified names."""
    owner = _owner(budget=2**20)
    writer = _RecordingWriter(owner=owner, names=[])
    with pytest.raises(ValueError, match="regime_id: int32 overflow"):
        safe_to_int_dtype(
            value=np.array([2**31], dtype=np.int64),
            name="regime_id",
            array_writer=writer,
        )
    invalid = cast(
        "FlatParams",
        MappingProxyType({"alive": MappingProxyType({"utility__bad": np.array([1j])})}),
    )
    with pytest.raises(InvalidParamsError, match="alive__utility__bad"):
        cast_params_to_canonical_dtypes(invalid, array_writer=writer)
    assert writer.names == []
    large = np.array([1e40], dtype=np.float64)
    if jax.config.jax_enable_x64:
        actual = safe_to_float_dtype(value=large, name="large", array_writer=writer)
        np.testing.assert_array_equal(actual, large)
    else:
        with pytest.raises(OverflowError, match="large: float32 overflow"):
            safe_to_float_dtype(value=large, name="large", array_writer=writer)
        assert writer.names == []


def test_profile_cache_does_not_prolong_closed_entry_owners() -> None:
    """Ready padding outputs are call-owned, and compiled profiles retain no arrays."""
    source = jax.device_put(np.array([1, 2, 3], dtype=np.int32))
    owner = _owner(budget=2**20, arrays=(source,))
    initial = MappingProxyType({"state": source})
    padded, _ = owner.pad(initial_conditions=initial, multiple=4)
    operations = owner.operations
    references = (weakref.ref(source), weakref.ref(padded["state"]), weakref.ref(owner))
    owner.close()
    del source, initial, padded, owner
    gc.collect()
    assert len(operations.cache) == 1
    assert all(reference() is None for reference in references)
