"""A call-local owner ledger reuses spans only inside one unchanged ownership epoch.

Every case here uses real JAX buffers on real devices. The ledger is metadata only:
it records measured address spans, never an array, and it lives no longer than the
`SimulationMemory` that owns it. Admission is always recomputed from the latest
snapshot; no verdict is ever cached.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.simulation import residency
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    OwnerLedger,
    measure_buffer_footprint,
)
from _lcm.simulation.value_reads import PeriodSimulationReads
from lcm.exceptions import ExecutionPlanningError


def _device() -> jax.Device:
    """Use the actual default device the test buffers are created on."""
    return jax.devices()[0]


def _scope(
    *,
    budget_bytes: int = 1_000_000_000,
    inputs: DeviceBufferFootprint | None = None,
) -> SimulationMemory:
    """Build one call-local budgeted accounting scope on the real default device."""
    device = _device()
    return SimulationMemory(
        budget_bytes=budget_bytes,
        devices=(device,),
        subject_devices=(device,),
        operations=ProfiledSimulationOperations(),
        inputs=DeviceBufferFootprint(spans={}) if inputs is None else inputs,
    )


def _double(**attributes: object) -> PeriodSimulationReads:
    """Stand in for a period owner with an explicit live-value inventory."""
    return cast("PeriodSimulationReads", SimpleNamespace(**attributes))


def _resident(*, memory: SimulationMemory) -> int:
    """Charge the current snapshot on the scope's single budgeted device."""
    return residency.resident_bytes_by_device(
        live=memory.budget_snapshot(),
        arguments=DeviceBufferFootprint(spans={}),
        devices=memory.devices,
    )[memory.devices[0]]


class _CountingMeasure:
    """Count full root walks without changing what a measurement returns."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, tree: object) -> DeviceBufferFootprint:
        self.calls += 1
        return measure_buffer_footprint(tree=tree)


@pytest.fixture
def counted_measure(monkeypatch: pytest.MonkeyPatch) -> _CountingMeasure:
    """Replace the measurement used by every ledger-integrated owner."""
    spy = _CountingMeasure()
    monkeypatch.setattr(residency, "measure_buffer_footprint", spy)
    monkeypatch.setattr(
        "_lcm.simulation.memory.measure_buffer_footprint", spy, raising=True
    )
    return spy


# --------------------------------------------------------------------------- ledger


def test_ledger_reuses_one_union_while_its_epoch_is_unchanged() -> None:
    """Repeated snapshots in a stable epoch return the identical merged metadata."""
    device = _device()
    ledger = OwnerLedger()
    ledger.bind(owner="a", footprint=DeviceBufferFootprint(spans={device: ((0, 16),)}))
    first = ledger.union()
    assert ledger.union() is first
    assert ledger.epoch == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ledger, device: ledger.bind(
            owner="b", footprint=DeviceBufferFootprint(spans={device: ((32, 48),)})
        ),
        lambda ledger, device: ledger.bind(
            owner="a", footprint=DeviceBufferFootprint(spans={device: ((0, 8),)})
        ),
        lambda ledger, _device: ledger.release(owner="a"),
        lambda ledger, _device: ledger.release_prefix(prefix="a"),
        lambda ledger, _device: ledger.clear(),
    ],
)
def test_every_ledger_mutation_advances_the_epoch_and_the_union(
    mutate: Callable[[OwnerLedger, jax.Device], None],
) -> None:
    """Publication, replacement and release each invalidate the cached union."""
    device = _device()
    ledger = OwnerLedger()
    ledger.bind(owner="a", footprint=DeviceBufferFootprint(spans={device: ((0, 16),)}))
    before_epoch = ledger.epoch
    before = ledger.union()
    mutate(ledger, device)
    assert ledger.epoch > before_epoch
    assert ledger.union() is not before


def test_ledger_projection_is_cached_per_requested_device_set() -> None:
    """A projected union and an unprojected union are separate stable answers."""
    device = _device()
    ledger = OwnerLedger()
    ledger.bind(owner="a", footprint=DeviceBufferFootprint(spans={device: ((0, 16),)}))
    projected = ledger.union(devices=(device,))
    assert ledger.union(devices=(device,)) is projected
    assert ledger.union() is not projected
    assert dict(projected.spans) == {device: ((0, 16),)}


def test_ledger_never_retains_the_measured_arrays() -> None:
    """A measured binding keeps address metadata only, not the array owner."""
    ledger = OwnerLedger()
    array = jnp.arange(8, dtype=jnp.int32)
    ledger.measure(owner="held:0", tree=(array,))
    assert dict(ledger.union().spans)
    assert not any(
        isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(ledger.union())
    )


# ------------------------------------------------------------------ real buffers


def test_aliased_buffers_are_charged_once_across_separate_bindings() -> None:
    """Two bindings of the same physical payload occupy one span, not two."""
    memory = _scope()
    array = jnp.arange(64, dtype=jnp.int32)
    memory.hold(tree=(array,))
    once = _resident(memory=memory)
    memory.hold(tree=({"alias": array},))
    assert _resident(memory=memory) == once == int(array.nbytes)


def test_a_larger_owner_under_a_smaller_view_keeps_its_whole_extent() -> None:
    """A retained base allocation stays charged while only a view is published."""
    memory = _scope()
    base = jnp.arange(256, dtype=jnp.int32)
    memory.hold(tree=(base,))
    memory.publish(tree=(base[:4],))
    assert _resident(memory=memory) >= int(base.nbytes)


def test_distinct_buffers_are_charged_separately() -> None:
    """Identical descriptors that do not alias occupy the sum of their payloads."""
    memory = _scope()
    first = jnp.arange(64, dtype=jnp.int32)
    second = jnp.arange(64, dtype=jnp.int32) + 1
    memory.hold(tree=(first,))
    memory.hold(tree=(second,))
    assert _resident(memory=memory) == int(first.nbytes) + int(second.nbytes)


# ------------------------------------------------------------------ stable epoch


def test_a_stable_epoch_does_no_further_root_walk(
    counted_measure: _CountingMeasure,
) -> None:
    """Repeated admission checks on unchanged owners re-measure nothing."""
    memory = _scope()
    memory.hold(tree=(jnp.arange(64, dtype=jnp.int32),))
    baseline = counted_measure.calls
    for _ in range(5):
        memory.budget_snapshot()
    assert counted_measure.calls == baseline


def test_holding_more_owners_measures_only_the_new_tree(
    counted_measure: _CountingMeasure,
) -> None:
    """The accumulated unit roots are never re-walked when one owner is added."""
    memory = _scope()
    for index in range(4):
        memory.hold(tree=(jnp.arange(8, dtype=jnp.int32) + index,))
        memory.budget_snapshot()
    assert counted_measure.calls == 4


def test_supplied_additional_roots_are_always_measured_afresh(
    counted_measure: _CountingMeasure,
) -> None:
    """A caller's transient tree is never assumed to belong to the stable epoch."""
    memory = _scope()
    memory.hold(tree=(jnp.arange(8, dtype=jnp.int32),))
    baseline = counted_measure.calls
    transient = jnp.arange(8, dtype=jnp.int32) + 5
    memory.budget_snapshot(additional=(transient,))
    memory.budget_snapshot(additional=(transient,))
    assert counted_measure.calls == baseline + 2


# -------------------------------------------------------------- owner mutations


def test_every_memory_mutation_advances_the_residency_epoch() -> None:
    """Each mutation class named by the ownership design invalidates the union."""
    memory = _scope()
    array = jnp.arange(16, dtype=jnp.int32)
    mutations = (
        lambda: memory.hold(tree=(array,)),
        lambda: memory.set_derived((array,)),
        lambda: memory.set_chunk_inputs(tree=(array,)),
        lambda: memory.publish(tree=(array,)),
        lambda: memory.replace_outputs(tree=(array,)),
        lambda: setattr(memory, "unit_inputs", (array,)),
        lambda: setattr(memory, "inputs", measure_buffer_footprint(tree=(array,))),
        lambda: setattr(
            memory,
            "period_owner",
            cast("PeriodSimulationReads", SimpleNamespace(live_values=())),
        ),
        lambda: setattr(memory, "period_owner", None),
        memory.close_unit,
    )
    for mutation in mutations:
        before = memory.residency_epoch
        mutation()
        assert memory.residency_epoch != before, mutation


def test_rebinding_unit_roots_releases_the_previous_unit_temporaries() -> None:
    """Assigning new unit roots discards earlier held temporaries, as before."""
    memory = _scope()
    memory.hold(tree=(jnp.arange(256, dtype=jnp.int32),))
    charged = _resident(memory=memory)
    memory.unit_inputs = (jnp.arange(8, dtype=jnp.int32),)
    assert _resident(memory=memory) < charged


def test_closing_a_unit_releases_only_call_local_temporaries() -> None:
    """Published outputs and entry inputs survive a unit's temporaries."""
    entry = jnp.arange(128, dtype=jnp.int32)
    memory = _scope(inputs=measure_buffer_footprint(tree=(entry,)))
    published = jnp.arange(64, dtype=jnp.int32)
    memory.publish(tree=(published,))
    memory.hold(tree=(jnp.arange(512, dtype=jnp.int32),))
    memory.close_unit()
    assert _resident(memory=memory) == int(entry.nbytes) + int(published.nbytes)


def test_an_exception_inside_a_unit_preserves_the_caller_owners() -> None:
    """A failed unit drops its temporaries and keeps caller and result owners."""
    entry = jnp.arange(128, dtype=jnp.int32)
    memory = _scope(inputs=measure_buffer_footprint(tree=(entry,)))
    memory.publish(tree=(jnp.arange(64, dtype=jnp.int32),))
    before = _resident(memory=memory)
    try:
        memory.hold(tree=(jnp.arange(4096, dtype=jnp.int32),))
        raise ExecutionPlanningError("unit failed")  # noqa: TRY301
    except ExecutionPlanningError:
        memory.close_unit()
    assert _resident(memory=memory) == before


def test_a_period_owner_generation_change_re_measures_its_live_values() -> None:
    """A materialized read inside the period owner is charged without a rebind."""
    memory = _scope()
    owner = SimpleNamespace(live_values=(), generation=0)
    memory.period_owner = cast("PeriodSimulationReads", owner)
    before = _resident(memory=memory)
    materialized = jnp.arange(256, dtype=jnp.int32)
    owner.live_values = (materialized,)
    owner.generation = 1
    assert _resident(memory=memory) == before + int(materialized.nbytes)


def test_an_uncertain_period_owner_is_re_measured_conservatively() -> None:
    """An owner publishing no generation is re-walked on every snapshot."""
    memory = _scope()
    owner = SimpleNamespace(live_values=())
    memory.period_owner = cast("PeriodSimulationReads", owner)
    before = _resident(memory=memory)
    materialized = jnp.arange(256, dtype=jnp.int32)
    owner.live_values = (materialized,)
    assert _resident(memory=memory) == before + int(materialized.nbytes)


# ------------------------------------------------------------------- admission


@pytest.mark.parametrize("slack", [-1, 0, 1])
def test_budgets_below_at_and_above_the_requirement(slack: int) -> None:
    """Admission refuses only strictly above the budget, at the exact boundary."""
    array = jnp.arange(64, dtype=jnp.int32)
    required = int(array.nbytes)
    memory = _scope(budget_bytes=required + slack)
    memory.hold(tree=(array,))
    if slack < 0:
        with pytest.raises(ExecutionPlanningError):
            memory.check_resident()
    else:
        memory.check_resident()


def test_admission_always_reads_the_latest_snapshot() -> None:
    """A previous acceptance is never reusable once an owner is published."""
    array = jnp.arange(64, dtype=jnp.int32)
    memory = _scope(budget_bytes=int(array.nbytes))
    memory.hold(tree=(array,))
    memory.check_resident()
    memory.publish(tree=(jnp.arange(64, dtype=jnp.int32) + 1,))
    with pytest.raises(ExecutionPlanningError):
        memory.check_resident()


def test_refusal_precedes_any_allocation() -> None:
    """A transfer that cannot fit is refused before its destination is created."""
    array = jnp.arange(64, dtype=jnp.int32)
    memory = _scope(budget_bytes=int(array.nbytes) + 8)
    memory.hold(tree=(array,))
    device = memory.devices[0]
    with pytest.raises(ExecutionPlanningError, match="before allocation"):
        residency.require_transfer_headroom(
            live=memory.budget_snapshot(),
            destination_bytes={device: 64},
            scratch_bytes={},
            budget_bytes=memory.budget_bytes,
            devices=memory.devices,
        )


def test_retained_same_backend_devices_stay_in_the_budget_projection() -> None:
    """A non-selected same-backend source device keeps its own ceiling."""
    device = _device()
    live = DeviceBufferFootprint(spans={device: ((0, 16),)})
    assert residency.resolve_budget_devices(execution_devices=(device,), live=live) == (
        device,
    )


def test_compiler_kept_inputs_are_subtracted_only_on_actual_intersections() -> None:
    """A distinct buffer of the same size never cancels a compiler argument."""
    memory = _scope()
    live_only = jnp.arange(64, dtype=jnp.int32)
    memory.hold(tree=(live_only,))
    unrelated = jnp.arange(64, dtype=jnp.int32) + 7
    charged: Mapping[jax.Device, int] = residency.resident_bytes_by_device(
        live=memory.budget_snapshot(),
        arguments=measure_buffer_footprint(tree=(unrelated,)),
        devices=memory.devices,
    )
    assert charged[memory.devices[0]] == int(live_only.nbytes)


def test_ledger_bindings_are_not_shared_between_two_scopes() -> None:
    """Two call-local scopes never observe each other's owners."""
    first = _scope()
    second = _scope()
    first.hold(tree=(jnp.arange(64, dtype=jnp.int32),))
    assert _resident(memory=second) == 0
    assert dict(second.budget_snapshot().spans) in ({}, {second.devices[0]: ()})


def test_axis_widths_stay_owned_by_the_scope() -> None:
    """The ledger hook does not disturb the existing immutable field contract."""
    memory = _scope()
    memory.axis_widths = MappingProxyType({"x": 4})
    assert dict(memory.axis_widths) == {"x": 4}
