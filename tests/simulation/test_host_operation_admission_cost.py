"""Budgeted simulation helpers admit through one axis-free plan, charging the same.

A budgeted forward call runs every pure simulation helper -- regime masks, key
splits, state advances, membership activation -- through
`ProfiledSimulationOperations.dispatch`, which admits each one against the live
inventory before it allocates. An unbudgeted call runs the same helpers as eager
primitives and admits nothing.

The cost of that difference is the admission arithmetic itself, so nothing here
removes a charge or a refusal. It pins the three places where the same answer was
being computed more than once per dispatch:

* a helper declares no width axis, so its frontier has one candidate; it is
  admitted by `plan_axis_free_workspace`, which applies the identical feasibility
  test and raises the identical messages as `plan_workspace(axes=())`;
* the operand tree is measured once and charged to both the placement headroom
  check and the dispatch admission, instead of twice;
* binding an owner name the ledger never held only adds address ranges, so it is
  folded into the cached projections rather than forcing a full re-merge.

Every case asserts that the charge on the budget device is never below what a
fresh measurement of the live owners would establish, that a budget below the
requirement still refuses, and that the simulated results are unchanged.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import (
    CompilerMemoryRecord,
    CompilerMemoryReservation,
    WorkspacePlan,
    plan_axis_free_workspace,
    plan_workspace,
)
from _lcm.simulation import host_operations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    OwnerLedger,
    measure_buffer_footprint,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from benchmarks.asv._simulation_witnesses import dissolution
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig

# The dissolution witness runs this many pure helper operations in one warm
# forward call. The count is a property of the fixture's regimes and periods,
# not of the admission route, and must not move when the route changes.
_WITNESS_HELPER_OPERATIONS = 51

_WITNESS_SEED = 6606


def _budget() -> ExecutionConfig:
    """Declare a budget far above the witness requirement on one device."""
    return ExecutionConfig(devices=(0,), device_memory_bytes=2**30)


def _reservation(*, peak: int, allocation: int) -> CompilerMemoryReservation:
    """Report one device's compiler accounting with an exact represented total."""
    return CompilerMemoryReservation(
        records=(
            CompilerMemoryRecord(
                peak_bytes=peak,
                argument_bytes=0,
                output_bytes=0,
                alias_bytes=0,
                temporary_bytes=allocation,
            ),
        )
    )


def _outcome(
    plan: Callable[[], WorkspacePlan[object]],
) -> tuple[object, ...] | str:
    """Reduce a planner call to its comparable verdict: selection or refusal text."""
    try:
        result = plan()
    except ExecutionPlanningError as error:
        return str(error)
    return (
        result.compiled,
        result.reservation_bytes,
        result.peak_bytes,
        dict(result.widths),
    )


class _CountingCompiler:
    """Return one candidate while counting how often the planner compiled it."""

    def __init__(self, *, result: object) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *_widths: object) -> object:
        """Serve both planner protocols: axis-free takes no width mapping."""
        self.calls += 1
        return self.result


# --------------------------------------------------------------- one admission route


def test_every_budgeted_helper_admits_through_the_axis_free_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One warm forward call plans each helper once, and never on the frontier."""
    counts: dict[str, int] = {"dispatch": 0, "axis_free": 0, "frontier": 0}
    recording = {"on": False}

    original_dispatch = host_operations.ProfiledSimulationOperations.dispatch
    original_axis_free = host_operations.plan_axis_free_workspace

    def counted_dispatch(
        self: host_operations.ProfiledSimulationOperations, **kwargs: Any
    ) -> object:
        if recording["on"]:
            counts["dispatch"] += 1
        return original_dispatch(self, **kwargs)

    def counted_axis_free(**kwargs: Any) -> object:
        if recording["on"]:
            counts["axis_free"] += 1
        return original_axis_free(**kwargs)

    def counted_frontier(**kwargs: Any) -> object:
        if recording["on"]:
            counts["frontier"] += 1
        return plan_workspace(**kwargs)

    monkeypatch.setattr(
        host_operations.ProfiledSimulationOperations, "dispatch", counted_dispatch
    )
    monkeypatch.setattr(host_operations, "plan_axis_free_workspace", counted_axis_free)
    monkeypatch.setattr(
        host_operations, "plan_workspace", counted_frontier, raising=False
    )

    model, params, initial = dissolution(execution_config=_budget())
    solution = model.solve(params=params, log_level="off")
    for _ in range(2):
        jax.block_until_ready(
            model.simulate(
                params=params,
                initial_conditions=initial,
                solution=solution,
                seed=_WITNESS_SEED,
                log_level="off",
            )
        )
    recording["on"] = True
    jax.block_until_ready(
        model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=_WITNESS_SEED,
            log_level="off",
        )
    )
    recording["on"] = False

    assert counts["dispatch"] == _WITNESS_HELPER_OPERATIONS
    assert counts["axis_free"] == _WITNESS_HELPER_OPERATIONS
    assert counts["frontier"] == 0


@pytest.mark.parametrize(
    ("allocation", "resident", "budget"),
    [
        (16, 0, 1_024),
        (16, 1_000, 1_024),
        (1_024, 0, 1_024),
        (1_025, 0, 1_024),
        (16, 1_024, 1_024),
        (16, 2_048, 1_024),
        (512, 600, 1_024),
    ],
)
def test_the_axis_free_plan_admits_exactly_what_the_frontier_planner_admits(
    *, allocation: int, resident: int, budget: int
) -> None:
    """The specialization grants and refuses the same positions, with one message."""
    memory = _reservation(peak=allocation + 7, allocation=allocation)
    marker = object()

    frontier_compiler = _CountingCompiler(result=marker)
    axis_free_compiler = _CountingCompiler(result=marker)

    frontier = _outcome(
        lambda: plan_workspace(
            axes=(),
            compile_candidate=frontier_compiler,
            budget_bytes=budget,
            resident_bytes=resident,
            memory_for=lambda _compiled: memory,
        )
    )
    axis_free = _outcome(
        lambda: plan_axis_free_workspace(
            compile_candidate=axis_free_compiler,
            budget_bytes=budget,
            resident_bytes=resident,
            memory_for=lambda _compiled: memory,
        )
    )
    assert axis_free == frontier
    if not isinstance(axis_free, str):
        assert axis_free[0] is marker
        assert axis_free[3] == {}
    # An exhausted position is refused before anything is compiled, on both routes.
    assert axis_free_compiler.calls == frontier_compiler.calls


def test_an_exhausted_position_refuses_before_the_helper_is_compiled() -> None:
    """No helper is compiled for a position whose residency already fills the budget."""
    compiler = _CountingCompiler(result=object())
    with pytest.raises(ExecutionPlanningError, match="leaving nothing of the"):
        plan_axis_free_workspace(
            compile_candidate=compiler,
            budget_bytes=1_024,
            resident_bytes=1_024,
            memory_for=lambda _compiled: _reservation(peak=1, allocation=1),
        )
    assert compiler.calls == 0


def test_a_budget_below_the_helper_requirement_still_refuses() -> None:
    """A budget no helper fits raises the planner's refusal, not a silent run."""
    model, params, initial = dissolution(
        execution_config=ExecutionConfig(devices=(0,), device_memory_bytes=1)
    )
    with pytest.raises(ExecutionPlanningError):
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=_WITNESS_SEED,
            log_level="off",
        )


# --------------------------------------------------------------------- owner ledger


def _footprint(
    *, device: jax.Device, spans: tuple[tuple[int, int], ...]
) -> DeviceBufferFootprint:
    """Bind explicit address ranges on one actual device."""
    return DeviceBufferFootprint(spans={device: spans})


def test_folding_a_new_owner_yields_the_full_re_merge_exactly() -> None:
    """An appended binding's projection equals a ledger merged from scratch."""
    device = jax.devices()[0]
    bindings = {
        "inputs": _footprint(device=device, spans=((0, 16), (64, 96))),
        "held:1": _footprint(device=device, spans=((16, 32),)),
        "held:2": _footprint(device=device, spans=((90, 128), (256, 300))),
        "held:3": _footprint(device=device, spans=((300, 301),)),
    }
    incremental = OwnerLedger()
    for owner, footprint in bindings.items():
        incremental.bind(owner=owner, footprint=footprint)
        # Force a projection to exist, so the next bind folds into it.
        incremental.union(devices=(device,))
    expected = union_buffer_footprints(
        footprints=tuple(bindings.values()), devices=(device,)
    )
    assert dict(incremental.union(devices=(device,)).spans) == dict(expected.spans)
    assert dict(incremental.union().spans) == dict(
        union_buffer_footprints(footprints=tuple(bindings.values())).spans
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ledger, device: ledger.bind(
            owner="held:1", footprint=_footprint(device=device, spans=((0, 8),))
        ),
        lambda ledger, _device: ledger.release(owner="held:1"),
        lambda ledger, _device: ledger.release_prefix(prefix="held:"),
        lambda ledger, _device: ledger.clear(),
        lambda ledger, _device: ledger.bump(),
    ],
)
def test_a_mutation_that_can_remove_bytes_never_reuses_a_folded_union(
    mutate: Callable[[OwnerLedger, jax.Device], None],
) -> None:
    """Only an addition folds; anything that can shrink an owner set re-merges."""
    device = jax.devices()[0]
    bindings = {
        "held:1": _footprint(device=device, spans=((0, 16),)),
        "inputs": _footprint(device=device, spans=((64, 96),)),
    }
    ledger = OwnerLedger()
    reference = OwnerLedger()
    for owner, footprint in bindings.items():
        ledger.bind(owner=owner, footprint=footprint)
        reference.bind(owner=owner, footprint=footprint)
    before = ledger.union(devices=(device,))
    before_epoch = ledger.epoch
    mutate(ledger, device)
    mutate(reference, device)
    assert ledger.epoch > before_epoch
    after = ledger.union(devices=(device,))
    assert after is not before
    # `reference` never served a projection, so it merges from its bindings.
    assert dict(after.spans) == dict(reference.union(devices=(device,)).spans)


def test_an_appended_owner_advances_the_epoch_and_the_charge() -> None:
    """A fold is still a new epoch, so no admission verdict survives the addition."""
    device = jax.devices()[0]
    ledger = OwnerLedger()
    ledger.bind(owner="inputs", footprint=_footprint(device=device, spans=((0, 16),)))
    before = ledger.union(devices=(device,))
    before_epoch = ledger.epoch
    ledger.bind(owner="held:1", footprint=_footprint(device=device, spans=((64, 96),)))
    assert ledger.epoch > before_epoch
    after = ledger.union(devices=(device,))
    assert after is not before
    assert dict(after.spans) == {device: ((0, 16), (64, 96))}


# ----------------------------------------------------------------- charged residency


class _ChargeWitness:
    """Record, per snapshot, the charge against a fresh walk of the live owners."""

    def __init__(self) -> None:
        self.samples: list[tuple[int, int]] = []

    def observe(
        self, *, scope: SimulationMemory, charged_footprint: DeviceBufferFootprint
    ) -> None:
        """Compare the scope's charge with a fresh measurement of its owner trees."""
        device = scope.devices[0]
        charged = resident_bytes_by_device(
            live=charged_footprint,
            arguments=DeviceBufferFootprint(spans={}),
            devices=scope.devices,
        )[device]
        owner = scope.period_owner
        trees: list[object] = [
            tuple(scope._held),
            scope.unit_inputs,
            scope.derived,
            () if owner is None else owner.live_values,
        ]
        fresh = resident_bytes_by_device(
            live=union_buffer_footprints(
                footprints=tuple(measure_buffer_footprint(tree=tree) for tree in trees),
                devices=scope.devices,
            ),
            arguments=DeviceBufferFootprint(spans={}),
            devices=scope.devices,
        )[device]
        self.samples.append((charged, fresh))


def test_the_charge_never_falls_below_a_fresh_measurement_of_the_live_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every budgeted snapshot of a real forward call over-charges, never under."""
    witness = _ChargeWitness()
    original = SimulationMemory.budget_snapshot

    # keyword-only-exempt: library-callback=SimulationMemory.budget_snapshot
    def observed(
        self: SimulationMemory, *, additional: object = ()
    ) -> DeviceBufferFootprint:
        result = original(self, additional=additional)
        witness.observe(scope=self, charged_footprint=result)
        return result

    monkeypatch.setattr(SimulationMemory, "budget_snapshot", observed)

    model, params, initial = dissolution(execution_config=_budget())
    jax.block_until_ready(
        model.simulate(
            params=params,
            initial_conditions=initial,
            seed=_WITNESS_SEED,
            log_level="off",
        )
    )
    assert len(witness.samples) > _WITNESS_HELPER_OPERATIONS
    assert all(charged >= fresh for charged, fresh in witness.samples)
    assert any(fresh > 0 for _charged, fresh in witness.samples)


# ------------------------------------------------------------------ operand charging


def _placement_arguments(*, device: jax.Device) -> Mapping[str, object]:
    """Mix an already-placed array with leaves placement still has to move."""
    return MappingProxyType(
        {
            "grid": jax.device_put(
                jnp.arange(32, dtype=jnp.int32),
                jax.sharding.SingleDeviceSharding(device),
            ),
            "scalars": (np.float32(1.5), 3),
            "states": jnp.arange(16, dtype=jnp.int32),
        }
    )


def _place(
    *,
    arguments: Mapping[str, object],
    device: jax.Device,
    budget_bytes: int,
    live: DeviceBufferFootprint,
    argument_footprint: DeviceBufferFootprint | None = None,
) -> Mapping[str, object]:
    """Place the same operands under the two spellings of the live inventory."""
    return place_simulation_arguments(
        arguments=arguments,
        subject_arg_names=("states",),
        value_reads=(),
        devices=(device,),
        budget_bytes=budget_bytes,
        live_footprint=live,
        argument_footprint=argument_footprint,
        budget_devices=(device,),
    )


def test_a_supplied_operand_measurement_charges_exactly_what_a_fresh_one_does() -> None:
    """Passing the operand footprint in changes placement's charge in no way."""
    device = jax.devices()[0]
    arguments = _placement_arguments(device=device)
    live = measure_buffer_footprint(tree=jnp.arange(8, dtype=jnp.int32))
    fresh = _place(arguments=arguments, device=device, budget_bytes=2**30, live=live)
    supplied = _place(
        arguments=arguments,
        device=device,
        budget_bytes=2**30,
        live=live,
        argument_footprint=measure_buffer_footprint(tree=arguments),
    )
    assert jax.tree.structure(fresh) == jax.tree.structure(supplied)
    for got, want in zip(
        jax.tree.leaves(fresh), jax.tree.leaves(supplied), strict=True
    ):
        assert got.sharding == want.sharding
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_a_supplied_operand_measurement_refuses_at_the_same_budget() -> None:
    """Both spellings of the live inventory refuse the same too-small budget."""
    device = jax.devices()[0]
    arguments = _placement_arguments(device=device)
    live = measure_buffer_footprint(tree=jnp.arange(8, dtype=jnp.int32))
    with pytest.raises(ExecutionPlanningError) as fresh:
        _place(arguments=arguments, device=device, budget_bytes=8, live=live)
    with pytest.raises(ExecutionPlanningError) as supplied:
        _place(
            arguments=arguments,
            device=device,
            budget_bytes=8,
            live=live,
            argument_footprint=measure_buffer_footprint(tree=arguments),
        )
    assert str(supplied.value) == str(fresh.value)


def test_a_budgeted_placement_still_requires_a_live_inventory() -> None:
    """An operand measurement alone does not authorize a budgeted placement."""
    device = jax.devices()[0]
    arguments = _placement_arguments(device=device)
    with pytest.raises(ExecutionPlanningError, match="live budget inventory"):
        place_simulation_arguments(
            arguments=arguments,
            subject_arg_names=("states",),
            value_reads=(),
            devices=(device,),
            budget_bytes=2**30,
            argument_footprint=measure_buffer_footprint(tree=arguments),
            budget_devices=(device,),
        )


# ----------------------------------------------------------------- result identity


def test_a_budgeted_forward_call_returns_the_unbudgeted_results_exactly() -> None:
    """Admitting each helper changes no simulated value and no result structure."""
    unbudgeted, params, initial = dissolution()
    expected = unbudgeted.simulate(
        params=params,
        initial_conditions=initial,
        seed=_WITNESS_SEED,
        log_level="off",
    )
    budgeted, params, initial = dissolution(execution_config=_budget())
    actual = budgeted.simulate(
        params=params,
        initial_conditions=initial,
        seed=_WITNESS_SEED,
        log_level="off",
    )
    assert jax.tree.structure(actual.raw_results) == jax.tree.structure(
        expected.raw_results
    )
    for got, want in zip(
        jax.tree.leaves(actual.raw_results),
        jax.tree.leaves(expected.raw_results),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
