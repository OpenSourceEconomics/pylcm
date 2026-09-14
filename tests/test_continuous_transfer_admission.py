"""Native ALL_GATHER admission and interrupted-copy ownership on eight CPUs.

This isolates the transport budget from the economic solver: the source has the
same canonical 3x3x24 layout as the public continuous witness, while a compiled
shape-only core demonstrably prunes both addressed input values. No topology is
set here. Run this file explicitly in a fresh eight-device CPU process.
"""

import dataclasses
import gc
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution import value_transfer
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.footprint import ResidentInventory, concrete_device_bytes
from _lcm.execution.output_layout import VALUE, PlannedCore, resolve_output_layout
from _lcm.execution.pending_work import PendingSolveWork
from _lcm.execution.scheduler import (
    BufferRegistry,
    PeriodTransferCache,
    shares_a_buffer,
)
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.execution.workspace_planning import (
    compiler_memory_reservation,
    plan_workspace,
)
from _lcm.solution import backward_induction
from lcm.exceptions import ExecutionPlanningError


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Case:
    source: jax.Array
    extra: jax.Array
    source_snapshot: np.ndarray
    program: ResolvedCoreProgram
    executable: jax.stages.Compiled
    inventory: ResidentInventory
    transfers: tuple[ResolvedValueTransfer, ...]
    compiler_bytes: int
    owner_bytes: int
    replica_bytes: int
    scratch_bytes: int


def _shape_only(*, values: Mapping[str, jax.Array]) -> jax.Array:
    first = values["first"]
    return jnp.arange(first.size, dtype=first.dtype).reshape(first.shape)


def _case(*, shared: bool) -> _Case:
    if jax.default_backend() != "cpu" or jax.device_count() != 8:
        pytest.skip("Requires an isolated eight-device CPU process")
    devices = tuple(jax.devices())
    ids = tuple(device.id for device in devices)
    mesh = jax.sharding.Mesh(np.array(devices), ("assets",))
    stored = jax.NamedSharding(mesh, jax.P(None, None, "assets"))
    dtype = np.dtype(jnp.zeros(()).dtype)
    source_snapshot = np.arange(216, dtype=dtype).reshape(3, 3, 24) - 300
    source = jax.device_put(source_snapshot, stored)
    extra = jax.device_put(source_snapshot - 1, stored)
    jax.block_until_ready((source, extra))
    assert not shares_a_buffer(first=source, second=extra)
    _assert_source_shards(source)
    _assert_source_shards(extra)
    # Resolve through the actual newly enabled ordinary full-read layout seam.
    kind, required = backward_induction._resolve_value_transfer_layout(
        stored_sharding=stored,
        source_execution_sharding=stored,
        require_full_replica=True,
    )
    assert kind is ValueTransferKind.ALL_GATHER
    assert required.is_fully_replicated
    address = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="done"
    )
    transfers = tuple(
        ResolvedValueTransfer(
            target=address,
            source=ValueConsumerAddress(
                source_period=0,
                source_regime="acting",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                argument="values",
                path=(name,),
            ),
            kind=kind,
            stored_sharding=stored,
            source_sharding=required,
            expected_shape=source.shape,
            expected_dtype=source.dtype,
            reused_by_several_consumers=shared,
        )
        for name in ("first", "second")
    )
    abstract = jax.ShapeDtypeStruct(source.shape, source.dtype, sharding=required)
    arguments = {"values": MappingProxyType({"first": abstract, "second": abstract})}
    executable = jax.jit(_shape_only, out_shardings=stored).lower(**arguments).compile()
    assert all(
        leaf is None for leaf in executable.input_shardings[1]["values"].values()
    )
    program = ResolvedCoreProgram(
        name="main",
        function=_shape_only,
        arguments=arguments,
        static_kwargs={},
        requirements=CoreExecutionRequirements(
            value_reads=tuple(
                ValueRead(target=address, source=transfer.source)
                for transfer in transfers
            )
        ),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths={},
        specialization_key=(),
        input_transfer_plan=transfers,
    )
    metadata = {
        ("acting", 0, "main"): backward_induction._ProgramExecutionMetadata(
            requirements=program.requirements,
            disposition=program.disposition,
            scope=program.scope,
            input_transfer_plan=transfers,
        )
    }
    copies = backward_induction._period_copy_reservations(period=0, metadata=metadata)
    scratch = backward_induction._period_transfer_scratch_reservations(
        period=0,
        metadata=metadata,
        device_ids=ids,
    )
    # Independent byte oracle: two distinct local 27-element owners; every
    # replicated destination and declared operator scratch holds all216 elements.
    full_bytes = 216 * dtype.itemsize
    owner_bytes = 2 * 27 * dtype.itemsize
    copy_count = 1 if shared else 2
    owners = concrete_device_bytes(tree=(source, extra))
    assert dict(owners) == dict.fromkeys(ids, owner_bytes)
    assert dict(scratch) == dict.fromkeys(ids, copy_count * full_bytes)
    inventory = ResidentInventory(
        device_ids=ids,
        live={},
        peer_bytes=dict.fromkeys(ids, 0),
        declared_inputs=(),
        fixed_bytes=owners,
        shared_copies=copies,
        transfer_scratch_bytes=scratch,
    )
    return _Case(
        source=source,
        extra=extra,
        source_snapshot=source_snapshot,
        program=program,
        executable=executable,
        inventory=inventory,
        transfers=transfers,
        compiler_bytes=compiler_memory_reservation(
            compiled=executable, widths={}
        ).reservation_bytes,
        owner_bytes=owner_bytes,
        replica_bytes=copy_count * full_bytes,
        scratch_bytes=copy_count * full_bytes,
    )


def _assert_source_shards(value: jax.Array) -> None:
    assert value.shape == (3, 3, 24)
    assert len(value.addressable_shards) == 8
    cells = np.arange(216).reshape(3, 3, 24)
    observed = []
    for shard in value.addressable_shards:
        assert shard.data.shape == (3, 3, 3)
        observed.extend(cells[shard.index].ravel().tolist())
    assert sorted(observed) == list(range(216))


def _resident(*, case: _Case, inventory: ResidentInventory) -> int:
    return backward_induction._candidate_resident_bytes(
        compiled=case.executable,
        program=case.program,
        internal_arguments={},
        inventory=inventory,
    )


def _core(*, case: _Case, owner: PendingSolveWork, shared: bool) -> PlannedCore:
    transfer = case.transfers[0]
    cache = (
        PeriodTransferCache(
            registry=BufferRegistry(),
            consumer_counts={(transfer.target, transfer.source_sharding): 1},
            before_delete=owner.before_delete,
        )
        if shared
        else None
    )
    return PlannedCore(
        compiled=case.executable,
        name="main",
        tile_widths={},
        layout=resolve_output_layout(
            core_key="main",
            value_template=case.source,
            state_order=("pref_type", "spousal_income", "assets"),
            output_roles=VALUE,
        ),
        input_transfer_plan=case.transfers,
        transfer_cache=cache,
        pending_work=owner,
    )


@pytest.mark.parametrize("shared", [False, True])
def test_pruned_full_replicas_and_scratch_have_an_exact_admission_threshold(
    *, shared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(shared=shared)
    expected_resident = case.owner_bytes + case.replica_bytes + case.scratch_bytes
    assert _resident(case=case, inventory=case.inventory) == expected_resident
    ceiling = case.compiler_bytes + expected_resident
    copies = []
    dispatches = []
    apply = value_transfer.apply_value_transfer
    call = jax.stages.Compiled.__call__

    def observe_copy(**kwargs: Any) -> jax.Array:
        copied = apply(**kwargs)
        assert kwargs["transfer"].kind is ValueTransferKind.ALL_GATHER
        assert copied.is_fully_replicated
        assert len(copied.addressable_shards) == 8
        assert all(
            shard.data.shape == case.source.shape for shard in copied.addressable_shards
        )
        np.testing.assert_array_equal(copied, case.source_snapshot)
        copies.append(copied)
        return copied

    def observe_dispatch(
        executable: jax.stages.Compiled, *args: Any, **kwargs: Any
    ) -> Any:
        if executable is case.executable:
            dispatches.append(True)
        return call(executable, *args, **kwargs)

    monkeypatch.setattr(value_transfer, "apply_value_transfer", observe_copy)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", observe_dispatch)

    def plan(*, budget: int, inventory: ResidentInventory) -> Any:
        return plan_workspace(
            axes=(),
            compile_candidate=lambda _widths: case.executable,
            budget_bytes=budget,
            resident_bytes_for=lambda _compiled: _resident(
                case=case, inventory=inventory
            ),
        )

    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        plan(budget=ceiling - 1, inventory=case.inventory)
    assert copies == []
    assert dispatches == []
    admitted = plan(budget=ceiling, inventory=case.inventory)
    assert admitted.compiled is case.executable
    # Negative control: the SAME just-below ceiling wrongly admits when only
    # operator scratch is omitted. No rejected/mutated candidate is dispatched.
    omitted = dataclasses.replace(case.inventory, transfer_scratch_bytes={})
    assert plan(budget=ceiling - 1, inventory=omitted).compiled is case.executable
    assert copies == []
    assert dispatches == []
    owner = PendingSolveWork()
    try:
        result = _core(case=case, owner=owner, shared=shared)(
            values={"first": case.source, "second": case.source}
        )
        jax.block_until_ready((result, copies))
        assert dispatches == [True]
        assert len(copies) == (1 if shared else 2)
        full_bytes = case.source.size * case.source.dtype.itemsize
        assert dict(concrete_device_bytes(tree=copies)) == dict.fromkeys(
            range(8), len(copies) * full_bytes
        )
        np.testing.assert_array_equal(result, np.arange(216).reshape(3, 3, 24))
    finally:
        owner.close()
    _assert_originals(case)


def _assert_originals(case: _Case) -> None:
    for value, expected in (
        (case.source, case.source_snapshot),
        (case.extra, case.source_snapshot - 1),
    ):
        assert not value.is_deleted()
        _assert_source_shards(value)
        np.testing.assert_array_equal(value, expected)


def test_interrupted_all_gather_materialization_keeps_copy_until_owner_close(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(shared=False)
    owner = PendingSolveWork()
    core = dataclasses.replace(
        _core(case=case, owner=owner, shared=False),
        # First transfer is real; its repeated locator then interrupts the plan
        # before dispatch. No numerical kernel or source artifact is modified.
        input_transfer_plan=(case.transfers[0], case.transfers[0]),
    )
    references = []
    apply = value_transfer.apply_value_transfer

    def observe_copy(**kwargs: Any) -> jax.Array:
        copied = apply(**kwargs)
        assert kwargs["transfer"].kind is ValueTransferKind.ALL_GATHER
        assert copied.is_fully_replicated
        assert not shares_a_buffer(first=copied, second=case.source)
        references.append(weakref.ref(copied))
        return copied

    monkeypatch.setattr(value_transfer, "apply_value_transfer", observe_copy)
    try:
        with pytest.raises(
            ValueError, match="Duplicate value-transfer consumer path"
        ) as caught:
            core(values={"first": case.source, "second": case.source})
        caught.value.__traceback__ = None
        del caught
        assert len(references) == 1
        gc.collect()
        assert references[0]() is not None
        owner.close()
        gc.collect()
        assert references[0]() is None
        _assert_originals(case)
    finally:
        owner.close()
