"""Admission of selected value views: a fresh block never stands in for its owner.

The public-path probe only observes normal solver calls; it does not forge or
modify model internals. The small synthetic case uses the same transfer and
inventory interfaces as test_continuous_transfer_admission.py.
"""

import dataclasses
import math
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ProgramScope,
    ResolvedCoreProgram,
    ValueRead,
)
from _lcm.execution.footprint import ResidentInventory
from _lcm.execution.output_layout import VALUE
from _lcm.execution.value_transfer import (
    CoordinateSelection,
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
    ValueViewDescriptor,
    ValueViewLeaf,
)
from _lcm.execution.workspace_planning import (
    compiler_memory_reservation,
    plan_workspace,
)
from _lcm.solution import backward_induction as bi
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from tests.solution.test_invariant_blocking import _independent_types_model
from tests.test_models import independent_types


def test_public_blocked_solve_never_credits_a_selected_input_as_its_owner(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compiler-live compact input cannot cover its separate stored owner."""
    original = bi._candidate_resident_bytes
    checked: list[tuple[int, int]] = []

    def observe(**kwargs: Any) -> int:
        got = original(**kwargs)
        program = kwargs["program"]
        transfers = program.input_transfer_plan
        # Isolate a normal unshared selected read, not a mixed/shared graph.
        if transfers and all(
            t.selects
            and t.kind is ValueTransferKind.ALIGNED_LOCAL
            and not t.reused_by_several_consumers
            for t in transfers
        ):
            shardings = kwargs["compiled"].input_shardings[1]
            if all(
                bi._compiler_reads_source(shardings=shardings, source=t.source)
                for t in transfers
            ):
                expected = kwargs["inventory"].resident_bytes(
                    consumes=(),
                    consumed_copies=frozenset(),
                    temporary_bytes={},
                )
                checked.append((got, expected))
                assert got == expected, (
                    "A selected argument is not the full stored owner; "
                    f"got resident={got}, owner-preserving resident={expected}"
                )
        return got

    monkeypatch.setattr(bi, "_candidate_resident_bytes", observe)
    model = _independent_types_model(
        execution_config=ExecutionConfig(
            devices=(0,),
            device_memory_bytes=2**30,
            invariant_block_widths={"pref_type": 1},
        )
    )
    model.solve(params=independent_types.get_params(), log_level="off")
    assert checked, "The public witness must reach a compiler-live selected read."


def _shape_only(*, next_regime_to_V_arr: dict[str, jax.Array]) -> jax.Array:
    value = next_regime_to_V_arr["terminal"]
    return jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Case:
    program: ResolvedCoreProgram
    compiled: Any
    inventory: ResidentInventory
    metadata: dict
    owner: jax.Array
    owner_bytes: int
    block_bytes: int
    compiler_bytes: int
    device_id: int


def _case(*, axis: int, code: int, cells: int, shared: bool) -> _Case:
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    shape = (3, cells) if axis == 0 else (cells, 3)
    names = ("pref_type", "assets") if axis == 0 else ("assets", "pref_type")
    owner = jax.device_put(
        np.arange(math.prod(shape), dtype=dtype).reshape(shape), sharding
    )
    target = ValueArtifactAddress(
        kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="terminal"
    )
    source = ValueConsumerAddress(
        source_period=0,
        source_regime="working",
        core_key="main",
        channel=ValueInputChannel.NEXT_REGIME_VALUE,
        path=("terminal",),
    )
    view = ValueViewDescriptor(
        artifact=target,
        leaf=ValueViewLeaf.SELECTED,
        stored_axis_names=names,
        stored_shape=shape,
        dtype=dtype,
        weak_type=False,
        consumer_shape=(cells,),
        required_sharding=sharding,
        selections=(
            CoordinateSelection(
                state_name="pref_type", start=code, width=1, codes=(code,)
            ),
        ),
    )
    transfer = ResolvedValueTransfer(
        target=target,
        source=source,
        kind=ValueTransferKind.ALIGNED_LOCAL,
        stored_sharding=sharding,
        source_sharding=sharding,
        expected_shape=shape,
        expected_dtype=dtype,
        view=view,
        reused_by_several_consumers=shared,
    )
    args = {
        "next_regime_to_V_arr": {
            "terminal": jax.ShapeDtypeStruct((cells,), dtype, sharding=sharding)
        }
    }
    compiled = jax.jit(_shape_only, out_shardings=sharding).lower(**args).compile()
    assert all(
        x is None for x in compiled.input_shardings[1]["next_regime_to_V_arr"].values()
    )
    requirements = CoreExecutionRequirements(
        value_reads=(ValueRead(target=target, source=source, view=view),)
    )
    program = ResolvedCoreProgram(
        name="main",
        function=_shape_only,
        arguments=MappingProxyType(dict(args)),
        static_kwargs=MappingProxyType({}),
        requirements=requirements,
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
        tile_widths=MappingProxyType({}),
        specialization_key=(),
        input_transfer_plan=(transfer,),
    )
    metadata = {
        ("working", 0, "main"): bi._ProgramExecutionMetadata(
            requirements=requirements,
            disposition=program.disposition,
            scope=ProgramScope.ANY,
            input_transfer_plan=(transfer,),
        )
    }
    if shared:
        second_source = dataclasses.replace(source, source_regime="second")
        second_transfer = dataclasses.replace(transfer, source=second_source)
        metadata[("second", 0, "main")] = bi._ProgramExecutionMetadata(
            requirements=CoreExecutionRequirements(
                value_reads=(
                    ValueRead(
                        target=target,
                        source=second_source,
                        view=view,
                    ),
                )
            ),
            disposition=program.disposition,
            scope=ProgramScope.ANY,
            input_transfer_plan=(second_transfer,),
        )
    scratch = bi._period_transfer_scratch_reservations(
        period=0,
        metadata=metadata,
        device_ids=(device.id,),
    )
    copies = bi._period_copy_reservations(period=0, metadata=metadata)
    owner_bytes, block_bytes = 3 * cells * dtype.itemsize, cells * dtype.itemsize
    inventory = ResidentInventory(
        device_ids=(device.id,),
        live={},
        peer_bytes={device.id: 0},
        declared_inputs=(),
        fixed_bytes=MappingProxyType({device.id: owner_bytes}),
        shared_copies=copies,
        transfer_scratch_bytes=scratch,
    )
    return _Case(
        program=program,
        compiled=compiled,
        inventory=inventory,
        metadata=metadata,
        owner=owner,
        owner_bytes=owner_bytes,
        block_bytes=block_bytes,
        device_id=device.id,
        compiler_bytes=compiler_memory_reservation(
            compiled=compiled, widths={}
        ).reservation_bytes,
    )


def test_only_a_real_owner_pass_through_gets_aligned_input_credit() -> None:
    case = _case(axis=0, code=2, cells=8, shared=False)
    selected = case.metadata[("working", 0, "main")]
    assert bi._aligned_input_artifacts(metadata=selected) == ()
    transfer = selected.input_transfer_plan[0]
    whole = dataclasses.replace(transfer, view=None)
    metadata = bi._ProgramExecutionMetadata(
        requirements=CoreExecutionRequirements(
            value_reads=(ValueRead(target=whole.target, source=whole.source),)
        ),
        disposition=CoreExecutionDisposition.PLANNED,
        scope=ProgramScope.ANY,
        input_transfer_plan=(whole,),
    )
    assert bi._aligned_input_artifacts(metadata=metadata) == (whole.target,)
    assert not bi._period_transfer_scratch_reservations(
        period=0,
        metadata={("working", 0, "main"): metadata},
        device_ids=(case.device_id,),
    )


@pytest.mark.parametrize(("axis", "code", "cells"), [(0, 0, 8), (0, 2, 17), (1, 2, 8)])
@pytest.mark.parametrize("shared", [False, True])
def test_pruned_selected_view_has_the_exact_conservative_admission_threshold(
    *,
    axis: int,
    code: int,
    cells: int,
    shared: bool,
) -> None:
    """Owner + destination + stage scratch, deduplicated only for a shared view.

    This is the existing planner's conservative reservation, not a claim that
    these objects are the allocator's exact simultaneous peak.
    """
    case = _case(axis=axis, code=code, cells=cells, shared=shared)
    assert dict(case.inventory.transfer_scratch_bytes) == {
        case.device_id: case.block_bytes
    }
    expected_resident = case.owner_bytes + 2 * case.block_bytes

    def resident(inventory: ResidentInventory) -> int:
        return bi._candidate_resident_bytes(
            compiled=case.compiled,
            program=case.program,
            internal_arguments={},
            inventory=inventory,
        )

    assert resident(case.inventory) == expected_resident
    ceiling = case.compiler_bytes + expected_resident

    def plan(*, budget: int, inventory: ResidentInventory) -> Any:
        return plan_workspace(
            axes=(),
            compile_candidate=lambda _widths: case.compiled,
            budget_bytes=budget,
            resident_bytes_for=lambda _compiled: resident(inventory),
        )

    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        plan(budget=ceiling - 1, inventory=case.inventory)
    assert plan(budget=ceiling, inventory=case.inventory).compiled is case.compiled
    # Negative control: omitted selection scratch incorrectly admits that same ceiling.
    omitted = dataclasses.replace(
        case.inventory, transfer_scratch_bytes=MappingProxyType({})
    )
    assert plan(budget=ceiling - 1, inventory=omitted).compiled is case.compiled
    assert not case.owner.is_deleted()
