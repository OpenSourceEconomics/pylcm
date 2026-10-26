"""Admission reserves a value assembled from invariant blocks once per cell.

A regime solved one invariant code at a time writes every block into one complete
destination, which stays live while the later block programs run. Those programs
report only their own block's output, so the cell's internal reservation carries
the destination: once, whatever the number of codes and width candidates. A
one-code value is its block and needs none. The public case solves the
independent-types model blocked on `pref_type` under a budget.
"""

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    InternalInputRef,
    InvariantBinding,
    ResolvedCoreProgram,
)
from _lcm.execution.output_layout import VALUE
from _lcm.solution import backward_induction
from _lcm.typing import PytreeValue, ShapeDtypePytree, StateName, ValueTemplate
from lcm import ExecutionConfig
from lcm.typing import ReferenceName, RegimeName
from tests.solution.test_invariant_blocking import _independent_types_model
from tests.test_models import independent_types

type _Candidate = tuple[tuple[RegimeName, int, str], tuple[tuple[str, int], ...]]


def _identity(*, value: PytreeValue) -> PytreeValue:
    return value


def _program(*, name: str, binding: InvariantBinding | None) -> ResolvedCoreProgram:
    return ResolvedCoreProgram(
        name=name,
        function=_identity,
        arguments=MappingProxyType({}),
        static_kwargs=MappingProxyType({}),
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="accounting metadata test",
        donation_candidates=(),
        tile_widths=MappingProxyType({}),
        specialization_key=(name,),
        input_transfer_plan=(),
        invariant_binding=binding,
    )


def _blocked_cell(
    *,
    codes: int,
    blocked_axis: int,
    dtype: str,
    sharding: jax.sharding.Sharding,
    bound: bool = True,
) -> tuple[
    dict[_Candidate, ResolvedCoreProgram],
    jax.ShapeDtypeStruct,
    tuple[StateName, ...],
]:
    """One `terminal` cell: a program per code, each at three width candidates."""
    shape = (codes, 8) if blocked_axis == 0 else (8, codes)
    names = ("pref_type", "wealth") if blocked_axis == 0 else ("wealth", "pref_type")
    template = jax.ShapeDtypeStruct(shape, np.dtype(dtype), sharding=sharding)
    programs: dict[_Candidate, ResolvedCoreProgram] = {}
    for code in range(codes if bound else 1):
        binding = (
            InvariantBinding(
                state_name="pref_type", start=code, code=code, family="value"
            )
            if bound
            else None
        )
        # Repeated width candidates must not repeat the destination reservation.
        for width in (1, 2, 4):
            candidate = (("terminal", 1, f"block_{code}"), (("cell", width),))
            programs[candidate] = _program(name=f"block_{code}", binding=binding)
    return programs, template, names


def _reserved(
    *,
    programs: Mapping[_Candidate, ResolvedCoreProgram],
    template: jax.ShapeDtypeStruct,
    names: tuple[StateName, ...],
    internal_templates: Mapping[_Candidate, Mapping[ReferenceName, ShapeDtypePytree]]
    | None = None,
) -> int:
    return backward_induction._internal_reservations_by_cell(
        programs=programs,
        templates=internal_templates or {candidate: {} for candidate in programs},
        value_templates={"terminal": template},
        value_axis_names={"terminal": names},
    )[("terminal", 1)]


@pytest.mark.parametrize("codes", [1, 2, 8])
@pytest.mark.parametrize("blocked_axis", [0, 1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_assembly_reserves_one_full_destination_per_cell(
    *, codes: int, blocked_axis: int, dtype: str
) -> None:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    programs, template, names = _blocked_cell(
        codes=codes, blocked_axis=blocked_axis, dtype=dtype, sharding=sharding
    )
    expected = 0 if codes == 1 else math.prod(template.shape) * np.dtype(dtype).itemsize
    assert _reserved(programs=programs, template=template, names=names) == expected


def test_an_unbound_complete_output_gets_no_assembly_reservation() -> None:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    programs, template, names = _blocked_cell(
        codes=8, blocked_axis=0, dtype="float32", sharding=sharding, bound=False
    )
    assert _reserved(programs=programs, template=template, names=names) == 0


def test_assembly_adds_to_the_internal_producer_reservations_once() -> None:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    programs, template, names = _blocked_cell(
        codes=8, blocked_axis=0, dtype="float32", sharding=sharding
    )
    requirements = CoreExecutionRequirements(
        internal_inputs={"scratch": InternalInputRef(producer="producer", label="bank")}
    )
    programs = {
        candidate: dataclasses.replace(program, requirements=requirements)
        for candidate, program in programs.items()
    }
    internal_templates = {
        candidate: {
            "scratch": jax.ShapeDtypeStruct(
                (7,), np.dtype("float32"), sharding=sharding
            )
        }
        for candidate in programs
    }
    assert (
        _reserved(
            programs=programs,
            template=template,
            names=names,
            internal_templates=internal_templates,
        )
        == 8 * 8 * 4 + 7 * 4
    )


@pytest.mark.parametrize("blocked_axis", [0, 1])
@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_each_device_is_charged_its_shard_of_the_destination(
    *, blocked_axis: int, replicated: bool, dtype: str
) -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Layout accounting needs at least two visible devices.")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("d",))
    # The partitioned axis is the unblocked one: a blocked axis is never split.
    spec = (
        jax.P(None, None)
        if replicated
        else (jax.P(None, "d") if blocked_axis == 0 else jax.P("d", None))
    )
    programs, template, names = _blocked_cell(
        codes=2,
        blocked_axis=blocked_axis,
        dtype=dtype,
        sharding=jax.NamedSharding(mesh, spec),
    )
    # A literal for this evenly divisible two-device layout, not the footprint
    # helper the reservation itself uses.
    expected = 2 * 8 * np.dtype(dtype).itemsize // (1 if replicated else 2)
    assert _reserved(programs=programs, template=template, names=names) == expected


def test_a_budgeted_blocked_solve_reserves_each_assembled_destination(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    reserve = backward_induction._internal_reservations_by_cell
    reserved: dict[tuple[RegimeName, int], int] = {}
    blocked: set[tuple[RegimeName, int]] = set()

    def _record(
        *,
        programs: Mapping[_Candidate, ResolvedCoreProgram],
        templates: Mapping[_Candidate, Mapping[ReferenceName, ShapeDtypePytree]],
        value_templates: Mapping[RegimeName, ValueTemplate],
        value_axis_names: Mapping[RegimeName, tuple[StateName, ...]],
    ) -> MappingProxyType[tuple[RegimeName, int], int]:
        result = reserve(
            programs=programs,
            templates=templates,
            value_templates=value_templates,
            value_axis_names=value_axis_names,
        )
        reserved.update(result)
        blocked.update(
            candidate[0][:2]
            for candidate, program in programs.items()
            if program.invariant_binding is not None
        )
        return result

    monkeypatch.setattr(backward_induction, "_internal_reservations_by_cell", _record)
    model = _independent_types_model(
        execution_config=ExecutionConfig(
            devices=(0,),
            device_memory_bytes=2**30,
            invariant_block_widths={"pref_type": 1},
        )
    )
    values = model.solve(params=independent_types.get_params(), log_level="off").values
    assert blocked, "the solve blocked no regime"
    for regime, period in blocked:
        value = values[period][regime]
        assert reserved[(regime, period)] >= value.size * value.dtype.itemsize
