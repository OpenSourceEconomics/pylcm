"""Admission reserves a value assembled from invariant blocks once per cell.

A regime solved one invariant code at a time writes every block into one complete
destination, which stays live while the later block programs run. Those programs
report only their own block's output, so the cell's internal reservation carries
the destination: once, whatever the number of codes and width candidates. A
one-code value is its block and needs none. The public cases solve models blocked
on `pref_type` under a budget: the independent-types model, and a two-period
model whose values dominate its memory, where a budget only the destination
exceeds must be refused.
"""

import dataclasses
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Unpack

import jax
import jax.numpy as jnp
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
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarInt,
)
from tests.solution._callback_types import MeasureVariantKwargs
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _BlockedAdmission:
    """What admission charged one blocked candidate, with and without the cell's
    internal reservation, beside the compiler's reservation for it."""

    cell: tuple[RegimeName, int]
    resident: int
    resident_without_internal: int
    reservation: int


def _record_blocked_admissions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_BlockedAdmission]:
    measure_variant = backward_induction._measure_variant
    records: list[_BlockedAdmission] = []

    def _measure(**kwargs: Unpack[MeasureVariantKwargs]) -> int:
        resident = measure_variant(**kwargs)
        candidate, triple = kwargs["candidate"], kwargs["triple"]
        program = kwargs["resolved_programs"][candidate]
        if program.invariant_binding is not None:
            records.append(
                _BlockedAdmission(
                    cell=triple[:2],
                    resident=resident,
                    resident_without_internal=backward_induction._candidate_resident_bytes(
                        compiled=kwargs["compiled"][kwargs["variant_key"]],
                        program=program,
                        internal_arguments=kwargs["internal_templates"][candidate],
                        inventory=dataclasses.replace(
                            kwargs["resident_inventory"][triple], internal_bytes=0
                        ),
                    ),
                    reservation=kwargs["memory_by_lowering_key"][
                        kwargs["variant_key"]
                    ].reservation_bytes,
                )
            )
        return resident

    monkeypatch.setattr(backward_induction, "_measure_variant", _measure)
    return records


def test_each_blocked_candidate_is_admitted_with_its_cells_destination(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _record_blocked_admissions(monkeypatch)
    model = _independent_types_model(
        execution_config=ExecutionConfig(
            devices=(0,),
            device_memory_bytes=2**30,
            invariant_block_widths={"pref_type": 1},
        )
    )
    values = model.solve(params=independent_types.get_params(), log_level="off").values
    assert records, "the solve admitted no blocked candidate"
    for record in records:
        regime, period = record.cell
        destination = values[period][regime].nbytes
        assert record.resident - record.resident_without_internal >= destination


# A two-period model whose 32-code values dwarf every other buffer, so admitting
# its working period turns on whether the assembled destination is charged.
_AGES = AgeGrid(start=0, inclusive_stop=1, step="Y")

_PrefType = categorical(ordered=False)(
    type("_PrefType", (), {"__annotations__": {f"t{i}": ScalarInt for i in range(32)}})
)


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    terminal: ScalarInt


def _utility(*, consumption: ContinuousAction, pref_type: DiscreteState) -> FloatND:
    return jnp.log1p(consumption) + 0.01 * pref_type


def _terminal_utility(*, wealth: ContinuousState, pref_type: DiscreteState) -> FloatND:
    return wealth + 0.01 * pref_type


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _affordable(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _value_dominated_model(*, budget: int) -> Model:
    states = {
        "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2**14),
        "pref_type": DiscreteGrid(category_class=_PrefType),
    }
    return Model(
        regimes={
            "working": UserRegime(
                states=states,
                state_transitions={
                    "wealth": _next_wealth,
                    "pref_type": fixed_transition("pref_type"),
                },
                actions={"consumption": LinSpacedGrid(start=0.0, stop=0.5, n_points=2)},
                functions={"utility": _utility},
                constraints={"affordable": _affordable},
            ),
            "terminal": UserRegime(
                states=states, functions={"utility": _terminal_utility}
            ),
        },
        ages=_AGES,
        regime_id_class=_RegimeId,
        initial_nodes=((_AGES.exact_values[0], "working"),),
        edges={"working": {"terminal": _AGES.exact_values[-2]}},
        execution_config=ExecutionConfig(
            devices=(0,),
            device_memory_bytes=budget,
            invariant_block_widths={"pref_type": 1},
        ),
    )


def test_a_budget_only_the_assembled_destination_exceeds_is_refused(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The working period fits its budget without its destination, but not with it.

    The budget covers everything admission charges the working candidates apart
    from the cell's internal reservation, plus their compiler reservation, plus
    half the destination: the size of the working value, taken from the solved
    array rather than from the accounting under test.
    """
    records = _record_blocked_admissions(monkeypatch)
    values = (
        _value_dominated_model(budget=2**30)
        .solve(params={"discount_factor": 0.9}, log_level="off")
        .values
    )
    working = [record for record in records if record.cell == ("working", 0)]
    assert working, "the solve admitted no blocked working candidate"
    destination = values[0]["working"].nbytes
    without_destination = [record.resident_without_internal for record in working]
    reservation = max(record.reservation for record in working)
    # The destination outweighs the candidates' own memory and their spread, so
    # half of it separates fitting without the destination from fitting with it.
    assert (
        reservation + max(without_destination) - min(without_destination)
        < destination // 2
    )
    budget = max(without_destination) + reservation + destination // 2
    monkeypatch.undo()
    with pytest.raises(ExecutionPlanningError, match="'working' at period 0"):
        _value_dominated_model(budget=budget).solve(
            params={"discount_factor": 0.9}, log_level="off"
        )
