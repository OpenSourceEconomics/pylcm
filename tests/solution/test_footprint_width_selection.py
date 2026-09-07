"""Width selection binds the compiler peak plus the plan's resident footprint.

One acting regime solves three periods into a terminal regime, and every
regime value is retained, so each earlier period finds one more value resident
on its device than the period after it. The peak the compiler reports already
counts the buffers the executable receives as arguments, so the resident term
counts only what the position holds *besides* the unit's own inputs.

With a fake compiler peak proportional to the streamed width product and a
budget that exactly fits the full extent at the position where nothing else is
resident, the latest acting period streams at the full extent and the earlier
ones stream narrower.
"""

import math
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
import pytest

from _lcm.execution.scheduler import DispatchUnit, ScheduledNode, plan_period_waves
from _lcm.solution import backward_induction
from _lcm.solution.solve_inputs import SolveInputMappings
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND, ScalarInt

# Points of the wealth grid, so one regime value is this many elements.
_N_WEALTH = 64

# Product extent of the streamed action axis: two work states times three.
_ACTION_EXTENT = 6


@categorical(ordered=False)
class RegimeId:
    """Regime identities of the two-regime lifecycle."""

    acting: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class Work:
    """Binary labour-supply action."""

    leisure: ScalarInt
    working: ScalarInt


def _next_regime(*, age: int) -> ScalarInt:
    """Leave the acting regime after the last acting age."""
    return jnp.where(age < 2, RegimeId.acting, RegimeId.done)


def _utility(*, consumption: float, work: ScalarInt, wealth: float) -> FloatND:
    """Value one action cell at one wealth node."""
    return jnp.log(consumption) - 0.1 * work + 0.01 * wealth


def _terminal_utility(*, wealth: float) -> float:
    """Value the terminal regime by its wealth."""
    return wealth


def _build_model(
    *,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> Model:
    """Build one acting regime over three periods into a terminal regime."""
    acting = Regime(
        transition=_next_regime,
        active=lambda age: age < 3,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH),
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
        },
        functions={"utility": _utility},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 3,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH),
        },
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
        execution_config=execution_config,
    )


def _value_bytes() -> int:
    """Bytes one regime value occupies at the working precision."""
    return _N_WEALTH * jnp.zeros(()).dtype.itemsize


def _fake_peak(*, compiled: object, widths: Mapping[str, int]) -> int:  # noqa: ARG001
    """Report a compiler peak proportional to the streamed width product."""
    return _value_bytes() * math.prod(widths.values())


def _solve_capturing_compilation(
    *, monkeypatch: pytest.MonkeyPatch, budget_bytes: int | None
) -> tuple[Mapping[str, Any], backward_induction._CompiledPrograms]:
    """Solve the model and return the compilation call's inputs and its result."""
    calls: list[tuple[Mapping[str, Any], backward_induction._CompiledPrograms]] = []
    original = backward_induction._compile_all_functions

    def capture(**kwargs: Any) -> backward_induction._CompiledPrograms:
        result = original(**kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(backward_induction, "compiler_peak_bytes", _fake_peak)
    monkeypatch.setattr(backward_induction, "_compile_all_functions", capture)
    model = _build_model(
        execution_config=ExecutionConfig(device_memory_bytes=budget_bytes)
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    model.solve(
        params=params,
        log_level="debug",
    )
    (call,) = calls
    return call


def _selected_width_products(
    *, monkeypatch: pytest.MonkeyPatch, budget_bytes: int
) -> dict[tuple[str, int], int]:
    """Return the selected streamed width product of every regime-period cell."""
    _, programs = _solve_capturing_compilation(
        monkeypatch=monkeypatch, budget_bytes=budget_bytes
    )
    return {
        cell: math.prod(cores["main"].tile_widths.values())
        for cell, cores in programs.executables.items()
    }


def _resident_bytes(
    *,
    monkeypatch: pytest.MonkeyPatch,
    extra_cores: Mapping[tuple[str, int], str] = MappingProxyType({}),
) -> Mapping[tuple[str, int, str], int]:
    """Predict every core triple's resident bytes from one unbudgeted solve.

    `extra_cores` adds a second core to the named cells, sharing the cell's
    resolved declarations, so a multi-core kernel can be described without a
    solver that declares one.
    """
    kwargs, programs = _solve_capturing_compilation(
        monkeypatch=monkeypatch, budget_bytes=None
    )
    metadata = dict(programs.metadata)
    for (regime_name, period), core_key in extra_cores.items():
        metadata[(regime_name, period, core_key)] = metadata[
            (regime_name, period, "main")
        ]
    return backward_induction._resident_bytes_by_triple(
        regimes=kwargs["regimes"],
        ledger=programs.input_liveness,
        templates=SolveInputMappings(
            next_regime_to_V_arr=kwargs["next_regime_to_V_arr"],
            next_regime_to_continuation=kwargs["next_regime_to_continuation"],
            next_edge_to_V_arr=kwargs["next_edge_to_V_arr"],
        ),
        program_metadata=MappingProxyType(metadata),
        device_ids=kwargs["execution"].device_ids,
    )


def test_two_cores_of_one_regime_form_a_single_dispatch_unit() -> None:
    """A kernel dispatches its cores together, so one cell is one schedule unit."""
    waves = plan_period_waves(
        nodes=(
            ScheduledNode(period=0, regime="acting", program="main"),
            ScheduledNode(period=0, regime="acting", program="replay"),
        ),
        same_period_dependencies=MappingProxyType({"acting": ()}),
        device_sets=MappingProxyType({"acting": frozenset({0})}),
    )

    assert waves == (
        (DispatchUnit(period=0, regime="acting", programs=("main", "replay")),),
    )


def test_program_keys_by_cell_groups_a_cells_cores_in_producer_order() -> None:
    """Every core of one regime-period cell is grouped under that cell."""
    assert backward_induction._program_keys_by_cell(
        triples=(("acting", 0, "main"), ("acting", 0, "replay"), ("done", 3, "main"))
    ) == {("acting", 0): ("main", "replay"), ("done", 3): ("main",)}


def test_resident_bytes_grow_by_one_retained_value_per_earlier_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each earlier period competes with one more retained value than the next."""
    resident = _resident_bytes(monkeypatch=monkeypatch)

    assert (
        resident[("done", 3, "main")],
        resident[("acting", 2, "main")],
        resident[("acting", 1, "main")],
        resident[("acting", 0, "main")],
    ) == (0, 0, _value_bytes(), 2 * _value_bytes())


def test_every_core_of_one_cell_sees_the_same_resident_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell's cores share its schedule position, so they share its number."""
    resident = _resident_bytes(
        monkeypatch=monkeypatch,
        extra_cores=MappingProxyType({("acting", 1): "replay"}),
    )

    assert resident[("acting", 1, "replay")] == resident[("acting", 1, "main")]


def test_a_second_core_of_one_cell_is_not_counted_as_a_concurrent_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kernel's own cores share one allocation window, not two."""
    resident = _resident_bytes(
        monkeypatch=monkeypatch,
        extra_cores=MappingProxyType({("acting", 1): "replay"}),
    )

    assert resident[("acting", 1, "main")] == _value_bytes()


def test_the_period_below_a_retained_value_streams_narrower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident bytes grow backward, so an earlier period fits a narrower width."""
    widths = _selected_width_products(
        monkeypatch=monkeypatch, budget_bytes=_ACTION_EXTENT * _value_bytes()
    )

    assert widths[("acting", 1)] < widths[("acting", 2)]


def test_the_position_holding_nothing_besides_its_argument_keeps_the_full_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget that exactly fits the widest peak is spent on it where it is free."""
    widths = _selected_width_products(
        monkeypatch=monkeypatch, budget_bytes=_ACTION_EXTENT * _value_bytes()
    )

    assert widths[("acting", 2)] == _ACTION_EXTENT


def test_a_large_budget_no_position_can_bind_keeps_every_period_at_full_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget far above every peak selects the full extent at every period."""
    widths = _selected_width_products(
        monkeypatch=monkeypatch, budget_bytes=100 * _value_bytes()
    )

    assert {
        cell: width for cell, width in widths.items() if cell[0] == "acting"
    } == dict.fromkeys((("acting", 0), ("acting", 1), ("acting", 2)), _ACTION_EXTENT)


def test_a_solve_without_a_budget_does_not_walk_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No budget consults no peak, so no position is predicted for any core."""
    walks: list[object] = []
    original = backward_induction.plan_resident_bytes

    def record(**kwargs: Any) -> object:
        walks.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(backward_induction, "plan_resident_bytes", record)
    _solve_capturing_compilation(monkeypatch=monkeypatch, budget_bytes=None)

    assert walks == []


def test_a_cell_the_budget_cannot_host_never_enters_a_compilation_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A position that already fills the budget lowers no width of its own.

    The wave loop's pending set is read through the per-wave triple count it
    builds, whose keys are the candidates of every triple still being lowered.
    """
    lowered: set[tuple[str, int]] = set()
    original = backward_induction._count_triples_per_lowering_key

    def record(*, lowering_keys: Mapping[Any, Any]) -> Any:
        lowered.update((triple[0], triple[1]) for triple, _width in lowering_keys)
        return original(lowering_keys=lowering_keys)

    monkeypatch.setattr(backward_induction, "_count_triples_per_lowering_key", record)
    with pytest.raises(ExecutionPlanningError):
        _selected_width_products(
            monkeypatch=monkeypatch, budget_bytes=2 * _value_bytes()
        )

    assert ("acting", 0) not in lowered


def test_a_cell_whose_position_fills_the_budget_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resident bytes reaching the budget leave nothing for any workspace."""
    with pytest.raises(
        ExecutionPlanningError, match=re.escape("Regime 'acting' at period 0")
    ):
        _selected_width_products(
            monkeypatch=monkeypatch, budget_bytes=2 * _value_bytes()
        )


@pytest.mark.parametrize(
    ("resident_bytes", "expected"),
    [(9, ("a", 0, "main")), (10, None), (11, None)],
)
def test_only_a_core_with_room_left_for_a_workspace_is_lowered(
    *, resident_bytes: int, expected: tuple[str, int, str] | None
) -> None:
    """A core is lowered only while its position leaves part of the budget free."""
    triple = ("a", 0, "main")

    assert backward_induction._triples_within_budget(
        candidates_by_triple={triple: [(triple, ())]},
        resident_bytes_by_triple={triple: resident_bytes},
        budget_bytes=10,
    ) == ((expected,) if expected is not None else ())


@pytest.mark.parametrize(
    "template",
    [
        "Regime 'acting' at period 0",
        "{resident} bytes resident",
        "{budget}-byte budget",
    ],
)
def test_the_refusal_names_the_cell_the_budget_and_the_resident_bytes(
    *, monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    """A cell fitting at no width names where it failed and what it competed with."""
    budget = 2 * _value_bytes() + 1
    pattern = template.format(resident=2 * _value_bytes(), budget=budget)

    with pytest.raises(ExecutionPlanningError, match=re.escape(pattern)):
        _selected_width_products(monkeypatch=monkeypatch, budget_bytes=budget)


def test_a_solve_with_a_budget_walks_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget is spent against a position, so every core's position is predicted."""
    walks: list[object] = []
    original = backward_induction.plan_resident_bytes

    def record(**kwargs: Any) -> object:
        walks.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(backward_induction, "plan_resident_bytes", record)
    _selected_width_products(monkeypatch=monkeypatch, budget_bytes=100 * _value_bytes())

    assert walks != []


def test_a_cell_the_budget_can_host_enters_a_compilation_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A position leaving room for a workspace lowers a width of its own."""
    lowered: set[tuple[str, int]] = set()
    original = backward_induction._count_triples_per_lowering_key

    def record(*, lowering_keys: Mapping[Any, Any]) -> Any:
        lowered.update((triple[0], triple[1]) for triple, _width in lowering_keys)
        return original(lowering_keys=lowering_keys)

    monkeypatch.setattr(backward_induction, "_count_triples_per_lowering_key", record)
    _selected_width_products(monkeypatch=monkeypatch, budget_bytes=100 * _value_bytes())

    assert ("acting", 0) in lowered
