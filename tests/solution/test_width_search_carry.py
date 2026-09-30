"""Exact width selection with the compatibility carry flag enabled or disabled."""

import math
from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import (
    CompilerMemoryReservation,
    workspace_width_candidates,
)
from _lcm.solution import backward_induction
from lcm import (
    AgeGrid,
    Choose,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    fixed_transition,
)
from lcm.execution import WidthSearchPolicy
from lcm.typing import ScalarInt
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.solution.test_footprint_width_selection import (
    RegimeId,
    Work,
    _terminal_utility,
    _utility,
)
from tests.test_models.schedules import until_exit

_N_WEALTH = 64
_N_PERIODS = 8


def _next_regime(*, age: int) -> ScalarInt:
    """Leave the acting regime after the last acting age."""
    return jnp.where(age < _N_PERIODS - 1, RegimeId.acting, RegimeId.done)


def _value_bytes() -> int:
    """Bytes one regime value occupies at the working precision."""
    return _N_WEALTH * jnp.zeros(()).dtype.itemsize


def _fake_peak(
    *, compiled: object, widths: Mapping[str, int]
) -> CompilerMemoryReservation:
    """Report a compiler peak proportional to the streamed width product."""
    del compiled
    return synthetic_memory(_value_bytes() * math.prod(widths.values()))


def _model(*, carry: bool, budget_bytes: int) -> Model:
    acting = Regime(
        regime_transitions=until_exit(
            _N_PERIODS,
            law=Choose(func=_next_regime, targets=("acting", "done")),
            exits=("done",),
        ),
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
        },
        functions={"utility": _utility},
    )
    done = Regime(
        regime_transitions=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH)},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=_N_PERIODS, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(
            device_memory_bytes=budget_bytes,
            width_search=WidthSearchPolicy(carry_across_periods=carry),
        ),
        initial_regimes={0: "acting"},
    )


def _solve(
    *,
    monkeypatch: pytest.MonkeyPatch,
    carry: bool,
    budget_bytes: int,
) -> dict:
    compiled_candidates: list[tuple] = []
    selected: dict = {}
    original_wave = backward_induction._lower_and_compile_wave
    original_compile = backward_induction._compile_all_functions

    def count_wave(**kwargs: Any) -> Any:
        compiled_candidates.extend(kwargs["new_lowerings"].values())
        return original_wave(**kwargs)

    def capture(**kwargs: Any) -> Any:
        result = original_compile(**kwargs)
        selected.update(
            {
                cell: tuple(cores["main"].tile_widths.items())
                for cell, cores in result.executables.items()
            }
        )
        return result

    monkeypatch.setattr(backward_induction, "compiler_memory_reservation", _fake_peak)
    monkeypatch.setattr(backward_induction, "_lower_and_compile_wave", count_wave)
    monkeypatch.setattr(backward_induction, "_compile_all_functions", capture)
    model = _model(carry=carry, budget_bytes=budget_bytes)
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    solution = model.solve(params=params, log_level="off")
    monkeypatch.undo()
    acting = [c for c in compiled_candidates if c[0][0] == "acting"]
    return {
        "compiled": len(acting),
        "selected": selected,
        "values": {
            p: np.asarray(v["acting"])
            for p, v in solution.values.items()
            if "acting" in v
        },
    }


# A budget of 50 values refuses the two widest ranks at every acting period.
_BUDGET_VALUES = 50


@pytest.fixture(scope="module")
def solves() -> dict[str, dict]:
    """Solve the fixture with each value of the compatibility flag."""
    monkeypatch = pytest.MonkeyPatch()
    budget = _BUDGET_VALUES * _value_bytes()
    return {
        "walk": _solve(monkeypatch=monkeypatch, carry=False, budget_bytes=budget),
        "carry": _solve(monkeypatch=monkeypatch, carry=True, budget_bytes=budget),
    }


def test_carry_flag_keeps_the_exhaustive_compilation_count(solves: dict) -> None:
    """The compatibility flag leaves the same candidates compiled."""
    assert solves["carry"]["compiled"] == solves["walk"]["compiled"]


def test_carry_flag_selects_the_walks_widths(solves: dict) -> None:
    """Every period keeps the width the exhaustive walk selects for it."""
    assert solves["carry"]["selected"] == solves["walk"]["selected"]


def test_carry_flag_leaves_the_values_bitwise_equal(
    solves: dict,
) -> None:
    """The solved acting values are bitwise those of the exhaustive walk."""
    assert all(
        solves["carry"]["values"][period].tobytes() == values.tobytes()
        for period, values in solves["walk"]["values"].items()
    )


@pytest.mark.parametrize(
    ("totals", "budget", "expected_ranks"),
    [
        (((12, 11, 8), (8, 11, 8)), 10, (2, 0)),
        (((754, 554, 752, 454), (654, 454, 652, 354)), 500, (3, 1)),
    ],
    ids=["hidden-earlier-admission", "two-axis-rank-is-not-monotone"],
)
@pytest.mark.parametrize("carry", [False, True])
def test_each_period_selects_its_first_admitted_rank(
    *,
    monkeypatch: pytest.MonkeyPatch,
    totals: tuple[tuple[int, ...], tuple[int, ...]],
    budget: int,
    expected_ranks: tuple[int, int],
    carry: bool,
) -> None:
    """An admitted earlier rank wins even when its narrower neighbour refuses.

    Admission totals are a table-driven protocol witness, not measured XLA memory.
    The model, candidate frontier, compilation and final width selection are real.
    """
    scale = 1_000_000
    expected = {}
    leader = []
    original_measure = backward_induction._measure_variant

    def measure(**kwargs: Any) -> int:
        triple = kwargs["triple"]
        if triple[0] != "acting":
            return original_measure(**kwargs)
        kwargs["memory_by_lowering_key"][kwargs["variant_key"]] = synthetic_memory(1)
        if not leader:
            leader.append(triple)
        row = 0 if triple == leader[0] else 1
        program = kwargs["resolved_programs"][kwargs["candidate"]]
        widths = workspace_width_candidates(
            axes=program.requirements.axes, budget_bytes=budget * scale
        )
        rank = widths.index(dict(program.tile_widths))
        expected[triple[:2]] = dict(widths[expected_ranks[row]])
        return totals[row][min(rank, len(totals[row]) - 1)] * scale - 1

    monkeypatch.setattr(backward_induction, "_measure_variant", measure)
    result = _solve(monkeypatch=monkeypatch, carry=carry, budget_bytes=budget * scale)
    assert len(expected) == _N_PERIODS
    assert {
        cell: dict(widths)
        for cell, widths in result["selected"].items()
        if cell[0] == "acting"
    } == expected
