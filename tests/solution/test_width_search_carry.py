"""Carrying one period's admitted width to the others of its carry group.

The fixture is one acting regime over eight periods whose ranked width frontier is
the same in every period. A synthetic compiler peak proportional to the streamed
width product, under a budget that refuses the two widest ranks in every period,
makes the exhaustive walk compile three candidates per period. Carrying the
admitted rank compiles it and its wider neighbour once per further period.
"""

import math
from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.solution import backward_induction
from lcm import (
    AgeGrid,
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
        transition=_next_regime,
        active=lambda age: age < _N_PERIODS,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=_N_WEALTH)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
        },
        functions={"utility": _utility},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= _N_PERIODS,
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
    )


def _solve(*, monkeypatch: pytest.MonkeyPatch, carry: bool, budget_bytes: int) -> dict:
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
    """Solve the fixture once under the walk and once with the carry."""
    monkeypatch = pytest.MonkeyPatch()
    budget = _BUDGET_VALUES * _value_bytes()
    return {
        "walk": _solve(monkeypatch=monkeypatch, carry=False, budget_bytes=budget),
        "carry": _solve(monkeypatch=monkeypatch, carry=True, budget_bytes=budget),
    }


def test_carrying_the_admitted_rank_compiles_fewer_candidates(solves: dict) -> None:
    """Later periods skip the ranks the first period's walk already refused."""
    assert solves["carry"]["compiled"] < solves["walk"]["compiled"]


def test_carrying_the_admitted_rank_selects_the_walks_widths(solves: dict) -> None:
    """Every period keeps the width the exhaustive walk selects for it."""
    assert solves["carry"]["selected"] == solves["walk"]["selected"]


def test_carrying_the_admitted_rank_leaves_the_values_bitwise_equal(
    solves: dict,
) -> None:
    """The solved acting values are bitwise those of the exhaustive walk."""
    assert all(
        np.array_equal(solves["carry"]["values"][period], values)
        for period, values in solves["walk"]["values"].items()
    )
