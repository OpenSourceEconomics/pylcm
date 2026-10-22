"""The only value arrays a solve keeps on the device are values it has computed.

No placeholder of a value array's shape is resident at any point of the solve,
neither before the first period nor while a later one runs. A computed value is
a regime's solved value array or a gated edge's folded continuation; an array
that was live before the solve began is not the solve's. The models are the
grid-search models of `test_period_target_value_reads`, whose value arrays have
two axes or more and so cannot be mistaken for a one-dimensional grid.
"""

from collections.abc import Callable
from typing import Unpack

import jax
import pytest

from _lcm.solution import backward_induction
from lcm import Model
from tests.solution._callback_types import (
    EvaluateEdgeFoldKwargs,
    RunPeriodKernelKwargs,
    RunPeriodKernelResult,
)
from tests.solution.test_period_target_value_reads import (
    _gated_model,
    _three_regime_model,
)

# Shape, dtype name and buffer address of one live array.
type _Resident = tuple[tuple[int, ...], str, int]


def _live_multi_axis_arrays() -> frozenset[_Resident]:
    return frozenset(
        (array.shape, array.dtype.name, array.unsafe_buffer_pointer())
        for array in jax.live_arrays()
        if array.ndim >= 2
    )


@pytest.mark.parametrize(
    "build_model",
    [_three_regime_model, _gated_model],
    ids=["three-regime", "gated-edge"],
)
def test_grid_search_solve_keeps_no_value_placeholder_resident(
    *, monkeypatch: pytest.MonkeyPatch, build_model: Callable[[], Model]
) -> None:
    """Every value-shaped array live at a period's start was computed by the solve."""
    computed: set[int] = set()
    computed_layouts: set[tuple[tuple[int, ...], str]] = set()
    samples: list[tuple[frozenset[int], frozenset[_Resident]]] = []
    run_period_kernel = backward_induction._run_period_kernel
    evaluate_edge_fold = backward_induction._evaluate_edge_fold

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        samples.append((frozenset(computed), _live_multi_axis_arrays()))
        output = run_period_kernel(**kwargs)
        assert isinstance(output.value, jax.Array)
        computed.add(output.value.unsafe_buffer_pointer())
        return output

    def _fold(**kwargs: Unpack[EvaluateEdgeFoldKwargs]) -> jax.Array:
        wbar = evaluate_edge_fold(**kwargs)
        computed.add(wbar.unsafe_buffer_pointer())
        computed_layouts.add((wbar.shape, wbar.dtype.name))
        return wbar

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _run)
    monkeypatch.setattr(backward_induction, "_evaluate_edge_fold", _fold)
    model = build_model()
    computed.update(resident[2] for resident in _live_multi_axis_arrays())
    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    value_layouts = computed_layouts | {
        (array.shape, array.dtype.name)
        for regime_to_value in solution.values.values()
        for array in regime_to_value.values()
    }
    assert [
        resident
        for computed_so_far, residents in samples
        for resident in residents
        if resident[:2] in value_layouts and resident[2] not in computed_so_far
    ] == []
