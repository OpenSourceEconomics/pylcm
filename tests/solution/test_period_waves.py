"""The solve loop dispatches each period's regimes in dependency waves.

The waves of a period are dispatched before any wave of the period before it.
"""

from typing import Unpack

import pytest

from _lcm.solution import backward_induction
from lcm import AgeGrid, Model
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
    _with_full_topology_laws,
)
from tests.solution._callback_types import RunPeriodKernelKwargs, RunPeriodKernelResult


def _model() -> Model:
    return Model(
        regimes=_make_full_topology_regimes(),
        edges=_with_full_topology_laws(
            {
                "single_f": {"married": 0, "single_f_p1": 0},
                "single_m": {"married": 0, "single_m_p1": 0},
                "single_f_p1": {"single_f_terminal": 1},
                "single_m_p1": {"single_m_terminal": (0, 1, 2)},
                "married": {"married_terminal": 1},
            }
        ),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
        initial_nodes={0: ("single_f", "single_m")},
    )


def _params() -> dict[str, float]:
    return {"discount_factor": 0.95}


def test_periods_are_dispatched_strictly_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No node of period t is dispatched before every node of period t + 1."""
    order: list[int] = []
    real = backward_induction._run_period_kernel

    def record(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        order.append(kwargs["period"])
        return real(**kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", record)
    _model().solve(params=_params(), log_level="off")

    assert order == sorted(order, reverse=True)
