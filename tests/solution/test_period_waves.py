"""The solve loop dispatches each period's regimes in dependency waves.

On one device every wave holds one regime, so the dispatch order is today's:
same-period references first, declaration order among independent regimes.
"""

from typing import Any

import pytest

from _lcm.solution import backward_induction
from lcm import AgeGrid, Model
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)


def _model() -> Model:
    return Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )


def _params() -> dict[str, float]:
    return {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}


def test_every_reference_is_dispatched_before_the_regime_that_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within a period, a same-period reference precedes its reader."""
    order: list[tuple[int, str]] = []
    real = backward_induction._run_period_kernel

    def record(**kwargs: Any) -> object:
        order.append((kwargs["period"], kwargs["regime_name"]))
        return real(**kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", record)
    model = _model()
    model.solve(params=_params(), log_level="off")
    positions = {key: index for index, key in enumerate(order)}

    assert all(
        positions[(period, reference)] < positions[(period, name)]
        for (period, name) in order
        for reference in model._regimes[name].same_period_ref_regimes
        if (period, reference) in positions
    )


def test_periods_are_dispatched_strictly_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No node of period t is dispatched before every node of period t + 1."""
    order: list[int] = []
    real = backward_induction._run_period_kernel

    def record(**kwargs: Any) -> object:
        order.append(kwargs["period"])
        return real(**kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", record)
    _model().solve(params=_params(), log_level="off")

    assert order == sorted(order, reverse=True)
