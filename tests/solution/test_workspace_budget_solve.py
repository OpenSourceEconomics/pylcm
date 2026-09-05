"""A device-memory budget selects the widest streamed action block that fits.

The planner reads compiler-reported peaks. Most tests here replace that reader with a
synthetic peak that is linear in the width product, so the selection rule is checked on
any backend; the fail-closed case reads the real compiler report. The selected widths
are observed through a period capture, which records exactly what the solve dispatched.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import cloudpickle
import numpy as np
import pytest

from _lcm.solution import backward_induction
from _lcm.solution.period_capture import _PAYLOAD_NAME
from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import replay_period
from lcm.solvers import GridSearch
from tests.conftest import assert_agrees_to_ulp
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 2
_N_CONSUMPTION = 3
_ACTION_EXTENT = (
    len(DiscreteGrid(category_class=LaborSupply).categories) * _N_CONSUMPTION
)
_CAPTURE_TARGET = "working_life@0"
# Synthetic compiler peak per unit of width product.
_BYTES_PER_ACTION = 1000


def _model(*, action_block_width: int | None = None) -> Model:
    """Build the two-period grid-search toy with a small streamed action product."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(
                        start=1, stop=3, n_points=_N_CONSUMPTION
                    ),
                },
                solver=GridSearch(action_block_width=action_block_width),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.fixture
def synthetic_peaks(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, int]]:
    """Report a peak of `_BYTES_PER_ACTION` per unit of width product."""
    seen: list[dict[str, int]] = []

    def peak(*, compiled: object, widths: Mapping[str, int]) -> int:
        del compiled
        seen.append(dict(widths))
        return _BYTES_PER_ACTION * math.prod(widths.values())

    monkeypatch.setattr(backward_induction, "compiler_peak_bytes", peak)
    return seen


def _solve_capturing(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: Model,
    device_memory_bytes: int | None,
):
    """Solve while capturing the first working-life period."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", _CAPTURE_TARGET)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    return model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
        execution_config=ExecutionConfig(device_memory_bytes=device_memory_bytes),
    )


def _captured_widths(tmp_path: Path) -> dict[str, dict[str, int]]:
    """Read the tile widths the capture recorded for each dispatched core."""
    with (tmp_path / _CAPTURE_TARGET / _PAYLOAD_NAME).open("rb") as stream:
        return cloudpickle.load(stream)["core_tile_widths"]


def test_no_budget_uses_the_bootstrap_width(*, monkeypatch, tmp_path) -> None:
    """Extent six streams in blocks of four; the whole product needs a budget."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=None,
    )

    assert _captured_widths(tmp_path) == {"main": {"action": 4}}


@pytest.mark.usefixtures("synthetic_peaks")
def test_budget_selects_the_widest_feasible_action_block(
    *, monkeypatch, tmp_path
) -> None:
    """Frontier 1, 2, 4, 6 with a budget of two width units selects width 2."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=2 * _BYTES_PER_ACTION,
    )

    assert _captured_widths(tmp_path) == {"main": {"action": 2}}


@pytest.mark.usefixtures("synthetic_peaks")
def test_budget_above_every_candidate_keeps_the_full_action_product(
    *, monkeypatch, tmp_path
) -> None:
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=_ACTION_EXTENT * _BYTES_PER_ACTION,
    )

    assert _captured_widths(tmp_path) == {"main": {"action": _ACTION_EXTENT}}


def test_budget_above_every_candidate_lowers_only_the_full_action_product(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    """A budget every core meets costs no more compilation than no budget."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=_ACTION_EXTENT * _BYTES_PER_ACTION,
    )

    assert [widths["action"] for widths in synthetic_peaks if widths] == [
        _ACTION_EXTENT
    ]


def test_budget_lowers_widths_descending_until_one_fits(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    """Only the candidates wider than the selected one are compiled and rejected."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=2 * _BYTES_PER_ACTION,
    )

    assert [widths["action"] for widths in synthetic_peaks if widths] == [6, 4, 2]


def test_budgeted_values_agree_with_the_unbudgeted_solve(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    """A narrower block partitions the same maximum; values agree to a few ULP."""
    del synthetic_peaks
    unbudgeted = _model().solve(
        params=get_params(n_periods=_N_PERIODS), log_level="off"
    )
    budgeted = _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=2 * _BYTES_PER_ACTION,
    )
    assert _captured_widths(tmp_path) == {"main": {"action": 2}}

    for period in range(_N_PERIODS - 1):
        assert_agrees_to_ulp(
            got=np.asarray(budgeted.values[period]["working_life"]),
            expected=np.asarray(unbudgeted.values[period]["working_life"]),
            n_ulp=16,
        )


def test_budget_below_every_candidate_fails_closed(*, monkeypatch, tmp_path) -> None:
    """One byte fits no compiled core, so planning refuses before backward induction."""
    with pytest.raises(ExecutionPlanningError, match="No workspace-width candidate"):
        _solve_capturing(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            model=_model(),
            device_memory_bytes=1,
        )

    assert not (tmp_path / _CAPTURE_TARGET).exists()


def test_fixed_action_block_width_is_the_only_candidate(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(action_block_width=4),
        device_memory_bytes=_ACTION_EXTENT * _BYTES_PER_ACTION,
    )

    assert {widths["action"] for widths in synthetic_peaks if widths} == {4}
    assert _captured_widths(tmp_path) == {"main": {"action": 4}}


@pytest.mark.usefixtures("synthetic_peaks")
def test_fixed_action_block_width_over_budget_names_the_request(
    *, monkeypatch, tmp_path
) -> None:
    with pytest.raises(ExecutionPlanningError, match="explicitly requested"):
        _solve_capturing(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            model=_model(action_block_width=4),
            device_memory_bytes=4 * _BYTES_PER_ACTION - 1,
        )


@pytest.mark.usefixtures("synthetic_peaks")
def test_replay_of_a_budgeted_capture_reproduces_its_value(
    *, monkeypatch, tmp_path
) -> None:
    """Replay lowers the captured width and returns the array the solve published."""
    budgeted = _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        model=_model(),
        device_memory_bytes=2 * _BYTES_PER_ACTION,
    )

    replay = replay_period(directory=tmp_path / _CAPTURE_TARGET)

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(budgeted.values[0]["working_life"]),
    )
