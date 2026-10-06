"""A device-memory budget selects the widest streamed action block that fits.

The planner reads compiler-reported peaks. Most tests here replace that reader with a
synthetic peak that is linear in the width product, so the selection rule is checked on
any backend; the fail-closed case reads the real compiler report. The selected widths
are observed through a period capture, which records exactly what the solve dispatched.
"""

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.execution_plan import visible_device_ids
from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.solution import backward_induction
from _lcm.solution.period_capture import _PAYLOAD_NAME
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import replay_period
from lcm.solvers import GridSearch
from tests.conftest import assert_agrees_to_ulp
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    next_regime,
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


def _fixed_owner_bytes() -> int:
    """Three wealth/consumption nodes, three params, four V cells, five int32s."""
    return 13 * jnp.zeros(()).dtype.itemsize + 5 * 4


def _model(
    *,
    execution_config: ExecutionConfig = ExecutionConfig(device_memory_bytes=None),  # noqa: B008
    enable_jit: bool = True,
) -> Model:
    """Hold the state-cell width at one to isolate action-width budget selection."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        edges={"working_life": {"dead": START_AGE}},
        regimes={
            "working_life": working_life.replace(
                regime_transitions=DeterministicTransition(func=next_regime),
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(
                        start=1, stop=3, n_points=_N_CONSUMPTION
                    ),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, inclusive_stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=replace(
            execution_config, axis_widths={"cell": 1, **execution_config.axis_widths}
        ),
        initial_nodes={18: "working_life"},
        enable_jit=enable_jit,
    )


@pytest.fixture
def synthetic_peaks(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, int]]:
    """Report a peak of `_BYTES_PER_ACTION` per unit of width product."""
    seen: list[dict[str, int]] = []

    def peak(
        *, compiled: object, widths: Mapping[str, int]
    ) -> CompilerMemoryReservation:
        del compiled
        seen.append(dict(widths))
        return synthetic_memory(_BYTES_PER_ACTION * math.prod(widths.values()))

    monkeypatch.setattr(backward_induction, "compiler_memory_reservation", peak)
    return seen


def _solve_capturing(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device_memory_bytes: int | None,
    axis_widths: Mapping[str, int] = MappingProxyType({}),
):
    """Solve while capturing the first working-life period."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", _CAPTURE_TARGET)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = _model(
        execution_config=ExecutionConfig(
            device_memory_bytes=device_memory_bytes,
            axis_widths=axis_widths,
        )
    )
    return model.solve(
        params=get_params(n_periods=_N_PERIODS),
        log_level="off",
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
        device_memory_bytes=None,
    )

    assert _captured_widths(tmp_path) == {"main": {"action_product": 4, "cell": 1}}


@pytest.mark.usefixtures("synthetic_peaks")
def test_budget_selects_the_widest_feasible_action_block(
    *, monkeypatch, tmp_path
) -> None:
    """Frontier 1, 2, 4, 6 with a budget of two width units selects width 2."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=_fixed_owner_bytes() + 2 * _BYTES_PER_ACTION,
    )

    assert _captured_widths(tmp_path) == {"main": {"action_product": 2, "cell": 1}}


@pytest.mark.usefixtures("synthetic_peaks")
def test_budget_above_every_candidate_keeps_the_full_action_product(
    *, monkeypatch, tmp_path
) -> None:
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=_fixed_owner_bytes() + _ACTION_EXTENT * _BYTES_PER_ACTION,
    )

    assert _captured_widths(tmp_path) == {
        "main": {"action_product": _ACTION_EXTENT, "cell": 1}
    }


def test_budget_above_every_candidate_lowers_only_the_full_action_product(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    """A budget every core meets costs no more compilation than no budget."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=_fixed_owner_bytes() + _ACTION_EXTENT * _BYTES_PER_ACTION,
    )

    assert [
        widths["action_product"]
        for widths in synthetic_peaks
        if "action_product" in widths
    ] == [_ACTION_EXTENT]


def test_budget_lowers_widths_descending_until_one_fits(
    *, synthetic_peaks, monkeypatch, tmp_path
) -> None:
    """Only the candidates wider than the selected one are compiled and rejected."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=_fixed_owner_bytes() + 2 * _BYTES_PER_ACTION,
    )

    assert [
        widths["action_product"]
        for widths in synthetic_peaks
        if "action_product" in widths
    ] == [
        6,
        4,
        2,
    ]


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
        device_memory_bytes=_fixed_owner_bytes() + 2 * _BYTES_PER_ACTION,
    )
    assert _captured_widths(tmp_path) == {"main": {"action_product": 2, "cell": 1}}

    for period in range(_N_PERIODS - 1):
        assert_agrees_to_ulp(
            got=np.asarray(budgeted.values[period]["working_life"]),
            expected=np.asarray(unbudgeted.values[period]["working_life"]),
            n_ulp=16,
        )


def test_budget_below_every_candidate_fails_closed(*, monkeypatch, tmp_path) -> None:
    """One byte fits no compiled core, so planning refuses before backward induction."""
    with pytest.raises(ExecutionPlanningError, match="leaving nothing"):
        _solve_capturing(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            device_memory_bytes=1,
        )

    assert not (tmp_path / _CAPTURE_TARGET).exists()


@pytest.fixture
def fixed_width_solve(*, synthetic_peaks, monkeypatch, tmp_path) -> dict[str, object]:
    """Solve the regression model with the action product pinned to width four."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=_fixed_owner_bytes() + _ACTION_EXTENT * _BYTES_PER_ACTION,
        axis_widths={"action_product": 4},
    )
    return {
        "compiled": {
            widths["action_product"]
            for widths in synthetic_peaks
            if "action_product" in widths
        },
        "dispatched": _captured_widths(tmp_path),
    }


def test_a_fixed_width_leaves_the_planner_one_candidate_to_compile(
    *, fixed_width_solve
) -> None:
    """A fixed axis width is the only width the planner ever compiles."""
    assert fixed_width_solve["compiled"] == {4}


def test_a_fixed_width_is_the_width_the_core_is_dispatched_at(
    *, fixed_width_solve
) -> None:
    """The dispatched core runs at exactly the width the config fixed."""
    assert fixed_width_solve["dispatched"] == {"main": {"action_product": 4, "cell": 1}}


def test_a_fixed_width_for_an_axis_no_program_declares_is_refused() -> None:
    """A misspelled axis name is refused at model build, listing the declared names."""
    with pytest.raises(ExecutionPlanningError, match="action_product"):
        _model(execution_config=ExecutionConfig(axis_widths={"action_produkt": 8}))


@pytest.mark.usefixtures("synthetic_peaks")
def test_fixed_action_product_width_over_budget_names_the_request(
    *, monkeypatch, tmp_path
) -> None:
    """A fixed width that no budget can serve is refused, naming the request."""
    with pytest.raises(ExecutionPlanningError, match="explicitly requested"):
        _solve_capturing(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            device_memory_bytes=_fixed_owner_bytes() + 4 * _BYTES_PER_ACTION - 1,
            axis_widths={"action_product": 4},
        )


@pytest.mark.usefixtures("synthetic_peaks")
def test_replay_of_a_budgeted_capture_reproduces_its_value(
    *, monkeypatch, tmp_path
) -> None:
    """Replay lowers the captured width and returns the array the solve published."""
    budgeted = _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        device_memory_bytes=2 * _BYTES_PER_ACTION,
    )

    replay = replay_period(directory=tmp_path / _CAPTURE_TARGET)

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(budgeted.values[0]["working_life"]),
    )


def _report_pool_limit(*, monkeypatch: pytest.MonkeyPatch, bytes_limit: int) -> None:
    """Make every visible device report `bytes_limit`, as a preallocated GPU does."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setattr(
        "lcm.model.visible_device_pool_limits",
        lambda: dict.fromkeys(visible_device_ids(), bytes_limit),
    )


def test_an_eager_solve_refuses_the_default_budget_and_runs_without_one(
    *, monkeypatch
) -> None:
    """`enable_jit=False` refuses the device default; opting out solves alike."""
    _report_pool_limit(monkeypatch=monkeypatch, bytes_limit=1 << 30)
    params = get_params(n_periods=_N_PERIODS)
    compiled = _model().solve(params=params, log_level="off")

    with pytest.raises(
        ExecutionPlanningError,
        match=r"requires JIT compilation.*device_memory_bytes=None",
    ):
        _model(execution_config=ExecutionConfig(), enable_jit=False).solve(
            params=params, log_level="off"
        )
    eager = _model(
        execution_config=ExecutionConfig(device_memory_bytes=None), enable_jit=False
    ).solve(params=params, log_level="off")

    for period in range(_N_PERIODS - 1):
        assert_agrees_to_ulp(
            got=np.asarray(eager.values[period]["working_life"]),
            expected=np.asarray(compiled.values[period]["working_life"]),
            n_ulp=16,
        )


def test_an_eager_solve_with_an_explicit_budget_is_refused() -> None:
    """An explicit budget requires admission, which needs a compiled program."""
    model = _model(
        execution_config=ExecutionConfig(device_memory_bytes=1 << 30),
        enable_jit=False,
    )

    with pytest.raises(ExecutionPlanningError, match="requires JIT compilation"):
        model.solve(params=get_params(n_periods=_N_PERIODS), log_level="off")


def test_a_refusal_under_the_default_budget_names_its_source_and_remedies(
    *, monkeypatch
) -> None:
    """A model too large for the device default is told what to change."""
    _report_pool_limit(monkeypatch=monkeypatch, bytes_limit=2)

    with pytest.raises(ExecutionPlanningError) as refusal:
        _model(execution_config=ExecutionConfig()).solve(
            params=get_params(n_periods=_N_PERIODS), log_level="off"
        )

    message = str(refusal.value)
    assert "derived from the device pool limit (default)" in message
    for remedy in (
        "device_memory_headroom_fraction",
        "axis_width_ceilings",
        "sharded_states",
        "`device_memory_bytes=None` disables admission",
    ):
        assert remedy in message
