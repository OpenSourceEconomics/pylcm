"""An opt-in ceiling caps the widths the planner may tile a solve or simulate axis at.

`ExecutionConfig.axis_width_ceilings` names an upper bound per planner axis. The
planner intersects its legal candidates with it: no axis extent shortens, no
output shape changes, and every other axis keeps the widths it had. Solve cell axes
and simulate subject axes are both bound, budgeted or not.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintViolation

import _lcm.simulation.runtime as simulation_runtime
from _lcm.execution.workspace_planning import (
    bootstrap_widths,
    workspace_width_candidates,
)
from _lcm.solution.period_capture import _PAYLOAD_NAME
from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.exceptions import ExecutionPlanningError
from lcm.solvers import GridSearch
from tests.conftest import assert_agrees_to_ulp
from tests.execution.test_workspace_planning import _axis, _tiled
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 2
_N_WEALTH = 8
_CAPTURE_TARGET = "working_life@0"
_CELL_CEILING = 2


def _candidates(*, ceilings: Mapping[str, int]) -> tuple[Mapping[str, int], ...]:
    """Enumerate the budgeted frontier of one reduced and one tiled axis."""
    return workspace_width_candidates(
        axes=(_axis(extent=6), _tiled(extent=8)),
        width_ceilings=ceilings,
        budget_bytes=1_000_000,
    )


def test_workspace_width_candidates_ceiling_caps_the_widest_cell_width() -> None:
    """No candidate tiles the ceiling-bound axis wider than the ceiling."""
    widest = max(widths["cell"] for widths in _candidates(ceilings={"cell": 2}))

    assert widest == 2


def test_workspace_width_candidates_ceiling_leaves_the_action_frontier_intact() -> None:
    """Bounding the cell axis changes no width the action axis is offered."""
    bound = {widths["action_product"] for widths in _candidates(ceilings={"cell": 2})}

    assert bound == {widths["action_product"] for widths in _candidates(ceilings={})}


def test_workspace_width_candidates_ceiling_below_the_floor_names_the_axis() -> None:
    """A ceiling under the narrowest legal width refuses, naming axis and value."""
    with pytest.raises(ExecutionPlanningError, match=r"'cell'.*\b2\b"):
        workspace_width_candidates(
            axes=(_tiled(extent=8, minimum_width=4),),
            width_ceilings={"cell": 2},
            budget_bytes=1_000_000,
        )


def test_bootstrap_widths_respects_the_ceiling() -> None:
    """An unbudgeted plan lowers the bounded axis at the ceiling, not its bootstrap."""
    widths = bootstrap_widths(axes=(_tiled(extent=8),), width_ceilings={"cell": 2})

    assert widths == {"cell": 2}


def test_execution_config_rejects_a_non_int_ceiling() -> None:
    with pytest.raises((BeartypeCallHintViolation, TypeError)):
        ExecutionConfig(axis_width_ceilings={"cell": 2.0})  # ty: ignore[invalid-argument-type]


def test_execution_config_rejects_a_non_positive_ceiling() -> None:
    with pytest.raises(ValueError, match="axis_width_ceilings"):
        ExecutionConfig(axis_width_ceilings={"cell": 0})


def test_execution_config_rejects_a_per_regime_ceiling() -> None:
    """Only the bare-integer form exists; a per-regime mapping is not a width."""
    with pytest.raises((BeartypeCallHintViolation, TypeError)):
        ExecutionConfig(axis_width_ceilings={"cell": {"working_life": 2}})  # ty: ignore[invalid-argument-type]


def _model(
    *,
    axis_width_ceilings: Mapping[str, int] = MappingProxyType({}),
    device_memory_bytes: int | None = None,
) -> Model:
    """Build a two-period GridSearch model whose cell axis has extent `_N_WEALTH`."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=_N_WEALTH)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(
            axis_width_ceilings=axis_width_ceilings,
            device_memory_bytes=device_memory_bytes,
        ),
    )


def _solve_capturing(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    axis_width_ceilings: Mapping[str, int],
):
    """Solve while capturing the first working-life period's dispatched widths."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", _CAPTURE_TARGET)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = _model(axis_width_ceilings=axis_width_ceilings)
    return model.solve(params=get_params(n_periods=_N_PERIODS), log_level="off")


def _captured_widths(tmp_path: Path) -> dict[str, dict[str, int]]:
    with (tmp_path / _CAPTURE_TARGET / _PAYLOAD_NAME).open("rb") as stream:
        return cloudpickle.load(stream)["core_tile_widths"]


def test_ceiling_caps_the_dispatched_cell_width(*, monkeypatch, tmp_path) -> None:
    """The solve tiles cells at the ceiling and keeps the unbounded action width."""
    _solve_capturing(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        axis_width_ceilings={"cell": _CELL_CEILING},
    )

    assert _captured_widths(tmp_path) == {
        "main": {"action_product": 4, "cell": _CELL_CEILING}
    }


def test_no_ceiling_keeps_the_bootstrap_cell_width(*, monkeypatch, tmp_path) -> None:
    """Without a ceiling the same model tiles cells at its bootstrap width."""
    _solve_capturing(monkeypatch=monkeypatch, tmp_path=tmp_path, axis_width_ceilings={})

    assert _captured_widths(tmp_path) == {"main": {"action_product": 4, "cell": 4}}


def test_ceiling_leaves_the_value_shape_unchanged() -> None:
    """Tiling narrower partitions the same output; the value array keeps its shape."""
    bounded = _model(axis_width_ceilings={"cell": _CELL_CEILING}).solve(
        params=get_params(n_periods=_N_PERIODS), log_level="off"
    )

    assert np.asarray(bounded.values[0]["working_life"]).shape == (_N_WEALTH,)


def test_ceiling_leaves_the_solved_values_unchanged() -> None:
    """A narrower cell tile partitions the same maximum, bit for bit."""
    unbounded = _model().solve(params=get_params(n_periods=_N_PERIODS), log_level="off")
    bounded = _model(axis_width_ceilings={"cell": _CELL_CEILING}).solve(
        params=get_params(n_periods=_N_PERIODS), log_level="off"
    )

    assert_agrees_to_ulp(
        got=np.asarray(bounded.values[0]["working_life"]),
        expected=np.asarray(unbounded.values[0]["working_life"]),
        n_ulp=0,
    )


_N_SUBJECTS = 8
_SUBJECT_CEILING = 2
_SIMULATION_BUDGETS = [None, 2**30]


def _simulate_recording_widths(
    *,
    monkeypatch: pytest.MonkeyPatch,
    axis_width_ceilings: Mapping[str, int],
    device_memory_bytes: int | None,
) -> tuple[pd.DataFrame, list[Mapping[str, int]]]:
    """Simulate `_N_SUBJECTS` subjects, recording every width a dispatch selects."""
    selected: list[Mapping[str, int]] = []
    plan_workspace = simulation_runtime.plan_workspace

    def record(**arguments: Any) -> Any:
        plan = plan_workspace(**arguments)
        selected.append(plan.widths)
        return plan

    monkeypatch.setattr(simulation_runtime, "plan_workspace", record)
    model = _model(
        axis_width_ceilings=axis_width_ceilings, device_memory_bytes=device_memory_bytes
    )
    params = get_params(n_periods=_N_PERIODS)
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.linspace(1.0, 3.0, _N_SUBJECTS),
            "age": jnp.full(_N_SUBJECTS, float(START_AGE)),
            "regime_id": jnp.full(_N_SUBJECTS, RegimeId.working_life, dtype=jnp.int32),
        },
        solution=model.solve(params=params, log_level="off"),
        seed=7,
        log_level="off",
    )
    return result.to_dataframe(), selected


def _subject_widths(selected: list[Mapping[str, int]]) -> set[int]:
    return {widths["subject"] for widths in selected if "subject" in widths}


@pytest.mark.parametrize("device_memory_bytes", _SIMULATION_BUDGETS)
def test_ceiling_caps_every_simulated_subject_width(
    *, monkeypatch: pytest.MonkeyPatch, device_memory_bytes: int | None
) -> None:
    """Every simulation dispatch tiles subjects at the declared ceiling."""
    _, selected = _simulate_recording_widths(
        monkeypatch=monkeypatch,
        axis_width_ceilings={"subject": _SUBJECT_CEILING},
        device_memory_bytes=device_memory_bytes,
    )

    assert _subject_widths(selected) == {_SUBJECT_CEILING}


@pytest.mark.parametrize("device_memory_bytes", _SIMULATION_BUDGETS)
def test_no_ceiling_tiles_subjects_wider_than_the_ceiling(
    *, monkeypatch: pytest.MonkeyPatch, device_memory_bytes: int | None
) -> None:
    """Without a ceiling the same simulation tiles subjects wider than it."""
    _, selected = _simulate_recording_widths(
        monkeypatch=monkeypatch,
        axis_width_ceilings={},
        device_memory_bytes=device_memory_bytes,
    )

    assert min(_subject_widths(selected)) > _SUBJECT_CEILING


@pytest.mark.parametrize("device_memory_bytes", _SIMULATION_BUDGETS)
def test_subject_ceiling_leaves_the_simulated_panel_unchanged(
    *, monkeypatch: pytest.MonkeyPatch, device_memory_bytes: int | None
) -> None:
    """Narrower subject tiles partition the same panel, value for value."""
    unbounded, _ = _simulate_recording_widths(
        monkeypatch=monkeypatch,
        axis_width_ceilings={},
        device_memory_bytes=device_memory_bytes,
    )
    bounded, _ = _simulate_recording_widths(
        monkeypatch=monkeypatch,
        axis_width_ceilings={"subject": _SUBJECT_CEILING},
        device_memory_bytes=device_memory_bytes,
    )

    pd.testing.assert_frame_equal(bounded, unbounded, check_exact=True)
