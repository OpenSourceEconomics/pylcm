"""An opt-in ceiling caps the widths the planner may tile a cell axis at.

`ExecutionConfig.axis_width_ceilings` names an upper bound per planner axis. The
planner intersects its legal candidates with it: no axis extent shortens, no
output shape changes, and every other axis keeps the widths it had.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import cloudpickle
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

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


def _model(*, axis_width_ceilings: Mapping[str, int] = MappingProxyType({})) -> Model:
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
        execution_config=ExecutionConfig(axis_width_ceilings=axis_width_ceilings),
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
