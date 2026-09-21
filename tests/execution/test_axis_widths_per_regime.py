"""`ExecutionConfig.axis_widths` may fix one axis width per regime.

A bare integer under an axis name broadcasts to every regime declaring that
axis; a mapping from regime name to width pins only the regimes it names and
leaves the planner free everywhere else.
"""

import functools
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from _lcm.solution import backward_induction
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import assert_agrees_to_ulp
from tests.test_models.processes import (
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

_N_PERIODS = 6
_CELL_AXIS = "cell"


def _base_model() -> Model:
    return get_multi_regime_model(n_periods=_N_PERIODS, distribution_type="normal")


def _model_with(config: ExecutionConfig) -> Model:
    base = _base_model()
    return Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=config,
    )


# keyword-only-exempt: library-callback=_group_cores_by_regime_period
def _capture_grouping(
    cores_by_triple: Any,
    *,
    original: Any,
    sink: dict[tuple[str, int, str], dict[str, int]],
) -> Any:
    """Record each compiled core's lowering widths under its (regime, period, core)."""
    for triple, core in cores_by_triple.items():
        sink[triple] = dict(core.tile_widths)
    return original(cores_by_triple)


def _solve_and_collect_widths(
    *, config: ExecutionConfig
) -> tuple[dict[tuple[str, int, str], dict[str, int]], Any]:
    """Solve the two-regime model and return the widths every core was lowered at."""
    model = _model_with(config)
    observed: dict[tuple[str, int, str], dict[str, int]] = {}
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            backward_induction,
            "_group_cores_by_regime_period",
            functools.partial(
                _capture_grouping,
                original=backward_induction._group_cores_by_regime_period,
                sink=observed,
            ),
        )
        solution = model.solve(
            params=get_multi_regime_params("normal"),
            log_level="off",
        )
    return observed, solution


def _cell_widths_by_regime(
    widths: dict[tuple[str, int, str], dict[str, int]],
) -> dict[str, set[int]]:
    """Collapse the per-core record to the cell widths each regime was lowered at."""
    by_regime: dict[str, set[int]] = {}
    for (regime_name, _period, _core), core_widths in widths.items():
        if _CELL_AXIS in core_widths:
            by_regime.setdefault(regime_name, set()).add(core_widths[_CELL_AXIS])
    return by_regime


def test_a_bare_integer_broadcasts_to_every_regime() -> None:
    """One integer under an axis name fixes that axis in every regime."""
    observed, _ = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: 2})
    )

    assert _cell_widths_by_regime(observed) == {"work": {2}, "retire": {2}}


@pytest.mark.parametrize("budget", [None, 100_000_000])
@pytest.mark.parametrize("override", [None, 4, {"retire": 4}])
def test_grid_search_action_width_defaults_to_one(
    *, budget: int | None, override: Any
) -> None:
    """Stream every action at width one unless its regime has an explicit width."""
    observed, solution = _solve_and_collect_widths(
        config=ExecutionConfig(
            device_memory_bytes=budget,
            axis_widths={} if override is None else {"action_product": override},
        )
    )
    action_widths = {
        regime: {
            widths["action_product"]
            for (name, _, _), widths in observed.items()
            if name == regime and "action_product" in widths
        }
        for regime in ("work", "retire")
    }
    assert action_widths == {
        "work": {4 if override == 4 else 1},
        "retire": {1 if override is None else 4},
    }
    _, reference = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={"action_product": 4})
    )
    for period, by_regime in reference._engine_view.values.items():
        for regime_name, expected in by_regime.items():
            assert_agrees_to_ulp(
                got=solution._engine_view.values[period][regime_name],
                expected=expected,
                n_ulp=8,
                operand_magnitude=float(np.abs(np.asarray(expected)).max()),
            )


def test_a_per_regime_width_leaves_every_other_regime_planned() -> None:
    """Pinning one regime's cell width does not narrow the other regime's."""
    planned, _ = _solve_and_collect_widths(config=ExecutionConfig())
    pinned, _ = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2}}),
    )

    planned_by_regime = _cell_widths_by_regime(planned)
    pinned_by_regime = _cell_widths_by_regime(pinned)

    assert pinned_by_regime["retire"] == {2}
    assert pinned_by_regime["work"] == planned_by_regime["work"]
    assert planned_by_regime["retire"] == planned_by_regime["work"] != {2}


def test_a_per_regime_width_preserves_the_solved_values() -> None:
    """Chunking one regime finer partitions the work without changing the answer."""
    _, planned_solution = _solve_and_collect_widths(config=ExecutionConfig())
    _, pinned_solution = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2}}),
    )

    planned_values = planned_solution._engine_view.values
    pinned_values = pinned_solution._engine_view.values
    assert set(pinned_values) == set(planned_values)
    for period, by_regime in planned_values.items():
        for regime_name, expected in by_regime.items():
            # Values near zero are born by cancellation between the flow utility and
            # the discounted continuation, so the gap is measured at the spacing of
            # those operands rather than at the compared element's own magnitude.
            assert_agrees_to_ulp(
                got=pinned_values[period][regime_name],
                expected=expected,
                n_ulp=8,
                err_msg=f"{regime_name} period {period}",
                operand_magnitude=float(np.abs(np.asarray(expected)).max()),
            )


def test_a_mapping_and_an_integer_may_share_one_declaration() -> None:
    """A regime the mapping does not name keeps the broadcast width."""
    observed, _ = _solve_and_collect_widths(
        config=ExecutionConfig(
            axis_widths={_CELL_AXIS: 4, "action_product": {"retire": 1}}
        ),
    )

    assert _cell_widths_by_regime(observed) == {"work": {4}, "retire": {4}}


def test_per_regime_widths_are_read_only_after_construction() -> None:
    """The caller's nested dict cannot be mutated into the stored configuration."""
    widths: dict[str, Any] = {_CELL_AXIS: {"retire": 2}}
    config = ExecutionConfig(axis_widths=widths)

    widths[_CELL_AXIS]["retire"] = 8

    assert config.axis_widths[_CELL_AXIS] == MappingProxyType({"retire": 2})
    with pytest.raises(TypeError):
        config.axis_widths[_CELL_AXIS]["retire"] = 8  # ty: ignore[invalid-assignment]


def test_two_configurations_with_equal_per_regime_widths_are_equal() -> None:
    """Configuration identity survives the nested form."""
    first = ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2, "work": 4}})
    second = ExecutionConfig(axis_widths={_CELL_AXIS: {"work": 4, "retire": 2}})

    assert first == second


def test_per_regime_widths_reject_a_non_positive_width() -> None:
    """A width of zero is refused at construction, naming axis and regime."""
    with pytest.raises(
        ValueError, match=r"axis_widths\['cell'\]\['retire'\] must be positive"
    ):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 0}})


def test_per_regime_widths_reject_a_bool_width() -> None:
    """Widths are exact ints, so a bool is refused."""
    with pytest.raises(TypeError, match=r"axis_widths\['cell'\]\['retire'\]"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": True}})


def test_per_regime_widths_reject_an_empty_mapping() -> None:
    """An empty mapping names no regime, so it cannot be what the caller meant."""
    with pytest.raises(ValueError, match=r"axis_widths\['cell'\] names no regime"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {}})


def test_per_regime_widths_reject_an_empty_regime_name() -> None:
    """Regime keys are non-empty strings."""
    with pytest.raises(TypeError, match=r"axis_widths\['cell'\] keys"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"": 2}})


def test_model_rejects_a_per_regime_width_for_an_unknown_regime() -> None:
    """A regime no model declares is refused at model build, listing the known ones."""
    with pytest.raises(
        ExecutionPlanningError, match=r"axis_widths\['cell'\] names regime 'nope'"
    ):
        _model_with(ExecutionConfig(axis_widths={_CELL_AXIS: {"nope": 2}}))


def test_model_rejects_a_per_regime_width_for_an_unknown_axis() -> None:
    """A per-regime width for an axis no core program declares is refused."""
    with pytest.raises(ExecutionPlanningError, match="axis_widths names 'nope'"):
        _model_with(ExecutionConfig(axis_widths={"nope": {"retire": 2}}))


def test_model_rejects_a_per_regime_width_for_a_simulation_only_axis() -> None:
    """Simulation plans without a regime in hand, so it takes the broadcast form."""
    with pytest.raises(ExecutionPlanningError, match="subject"):
        _model_with(ExecutionConfig(axis_widths={"subject": {"retire": 2}}))
