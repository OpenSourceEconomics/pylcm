"""A GridSearch cell width is halved while its compiled reduce materialises a gather.

The compiled-program check is replaced by a fake that answers from the cell width,
so the planner's response is exercised on any backend.
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

_N_PERIODS = 3
_CELL_AXIS = "cell"


def _model_with(config: ExecutionConfig) -> Model:
    base = get_multi_regime_model(n_periods=_N_PERIODS, distribution_type="normal")
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
    for triple, core in cores_by_triple.items():
        sink[triple] = dict(core.tile_widths)
    return original(cores_by_triple)


def _materialised_above(*, compiled: object, widths: Any, limit: int) -> str | None:
    """Report a materialised fusion whenever the cell width exceeds `limit`."""
    del compiled
    return "loop_reduce_fusion" if widths.get(_CELL_AXIS, 0) > limit else None


def _solve(
    *, config: ExecutionConfig, limit: int | None
) -> tuple[dict[str, set[int]], Any]:
    """Solve under a fake check and return each regime's cell widths and the result."""
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
        if limit is not None:
            patcher.setattr(
                backward_induction,
                "_materialised_gather_fusion",
                functools.partial(_materialised_above, limit=limit),
            )
        solution = _model_with(config).solve(
            params=get_multi_regime_params("normal"), log_level="off"
        )
    by_regime: dict[str, set[int]] = {}
    for (regime_name, _period, _core), widths in observed.items():
        if _CELL_AXIS in widths:
            by_regime.setdefault(regime_name, set()).add(widths[_CELL_AXIS])
    return by_regime, solution


@functools.cache
def _planned_width() -> int:
    """The single cell width the planner picks when every program fuses."""
    widths, _ = _solve(config=ExecutionConfig(), limit=None)
    (width,) = set().union(*widths.values())
    assert width >= 4, "the halving tests need a planned width of at least four"
    return width


def test_a_fused_program_keeps_the_planned_width() -> None:
    widths, _ = _solve(config=ExecutionConfig(), limit=10**9)

    assert set().union(*widths.values()) == {_planned_width()}


def test_a_materialised_program_is_halved_until_it_fuses() -> None:
    """Materialising above a quarter of the planned width halves it twice."""
    planned = _planned_width()
    widths, _ = _solve(config=ExecutionConfig(), limit=planned // 4)

    assert set().union(*widths.values()) == {planned // 4}


def test_halving_the_cell_width_preserves_the_solved_values() -> None:
    planned = _planned_width()
    _, expected_solution = _solve(config=ExecutionConfig(), limit=None)
    _, halved_solution = _solve(config=ExecutionConfig(), limit=planned // 2)

    expected_values = expected_solution._engine_view.values
    halved_values = halved_solution._engine_view.values
    assert set(halved_values) == set(expected_values)
    for period, by_regime in expected_values.items():
        for regime_name, expected in by_regime.items():
            assert_agrees_to_ulp(
                got=halved_values[period][regime_name],
                expected=expected,
                n_ulp=8,
                err_msg=f"{regime_name} period {period}",
                operand_magnitude=float(np.abs(np.asarray(expected)).max()),
            )


def test_a_program_materialising_at_its_narrowest_width_fails_loudly() -> None:
    with pytest.raises(ExecutionPlanningError, match="loop_reduce_fusion"):
        _solve(config=ExecutionConfig(), limit=0)


def test_the_opt_out_keeps_a_materialised_width() -> None:
    widths, _ = _solve(
        config=ExecutionConfig(halve_on_materialised_gather=False), limit=0
    )

    assert set().union(*widths.values()) == {_planned_width()}


def test_a_fixed_cell_width_is_never_halved() -> None:
    widths, _ = _solve(
        config=ExecutionConfig(axis_widths=MappingProxyType({_CELL_AXIS: 2})),
        limit=0,
    )

    assert set().union(*widths.values()) == {2}


def test_execution_config_refuses_a_non_bool_halving_switch() -> None:
    with pytest.raises(TypeError, match="halve_on_materialised_gather"):
        ExecutionConfig(halve_on_materialised_gather=1)  # ty: ignore[invalid-argument-type]
