"""A GridSearch cell width is halved while its compiled reduce materialises a gather.

The compiled-program check is replaced by a fake that answers from the cell width,
so the planner's response is exercised on any backend.
"""

import ast
import collections
import functools
import inspect
import itertools
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintViolation

from _lcm.execution.hlo_fusions import UnrecognisedHloError
from _lcm.solution import backward_induction
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import X64_ENABLED, assert_agrees_to_ulp
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
    *,
    config: ExecutionConfig,
    limit: int | None,
    check: Callable[..., str | None] | None = None,
    model: Model | None = None,
) -> tuple[dict[str, set[int]], Any]:
    """Solve under a fake check and return each regime's cell widths and the result.

    `check`, when given, replaces the width-limit fake as the compiled-program check;
    `model`, when given, is solved instead of a fresh model built from `config`.
    """
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
        if check is not None:
            patcher.setattr(backward_induction, "_materialised_gather_fusion", check)
        elif limit is not None:
            patcher.setattr(
                backward_induction,
                "_materialised_gather_fusion",
                functools.partial(_materialised_above, limit=limit),
            )
        solution = (model or _model_with(config)).solve(
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
    with pytest.raises(BeartypeCallHintViolation, match="halve_on_materialised_gather"):
        ExecutionConfig(halve_on_materialised_gather=1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("dtype", "exponent"), [(np.float32, 20), (np.float64, 49)])
def test_halving_ulp_check_rejects_an_unrelated_large_state_scale(
    *, dtype: type[np.float32 | np.float64], exponent: int
) -> None:
    """An unrelated state's level must not relax the small state's ULP budget."""
    expected = np.asarray([2.0**exponent, 1.0], dtype=dtype)
    got = np.asarray([2.0**exponent, 1.5], dtype=dtype)
    with pytest.raises(AssertionError, match="ULP"):
        assert_agrees_to_ulp(got=got, expected=expected, n_ulp=8)


@pytest.mark.parametrize(("dtype", "exponent"), [(np.float32, 20), (np.float64, 49)])
@pytest.mark.parametrize(
    ("scale", "sign", "position"), itertools.product((-8, 0, 8), (-1, 1), (0, 1))
)
def test_halving_ulp_check_rejects_a_local_error_beside_any_unrelated_outlier(
    *,
    dtype: type[np.float32 | np.float64],
    exponent: int,
    scale: int,
    sign: int,
    position: int,
) -> None:
    """A small state off by half its value fails, whatever its sign, scale or slot."""
    expected = [sign * 2.0**scale]
    got = [sign * 1.5 * 2.0**scale]
    expected.insert(position, 2.0 ** (exponent + scale))
    got.insert(position, 2.0 ** (exponent + scale))
    with pytest.raises(AssertionError, match="ULP"):
        assert_agrees_to_ulp(
            got=np.asarray(got, dtype=dtype),
            expected=np.asarray(expected, dtype=dtype),
            n_ulp=8,
        )


def test_the_halving_value_check_measures_each_element_at_its_own_spacing() -> None:
    """The solved-value comparison passes no array-wide operand magnitude."""
    tree = ast.parse(
        inspect.getsource(test_halving_the_cell_width_preserves_the_solved_values)
    )
    keywords = {
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_agrees_to_ulp"
        for keyword in node.keywords
    }

    assert keywords == {"got", "expected", "n_ulp", "err_msg"}


def _simulated_choices(*, limit: int | None) -> pd.DataFrame:
    """Solve, then replay over every starting wealth and health on the grid."""
    model = _model_with(ExecutionConfig())
    _, solution = _solve(config=ExecutionConfig(), limit=limit, model=model)
    wealth, health = np.meshgrid(np.linspace(1, 5, 5), np.arange(2))
    n_subjects = wealth.size
    return model.simulate(
        params=get_multi_regime_params("normal"),
        initial_conditions={
            "health": jnp.asarray(health.ravel(), dtype=jnp.int32),
            "income": jnp.zeros(n_subjects),
            "wealth": jnp.asarray(wealth.ravel()),
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(n_subjects, MultiRegimeId.work, dtype=jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=463,
    ).to_dataframe()


@pytest.mark.skipif(not X64_ENABLED, reason="Not working with 32-Bit because of RNG")
@pytest.mark.parametrize("column", ["regime_name", "consumption"])
def test_halving_the_cell_width_replays_the_same_choices(*, column: str) -> None:
    """The halved and planned solutions route and choose identically in simulation."""
    pd.testing.assert_series_equal(
        _simulated_choices(limit=_planned_width() // 2)[column],
        _simulated_choices(limit=None)[column],
    )


def _unreadable(*, compiled: object, widths: Any) -> str | None:
    del compiled, widths
    msg = "reduce fusions ['loop_reduce_fusion'] were not read completely"
    raise UnrecognisedHloError(msg)


def test_a_program_the_check_cannot_read_keeps_the_planned_width() -> None:
    widths, _ = _solve(config=ExecutionConfig(), limit=None, check=_unreadable)

    assert set().union(*widths.values()) == {_planned_width()}


def _counting(
    *, compiled: object, widths: Any, counts: collections.Counter[int], limit: int
) -> str | None:
    counts[id(compiled)] += 1
    return _materialised_above(compiled=compiled, widths=widths, limit=limit)


@pytest.mark.parametrize("halvings", [0, 1])
def test_each_compiled_program_is_classified_once_per_solve(*, halvings: int) -> None:
    """Cores sharing an executable reuse its verdict instead of reading it again."""
    counts: collections.Counter[int] = collections.Counter()
    _solve(
        config=ExecutionConfig(),
        limit=None,
        check=functools.partial(
            _counting, counts=counts, limit=_planned_width() // 2**halvings
        ),
    )

    assert set(counts.values()) == {1}
