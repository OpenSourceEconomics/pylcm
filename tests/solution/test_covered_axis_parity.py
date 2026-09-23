"""Covering a planner axis changes its tile width and, at most, the last ulp.

Two tiny models expose a reduced axis whose extent (20) the power-of-two
bootstrap width (16) does not divide:

- the two-period NB-EGM multi-discrete toy with three discrete actions, whose
  `branch` axis enumerates 2 x 2 x 5 = 20 action combinations;
- the GridSearch work-and-consume regression model, whose streamed
  `action_product` axis holds 2 labor-supply x 10 consumption = 20 points.

Each is solved with and without covering its axis, once unbudgeted and once
under a budget with the bounded width search: the two arms that start from the
seed. A budgeted exhaustive search already dispatches the full extent, so
covering cannot change it there. Dispatched widths are read from the planner
through `CensusRecorder`, and every published array is compared byte for byte.

The NB-EGM toy at its builder's default sizes, three periods and a budgeted
bounded search, publishes the same bytes covered and uncovered on the CPU
backend. A wider tile can reorder rounding in fused reductions on other
backends, so every finite value is also held to `_MAX_ULP` of the uncovered
solve; shapes, dtypes and finite masks must stay exact.

Each distinct solve runs once per worker and is shared across tests.
"""

import functools
from typing import Any

import numpy as np
import pytest

from lcm import ExecutionConfig, LinSpacedGrid
from lcm.execution import WidthSearch, WidthSearchPolicy
from lcm.solvers import ACTION_PRODUCT_AXIS, BRANCH_AXIS
from tests.solution._candidate_census import Census, CensusRecorder
from tests.test_models import nbegm_multi_discrete_toy
from tests.test_models.deterministic import regression

_EXTENT = 20
_SEED = 16
_BUDGET = 10**9
_DEFAULT_SIZE_BUDGET = 40 * 1024**3
_MAX_ULP = 4
_NBEGM = "nbegm"
_GRID_SEARCH = "grid_search"
_AXIS = {_NBEGM: BRANCH_AXIS, _GRID_SEARCH: ACTION_PRODUCT_AXIS}
_UNBUDGETED = "unbudgeted"
_BOUNDED = "bounded"
_SOLVERS = (_NBEGM, _GRID_SEARCH)
_ARMS = (_UNBUDGETED, _BOUNDED)


def _execution(*, arm: str, covered: tuple[str, ...]) -> ExecutionConfig:
    """Build the execution config of one arm, covering `covered`."""
    budget = None if arm == _UNBUDGETED else _BUDGET
    return ExecutionConfig(
        device_memory_bytes=budget,
        width_search=WidthSearchPolicy(kind=WidthSearch.BOUNDED),
        covered_axes=covered,
    )


@functools.cache
def _solve(*, solver: str, arm: str, covered: bool) -> tuple[Any, Census]:
    """Solve one model under one arm; return the result and the planner census."""
    execution = _execution(arm=arm, covered=(_AXIS[solver],) if covered else ())
    recorder = CensusRecorder()
    with pytest.MonkeyPatch.context() as monkeypatch:
        recorder.install(monkeypatch=monkeypatch)
        result = _build_and_solve(solver=solver, execution=execution)
    return result, recorder.census()


def _build_and_solve(*, solver: str, execution: ExecutionConfig) -> Any:
    """Build one of the two models under `execution` and solve it."""
    if solver == _NBEGM:
        model = nbegm_multi_discrete_toy.build_model(
            variant="nbegm",
            n_actions=3,
            n_periods=2,
            n_liquid=8,
            n_savings=10,
            n_consumption=12,
            envelope_arithmetic="ordinary",
            execution_config=execution,
        )
        params = nbegm_multi_discrete_toy.build_params(n_actions=3)
    else:
        model = regression.get_model(
            n_periods=3,
            wealth_grid=LinSpacedGrid(start=1, stop=400, n_points=12),
            consumption_grid=LinSpacedGrid(start=1, stop=400, n_points=10),
            execution_config=execution,
        )
        params = regression.get_params(n_periods=3)
    return model.solve(params=params, log_level="off")


def _dispatched_widths(*, census: Census, axis: str) -> set[int]:
    """Distinct widths of `axis` across every dispatched core that declares it."""
    return {
        dict(core.tile_widths)[axis]
        for core in census.selected_cores.values()
        if axis in dict(core.tile_widths)
    }


def _published_arrays(result) -> dict[object, np.ndarray]:
    """Collect every array the solve publishes, keyed by where it is published."""
    arrays: dict[object, np.ndarray] = {}
    for period, by_regime in result.values.items():
        for regime, value in by_regime.items():
            arrays[("value", period, regime)] = np.asarray(value)
    for ref, artifact in result.replay_artifacts.items():
        arrays[("replay", ref)] = np.asarray(artifact)
    for ref, artifact in result.auxiliary_artifacts.items():
        arrays[("auxiliary", ref)] = np.asarray(artifact)
    return arrays


def _bytes(result) -> dict[object, tuple[tuple[int, ...], np.dtype, bytes]]:
    """Shape, dtype and raw bytes of every published array, for bitwise comparison."""
    return {
        key: (array.shape, array.dtype, array.tobytes())
        for key, array in _published_arrays(result).items()
    }


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("solver", _SOLVERS)
def test_uncovered_solve_dispatches_the_power_of_two_seed(
    *, solver: str, arm: str
) -> None:
    """Without covering, the 20-wide axis streams at the power-of-two seed 16."""
    _, census = _solve(solver=solver, arm=arm, covered=False)

    assert _dispatched_widths(census=census, axis=_AXIS[solver]) == {_SEED}


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("solver", _SOLVERS)
def test_covered_solve_dispatches_the_full_extent(*, solver: str, arm: str) -> None:
    """Covering the axis dispatches it at its full extent of 20 on every core."""
    _, census = _solve(solver=solver, arm=arm, covered=True)

    assert _dispatched_widths(census=census, axis=_AXIS[solver]) == {_EXTENT}


@pytest.mark.parametrize("arm", _ARMS)
@pytest.mark.parametrize("solver", _SOLVERS)
def test_covering_leaves_every_published_array_bitwise_unchanged(
    *, solver: str, arm: str
) -> None:
    """Values and artifacts agree byte for byte with and without covering."""
    uncovered, _ = _solve(solver=solver, arm=arm, covered=False)
    covered, _ = _solve(solver=solver, arm=arm, covered=True)

    assert _bytes(covered) == _bytes(uncovered)


@functools.cache
def _solve_default_size_toy(*, covered: bool) -> tuple[Any, Census]:
    """Solve the NB-EGM toy at its default sizes under a budgeted bounded search."""
    execution = ExecutionConfig(
        device_memory_bytes=_DEFAULT_SIZE_BUDGET,
        width_search=WidthSearchPolicy(kind=WidthSearch.BOUNDED),
        covered_axes=(BRANCH_AXIS,) if covered else (),
    )
    recorder = CensusRecorder()
    with pytest.MonkeyPatch.context() as monkeypatch:
        recorder.install(monkeypatch=monkeypatch)
        model = nbegm_multi_discrete_toy.build_model(
            variant="nbegm",
            n_actions=3,
            envelope_arithmetic="ordinary",
            execution_config=execution,
        )
        result = model.solve(
            params=nbegm_multi_discrete_toy.build_params(n_actions=3),
            log_level="off",
        )
    return result, recorder.census()


def _masks_and_non_float_bytes(result) -> dict[object, tuple]:
    """Shape, dtype and finite mask of float arrays; raw bytes of every other array."""
    out: dict[object, tuple] = {}
    for key, array in _published_arrays(result).items():
        if np.issubdtype(array.dtype, np.floating):
            out[key] = (array.shape, array.dtype, np.isfinite(array).tobytes())
        else:
            out[key] = (array.shape, array.dtype, array.tobytes())
    return out


def _max_ulp_over_finite_floats(*, left, right) -> int:
    """Largest ULP distance between two results' finite float entries."""
    worst = 0
    right_arrays = _published_arrays(right)
    for key, a in _published_arrays(left).items():
        if not np.issubdtype(a.dtype, np.floating):
            continue
        b = right_arrays[key]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            distances = np.testing.assert_array_max_ulp(
                a[mask], b[mask], maxulp=np.iinfo(np.int32).max
            )
            worst = max(worst, int(np.max(distances)))
    return worst


def test_default_size_toy_covered_solve_dispatches_the_full_branch_extent() -> None:
    """At default sizes the covered solve still dispatches `branch` at 20."""
    _, census = _solve_default_size_toy(covered=True)

    assert _dispatched_widths(census=census, axis=BRANCH_AXIS) == {_EXTENT}


def test_default_size_toy_covering_keeps_shapes_dtypes_and_finite_masks() -> None:
    """Covering leaves every shape, dtype, finite mask and non-float array exact."""
    uncovered, _ = _solve_default_size_toy(covered=False)
    covered, _ = _solve_default_size_toy(covered=True)

    assert _masks_and_non_float_bytes(covered) == _masks_and_non_float_bytes(uncovered)


def test_default_size_toy_covering_moves_no_value_beyond_the_ulp_bound() -> None:
    """Every finite published value agrees with the uncovered solve to `_MAX_ULP`."""
    uncovered, _ = _solve_default_size_toy(covered=False)
    covered, _ = _solve_default_size_toy(covered=True)

    assert _max_ulp_over_finite_floats(left=covered, right=uncovered) <= _MAX_ULP


def test_default_size_toy_covering_leaves_every_array_bitwise_unchanged() -> None:
    """Covering at default sizes publishes the same bytes as not covering."""
    uncovered, _ = _solve_default_size_toy(covered=False)
    covered, _ = _solve_default_size_toy(covered=True)

    assert _bytes(covered) == _bytes(uncovered)
