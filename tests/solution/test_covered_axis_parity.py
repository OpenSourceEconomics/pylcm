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
covering cannot change it there. The NB-EGM toy is also solved at its builder's
default sizes under a budgeted bounded search. Dispatched widths are read from
the planner through `CensusRecorder`.

Parity of the published arrays is a two-tier gate:

- on the CPU backend, every published array is byte-for-byte unchanged;
- on every backend, shapes and dtypes are unchanged, the NaN, +Inf and -Inf
  entries of every float array sit at exactly the same places, every integer or
  boolean array is exactly unchanged, and every finite float moves by at most
  `_MAX_ULP[dtype]`: 4 ULP in float64, 1 ULP in float32. A wider tile can
  reorder rounding in fused reductions off the CPU backend.

The NB-EGM toy publishes no discrete action code: its EGM carries are pinned
to the float dtype, and both of its regimes replay by grid recomputation, which
retains no policy artifact. The integer and boolean tier compares nothing for
that model; it applies to any integer or boolean array a solve publishes.

Covered and uncovered solves must also lead forward simulation to the same
discrete choices. That check runs on specimens whose parameters make the
simulated discrete actions vary across subjects:

- the GridSearch model with a lower disutility of work, so that some subjects
  work and others retire;
- the NB-EGM toy at its default sizes with a higher insurance premium, so that
  some subjects buy private insurance and others do not.

Each distinct solve runs once per worker and is shared across tests.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.execution import WidthSearch, WidthSearchPolicy
from lcm.solvers import ACTION_PRODUCT_AXIS, BRANCH_AXIS
from lcm.typing import UserParams
from tests.solution._candidate_census import Census, CensusRecorder
from tests.test_models import nbegm_multi_discrete_toy
from tests.test_models.deterministic import regression

_EXTENT = 20
_SEED = 16
_BUDGET = 10**9
_DEFAULT_SIZE_BUDGET = 40 * 1024**3
_MAX_ULP = {np.dtype(np.float64): 4, np.dtype(np.float32): 1}
_NBEGM = "nbegm"
_GRID_SEARCH = "grid_search"
_AXIS = {_NBEGM: BRANCH_AXIS, _GRID_SEARCH: ACTION_PRODUCT_AXIS}
_UNBUDGETED = "unbudgeted"
_BOUNDED = "bounded"
_DEFAULT_SIZE = "default_size"
_ARM_BUDGET = {
    _UNBUDGETED: None,
    _BOUNDED: _BUDGET,
    _DEFAULT_SIZE: _DEFAULT_SIZE_BUDGET,
}
_SMALL_CASES = tuple(
    (solver, arm)
    for solver in (_NBEGM, _GRID_SEARCH)
    for arm in (_UNBUDGETED, _BOUNDED)
)
_CASES = (*_SMALL_CASES, (_NBEGM, _DEFAULT_SIZE))
# Simulation specimens, one per covered case that yields a non-degenerate panel.
# The NB-EGM toy's two-period sizes are left out: forward simulation of its
# single alive period returns a NaN value for every subject, and each discrete
# action then sits at its first code, so the comparison would be vacuous.
_SIMULATION_CASES = (
    (_GRID_SEARCH, _UNBUDGETED),
    (_GRID_SEARCH, _BOUNDED),
    (_NBEGM, _DEFAULT_SIZE),
)
_SIMULATION_PARAMS: dict[str, dict[str, float]] = {
    _GRID_SEARCH: {"disutility_of_work": 0.25},
    _NBEGM: {"premium": 3.0},
}
_N_SUBJECTS = 400
_SIMULATION_SEED = 1
_NON_FINITE_CLASSES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "nan": np.isnan,
    "posinf": np.isposinf,
    "neginf": np.isneginf,
}


def _case_id(case: tuple[str, str]) -> str:
    """Name a case as `<solver>-<arm>`."""
    return "-".join(case)


@functools.cache
def _solve(*, solver: str, arm: str, covered: bool) -> tuple[Any, Census]:
    """Solve one model under one arm; return the result and the planner census."""
    execution = ExecutionConfig(
        device_memory_bytes=_ARM_BUDGET[arm],
        width_search=WidthSearchPolicy(kind=WidthSearch.BOUNDED),
        covered_axes=(_AXIS[solver],) if covered else (),
    )
    model, params = _build(solver=solver, arm=arm, execution=execution, overrides={})
    return _solve_with_census(model=model, params=params)


def _solve_with_census(*, model: Model, params: UserParams) -> tuple[Any, Census]:
    """Solve `model`; return the result and the planner census of the solve."""
    recorder = CensusRecorder()
    with pytest.MonkeyPatch.context() as monkeypatch:
        recorder.install(monkeypatch=monkeypatch)
        result = model.solve(params=params, log_level="off")
    return result, recorder.census()


def _build(
    *,
    solver: str,
    arm: str,
    execution: ExecutionConfig,
    overrides: Mapping[str, float],
) -> tuple[Model, UserParams]:
    """Build one of the models under `execution` and its parameters."""
    if solver == _NBEGM:
        sizes: dict[str, Any] = (
            {}
            if arm == _DEFAULT_SIZE
            else {"n_periods": 2, "n_liquid": 8, "n_savings": 10, "n_consumption": 12}
        )
        model = nbegm_multi_discrete_toy.build_model(
            variant="nbegm",
            n_actions=3,
            envelope_arithmetic="ordinary",
            execution_config=execution,
            **sizes,
        )
        params = nbegm_multi_discrete_toy.build_params(n_actions=3, **overrides)
    else:
        model = regression.get_model(
            n_periods=3,
            wealth_grid=LinSpacedGrid(start=1, stop=400, n_points=12),
            consumption_grid=LinSpacedGrid(start=1, stop=400, n_points=10),
            execution_config=execution,
        )
        params = regression.get_params(n_periods=3, **overrides)
    return model, params


def _covered_and_uncovered(case: tuple[str, str]) -> tuple[Any, Any]:
    """Return the covered and the uncovered result of one case."""
    solver, arm = case
    covered, _ = _solve(solver=solver, arm=arm, covered=True)
    uncovered, _ = _solve(solver=solver, arm=arm, covered=False)
    return covered, uncovered


@functools.cache
def _simulate(
    *, solver: str, arm: str, covered: bool
) -> tuple[Model, Census, pd.DataFrame]:
    """Solve a simulation specimen and simulate it from fixed initial conditions.

    Return the model, the planner census of its solve and the simulated panel,
    with discrete variables as labelled categoricals.
    """
    execution = ExecutionConfig(
        device_memory_bytes=_ARM_BUDGET[arm],
        width_search=WidthSearchPolicy(kind=WidthSearch.BOUNDED),
        covered_axes=(_AXIS[solver],) if covered else (),
    )
    model, params = _build(
        solver=solver,
        arm=arm,
        execution=execution,
        overrides=_SIMULATION_PARAMS[solver],
    )
    result, census = _solve_with_census(model=model, params=params)
    simulation = model.simulate(
        params=params,
        initial_conditions=_initial_conditions(solver),
        solution=result,
        log_level="off",
        seed=_SIMULATION_SEED,
    )
    return model, census, simulation.to_dataframe(terminal_rows="all")


def _initial_conditions(solver: str) -> dict[str, jax.Array]:
    """Subjects spread over the model's wealth-like state, all alive at the start."""
    regime_id = jnp.zeros(_N_SUBJECTS, dtype=jnp.int32)
    if solver == _NBEGM:
        return {
            "liquid": jnp.linspace(0.0, 30.0, _N_SUBJECTS),
            "income": jnp.zeros(_N_SUBJECTS),
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": regime_id,
        }
    return {
        "wealth": jnp.linspace(1.0, 150.0, _N_SUBJECTS),
        "age": jnp.full(_N_SUBJECTS, float(regression.START_AGE)),
        "regime_id": regime_id,
    }


def _discrete_action_names(model: Model) -> tuple[str, ...]:
    """Names of every discrete action of any regime, sorted."""
    return tuple(
        sorted(
            {
                name
                for regime in model.user_regimes.values()
                for name, grid in regime.actions.items()
                if isinstance(grid, DiscreteGrid)
            }
        )
    )


def _discrete_columns(panel: pd.DataFrame) -> tuple[str, ...]:
    """Columns of a labelled panel holding a discrete state, action or regime."""
    return tuple(panel.select_dtypes(include="category").columns)


def _dispatched_widths(*, census: Census, axis: str) -> set[int]:
    """Distinct widths of `axis` across every dispatched core that declares it."""
    return {
        dict(core.tile_widths)[axis]
        for core in census.selected_cores.values()
        if axis in dict(core.tile_widths)
    }


def _published_arrays(result) -> dict[object, np.ndarray]:
    """Collect every array leaf the solve publishes, keyed by where it is published."""
    arrays: dict[object, np.ndarray] = {}
    for period, by_regime in result.values.items():
        for regime, value in by_regime.items():
            arrays[("value", period, regime)] = np.asarray(value)
    for channel, store in (
        ("replay", result.replay_artifacts),
        ("auxiliary", result.auxiliary_artifacts),
    ):
        for ref, artifact in store.items():
            for index, leaf in enumerate(jax.tree_util.tree_leaves(artifact)):
                arrays[(channel, ref, index)] = np.asarray(leaf)
    return arrays


def _float_arrays(result) -> dict[object, np.ndarray]:
    """The published arrays of a floating dtype."""
    return {
        key: array
        for key, array in _published_arrays(result).items()
        if np.issubdtype(array.dtype, np.floating)
    }


def _discrete_bytes(result) -> dict[object, bytes]:
    """Raw bytes of every published integer or boolean array."""
    return {
        key: array.tobytes()
        for key, array in _published_arrays(result).items()
        if not np.issubdtype(array.dtype, np.floating)
    }


def _bytes(result) -> dict[object, tuple[tuple[int, ...], np.dtype, bytes]]:
    """Shape, dtype and raw bytes of every published array, for bitwise comparison."""
    return {
        key: (array.shape, array.dtype, array.tobytes())
        for key, array in _published_arrays(result).items()
    }


def _shapes_and_dtypes(result) -> dict[object, tuple[tuple[int, ...], np.dtype]]:
    """Shape and dtype of every published array."""
    return {
        key: (array.shape, array.dtype)
        for key, array in _published_arrays(result).items()
    }


def _class_masks(*, result, non_finite: str) -> dict[object, bytes]:
    """Where each float array holds the `non_finite` class, as raw mask bytes."""
    predicate = _NON_FINITE_CLASSES[non_finite]
    return {
        key: predicate(array).tobytes() for key, array in _float_arrays(result).items()
    }


def _ulp_excess(*, covered, uncovered) -> dict[object, tuple[int, int]]:
    """Worst ULP distance and bound of every float array that exceeds its bound."""
    excess: dict[object, tuple[int, int]] = {}
    reference = _float_arrays(uncovered)
    for key, a in _float_arrays(covered).items():
        b = reference[key]
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            continue
        distances = np.testing.assert_array_max_ulp(
            a[mask], b[mask], maxulp=np.iinfo(np.int32).max
        )
        worst = int(np.max(distances))
        if worst > _MAX_ULP[a.dtype]:
            excess[key] = (worst, _MAX_ULP[a.dtype])
    return excess


@pytest.mark.parametrize("case", _SMALL_CASES, ids=_case_id)
def test_uncovered_solve_dispatches_the_power_of_two_seed(
    *, case: tuple[str, str]
) -> None:
    """Without covering, the 20-wide axis streams at the power-of-two seed 16."""
    solver, arm = case
    _, census = _solve(solver=solver, arm=arm, covered=False)

    assert _dispatched_widths(census=census, axis=_AXIS[solver]) == {_SEED}


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covered_solve_dispatches_the_full_extent(*, case: tuple[str, str]) -> None:
    """Covering the axis dispatches it at its full extent of 20 on every core."""
    solver, arm = case
    _, census = _solve(solver=solver, arm=arm, covered=True)

    assert _dispatched_widths(census=census, axis=_AXIS[solver]) == {_EXTENT}


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="Byte parity is the CPU-backend tier."
)
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covering_leaves_every_published_array_bitwise_unchanged_on_cpu(
    *, case: tuple[str, str]
) -> None:
    """On the CPU backend, values and artifacts agree byte for byte."""
    covered, uncovered = _covered_and_uncovered(case)

    assert _bytes(covered) == _bytes(uncovered)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covering_keeps_every_shape_and_dtype(*, case: tuple[str, str]) -> None:
    """Every published array keeps its shape and dtype."""
    covered, uncovered = _covered_and_uncovered(case)

    assert _shapes_and_dtypes(covered) == _shapes_and_dtypes(uncovered)


@pytest.mark.parametrize("non_finite", tuple(_NON_FINITE_CLASSES))
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covering_keeps_every_non_finite_entry_in_place(
    *, case: tuple[str, str], non_finite: str
) -> None:
    """NaN, +Inf and -Inf each occupy exactly the same entries of every float
    array."""
    covered, uncovered = _covered_and_uncovered(case)

    assert _class_masks(result=covered, non_finite=non_finite) == _class_masks(
        result=uncovered, non_finite=non_finite
    )


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covering_keeps_every_discrete_array_exact(*, case: tuple[str, str]) -> None:
    """Every published integer or boolean array is unchanged."""
    covered, uncovered = _covered_and_uncovered(case)

    assert _discrete_bytes(covered) == _discrete_bytes(uncovered)


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_covering_moves_no_finite_value_beyond_the_dtype_ulp_bound(
    *, case: tuple[str, str]
) -> None:
    """Every finite float agrees with the uncovered solve to 4 ULP in float64 and
    1 ULP in float32."""
    covered, uncovered = _covered_and_uncovered(case)

    assert _ulp_excess(covered=covered, uncovered=uncovered) == {}


@pytest.mark.parametrize("case", _SIMULATION_CASES, ids=_case_id)
def test_covering_keeps_simulated_discrete_choices_identical(
    *, case: tuple[str, str]
) -> None:
    """Simulating from a covered and an uncovered solve, with the same initial
    conditions and seed, yields the same label for every discrete action, discrete
    state and regime of every subject in every period."""
    solver, arm = case
    model, covered_census, covered = _simulate(solver=solver, arm=arm, covered=True)
    _, uncovered_census, uncovered = _simulate(solver=solver, arm=arm, covered=False)
    axis = _AXIS[solver]
    assert _dispatched_widths(census=covered_census, axis=axis) == {_EXTENT}
    assert _dispatched_widths(census=uncovered_census, axis=axis) == {_SEED}
    finite = covered[np.isfinite(covered["value"].to_numpy())]
    assert any(finite[name].nunique() >= 2 for name in _discrete_action_names(model)), (
        "no discrete action takes two distinct values among finite-value rows"
    )

    columns = sorted({*_discrete_columns(uncovered), *_discrete_action_names(model)})
    assert covered[columns].equals(uncovered[columns])
