"""Halving the GridSearch cell width keeps every published array and simulated choice.

A controlled compiled-program check replaces the HLO classifier, so planning,
compilation, the solve and the replay all run for real on any backend, while
the halving decision is forced. Each arm is compared with a solve that has
halving disabled:

- a non-firing arm publishes byte-identical arrays;
- a firing arm publishes identical non-float arrays and floats that agree
  element-locally to eight units in the last place.
"""

import functools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.solution import backward_induction
from lcm import ExecutionConfig, Model
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_collective_cell_ceiling import _published_arrays
from tests.solution.test_grid_search_cell_axis import (
    _model as _fixture_model,
)
from tests.solution.test_grid_search_cell_axis import (
    _params,
    _RegimeId,
)

_KINDS = ("singleton", "ev1", "collective")
_BUDGET = 10**8
# (kind, device-memory budget) of every firing comparison.
_FIRING_CASES = [*((kind, None) for kind in _KINDS), ("collective", _BUDGET)]


def _model(*, kind: str, enabled: bool, budget: int | None) -> Model:
    """The two-state fixture without its cell-width pin, which would bypass halving."""
    base = _fixture_model(kind=kind, width=6)
    return Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=_RegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=ExecutionConfig(
            halve_on_materialised_gather=enabled, device_memory_bytes=budget
        ),
    )


# keyword-only-exempt: library-callback=_group_cores_by_regime_period
def _capture_grouping(
    cores_by_triple: Any, *, original: Any, sink: dict[Any, int]
) -> Any:
    for triple, core in cores_by_triple.items():
        if "cell" in core.tile_widths:
            sink[triple] = core.tile_widths["cell"]
    return original(cores_by_triple)


def _materialised_while_wider_than_one(*, compiled: object, widths: Any) -> str | None:
    del compiled
    return "controlled_materialised_gather" if widths.get("cell", 1) > 1 else None


def _never_materialised(*, compiled: object, widths: Any) -> str | None:
    del compiled, widths


@functools.cache
def _solve(
    *, kind: str, enabled: bool, firing: bool, budget: int | None
) -> tuple[Model, Any, dict[Any, int]]:
    """Solve one arm; return its model, solution and dispatched cell widths."""
    model = _model(kind=kind, enabled=enabled, budget=budget)
    widths: dict[Any, int] = {}
    check = _materialised_while_wider_than_one if firing else _never_materialised
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            backward_induction,
            "_group_cores_by_regime_period",
            functools.partial(
                _capture_grouping,
                original=backward_induction._group_cores_by_regime_period,
                sink=widths,
            ),
        )
        patcher.setattr(backward_induction, "_materialised_gather_fusion", check)
        solution = model.solve(params=_params(kind), log_level="off")
    return model, solution, widths


def _reference(*, kind: str, budget: int | None = None) -> tuple[Model, Any, Any]:
    return _solve(kind=kind, enabled=False, firing=False, budget=budget)


def _candidate(
    *, kind: str, firing: bool, budget: int | None = None
) -> tuple[Model, Any, Any]:
    return _solve(kind=kind, enabled=True, firing=firing, budget=budget)


def _publication_mismatches(*, got: Any, expected: Any, firing: bool) -> list[str]:
    """Name every published array that breaks the arm's comparison contract."""
    actual, reference = _published_arrays(got), _published_arrays(expected)
    if not reference or actual.keys() != reference.keys():
        return [
            f"inventory: {sorted(map(str, actual))} vs {sorted(map(str, reference))}"
        ]
    mismatches = []
    for key, exp in reference.items():
        obs = actual[key]
        if (obs.shape, obs.dtype) != (exp.shape, exp.dtype):
            mismatches.append(
                f"{key}: {obs.shape} {obs.dtype} vs {exp.shape} {exp.dtype}"
            )
        elif not firing or exp.dtype.kind != "f":
            if obs.tobytes(order="C") != exp.tobytes(order="C"):
                mismatches.append(f"{key}: bytes differ")
        else:
            try:
                assert_agrees_to_ulp(got=obs, expected=exp, n_ulp=8, err_msg=str(key))
            except AssertionError as error:
                mismatches.append(f"{key}: {error}")
    return mismatches


def _choices(*, model: Model, solution: Any) -> pd.DataFrame:
    """Simulate every state of the six-cell grid in the acting regime."""
    first, second = np.meshgrid(np.array([1.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    n_subjects = first.size
    return model.simulate(
        params=_params("singleton"),
        solution=solution,
        initial_conditions={
            "first": jnp.asarray(first.ravel()),
            "second": jnp.asarray(second.ravel()),
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(n_subjects, _RegimeId.acting, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()


@pytest.mark.parametrize("kind", _KINDS)
def test_reference_solve_dispatches_a_cell_width_above_one(*, kind: str) -> None:
    """The unpinned fixture tiles cells wider than one, so halving has room to act."""
    _, _, widths = _reference(kind=kind)
    assert max(widths.values(), default=0) > 1


@pytest.mark.parametrize("kind", _KINDS)
def test_halving_keeps_every_cell_width_when_nothing_materialises(*, kind: str) -> None:
    _, _, expected = _reference(kind=kind)
    _, _, got = _candidate(kind=kind, firing=False)
    assert got == expected


@pytest.mark.parametrize(("kind", "budget"), _FIRING_CASES)
def test_halving_narrows_every_cell_width_to_one_while_materialised(
    *, kind: str, budget: int | None
) -> None:
    _, _, expected = _reference(kind=kind, budget=budget)
    _, _, got = _candidate(kind=kind, firing=True, budget=budget)
    assert got == dict.fromkeys(expected, 1)


@pytest.mark.parametrize("kind", _KINDS)
def test_non_firing_halving_publishes_byte_identical_arrays(*, kind: str) -> None:
    _, expected, _ = _reference(kind=kind)
    _, got, _ = _candidate(kind=kind, firing=False)
    assert _publication_mismatches(got=got, expected=expected, firing=False) == []


@pytest.mark.parametrize(("kind", "budget"), _FIRING_CASES)
def test_firing_halving_publishes_equal_arrays_to_eight_ulp(
    *, kind: str, budget: int | None
) -> None:
    """Non-float arrays are identical; floats agree element-locally to eight ULP."""
    _, expected, _ = _reference(kind=kind, budget=budget)
    _, got, _ = _candidate(kind=kind, firing=True, budget=budget)
    assert _publication_mismatches(got=got, expected=expected, firing=True) == []


def test_collective_fixture_publishes_both_dissolution_outcomes() -> None:
    """The flag comparison is nonvacuous: both flag values are published."""
    _, solution, _ = _reference(kind="collective")
    flags = np.concatenate(
        [
            array.ravel()
            for array in _published_arrays(solution).values()
            if array.dtype.kind == "b"
        ]
        or [np.zeros(0, dtype=bool)]
    )
    assert set(flags.tolist()) == {False, True}


def test_singleton_reference_simulation_chooses_to_work() -> None:
    """The fixture's discrete optimum is separated, so the choice is deterministic."""
    model, solution, _ = _reference(kind="singleton")
    choices = _choices(model=model, solution=solution)
    assert set(choices["work"].dropna().astype(str).unique()) == {"on"}


@pytest.mark.parametrize("firing", [False, True])
@pytest.mark.parametrize("column", ["regime_name", "work"])
def test_halving_keeps_simulated_singleton_choices(
    *, firing: bool, column: str
) -> None:
    reference_model, expected, _ = _reference(kind="singleton")
    candidate_model, got, _ = _candidate(kind="singleton", firing=firing)
    pd.testing.assert_series_equal(
        _choices(model=candidate_model, solution=got)[column],
        _choices(model=reference_model, solution=expected)[column],
    )
