"""Tests for simulate-AOT compilation via `Model.n_subjects`.

When `Model(n_subjects=N)` is set, the first matching `simulate(...)` call
parallel-compiles all simulate functions for batch shape `N`. Subsequent calls
with size `N` reuse the cache; calls with a mismatching size warn once per size
and fall back to the runtime-traced path. AOT works under both `x64=False`
and `x64=True` because integer leaves are normalised to `int32` at every
boundary by `lcm.params.processing` and the simulate state pool.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import cloudpickle
import jax.numpy as jnp
import jax.stages
import pytest
from jax import Array

from lcm import Model
from lcm.api.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from tests.test_models.deterministic.regression import (
    RegimeId,
    dead,
    get_params,
    working_life,
)


def _build_test_model(*, n_periods: int, n_subjects: int | None = None) -> Model:
    """Construct the small 2-regime regression model with optional n_subjects."""
    final_age_alive = 18 + n_periods - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=18, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        n_subjects=n_subjects,
    )


def _build_initial_conditions(*, n_subjects: int) -> dict[str, Array]:
    """Subject array of size `n_subjects` matching the regression test model."""
    wealths = jnp.linspace(20.0, 320.0, num=n_subjects)
    return {
        "wealth": wealths,
        "age": jnp.full((n_subjects,), 18.0),
        "regime_id": jnp.array([RegimeId.working_life] * n_subjects),
    }


@pytest.mark.parametrize("invalid", [0, -3])
def test_n_subjects_validation_rejects_non_positive(invalid: int) -> None:
    """`Model(n_subjects=0)` and negative values raise `ValueError`."""
    with pytest.raises(ValueError, match="n_subjects"):
        _build_test_model(n_periods=3, n_subjects=invalid)


def test_n_subjects_validation_rejects_non_int() -> None:
    """`Model(n_subjects=1.5)` raises `ModelInitializationError`."""
    with pytest.raises(ModelInitializationError, match="n_subjects"):
        _build_test_model(n_periods=3, n_subjects=1.5)  # ty: ignore[invalid-argument-type]


def test_n_subjects_none_leaves_aot_cache_empty_after_simulate() -> None:
    """`Model(n_subjects=None)` keeps `_simulate_compile_cache` empty after simulate."""
    n_periods = 3
    model = _build_test_model(n_periods=n_periods, n_subjects=None)
    params = get_params(n_periods=n_periods)

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=_build_initial_conditions(n_subjects=4),
    )

    assert dict(model._simulate_compile_cache) == {}


def test_n_subjects_none_yields_simulate_result_sized_to_actual() -> None:
    """`Model(n_subjects=None).simulate(...)` returns a result sized to the input."""
    n_periods = 3
    model = _build_test_model(n_periods=n_periods, n_subjects=None)
    params = get_params(n_periods=n_periods)

    result = model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=_build_initial_conditions(n_subjects=4),
    )

    assert result.n_subjects == 4


def test_simulate_second_matching_call_does_not_invoke_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching second `simulate(...)` invokes `Lowered.compile` zero times."""
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)

    counter = {"count": 0}
    original_compile = jax.stages.Lowered.compile

    def counting_compile(
        self: jax.stages.Lowered, *args: Any, **kwargs: Any
    ) -> jax.stages.Compiled:
        counter["count"] += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)

    initial_conditions = _build_initial_conditions(n_subjects=n_subjects)

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )
    counter["count"] = 0

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )

    assert counter["count"] == 0


def test_simulate_first_matching_call_populates_aot_cache() -> None:
    """Matching first `simulate(...)` populates the cache for that size."""
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)

    assert n_subjects not in model._simulate_compile_cache

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=_build_initial_conditions(n_subjects=n_subjects),
    )

    assert n_subjects in model._simulate_compile_cache


def test_solve_does_not_populate_simulate_compile_cache() -> None:
    """`solve(...)` does not touch simulate-side compile state.

    Simulate AOT compilation is driven entirely by `simulate(...)`; calling
    `solve(...)` alone leaves `_simulate_compile_cache` empty.
    """
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)

    model.solve(params=params)

    assert dict(model._simulate_compile_cache) == {}


_DECLARED_N = 4
_ACTUAL_N = 7


@dataclass(frozen=True)
class _MismatchOutcome:
    """Captured simulate-with-mismatch artefacts for assertion."""

    warnings: list[logging.LogRecord]
    model: Model


@pytest.fixture(name="mismatch_outcome")
def _mismatch_outcome(
    caplog: pytest.LogCaptureFixture,
) -> _MismatchOutcome:
    """Run one mismatching `simulate(...)` and capture the WARNING records."""
    n_periods = 3
    model = _build_test_model(n_periods=n_periods, n_subjects=_DECLARED_N)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.simulate(
            params=params,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            initial_conditions=_build_initial_conditions(n_subjects=_ACTUAL_N),
        )

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "n_subjects" in r.getMessage()
    ]
    return _MismatchOutcome(warnings=warnings, model=model)


def test_simulate_mismatch_emits_one_warning(
    mismatch_outcome: _MismatchOutcome,
) -> None:
    """A single mismatching call logs exactly one WARNING."""
    assert len(mismatch_outcome.warnings) == 1


def test_simulate_mismatch_warning_names_declared_n(
    mismatch_outcome: _MismatchOutcome,
) -> None:
    """The mismatch warning message contains the declared `n_subjects`."""
    msg = mismatch_outcome.warnings[0].getMessage()
    assert str(_DECLARED_N) in msg


def test_simulate_mismatch_warning_names_actual_n(
    mismatch_outcome: _MismatchOutcome,
) -> None:
    """The mismatch warning message contains the actual `n_subjects`."""
    msg = mismatch_outcome.warnings[0].getMessage()
    assert str(_ACTUAL_N) in msg


def test_simulate_mismatch_does_not_populate_cache(
    mismatch_outcome: _MismatchOutcome,
) -> None:
    """A mismatching `n_subjects` falls back to the lazy path — no cache entry."""
    assert _ACTUAL_N not in mismatch_outcome.model._simulate_compile_cache


def test_simulate_warns_only_once_per_mismatching_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two calls with the same mismatching size produce only one WARNING."""
    n_periods = 3
    model = _build_test_model(n_periods=n_periods, n_subjects=_DECLARED_N)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)
    initial_conditions = _build_initial_conditions(n_subjects=_ACTUAL_N)

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.simulate(
            params=params,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            initial_conditions=initial_conditions,
        )
        model.simulate(
            params=params,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            initial_conditions=initial_conditions,
        )

    mismatch_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "n_subjects" in r.getMessage()
    ]
    assert len(mismatch_warnings) == 1


def test_simulate_result_pickles_when_n_subjects_matches() -> None:
    """`simulate(...)` returns a result that round-trips through cloudpickle.

    With `n_subjects` matching the batch shape, the simulate path runs
    AOT-compiled callables that wrap `LoadedExecutable` (unpicklable).
    `to_dataframe` doesn't need those callables, so the returned result
    must carry the lazy regimes — otherwise downstream pickling
    (e.g. pytask handing the result to the next task) fails.
    """
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)

    result = model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=_build_initial_conditions(n_subjects=n_subjects),
    )

    restored = cloudpickle.loads(cloudpickle.dumps(result))
    assert restored.n_subjects == n_subjects


def test_unpickled_model_can_simulate_with_aot() -> None:
    """A cloudpickle round-tripped `Model` still drives `simulate(...)` with AOT."""
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)
    initial_conditions = _build_initial_conditions(n_subjects=n_subjects)

    # Populate the AOT cache before pickling — confirms __getstate__ drops it.
    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )
    assert n_subjects in model._simulate_compile_cache

    restored = cloudpickle.loads(cloudpickle.dumps(model))

    # The restored Model starts with empty AOT state and a fresh lock.
    assert dict(restored._simulate_compile_cache) == {}
    assert restored._warned_n_subjects == set()
    assert isinstance(restored._simulate_compile_lock, type(threading.Lock()))

    # Simulate works post-unpickle and re-populates the cache for that size.
    restored.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )
    assert n_subjects in restored._simulate_compile_cache
