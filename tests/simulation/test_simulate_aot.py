"""Tests for simulate-AOT compilation via `Model.n_subjects`.

When `Model(n_subjects=N)` is set, the first matching `simulate(...)` call
parallel-compiles all simulate functions for batch shape `N`. Subsequent calls
with size `N` reuse the cache; calls with a mismatching size warn once per size
and fall back to the runtime-traced path.

The AOT path pins integer dtypes to int32, which is incompatible with
`jax_enable_x64=True`. The conftest enables x64 by default, so every test in
this file disables it via the autouse `_disable_x64` fixture.
"""

import logging
from collections.abc import Iterator
from typing import Any

import jax.numpy as jnp
import jax.stages
import pytest
from jax import Array
from jax import config as jax_config

from lcm import Model
from lcm.ages import AgeGrid
from tests.test_models.deterministic.regression import (
    RegimeId,
    dead,
    get_params,
    working_life,
)


@pytest.fixture(autouse=True)
def _disable_x64() -> Iterator[None]:
    """Disable x64 for AOT tests (conftest enables it globally)."""
    previous = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", val=False)
    try:
        yield
    finally:
        jax_config.update("jax_enable_x64", val=previous)


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
        "regime": jnp.array([RegimeId.working_life] * n_subjects),
    }


@pytest.mark.parametrize("invalid", [0, -3])
def test_n_subjects_validation_rejects_non_positive(invalid: int) -> None:
    """`Model(n_subjects=0)` and negative values raise `ValueError`."""
    with pytest.raises(ValueError, match="n_subjects"):
        _build_test_model(n_periods=3, n_subjects=invalid)


def test_n_subjects_validation_rejects_non_int() -> None:
    """`Model(n_subjects=1.5)` raises `TypeError`."""
    with pytest.raises(TypeError, match="n_subjects"):
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


def test_simulate_warns_on_n_subjects_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mismatching size logs WARNING naming both N and M, falls back to lazy path."""
    n_periods = 3
    declared_n = 4
    actual_n = 7
    model = _build_test_model(n_periods=n_periods, n_subjects=declared_n)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.simulate(
            params=params,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            initial_conditions=_build_initial_conditions(n_subjects=actual_n),
        )

    mismatch_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "n_subjects" in r.getMessage()
    ]
    assert len(mismatch_warnings) == 1
    msg = mismatch_warnings[0].getMessage()
    assert str(declared_n) in msg
    assert str(actual_n) in msg
    # Cache is NOT populated for mismatching size — fallback path was taken.
    assert actual_n not in model._simulate_compile_cache


def test_simulate_warns_only_once_per_mismatching_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two calls with the same mismatching size produce only one WARNING."""
    n_periods = 3
    declared_n = 4
    actual_n = 7
    model = _build_test_model(n_periods=n_periods, n_subjects=declared_n)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)
    initial_conditions = _build_initial_conditions(n_subjects=actual_n)

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
