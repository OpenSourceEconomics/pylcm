"""Tests for simulate-AOT compilation via `Model.n_subjects`.

When `Model(n_subjects=N)` is set, the first matching `simulate(...)` call
parallel-compiles all simulate functions for batch shape `N`. Subsequent calls
with size `N` reuse the cache; calls with a mismatching size warn once per size
and fall back to the runtime-traced path.
"""

import logging

import jax.numpy as jnp
import jax.stages
import pytest

from lcm import Model
from lcm.ages import AgeGrid
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


def _build_initial_conditions(*, n_subjects: int) -> dict:
    """Subject array of size `n_subjects` matching the regression test model."""
    wealths = jnp.linspace(20.0, 320.0, num=n_subjects)
    return {
        "wealth": wealths,
        "age": jnp.full((n_subjects,), 18.0),
        "regime": jnp.array([RegimeId.working_life] * n_subjects),
    }


@pytest.mark.parametrize("invalid", [0, -3])
def test_n_subjects_validation_rejects_non_positive(invalid: int) -> None:
    with pytest.raises(ValueError, match="n_subjects"):
        _build_test_model(n_periods=3, n_subjects=invalid)


def test_n_subjects_validation_rejects_non_int() -> None:
    with pytest.raises(TypeError, match="n_subjects"):
        _build_test_model(n_periods=3, n_subjects=1.5)  # ty: ignore[invalid-argument-type]


def test_n_subjects_none_keeps_lazy_behavior() -> None:
    """Without n_subjects, simulate works and no AOT cache is populated."""
    n_periods = 3
    model = _build_test_model(n_periods=n_periods, n_subjects=None)
    params = get_params(n_periods=n_periods)

    result = model.simulate(
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions=_build_initial_conditions(n_subjects=4),
    )

    assert result.n_subjects == 4
    assert model.n_subjects is None
    assert not getattr(model, "_simulate_compile_cache", {})


def test_simulate_compiles_only_once_with_matching_n_subjects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First simulate call AOT-compiles; second call hits the cache."""
    n_periods = 3
    n_subjects = 4
    model = _build_test_model(n_periods=n_periods, n_subjects=n_subjects)
    params = get_params(n_periods=n_periods)
    period_to_regime_to_V_arr = model.solve(params=params)

    counter = {"count": 0}
    original_compile = jax.stages.Lowered.compile

    def counting_compile(self: jax.stages.Lowered, *args, **kwargs):
        counter["count"] += 1
        return original_compile(self, *args, **kwargs)

    monkeypatch.setattr(jax.stages.Lowered, "compile", counting_compile)

    initial_conditions = _build_initial_conditions(n_subjects=n_subjects)

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )
    n_first = counter["count"]
    counter["count"] = 0

    model.simulate(
        params=params,
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        initial_conditions=initial_conditions,
    )
    n_second = counter["count"]

    assert n_first > 0, "First simulate must trigger compilation."
    assert n_second == 0, "Second simulate must hit the AOT cache."
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


def test_simulate_caches_recompiled_size_no_second_warning(
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
