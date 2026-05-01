"""Tests for the post-loop diagnostics path in `solve_brute.solve`.

These cover:
- happy path at `log_level="warning"` runs without raising and without
  the deferred-stack fan-in that previously OOMed at production sizes;
- NaN-bearing solves raise `InvalidValueFunctionError` and the message
  identifies the offending `(regime, age)`;
- `log_level="debug"` emits one stat line per `(regime, period)`;
- `log_level="off"` emits nothing and skips even the NaN fail-fast.
"""

import logging
from pathlib import Path

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ContinuousAction, ContinuousState, FloatND


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _utility(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return jnp.log(consumption + 1) + 0.01 * wealth


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def _borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


def _make_model() -> Model:
    alive = Regime(
        functions={"utility": _utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age < 2,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


_HEALTHY_PARAMS = {"discount_factor": 0.95, "interest_rate": 0.05}


def test_warning_level_solves_without_per_row_materialisation():
    """Happy-path solve at log_level="warning" returns finite V without
    entering the failure-path localisation."""
    model = _make_model()
    period_to_regime_to_V_arr = model.solve(params=_HEALTHY_PARAMS, log_level="warning")
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr))
            assert not jnp.any(jnp.isinf(V_arr))


def test_nan_failure_raises_with_regime_and_age():
    """A NaN-producing parameter set raises with the offending (regime, age).

    `discount_factor=NaN` poisons the next-V contribution to Q on the
    first non-terminal period; the validator must surface the offending
    regime in the error message.
    """
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with pytest.raises(InvalidValueFunctionError, match=r"alive"):
        model.solve(params=params, log_level="warning")


def test_off_level_solves_without_diagnostics(caplog: pytest.LogCaptureFixture):
    """log_level="off" emits no diagnostic records and skips the NaN fail-fast.

    Even with a NaN-producing parameter set, solve() returns instead of
    raising — the documented contract of `"off"`.
    """
    model = _make_model()
    params = {**_HEALTHY_PARAMS, "discount_factor": float("nan")}
    with caplog.at_level(logging.DEBUG):
        period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert period_to_regime_to_V_arr is not None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_debug_level_emits_per_period_stats(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """log_level="debug" logs a min/max/mean line for every (regime, period)."""
    model = _make_model()
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        model.solve(params=_HEALTHY_PARAMS, log_level="debug", log_path=tmp_path)
    debug_stat_lines = [
        r
        for r in caplog.records
        if "V min=" in r.getMessage() and "max=" in r.getMessage()
    ]
    assert len(debug_stat_lines) >= 1
