"""Tests for runtime-supplied stochastic-process parameters."""

from typing import Any

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    TauchenAR1Process,
    UniformIIDProcess,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(
    wealth: ContinuousState,
    income: ContinuousState,
    consumption: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) + 0.01 * (wealth + jnp.exp(income))


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    return wealth - consumption


def _constraint(consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return consumption <= wealth


_TAUCHEN_PARAMS: dict[str, Any] = {"rho": 0.9, "sigma": 1.0, "mu": 0.0, "n_std": 2}


def _make_model(*, fixed_params=None):
    """Create a process model with all process params supplied at runtime."""
    alive = UserRegime(
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
            "income": TauchenAR1Process(n_points=3, gauss_hermite=False),
        },
        state_transitions={
            "wealth": _next_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4)},
        functions={"utility": _utility},
        constraints={"borrowing": _constraint},
        transition=lambda period: jnp.where(period >= 1, RegimeId.dead, RegimeId.alive),
        active=lambda age: age < 2,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 2,
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params or {},
    )


def test_runtime_process_params_property():
    """Tauchen without params reports all params as runtime-supplied."""
    grid = TauchenAR1Process(n_points=5, gauss_hermite=False)
    for name in ("rho", "sigma", "mu", "n_std"):
        assert name in grid.params_to_pass_at_runtime
    assert not grid.is_fully_specified


def test_fully_specified_process():
    """Tauchen with all params should have no runtime-supplied params."""
    grid = TauchenAR1Process(
        n_points=5,
        gauss_hermite=False,
        batch_size=0,
        distributed=False,
        **_TAUCHEN_PARAMS,
    )
    assert grid.params_to_pass_at_runtime == ()
    assert grid.is_fully_specified


@pytest.mark.parametrize(
    ("grid_cls", "extra_kw"),
    [
        (UniformIIDProcess, {}),
        (NormalIIDProcess, {"gauss_hermite": True}),
    ],
)
def test_process_without_params_is_not_fully_specified(grid_cls, extra_kw):
    """All distributions require explicit params — nothing is defaulted."""
    grid = grid_cls(n_points=5, **extra_kw)
    assert not grid.is_fully_specified
    assert grid.params_to_pass_at_runtime


def test_runtime_process_in_params_template():
    """Runtime-supplied process params appear in params_template."""
    model = _make_model()
    alive_template = model._params_template["alive"]
    assert "income" in alive_template
    for name in ("rho", "sigma", "mu", "n_std"):
        assert name in alive_template["income"]


def test_solve_with_runtime_process():
    """Solve should work with runtime-supplied process params."""
    model = _make_model()
    params = {"discount_factor": 1.0, **_TAUCHEN_PARAMS}
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0


def test_runtime_process_with_fixed_params():
    """Process params provided via fixed_params are removed from template."""
    model = _make_model(fixed_params=_TAUCHEN_PARAMS)

    alive_template = model._params_template.get("alive", {})
    income_params = alive_template.get("income", {})
    for name in _TAUCHEN_PARAMS:
        assert name not in income_params

    params = {"discount_factor": 1.0}
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    assert len(period_to_regime_to_V_arr) > 0
