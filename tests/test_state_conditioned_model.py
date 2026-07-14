"""End-to-end: a minimal 2-regime-uncertainty model with state-conditioned income sigma.

The acceptance test for the DAG wiring: a precautionary-savings model whose IID income
shock has a `sigma` conditioned on a discrete `uncertainty` state must build and solve,
its value must depend on `uncertainty` iff the two regime sigmas differ, and the
unsupported families (Gauss-Hermite IID, Rouwenhorst) must be rejected at construction.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import StateConditioned
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> FloatND:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def next_uncertainty(uncertainty: DiscreteState) -> FloatND:
    """Absorbing: each uncertainty regime stays put."""
    return jnp.where(
        uncertainty == Uncertainty.low,
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def wealth_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _income_process(sigma_low: float, sigma_high: float, n_points: int = 5):
    grid_sigma = max(sigma_low, sigma_high)  # fixed common grid = widest regime
    return NormalIIDProcess(
        n_points=n_points,
        gauss_hermite=False,
        mu=0.0,
        sigma=grid_sigma,
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": sigma_low, "high": sigma_high}
        ),
    )


@functools.cache
def _get_model(sigma_low: float, sigma_high: float, n_periods: int = 5) -> Model:
    final_age_alive = 20 + (n_periods - 2) * 10
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": _income_process(sigma_low, sigma_high),
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 10, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def _params():
    return {
        "discount_factor": 0.95,
        "alive": {"next_wealth": {"interest_rate": 0.03}},
    }


def _solve(sigma_low, sigma_high):
    return _get_model(sigma_low, sigma_high).solve(
        params=_params(), log_level="warning"
    )


def _uncertainty_axis_maxdiff(V) -> float:
    """Max |V(...,uncertainty=low) - V(...,uncertainty=high)| over the alive-regime
    value leaves (the only state axis of size 2 is `uncertainty`)."""
    m = 0.0
    for leaf in jax.tree_util.tree_leaves(V):
        a = np.asarray(leaf)
        if a.ndim >= 1 and 2 in a.shape:
            ax = list(a.shape).index(2)
            m = max(m, float(np.abs(np.take(a, 0, ax) - np.take(a, 1, ax)).max()))
    return m


def test_conditioned_model_solves():
    """The milestone: the state-conditioned model builds and solves, V finite."""
    V = _solve(0.05, 0.30)
    assert V is not None
    for leaf in jax.tree_util.tree_leaves(V):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_equal_sigma_makes_uncertainty_irrelevant():
    """Degeneracy: if both regimes share sigma, the conditioning collapses and the value
    is *exactly* independent of the uncertainty state."""
    assert _uncertainty_axis_maxdiff(_solve(0.30, 0.30)) == 0.0


def test_higher_uncertainty_changes_value():
    """Conditioning is live: distinct per-regime sigmas make the value depend on the
    uncertainty state (the precautionary response flows through the transition CDF)."""
    assert _uncertainty_axis_maxdiff(_solve(0.02, 0.60)) > 1e-3


def _model_with_income(income_proc) -> Model:
    """Build the same alive/dead model with a swapped-in income process."""
    alive = Regime(
        active=lambda age: age <= 21,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": income_proc,
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None, active=lambda age: age > 21, functions={"utility": lambda: 0.0}
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=40, step="10Y"),
        fixed_params={"final_age_alive": 21},
    )


def test_gauss_hermite_state_conditioned_rejected():
    """GH + StateConditioned must raise: its nodes scale with sigma (audit F3)."""
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=True,
        mu=0.0,
        sigma=0.3,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="Gauss-Hermite"):
        _model_with_income(income).solve(params=_params())


def test_rouwenhorst_state_conditioned_rejected():
    """Rouwenhorst + StateConditioned must raise (rho-only transition, audit F2)."""
    income = RouwenhorstAR1Process(
        n_points=5,
        rho=0.9,
        sigma=0.3,
        mu=0.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="only supported for CDF-binned"):
        _model_with_income(income).solve(params=_params())
