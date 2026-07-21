"""NBEGM solves a stochastic-survival Epstein-Zin model, matching the brute oracle.

Each period a living agent survives with probability `survival` and otherwise dies
and receives a bequest. Under Epstein-Zin recursive preferences the certainty
equivalent is taken over the JOINT lottery of the income shock and the
survive-versus-die regime split: `nu = (p_alive E[V_alive^{1-gamma}] + p_dead
V_dead^{1-gamma})^{1/(1-gamma)}`. A linear regime-probability blend of the two
continuations is a different (wrong) object, so this pins the regime-level joint
CE. The NBEGM solve reproduces the dense `GridSearch` value up to interpolation
tolerance.

Reference: Alan Lujan, "The Endogenous Grid Method for Epstein-Zin Preferences,"
arXiv:2601.04438 (2026).
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    NBEGM,
    AgeGrid,
    GridSearch,
    H_epstein_zin,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.solvers import Solver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_N_PERIODS = 3
_N_INCOME_NODES = 5
_INCOME_SCALE = 0.3
_RETURN = 0.03
_SURVIVAL = 0.9

_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=15)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=15.0, n_points=60)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=80)
_INCOME = NormalIIDProcess(n_points=_N_INCOME_NODES, gauss_hermite=True)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    """Consumption flow — positive units for the power certainty equivalent."""
    return consumption


def _resources(liquid: ContinuousState, base_income: float) -> FloatND:
    return liquid + base_income


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= resources


def _next_liquid(savings: FloatND, income: ContinuousState) -> ContinuousState:
    return (1.0 + _RETURN) * savings + _INCOME_SCALE * jnp.exp(income)


def _bequest(liquid: ContinuousState) -> FloatND:
    """Strictly positive terminal estate value the power CE requires."""
    return jnp.sqrt(liquid + 1.0)


def _prob_alive(age: int, final_age_alive: float) -> FloatND:
    """Survive into the next living period; zero once past the last living age."""
    return jnp.where(age >= final_age_alive, 0.0, _SURVIVAL)


def _prob_dead(age: int, final_age_alive: float) -> FloatND:
    """Die into the bequest regime; certain once past the last living age."""
    return jnp.where(age >= final_age_alive, 1.0, 1.0 - _SURVIVAL)


def _build_model(*, solver: Solver) -> Model:
    final_age_alive = float(20 + (_N_PERIODS - 2) * 5)
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"liquid": _LIQUID_GRID, "income": _INCOME},
        state_transitions={"liquid": {"alive": _next_liquid, "dead": _next_liquid}},
        actions={"consumption": _CONSUMPTION_GRID},
        transition={
            "alive": MarkovTransition(_prob_alive),
            "dead": MarkovTransition(_prob_dead),
        },
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        constraints={"feasible": _feasible},
        certainty_equivalent=PowerMean(),
        solver=solver,
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states={"liquid": _LIQUID_GRID},
        functions={"utility": _bequest},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=20, stop=20 + (_N_PERIODS - 1) * 5, step="5Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


_PARAMS = {
    "alive": {
        "H": {
            "discount_factor": 0.95,
            "intertemporal_elasticity_of_substitution": 1.5,
        },
        "certainty_equivalent": {"risk_aversion": 4.0},
        "resources": {"base_income": 1.0},
        "income": {"mu": 0.0, "sigma": 0.2},
    },
    "dead": {},
}


def test_nbegm_epstein_zin_multi_regime_matches_brute_force() -> None:
    """NBEGM matches the dense grid-search value under a stochastic survival split."""
    nbegm = _build_model(
        solver=NBEGM(
            post_decision_function="savings",
            budget_target="resources",
            savings_grid=_SAVINGS_GRID,
            continuous_state="liquid",
        )
    ).solve(params=_PARAMS, log_level="off")
    brute = _build_model(solver=GridSearch()).solve(params=_PARAMS, log_level="off")
    for period in (0, 1):
        nbegm_V = np.asarray(nbegm[period]["alive"])
        brute_V = np.asarray(brute[period]["alive"])
        assert not np.isnan(nbegm_V).any(), f"period {period}"
        np.testing.assert_allclose(nbegm_V, brute_V, rtol=2e-2, atol=1e-2)
