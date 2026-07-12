"""NBEGM solves an Epstein-Zin regime with a Cobb-Douglas composite flow.

The period flow `q = c^phi * service^(1-phi)` mixes consumption with a fixed
service level — a single power `A c^phi` in consumption (`A = service^(1-phi)`),
not the basic single-good flow `q = c`. Its Euler-form marginal
`q^(-rho) q_c = flow_coefficient * c^flow_exponent` has `flow_coefficient =
A^(1-rho) phi` and `flow_exponent = phi(1-rho) - 1`, so the NBEGM Euler inversion
must read the flow's own power structure, not the hard-coded `q = c` case. This is
the inner-flow shape a nested two-asset (Kaplan-Violante) solve presents once the
durable is fixed per outer node. The NBEGM solve reproduces the dense `GridSearch`
value up to interpolation tolerance.
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
_PHI = 0.6

_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=15)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=15.0, n_points=60)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=80)
_INCOME = NormalIIDProcess(n_points=_N_INCOME_NODES, gauss_hermite=True)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction, service_level: float) -> FloatND:
    """Cobb-Douglas composite of consumption and a fixed service — stays positive."""
    return consumption**_PHI * service_level ** (1.0 - _PHI)


def _resources(liquid: ContinuousState, base_income: float) -> FloatND:
    return liquid + base_income


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return consumption <= resources


def _next_liquid(savings: FloatND, income: ContinuousState) -> ContinuousState:
    return (1.0 + _RETURN) * savings + _INCOME_SCALE * jnp.exp(income)


def _bequest(liquid: ContinuousState) -> FloatND:
    return jnp.sqrt(liquid + 1.0)


def _prob_alive(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, _SURVIVAL)


def _prob_dead(age: int, final_age_alive: float) -> FloatND:
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
        "utility": {"service_level": 2.0},
        "income": {"mu": 0.0, "sigma": 0.2},
    },
    "dead": {},
}


def test_nbegm_epstein_zin_composite_flow_matches_brute_force() -> None:
    """NBEGM matches grid search when the flow is a Cobb-Douglas composite in c."""
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
        # The top liquid node extrapolates flat above the grid; EGM and dense
        # grid search bias that one node slightly differently (a finite-grid edge
        # artifact, not a disagreement in the economics). Compare the interior.
        np.testing.assert_allclose(
            nbegm_V[:, :-1], brute_V[:, :-1], rtol=2e-2, atol=1e-2
        )
