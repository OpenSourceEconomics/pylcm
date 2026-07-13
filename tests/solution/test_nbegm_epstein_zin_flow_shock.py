"""NBEGM matches the brute oracle when an IID shock scales the flow under EZ.

An IID node that multiplies the period flow but never enters the budget
(`resources` reads only the liquid state) leaves the child's carry rows on the
shared liquid grid with savings-independent weights — the configuration where a
linear-expected-utility read may fold the node's rows into their weighted
average before interpolating. Under a nonlinear certainty equivalent that fold
is not available: the power mean must transform every node's value before the
lottery sum (`E[g(V)] != g(E[V])`), so the NBEGM read keeps the node axis and
transforms per node. With the flow at half the aggregator weight and a
unit-variance shock, folding raw rows would misstate the value by far more
than the interpolation tolerance asserted here.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    NBEGM,
    AgeGrid,
    GridSearch,
    H_epstein_zin,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.solvers import Solver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_N_PERIODS = 3
_RETURN = 0.03
_BASE_INCOME = 1.0

_LIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=15)
# The oracle's action grid contains every liquid point's exact cash-on-hand:
# at a binding borrowing constraint the optimum is the corner `c = coh`, where
# the value is first-order in the action-grid gap — a dense grid alone leaves
# the brute reference visibly below the true corner value.
_CONSUMPTION_GRID = IrregSpacedGrid(
    points=tuple(
        np.unique(
            np.concatenate(
                [
                    np.linspace(0.1, 21.0, 160),
                    np.linspace(0.5, 20.0, 15) + _BASE_INCOME,
                ]
            )
        ).tolist()
    )
)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=18.0, n_points=80)
_HEALTH = NormalIIDProcess(n_points=5, gauss_hermite=True)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption: ContinuousAction, health: ContinuousState) -> FloatND:
    """Flow scaled by the IID node — positive, and single-power in consumption."""
    return consumption * jnp.exp(health)


def _resources(liquid: ContinuousState) -> FloatND:
    return liquid + _BASE_INCOME


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _feasible(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return consumption <= resources


def _next_liquid(savings: FloatND) -> ContinuousState:
    return (1.0 + _RETURN) * savings


def _terminal_value(liquid: ContinuousState) -> FloatND:
    return jnp.sqrt(liquid + 1.0)


def _next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)


def _build_model(*, solver: Solver) -> Model:
    final_age_alive = float(20 + (_N_PERIODS - 2) * 5)
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"liquid": _LIQUID_GRID, "health": _HEALTH},
        state_transitions={"liquid": _next_liquid},
        actions={"consumption": _CONSUMPTION_GRID},
        transition=_next_regime,
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
        functions={"utility": _terminal_value},
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
            "discount_factor": 0.5,
            "intertemporal_elasticity_of_substitution": 1.5,
        },
        "certainty_equivalent": {"risk_aversion": 8.0},
        "health": {"mu": 0.0, "sigma": 1.0},
    },
    "dead": {},
}


def test_nbegm_epstein_zin_flow_shock_matches_brute_force() -> None:
    """The joint CE over a flow-scaling IID node matches the dense grid search."""
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
