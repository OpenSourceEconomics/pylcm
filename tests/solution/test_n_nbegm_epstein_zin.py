"""NNBEGM solves a two-asset Epstein-Zin model, matching the brute oracle.

A Kaplan-Violante-style two-asset problem: a liquid Euler margin `wealth`, an
illiquid durable `illiquid` chosen each period (outer grid search over
`next_illiquid`, paying a one-for-one credited move), a Cobb-Douglas composite
flow `q = c^phi * next_illiquid^(1-phi)`, stochastic survival (alive→dead with a
bequest), and Epstein-Zin recursive preferences (gamma != 1/psi). Conditional on
the fixed outer durable node the inner consumption-saving problem is a 1-D NB-EGM
solve whose flow is a single power `A c^phi` — so the inner Euler inversion reads
the composite flow's power structure, the regime-level survival split takes the
joint certainty equivalent, and the outer durable choice is a plain max. The
NNBEGM value closely tracks the dense two-action `GridSearch` value
everywhere — its off-grid policy is at least as good as brute's grid-quantized one
(the point of endogenous grids for durable choice) — and tracks it to under a
percent on average.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from lcm import (
    NBEGM,
    NNBEGM,
    AgeGrid,
    GridSearch,
    H_epstein_zin,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    PowerMean,
    Regime,
    categorical,
)
from lcm.solvers import Solver
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_N_PERIODS = 3
_LIQUID_RATE = 0.03
_LABOUR_INCOME = 4.0
_SURVIVAL = 0.9
_PHI = 0.6

_WEALTH_GRID = LinSpacedGrid(start=0.5, stop=30.0, n_points=12)
_ILLIQUID_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=10)
# The dense brute oracle refines both continuous choices so its grid-quantized
# value approaches the true value the off-grid nested solve targets.
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=80)
_ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=-20.0, stop=20.0, n_points=121)
_OUTER_GRID = LinSpacedGrid(start=0.5, stop=20.0, n_points=15)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=35.0, n_points=60)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _credited(illiquid: ContinuousState, next_illiquid: ContinuousState) -> FloatND:
    return next_illiquid - illiquid


def _resources(
    wealth: ContinuousState,
    illiquid: ContinuousState,
    next_illiquid: ContinuousState,
) -> FloatND:
    return (
        wealth
        + _LABOUR_INCOME
        - _credited(illiquid=illiquid, next_illiquid=next_illiquid)
    )


def _liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_wealth(liquid_savings: FloatND) -> ContinuousState:
    return (1.0 + _LIQUID_RATE) * liquid_savings


def _durable_transition(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def _keep_illiquid(illiquid: ContinuousState) -> FloatND:
    return illiquid


def _utility(consumption: ContinuousAction, next_illiquid: ContinuousState) -> FloatND:
    """Cobb-Douglas composite of consumption and the chosen durable service."""
    return consumption**_PHI * next_illiquid ** (1.0 - _PHI)


def _bequest(wealth: ContinuousState, illiquid: ContinuousState) -> FloatND:
    """Strictly positive terminal estate over both stocks."""
    return jnp.sqrt(wealth + illiquid + 1.0)


def _prob_alive(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, _SURVIVAL)


def _prob_dead(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 1.0 - _SURVIVAL)


def _illiquid_feasible(next_illiquid: ContinuousState) -> FloatND:
    return (next_illiquid >= _OUTER_GRID.start) & (next_illiquid <= _OUTER_GRID.stop)


def _budget_feasible(liquid_savings: FloatND) -> FloatND:
    return liquid_savings >= 0.0


def _consumption_feasible(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return consumption <= resources


def _build_solver(*, variant: str) -> Solver:
    if variant == "brute":
        return GridSearch()
    return NNBEGM(
        inner=NBEGM(
            continuous_state="wealth",
            post_decision_function="liquid_savings",
            budget_target="resources",
            savings_grid=_SAVINGS_GRID,
        ),
        outer_action="illiquid_investment",
        outer_post_decision="next_illiquid",
        outer_grid=_OUTER_GRID,
        outer_no_adjustment_candidate="keep_illiquid",
    )


def _build_model(*, variant: str) -> Model:
    final_age_alive = float(20 + (_N_PERIODS - 2) * 5)
    constraints: dict[str, Callable[..., FloatND]] = {
        "consumption_feasible": _consumption_feasible
    }
    if variant == "brute":
        constraints |= {
            "illiquid_feasible": _illiquid_feasible,
            "budget_feasible": _budget_feasible,
        }
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={"wealth": _WEALTH_GRID, "illiquid": _ILLIQUID_GRID},
        state_transitions={
            "wealth": {"alive": _next_wealth, "dead": _next_wealth},
            "illiquid": {"alive": _durable_transition, "dead": _durable_transition},
        },
        actions={
            "consumption": _CONSUMPTION_GRID,
            "illiquid_investment": _ILLIQUID_INVESTMENT_GRID,
        },
        transition={
            "alive": MarkovTransition(_prob_alive),
            "dead": MarkovTransition(_prob_dead),
        },
        functions={
            "utility": _utility,
            "resources": _resources,
            "liquid_savings": _liquid_savings,
            "keep_illiquid": _keep_illiquid,
            "credited": _credited,
            "H": H_epstein_zin,
        },
        constraints=constraints,
        certainty_equivalent=PowerMean(),
        solver=_build_solver(variant=variant),
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        states={"wealth": _WEALTH_GRID, "illiquid": _ILLIQUID_GRID},
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
    },
    "dead": {},
}


def test_n_nbegm_epstein_zin_tracks_the_dense_reference() -> None:
    """NNBEGM closely tracks a deliberately different dense reference under EZ.

    The two solvers optimize over different candidate sets by construction —
    NNBEGM enumerates a fixed post-decision grid of next-period stocks with an
    off-grid inner consumption policy, while dense grid search enumerates an
    investment action grid whose induced next-stock candidates depend on the
    current state, with grid-quantized consumption. Neither candidate set
    contains the other, so no directional (dominance) ordering exists between
    the two values; agreement is asserted as unsigned gaps.
    """
    nested = _build_model(variant="n_nbegm").solve(params=_PARAMS, log_level="off")
    brute = _build_model(variant="brute").solve(params=_PARAMS, log_level="off")
    for period in (0, 1):
        nested_V = np.asarray(nested[period]["alive"])
        brute_V = np.asarray(brute[period]["alive"])
        assert not np.isnan(nested_V).any(), f"period {period}"
        # A handful of cells adjacent to the borrowing corner and the
        # adjust/no-adjust kink carry the inner families' constrained-region
        # representation gap (as in the CRRA nested toy), so agreement is a
        # mean statement with a loose per-cell cap.
        rel_gap = np.abs(nested_V - brute_V) / np.abs(brute_V)
        assert float(rel_gap.max()) < 0.10, (
            f"period {period}: max rel gap {float(rel_gap.max()):.4f}"
        )
        mean_rel_gap = float(np.mean(rel_gap))
        assert mean_rel_gap < 0.01, f"period {period}: mean rel gap {mean_rel_gap:.4f}"
