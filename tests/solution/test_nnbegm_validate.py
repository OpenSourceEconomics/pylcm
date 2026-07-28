"""NNBEGM applies the same case-piece smoothness gate as its inner NBEGM.

The nested solver runs NBEGM kernels on the inner margin, so a piece that hides
a branch breaks the inner Euler inversion exactly as it would under a bare
NBEGM. Building the model must reject it either way.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.case_piece import boundary, case_boundary, piece
from lcm.exceptions import NBEGMCaseError
from lcm.regime import Regime
from lcm.solvers import NBEGM, NNBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=20.0, n_points=8)
ILLIQUID_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=3)
SAVINGS_GRID = LinSpacedGrid(start=0.05, stop=19.0, n_points=8)
CONSUMPTION_GRID = LinSpacedGrid(start=0.05, stop=19.0, n_points=8)
ILLIQUID_INVESTMENT_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=3)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@case_boundary(boundary("wealth", "means_test", equality="otherwise", kind="jump"))
def eligible(wealth: ContinuousState, means_test: float) -> FloatND:
    return wealth < means_test


@piece("subsidy", when=eligible)
def subsidy_eligible(medical_expense: float) -> FloatND:
    return jnp.asarray(0.1 * medical_expense)


@piece("subsidy", otherwise=eligible)
def subsidy_private(medical_expense: float) -> FloatND:
    """Hide a branch behind a `jnp.where` the AST gate cannot see as Python."""
    return jnp.where(medical_expense > 0.0, 0.9 * medical_expense, 0.0)


def resources(wealth: ContinuousState, subsidy: FloatND) -> FloatND:
    return wealth + subsidy


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    return 1.02 * liquid_savings


def durable_transition(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def keep_illiquid(illiquid: ContinuousState) -> FloatND:
    return illiquid


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState, illiquid: ContinuousState) -> FloatND:
    return jnp.log(wealth + illiquid + 1.0)


def next_regime(age: int) -> ScalarInt:
    return jnp.where(age < 25, RegimeId.alive, RegimeId.dead)


def _inner_nbegm() -> NBEGM:
    return NBEGM(
        continuous_state="wealth",
        post_decision_function="liquid_savings",
        budget_target="resources",
        savings_grid=SAVINGS_GRID,
    )


def _build_model(*, solver):
    alive = Regime(
        active=lambda age: age <= 20,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        state_transitions={"wealth": next_wealth, "illiquid": durable_transition},
        actions={
            "consumption": CONSUMPTION_GRID,
            "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
        },
        transition=next_regime,
        functions={
            "utility": utility,
            "resources": resources,
            "liquid_savings": liquid_savings,
            "keep_illiquid": keep_illiquid,
            "eligible": eligible,
            "subsidy_eligible": subsidy_eligible,
            "subsidy_private": subsidy_private,
        },
        solver=solver,
    )
    dead = Regime(
        transition=None,
        active=lambda age: age > 20,
        states={"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID},
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=25, step="5Y"),
        fixed_params={"means_test": 5.0, "medical_expense": 1.0},
    )


def test_nbegm_rejects_a_piece_that_hides_a_branch():
    """A `jnp.where` inside a smooth piece fails NBEGM's smoothness gate."""
    with pytest.raises(NBEGMCaseError, match="smoothness gate"):
        _build_model(solver=_inner_nbegm())


def test_nnbegm_rejects_a_piece_that_hides_a_branch():
    """The nested solver applies its inner NBEGM's smoothness gate too."""
    solver = NNBEGM(
        inner=_inner_nbegm(),
        outer_action="illiquid_investment",
        outer_post_decision="next_illiquid",
        outer_grid=ILLIQUID_GRID,
        outer_no_adjustment_candidate="keep_illiquid",
    )
    with pytest.raises(NBEGMCaseError, match="smoothness gate"):
        _build_model(solver=solver)
