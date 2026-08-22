"""NNBEGM applies the same case-piece smoothness gate as its inner NBEGM.

The nested solver runs NBEGM kernels on the inner margin, so a piece that hides
a branch breaks the inner Euler inversion exactly as it would under a bare
NBEGM. Building the model must reject it either way.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    categorical,
)
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


@case_boundary(
    boundary(
        variable="wealth", threshold="means_test", equality="otherwise", kind="jump"
    )
)
def eligible(wealth: ContinuousState, means_test: float) -> FloatND:
    return wealth < means_test


@piece(output="subsidy", when=eligible)
def subsidy_eligible(medical_expense: float) -> FloatND:
    return jnp.asarray(0.1 * medical_expense)


@piece(output="subsidy", otherwise=eligible)
def subsidy_private(medical_expense: float) -> FloatND:
    """Hide a branch behind a `jnp.where` the AST gate cannot see as Python."""
    return jnp.where(medical_expense > 0.0, 0.9 * medical_expense, 0.0)


def resources(wealth: ContinuousState, subsidy: FloatND) -> FloatND:
    return wealth + subsidy


def liquid_savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(liquid_savings: FloatND) -> ContinuousState:
    return 1.02 * liquid_savings


def new_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """The durable stock chosen this period."""
    return illiquid + illiquid_investment


def durable_transition(new_illiquid: ContinuousState) -> ContinuousState:
    return new_illiquid


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
        savings_grid=SAVINGS_GRID,
    )


def _build_model(*, solver: NBEGM | NNBEGM) -> Model:
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="liquid_savings",
    )
    active = lambda age: age <= 20  # noqa: E731
    states = {"wealth": WEALTH_GRID, "illiquid": ILLIQUID_GRID}
    state_transitions = {"wealth": next_wealth, "illiquid": durable_transition}
    actions = {
        "consumption": CONSUMPTION_GRID,
        "illiquid_investment": ILLIQUID_INVESTMENT_GRID,
    }
    functions = {
        "utility": utility,
        "new_illiquid": new_illiquid,
        "resources": resources,
        "liquid_savings": liquid_savings,
        "keep_illiquid": keep_illiquid,
        "eligible": eligible,
        "subsidy_eligible": subsidy_eligible,
        "subsidy_private": subsidy_private,
    }
    # Built per branch rather than through a shared class and margin mapping:
    # the two regime classes take different margins and narrow `solver`
    # differently, and neither distinction survives a `**kwargs` splat.
    if isinstance(solver, NNBEGM):
        alive = NestedConsumptionSavingsRegime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=next_regime,
            functions=functions,
            solver=solver,
            liquid=liquid,
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="illiquid_investment",
                post_decision_state="new_illiquid",
                no_adjustment="keep_illiquid",
            ),
        )
    else:
        alive = ConsumptionSavingsRegime(
            active=active,
            states=states,
            state_transitions=state_transitions,
            actions=actions,
            transition=next_regime,
            functions=functions,
            solver=solver,
            liquid=liquid,
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
        outer_grid=ILLIQUID_GRID,
    )
    with pytest.raises(NBEGMCaseError, match="smoothness gate"):
        _build_model(solver=solver)
