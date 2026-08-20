"""A model is refused when a constraint cannot be met on the route it lands on.

Which constraints a regime can state is a property of the pipeline its solver
walks, not of the constraint. A grid search has whole candidates in hand, so it
can evaluate anything; DC-EGM builds its feasibility predicate once per
discrete combination, before the inversion produces the continuous action, so a
constraint reading that action cannot be met there. The same declaration is
therefore accepted in one regime and refused in another, and the refusal names
the route it could not be met on.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    post_decision_lower_bound,
    ref,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import ConsumptionSavingsRegime, LiquidMargin, Regime
from lcm.solvers import DCEGM, GridSearch, OneMarginSolver
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)

_LIQUID = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)
_WEALTH_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_ACTION_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class Work:
    working: ScalarInt
    retired: ScalarInt


def utility(consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) + 0.0 * work


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(jnp.clip(wealth, 1e-8))


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(savings: FloatND) -> FloatND:
    return savings


def next_regime(_age: float) -> ScalarInt:
    return RegimeId.done


def spends_within_reason(consumption: ContinuousAction) -> FloatND:
    """Reads the continuous action, which the inversion produces."""
    return consumption <= 3.0


def works_or_retires(work: DiscreteAction) -> FloatND:
    """Reads a discrete action, which is bound per combination."""
    return work >= 0


def _model(*, solver: OneMarginSolver | GridSearch, constraint: UserFunction) -> Model:
    """Build a one-period model whose saving regime declares `constraint`."""
    saving_regime = ConsumptionSavingsRegime(
        actions={"consumption": _ACTION_GRID, "work": DiscreteGrid(Work)},
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"done": next_wealth}},
        constraints={"declared": constraint},
        transition=next_regime,
        functions={"utility": utility, "resources": resources, "savings": savings},
        active=lambda age: age == 0,
        solver=solver,
        liquid=_LIQUID,
    )
    done_regime = Regime(
        actions={},
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving_regime, "done": done_regime},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def _dcegm() -> DCEGM:
    return DCEGM(savings_grid=LinSpacedGrid(start=-2.0, stop=4.0, n_points=12))


def test_a_constraint_a_dcegm_regime_cannot_read_is_refused() -> None:
    """The continuous action is produced by the inversion, not bound before it."""
    with pytest.raises(ModelInitializationError, match=r"dcegm solve.*still needs"):
        _model(solver=_dcegm(), constraint=spends_within_reason)


def test_the_refusal_names_the_route_the_constraint_could_not_be_met_on() -> None:
    """Which pipeline could not meet it is the actionable part of the message."""
    with pytest.raises(ModelInitializationError, match="dcegm solve"):
        _model(solver=_dcegm(), constraint=spends_within_reason)


def test_the_same_constraint_is_accepted_on_a_grid_search_regime() -> None:
    """Grid search holds whole candidates, so the identical declaration builds.

    That one declaration is refused under one solver and accepted under another
    is the whole point of deciding per route: it is a fact about the pipeline,
    never about the constraint.
    """
    model = _model(solver=GridSearch(), constraint=spends_within_reason)

    assert "saving" in model.user_regimes


def test_a_constraint_over_combination_inputs_is_accepted_by_dcegm() -> None:
    """A discrete action is bound per combination, so a constraint on it builds."""
    model = _model(solver=_dcegm(), constraint=works_or_retires)

    assert "saving" in model.user_regimes


def test_a_bound_on_the_post_decision_state_builds_under_dcegm() -> None:
    """The savings grid's lowest node is the limit, so the solve need not call it."""
    model = _model(
        solver=_dcegm(),
        constraint=post_decision_lower_bound(margin=_LIQUID, lower=-2.0),
    )

    assert "saving" in model.user_regimes


def test_a_proved_bound_is_still_checked_against_the_grid() -> None:
    """Proving a bound is not ignoring it: a declaration the grid cannot meet errors.

    The proof discharges the constraint precisely because the grid enforces the
    same number, so the claim that it does has to keep being checked. Were the
    check to lapse, a model would declare one limit, solve against another, and
    publish a policy with nothing to say the two had diverged.
    """
    with pytest.raises(ModelInitializationError, match="savings"):
        _model(
            solver=_dcegm(),
            constraint=post_decision_lower_bound(margin=_LIQUID, lower=0.0),
        )


def test_a_bound_on_the_euler_state_is_refused_by_dcegm() -> None:
    """A bound elsewhere says nothing about the savings grid, so nothing proved it."""
    with pytest.raises(
        ModelInitializationError, match=r"dcegm solve.*still needs \['wealth'\]"
    ):
        _model(solver=_dcegm(), constraint=ref("wealth") >= 0.0)


def test_a_bound_on_the_post_decision_state_is_evaluated_by_grid_search() -> None:
    """Grid search inverts on no savings grid, so the identical bound is ordinary."""
    model = _model(solver=GridSearch(), constraint=ref("savings") >= -2.0)

    assert "saving" in model.user_regimes
