"""Spec for NNBEGM build-time validation (the fail-loudly nesting contract).

A regime with `solver=NNBEGM(...)` must declare a real outer margin that is
distinct from the margin the inner NB-EGM consumes. Violations raise
`ModelInitializationError` at model build, naming the offending feature and the
correct alternative solver. The checks run on the user regimes directly
(`validate_nnbegm_regimes`), so each case asserts the outcome without building
kernels or solving.

The cases mutate the valid smooth two-asset toy one rule at a time:

1. no outer margin (outer action absent) → use `NBEGM`,
2. outer post-decision naming no declared function of the regime,
3. outer action equals the inner consumption action → reject (distinct
   margins).

Two shapes NEGM rejects are *accepted* here, because the outer sweep re-solves
the whole inner problem per outer node instead of lifting a frozen inversion:
an Euler-state law that reads the outer post-decision, and a utility that
multiplies consumption by it.
"""

import dataclasses
from typing import cast

import pytest

from _lcm.egm.nnbegm_validation import validate_nnbegm_regimes
from lcm import AgeGrid, Model
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import NNBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models import n_nbegm_toy

_SOLVER = cast("NNBEGM", n_nbegm_toy.build_solver(variant="n_nbegm"))

_VALID = UserRegime(
    active=lambda age: age <= 20,
    states={
        "wealth": n_nbegm_toy.WEALTH_GRID,
        "illiquid": n_nbegm_toy.ILLIQUID_GRID,
    },
    state_transitions={
        "wealth": n_nbegm_toy.next_wealth,
        "illiquid": n_nbegm_toy.durable_transition,
    },
    actions={
        "consumption": n_nbegm_toy.CONSUMPTION_GRID,
        "illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID,
    },
    transition=n_nbegm_toy.next_regime,
    functions={
        "utility": n_nbegm_toy.utility,
        "new_illiquid": n_nbegm_toy.new_illiquid,
        "resources": n_nbegm_toy.resources,
        "liquid_savings": n_nbegm_toy.liquid_savings,
        "keep_illiquid": n_nbegm_toy.keep_illiquid,
        "credited": n_nbegm_toy.credited,
    },
    solver=_SOLVER,
)


def _validate(regime: UserRegime) -> None:
    """Run the NNBEGM contract check on a single-regime mapping."""
    validate_nnbegm_regimes(user_regimes={"alive": regime})


def test_valid_two_asset_toy_nnbegm_regime_passes_validation():
    """The smooth two-asset toy satisfies the nesting contract."""
    _validate(_VALID)


def test_outer_action_absent_is_rejected_with_nbegm_pointer():
    """A regime with no outer continuous action is a pure 1-D problem.

    Dropping the durable action leaves a single continuous action; NNBEGM would
    silently run as plain NB-EGM, so it is rejected with a pointer to `NBEGM`.
    """
    regime = _VALID.replace(actions={"consumption": n_nbegm_toy.CONSUMPTION_GRID})
    with pytest.raises(ModelInitializationError, match="use `NBEGM`"):
        _validate(regime)


def test_outer_post_decision_not_declared_is_rejected():
    """An outer post-decision naming no function of the regime fails."""
    regime = _VALID.replace(
        solver=dataclasses.replace(_SOLVER, outer_post_decision="not_a_function")
    )
    with pytest.raises(ModelInitializationError, match="not a declared function"):
        _validate(regime)


def test_outer_action_equal_to_the_inner_consumption_action_is_rejected():
    """The outer margin must not be the action the inner NB-EGM consumes.

    The inner solver claims the regime's first continuous action as its
    consumption margin, so an outer action pointing at it swaps the two
    margins.
    """
    regime = _VALID.replace(
        solver=dataclasses.replace(_SOLVER, outer_action="consumption")
    )
    with pytest.raises(ModelInitializationError, match="coincides with the inner"):
        _validate(regime)


def _euler_law_reading_outer_margin(
    liquid_savings: FloatND, new_illiquid: ContinuousState
) -> ContinuousState:
    """A liquid Euler law whose return depends on the chosen durable stock."""
    return (1.0 + n_nbegm_toy.LIQUID_RATE) * liquid_savings + 0.01 * new_illiquid


def test_an_euler_law_reading_the_outer_margin_is_accepted():
    """A liquid law that reads the outer post-decision stays in scope.

    The outer sweep binds the outer post-decision as a flat param and re-runs
    the whole inner solve per outer node, so the law sees a constant and the
    inner Euler inversion is exact conditional on that node.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": _euler_law_reading_outer_margin,
            "illiquid": n_nbegm_toy.durable_transition,
        },
    )
    _validate(regime)


def _utility_coupling_consumption_and_durable_move(
    consumption: ContinuousAction, new_illiquid: ContinuousState
) -> FloatND:
    """A Cobb-Douglas composite of consumption and the chosen durable service."""
    composite = consumption**0.8 * new_illiquid**0.2
    return composite ** (1.0 - n_nbegm_toy.RISK_AVERSION) / (
        1.0 - n_nbegm_toy.RISK_AVERSION
    )


def test_a_utility_composite_of_consumption_and_the_durable_is_accepted():
    """A multiplicative composite flow stays in scope.

    With the durable node bound, the composite collapses to a single power of
    consumption, so the inner inversion is well defined — this is the
    Kaplan-Violante flow the two-asset models use.
    """
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "utility": _utility_coupling_consumption_and_durable_move,
        },
    )
    _validate(regime)


def test_a_regime_with_a_non_nested_solver_is_left_alone():
    """The NNBEGM contract check only inspects regimes with an `NNBEGM` solver."""
    regime = _VALID.replace(
        solver=n_nbegm_toy.build_solver(variant="brute"),
        actions={"illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID},
    )
    _validate(regime)


def test_model_build_runs_the_nnbegm_contract_check():
    """`Model(...)` rejects a swapped-margin NNBEGM regime, not only the check."""
    alive = _VALID.replace(
        solver=dataclasses.replace(_SOLVER, outer_action="consumption")
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age > 20,
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        functions={"utility": n_nbegm_toy.terminal_utility},
    )
    with pytest.raises(ModelInitializationError, match="coincides with the inner"):
        Model(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=n_nbegm_toy.RegimeId,
            ages=AgeGrid(start=20, stop=25, step="5Y"),
            fixed_params={"final_age_alive": 20},
        )


def _outer_law_reading_the_inner_savings(
    new_illiquid: ContinuousState, liquid_savings: FloatND
) -> ContinuousState:
    """A durable law whose carried stock depends on the inner savings choice."""
    return new_illiquid + 0.01 * liquid_savings


def test_an_outer_law_reading_the_inner_savings_margin_is_rejected():
    """A durable law depending on the inner savings choice is not an NNBEGM model.

    The solver evaluates the declared law to find what the next period carries.
    Reading the inner post-decision ties the stock carried forward to the
    consumption the inner Euler inversion is solving for, so the outer max stops
    ranging over independent problems and the nesting is no longer valid.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_the_inner_savings,
        },
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        _validate(regime)


def _outer_law_reading_a_sibling_law(
    new_illiquid: ContinuousState, next_wealth: ContinuousState
) -> ContinuousState:
    """A durable law reaching the inner margin through the Euler state's law."""
    return new_illiquid + 0.01 * next_wealth


def test_an_outer_law_reaching_the_inner_margin_through_a_sibling_is_rejected():
    """The coupling is caught even when a sibling law stands between.

    `next_illiquid` reads `next_wealth`, which reads the inner post-decision. A
    check that stopped at the sibling's name would see a law reading only states
    and miss the coupling entirely.
    """
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_a_sibling_law,
        },
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        _validate(regime)


def test_a_depreciating_outer_law_is_accepted():
    """A law reading only the chosen stock stays in scope.

    `next_z = alpha * s'` is the ordinary durable law; the guard must not reject
    it while catching the coupled shapes above.
    """

    def depreciating(new_illiquid: ContinuousState) -> ContinuousState:
        return 0.7 * new_illiquid

    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": depreciating,
        },
    )
    _validate(regime)
