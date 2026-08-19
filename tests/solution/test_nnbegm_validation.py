"""Validation tests for the regime-owned NNBEGM nesting contract.

`NestedConsumptionSavingsRegime` owns the two structural margins.  Its three
validation tiers establish their kind, existence, and disjointness before the
model-stage NNBEGM validator runs.  The solver-side validator therefore owns
only the dynamic condition that the outer carried-state law must not depend on
the inner liquid post-decision margin, directly or through a sibling law.
"""

import pytest

from _lcm.egm.nnbegm_validation import validate_nnbegm_regimes
from lcm import AgeGrid, Model
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Regime,
)
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.test_models import n_nbegm_toy


def _valid_regime() -> NestedConsumptionSavingsRegime:
    return NestedConsumptionSavingsRegime(
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
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_illiquid",
            no_adjustment="keep_illiquid",
        ),
        solver=n_nbegm_toy.build_solver(variant="n_nbegm"),
    )


_VALID = _valid_regime()


def _validate(regime: Regime) -> None:
    """Run the NNBEGM dynamic contract check on a single-regime mapping."""
    validate_nnbegm_regimes(user_regimes={"alive": regime})


def test_valid_two_asset_toy_nnbegm_regime_passes_validation() -> None:
    """The smooth two-asset toy satisfies the nesting contract."""
    _validate(_VALID)


def test_margin_collision_is_rejected_before_solver_validation() -> None:
    """Tier one rejects a role name shared by the liquid and outer margins."""
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _VALID.replace(
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="consumption",
                post_decision_state="new_illiquid",
                no_adjustment="keep_illiquid",
            )
        )


def test_explicitly_masked_outer_function_is_rejected_before_model_build() -> None:
    """Tier two rejects an explicitly masked outer post-decision function."""
    with pytest.raises(RegimeInitializationError, match="explicitly masked"):
        _VALID.replace(
            functions={**dict(_VALID.functions), "new_illiquid": None},
        )


def _euler_law_reading_outer_margin(
    liquid_savings: FloatND, new_illiquid: ContinuousState
) -> ContinuousState:
    """A liquid Euler law whose return depends on the chosen durable stock."""
    return (1.0 + n_nbegm_toy.LIQUID_RATE) * liquid_savings + 0.01 * new_illiquid


def test_an_euler_law_reading_the_outer_margin_is_accepted() -> None:
    """The outer node is fixed while the inner solve runs, so this is valid."""
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
    """A Cobb-Douglas composite of consumption and chosen durable service."""
    composite = consumption**0.8 * new_illiquid**0.2
    return composite ** (1.0 - n_nbegm_toy.RISK_AVERSION) / (
        1.0 - n_nbegm_toy.RISK_AVERSION
    )


def test_a_utility_composite_of_consumption_and_the_durable_is_accepted() -> None:
    """Conditional on an outer node, utility remains a 1-D inner problem."""
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "utility": _utility_coupling_consumption_and_durable_move,
        },
    )
    _validate(regime)


def test_a_regime_with_a_non_nested_solver_is_left_alone() -> None:
    """The dynamic NNBEGM check ignores regimes not bound to NNBEGM."""
    regime = Regime(
        active=_VALID.active,
        states=_VALID.states,
        state_transitions=_VALID.state_transitions,
        actions={"illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID},
        transition=_VALID.transition,
        functions=_VALID.functions,
        solver=n_nbegm_toy.build_solver(variant="brute"),
    )
    _validate(regime)


def _outer_law_reading_the_inner_savings(
    new_illiquid: ContinuousState, liquid_savings: FloatND
) -> ContinuousState:
    """A durable law whose carried stock depends on the inner savings choice."""
    return new_illiquid + 0.01 * liquid_savings


def test_an_outer_law_reading_the_inner_savings_margin_is_rejected() -> None:
    """Direct dependence on the inner post-decision axis breaks nesting."""
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
    """A durable law reaching the inner margin through the Euler-state law."""
    return new_illiquid + 0.01 * next_wealth


def test_an_outer_law_reaching_the_inner_margin_through_a_sibling_is_rejected() -> None:
    """The dependency traversal follows sibling state-transition laws."""
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_a_sibling_law,
        },
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        _validate(regime)


def test_a_depreciating_outer_law_is_accepted() -> None:
    """A law reading only the chosen outer stock stays in scope."""

    def depreciating(new_illiquid: ContinuousState) -> ContinuousState:
        return 0.7 * new_illiquid

    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": depreciating,
        },
    )
    _validate(regime)


def test_model_build_runs_the_dynamic_nnbegm_contract_check() -> None:
    """`Model(...)` invokes the same dynamic coupling guard."""
    alive = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_the_inner_savings,
        }
    )
    dead = Regime(
        transition=None,
        active=lambda age: age > 20,
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        functions={"utility": n_nbegm_toy.terminal_utility},
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        Model(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=n_nbegm_toy.RegimeId,
            ages=AgeGrid(start=20, stop=25, step="5Y"),
            fixed_params={"final_age_alive": 20},
        )
