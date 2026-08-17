"""The consumption-savings specialization owns roles without shrinking the DAG."""

from typing import cast

import pytest

from _lcm.regime_building.finalize import finalize_regimes
from lcm import (
    DCEGM,
    EGM,
    NEGM,
    ConsumptionSavingsRegime,
    FUESEnvelope,
    GridSearch,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
)
from lcm.exceptions import RegimeInitializationError

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)


def utility(consumption_bonus):
    return consumption_bonus


def consumption_bonus(consumption, bonus=1.0):
    return consumption + bonus


def resources(wealth, transfer=0.0):
    return wealth + transfer


def savings(resources, consumption):
    return resources - consumption


def next_wealth(savings, interest_rate=0.0):
    return (1 + interest_rate) * savings


def _regime(*, solver=None, functions=None) -> ConsumptionSavingsRegime:
    if solver is None:
        solver = GridSearch()
    return ConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"wealth": _GRID},
        actions={"consumption": _GRID},
        functions=(
            {
                "utility": utility,
                "consumption_bonus": consumption_bonus,
                "resources": resources,
                "savings": savings,
            }
            if functions is None
            else functions
        ),
        state_transitions={"wealth": next_wealth},
        solver=solver,
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
    )


def _fues_dcegm(**kwargs) -> DCEGM:
    return DCEGM(
        savings_grid=_GRID,
        envelope=FUESEnvelope(),
        **kwargs,
    )


def test_dcegm_omits_roles_owned_by_consumption_savings_regime():
    """The regime binds all four canonical roles into a terse DC-EGM config."""
    regime = _regime(solver=_fues_dcegm())

    assert regime.solver == _fues_dcegm(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
    )


def test_plain_egm_omits_the_post_decision_role_owned_by_the_regime():
    """Plain EGM receives its one named accounting role from the regime."""
    regime = _regime(solver=EGM(savings_grid=_GRID))

    assert cast("EGM", regime.solver).post_decision_function == "savings"


def test_negm_inner_solver_uses_the_same_regime_owned_roles():
    """NEGM binds the role contract into its nested DC-EGM configuration."""
    solver = NEGM(
        inner=_fues_dcegm(),
        outer_action="new_durable",
        outer_state="durable",
        outer_post_decision="durable_after_choice",
        outer_grid=_GRID,
    )
    regime = _regime(solver=solver)

    inner = cast("NEGM", regime.solver).inner
    assert inner.continuous_state == "wealth"
    assert inner.continuous_action == "consumption"
    assert inner.resources == "resources"
    assert inner.post_decision_function == "savings"


def test_conflicting_solver_duplicate_is_rejected_at_regime_construction():
    """There is one owner: an explicit duplicate may agree but never conflict."""
    with pytest.raises(RegimeInitializationError, match="owns this role as 'wealth'"):
        _regime(solver=_fues_dcegm(continuous_state="cash_on_hand"))


def test_specialization_retains_arbitrary_helper_functions():
    """Owning the accounting seam leaves the rest of the function DAG general."""
    regime = _regime()

    assert "consumption_bonus" in regime.get_all_functions()
    assert regime.get_all_functions()["utility"] is utility


def test_finalized_specialization_rejects_a_missing_role_function():
    """Role references are validated after model-level slots have been merged."""
    regime = _regime(
        functions={
            "utility": utility,
            "consumption_bonus": consumption_bonus,
            "savings": savings,
        }
    )

    with pytest.raises(RegimeInitializationError, match="resources 'resources'"):
        finalize_regimes(
            user_regimes={"working": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_replace_preserves_the_specialized_type_and_role_binding():
    """Regime replacement keeps the specialization and rebinds a new solver."""
    replaced = _regime().replace(solver=EGM(savings_grid=_GRID))

    assert isinstance(replaced, ConsumptionSavingsRegime)
    assert cast("EGM", replaced.solver).post_decision_function == "savings"
