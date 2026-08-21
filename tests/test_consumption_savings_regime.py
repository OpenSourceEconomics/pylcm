"""Regime-owned margin declarations for the EGM solver family."""

from dataclasses import fields
from types import MappingProxyType
from typing import Any, cast

import pytest

from _lcm.regime_building.finalize import finalize_regimes
from lcm import (
    DiscreteGrid,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Regime,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import (
    DCEGM,
    EGM,
    NBEGM,
    NEGM,
    NNBEGM,
    FiniteOuterGrid,
    FUESEnvelope,
    GridSearch,
)
from lcm.typing import ScalarInt

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)


@categorical(ordered=False)
class _Employment:
    working: ScalarInt
    retired: ScalarInt


_DISCRETE_GRID = DiscreteGrid(_Employment)

_LIQUID = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)
_OUTER = OuterContinuousMargin(
    state="durable",
    action="new_durable",
    post_decision_state="durable_after_choice",
    no_adjustment=outer_unchanged,
)


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


def durable_after_choice(new_durable):
    return new_durable


def next_durable(durable_after_choice):
    return durable_after_choice


def _functions() -> dict[str, Any]:
    return {
        "utility": utility,
        "consumption_bonus": consumption_bonus,
        "resources": resources,
        "savings": savings,
    }


def _regime(
    *,
    solver: Any = None,
    functions: dict[str, Any] | None = None,
    liquid: LiquidMargin = _LIQUID,
) -> ConsumptionSavingsRegime:
    if solver is None:
        solver = GridSearch()
    return ConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"wealth": _GRID},
        actions={"consumption": _GRID},
        functions=_functions() if functions is None else functions,
        state_transitions={"wealth": next_wealth},
        solver=solver,
        liquid=liquid,
    )


def _nested_regime(
    *,
    solver: Any = None,
    liquid: LiquidMargin = _LIQUID,
    outer: OuterContinuousMargin = _OUTER,
    functions: dict[str, Any] | None = None,
) -> NestedConsumptionSavingsRegime:
    if solver is None:
        solver = GridSearch()
    all_functions = _functions()
    all_functions |= {
        "durable_after_choice": durable_after_choice,
    }
    if functions is not None:
        all_functions = functions
    return NestedConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"wealth": _GRID, "durable": _GRID},
        actions={"consumption": _GRID, "new_durable": _GRID},
        functions=all_functions,
        state_transitions={"wealth": next_wealth, "durable": next_durable},
        solver=solver,
        liquid=liquid,
        outer_continuous=outer,
    )


def _fues_dcegm() -> DCEGM:
    return DCEGM(savings_grid=_GRID, envelope=FUESEnvelope())


def test_public_solver_dataclasses_contain_numerical_configuration_only():
    assert {item.name for item in fields(EGM)} == {"savings_grid"}
    assert {
        "budget_target",
        "continuous_state",
        "continuous_action",
        "post_decision_function",
        "resources",
    }.isdisjoint(item.name for item in fields(DCEGM))
    assert {
        "budget_target",
        "continuous_state",
        "continuous_action",
        "post_decision_function",
    }.isdisjoint(item.name for item in fields(NBEGM))
    assert {
        "outer_action",
        "outer_state",
        "outer_post_decision",
        "outer_no_adjustment_candidate",
        "outer_cost",
    }.isdisjoint(item.name for item in fields(NEGM))
    assert {
        "outer_action",
        "outer_state",
        "outer_post_decision",
        "outer_no_adjustment_candidate",
    }.isdisjoint(item.name for item in fields(NNBEGM))


def test_dcegm_receives_all_four_names_from_the_liquid_margin():
    regime = _regime(solver=_fues_dcegm())

    solver = cast("Any", regime.solver)
    assert solver.continuous_state == "wealth"
    assert solver.continuous_action == "consumption"
    assert solver.resources == "resources"
    assert solver.post_decision_function == "savings"


def test_plain_egm_receives_its_post_decision_name_from_the_margin():
    regime = _regime(solver=EGM(savings_grid=_GRID))

    assert cast("Any", regime.solver).post_decision_function == "savings"


def test_nbegm_receives_all_liquid_names_from_the_margin():
    regime = _regime(solver=NBEGM(savings_grid=_GRID))

    solver = cast("Any", regime.solver)
    assert solver.continuous_state == "wealth"
    assert solver.continuous_action == "consumption"
    assert solver.budget_target == "resources"
    assert solver.post_decision_function == "savings"


def test_nested_regime_binds_both_margins_into_negm():
    regime = _nested_regime(solver=NEGM(inner=_fues_dcegm(), outer_grid=_GRID))

    solver = cast("Any", regime.solver)
    assert solver.inner.continuous_state == "wealth"
    assert solver.inner.continuous_action == "consumption"
    assert solver.inner.resources == "resources"
    assert solver.inner.post_decision_function == "savings"
    assert solver.outer_state == "durable"
    assert solver.outer_action == "new_durable"
    assert solver.outer_post_decision == "durable_after_choice"
    assert solver.outer_no_adjustment_candidate is None


def test_nested_regime_binds_both_margins_into_nnbegm():
    regime = _nested_regime(
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_GRID),
            outer_search=FiniteOuterGrid(grid=_GRID),
        )
    )

    solver = cast("Any", regime.solver)
    assert solver.inner.continuous_state == "wealth"
    assert solver.inner.continuous_action == "consumption"
    assert solver.inner.budget_target == "resources"
    assert solver.inner.post_decision_function == "savings"
    assert solver.outer_state == "durable"
    assert solver.outer_action == "new_durable"
    assert solver.outer_post_decision == "durable_after_choice"
    # `_OUTER` declares `no_adjustment=outer_unchanged`, the public spelling of
    # "the keeper is the state held still". The public->bound seam resolves that
    # sentinel to `None`, which is what a solver reading the bound margin sees:
    # no *named* no-adjustment function to call. Asserting the sentinel here
    # would pin a value that can no longer reach the solver.
    assert solver.outer_no_adjustment_candidate is None


def test_pairing_check_rejects_two_margin_solver_on_one_margin_regime():
    with pytest.raises(RegimeInitializationError, match="OneMarginSolver"):
        _regime(solver=NEGM(inner=_fues_dcegm(), outer_grid=_GRID))


def test_pairing_check_rejects_one_margin_solver_on_nested_regime():
    with pytest.raises(RegimeInitializationError, match="TwoMarginSolver"):
        _nested_regime(solver=_fues_dcegm())


def test_plain_regime_rejects_an_unbound_egm_family_solver():
    with pytest.raises(RegimeInitializationError, match="margin declarations"):
        Regime(transition=lambda: 0, solver=_fues_dcegm())


def test_tier_one_liquid_names_are_pairwise_distinct():
    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        LiquidMargin(
            state="wealth",
            action="wealth",
            resources="resources",
            post_decision_state="savings",
        )


def test_tier_one_liquid_outer_collisions_are_rejected():
    outer = OuterContinuousMargin(
        state="wealth",
        action="new_durable",
        post_decision_state="durable_after_choice",
        no_adjustment=outer_unchanged,
    )
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _nested_regime(outer=outer)


def test_specialization_retains_arbitrary_helper_functions():
    regime = _regime()

    assert "consumption_bonus" in regime.get_all_functions()
    assert regime.get_all_functions()["utility"] is utility


def test_finalized_specialization_rejects_a_missing_broadcastable_function():
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


def test_composed_resources_are_injected_only_at_model_finalization():
    def gross_resources(wealth):
        return wealth

    def adjustment_cost(durable_after_choice):
        return 0.1 * durable_after_choice

    composed = NetOfAdjustmentCost(
        output="resources",
        before_cost="gross_resources",
        cost="adjustment_cost",
    )
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources=composed,
        post_decision_state="savings",
    )
    functions = _functions()
    functions.pop("resources")
    functions |= {
        "gross_resources": gross_resources,
        "adjustment_cost": adjustment_cost,
        "durable_after_choice": durable_after_choice,
    }
    regime = _nested_regime(
        solver=NEGM(inner=_fues_dcegm(), outer_grid=_GRID),
        liquid=liquid,
        functions=functions,
    )

    finalized = finalize_regimes(
        user_regimes={"working": regime},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["working"]

    assert "resources" in finalized.functions


def test_composition_exclusion_reports_the_complete_rule():
    composed = NetOfAdjustmentCost(
        output="resources",
        before_cost="gross_resources",
        cost="adjustment_cost",
    )
    liquid = LiquidMargin(
        state="wealth",
        action="consumption",
        resources=composed,
        post_decision_state="savings",
    )
    functions = _functions()
    functions |= {
        "gross_resources": resources,
        "adjustment_cost": lambda durable_after_choice: durable_after_choice,
        "durable_after_choice": durable_after_choice,
    }
    regime = _nested_regime(liquid=liquid, functions=functions)

    with pytest.raises(
        Exception,
        match=r"must not exist.*must exist.*composes",
    ):
        finalize_regimes(
            user_regimes={"working": regime},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_replace_preserves_the_specialized_type_and_rebinds_the_solver():
    replaced = _regime().replace(solver=EGM(savings_grid=_GRID))

    assert isinstance(replaced, ConsumptionSavingsRegime)
    assert cast("Any", replaced.solver).post_decision_function == "savings"


def test_liquid_state_message_states_the_property_it_checks():
    """The rejection names the grid property the guard actually tests."""
    regime = _regime(
        functions={"utility": utility, "resources": resources, "savings": savings}
    )
    object.__setattr__(regime, "states", MappingProxyType({"wealth": _DISCRETE_GRID}))

    with pytest.raises(RegimeInitializationError) as excinfo:
        regime._fail_if_local_liquid_state_is_not_continuous()

    assert "non-process" not in str(excinfo.value)


def test_a_single_finalization_error_renders_without_enumeration():
    """One error reads as a plain sentence, not as a numbered list of one."""
    regime = _regime()
    object.__setattr__(
        regime, "actions", MappingProxyType({"consumption": _DISCRETE_GRID})
    )

    with pytest.raises(RegimeInitializationError) as excinfo:
        regime._validate_finalized_structure(regime_name="working")

    assert "The following errors occurred" not in str(excinfo.value)
    assert "liquid.action" in str(excinfo.value)


def test_several_finalization_errors_are_enumerated():
    """Two or more errors are numbered, so neither hides inside a run-on line."""
    regime = _regime()
    object.__setattr__(regime, "states", MappingProxyType({"wealth": _DISCRETE_GRID}))
    object.__setattr__(
        regime, "actions", MappingProxyType({"consumption": _DISCRETE_GRID})
    )

    with pytest.raises(RegimeInitializationError) as excinfo:
        regime._validate_finalized_structure(regime_name="working")

    message = str(excinfo.value)
    assert "The following errors occurred" in message
    assert "1. " in message
    assert "2. " in message
