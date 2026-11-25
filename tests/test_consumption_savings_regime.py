"""Regime-owned margin declarations for the EGM solver family."""

from dataclasses import fields
from types import MappingProxyType
from typing import Literal, cast

import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_law import bind_regime_law
from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.solution.egm import _BoundEGM
from _lcm.solution.nbegm import _BoundNBEGM
from _lcm.solution.negm import _BoundNEGM
from _lcm.solution.nnbegm import _BoundNNBEGM
from lcm import (
    CollectiveUtility,
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
    OneMarginSolver,
    TwoMarginSolver,
)
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserFunction,
)


def _next_regime_code() -> int:
    """Stand in for the law of a non-terminal regime."""
    return 0


_NON_TERMINAL_LAW = bind_regime_law(_next_regime_code)

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)


@categorical(ordered=False)
class _Employment:
    working: ScalarInt
    retired: ScalarInt


_DISCRETE_GRID = DiscreteGrid(category_class=_Employment)

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


def utility(consumption_bonus: FloatND) -> FloatND:
    return consumption_bonus


def consumption_bonus(*, consumption: ContinuousAction, bonus: float = 1.0) -> FloatND:
    return consumption + bonus


def resources(*, wealth: ContinuousState, transfer: float = 0.0) -> FloatND:
    return wealth + transfer


def savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(*, savings: FloatND, interest_rate: float = 0.0) -> ContinuousState:
    return (1 + interest_rate) * savings


def durable_after_choice(new_durable: ContinuousAction) -> FloatND:
    return new_durable


def next_durable(durable_after_choice: FloatND) -> ContinuousState:
    return durable_after_choice


def _functions() -> dict[str, UserFunction]:
    return {
        "utility": utility,
        "consumption_bonus": consumption_bonus,
        "resources": resources,
        "savings": savings,
    }


def _regime(
    *,
    solver: OneMarginSolver | GridSearch | None = None,
    functions: dict[str, UserFunction] | None = None,
    liquid: LiquidMargin = _LIQUID,
) -> ConsumptionSavingsRegime:
    if solver is None:
        solver = GridSearch()
    return ConsumptionSavingsRegime(
        states={"wealth": _GRID},
        actions={"consumption": _GRID},
        functions=_functions() if functions is None else functions,
        state_transitions={"wealth": next_wealth},
        solver=solver,
        liquid=liquid,
    )


def _nested_regime(
    *,
    solver: TwoMarginSolver | GridSearch | None = None,
    liquid: LiquidMargin = _LIQUID,
    outer: OuterContinuousMargin = _OUTER,
    functions: dict[str, UserFunction] | None = None,
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


def test_public_solver_dataclasses_contain_numerical_configuration_only() -> None:
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


def test_dcegm_receives_all_four_names_from_the_liquid_margin() -> None:
    regime = _regime(solver=_fues_dcegm())

    solver = cast("_BoundDCEGM", regime.solver)
    assert solver.continuous_state == "wealth"
    assert solver.continuous_action == "consumption"
    assert solver.resources == "resources"
    assert solver.post_decision_function == "savings"


def test_plain_egm_receives_its_post_decision_name_from_the_margin() -> None:
    regime = _regime(solver=EGM(savings_grid=_GRID))

    assert cast("_BoundEGM", regime.solver).post_decision_function == "savings"


def test_nbegm_receives_all_liquid_names_from_the_margin() -> None:
    regime = _regime(solver=NBEGM(savings_grid=_GRID))

    solver = cast("_BoundNBEGM", regime.solver)
    assert solver.continuous_state == "wealth"
    assert solver.continuous_action == "consumption"
    assert solver.budget_target == "resources"
    assert solver.post_decision_function == "savings"


def test_nested_regime_binds_both_margins_into_negm() -> None:
    regime = _nested_regime(solver=NEGM(inner=_fues_dcegm(), outer_grid=_GRID))

    solver = cast("_BoundNEGM", regime.solver)
    assert solver.inner.continuous_state == "wealth"
    assert solver.inner.continuous_action == "consumption"
    assert solver.inner.resources == "resources"
    assert solver.inner.post_decision_function == "savings"
    assert solver.outer_state == "durable"
    assert solver.outer_action == "new_durable"
    assert solver.outer_post_decision == "durable_after_choice"
    assert solver.outer_no_adjustment_candidate is None


def test_nested_regime_binds_both_margins_into_nnbegm() -> None:
    regime = _nested_regime(
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_GRID),
            outer_search=FiniteOuterGrid(grid=_GRID),
        )
    )

    solver = cast("_BoundNNBEGM", regime.solver)
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


def test_plain_regime_rejects_an_unbound_egm_family_solver() -> None:
    with pytest.raises(RegimeInitializationError, match="margin declarations"):
        Regime(solver=_fues_dcegm())


def test_tier_one_liquid_names_are_pairwise_distinct() -> None:
    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        LiquidMargin(
            state="wealth",
            action="wealth",
            resources="resources",
            post_decision_state="savings",
        )


def test_tier_one_liquid_outer_collisions_are_rejected() -> None:
    outer = OuterContinuousMargin(
        state="wealth",
        action="new_durable",
        post_decision_state="durable_after_choice",
        no_adjustment=outer_unchanged,
    )
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _nested_regime(outer=outer)


def test_specialization_retains_arbitrary_helper_functions() -> None:
    regime = _regime()

    assert "consumption_bonus" in regime.get_all_functions()
    assert regime.get_all_functions()["utility"] is utility


def test_finalized_specialization_rejects_a_missing_broadcastable_function() -> None:
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
            laws={"working": _NON_TERMINAL_LAW},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_composed_resources_are_injected_only_at_model_finalization() -> None:
    def gross_resources(wealth: ContinuousState) -> FloatND:
        return wealth

    def adjustment_cost(durable_after_choice: FloatND) -> FloatND:
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
        laws={"working": _NON_TERMINAL_LAW},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["working"]

    assert "resources" in finalized.functions


def test_composition_exclusion_reports_the_complete_rule() -> None:
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
            laws={"working": _NON_TERMINAL_LAW},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
        )


def test_replace_preserves_the_specialized_type_and_rebinds_the_solver() -> None:
    replaced = _regime().replace(solver=EGM(savings_grid=_GRID))

    assert isinstance(replaced, ConsumptionSavingsRegime)
    assert cast("_BoundEGM", replaced.solver).post_decision_function == "savings"


def test_liquid_state_message_states_the_property_it_checks() -> None:
    """The rejection names the grid property the guard actually tests."""
    regime = _regime(
        functions={"utility": utility, "resources": resources, "savings": savings}
    )
    object.__setattr__(regime, "states", MappingProxyType({"wealth": _DISCRETE_GRID}))

    with pytest.raises(RegimeInitializationError) as excinfo:
        regime._fail_if_local_liquid_state_is_not_continuous()

    assert "non-process" not in str(excinfo.value)


def test_a_single_finalization_error_renders_without_enumeration() -> None:
    """One error reads as a plain sentence, not as a numbered list of one."""
    regime = _regime()
    object.__setattr__(
        regime, "actions", MappingProxyType({"consumption": _DISCRETE_GRID})
    )

    with pytest.raises(RegimeInitializationError) as excinfo:
        regime._validate_finalized_structure(regime_name="working")

    assert "The following errors occurred" not in str(excinfo.value)
    assert "liquid.action" in str(excinfo.value)


def test_several_finalization_errors_are_enumerated() -> None:
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


type _KeyedSlot = Literal["koopmans_aggregator", "certainty_equivalent"]


def _named_member_regime(*, solver: OneMarginSolver | TwoMarginSolver) -> Regime:
    """The one member `alice`, under an endogenous-grid solver of either family."""
    regime = (
        _nested_regime(solver=solver)
        if isinstance(solver, TwoMarginSolver)
        else _regime(solver=solver)
    )
    return regime.replace(
        functions={
            **regime.functions,
            "utility": CollectiveUtility(utilities={"alice": utility}),
        }
    )


def _key_by_member(*, regime: Regime, slot: _KeyedSlot) -> Regime:
    """Declare `slot` keyed by the regime's one member `alice`."""
    if slot == "koopmans_aggregator":
        return regime.replace(koopmans_aggregator={"alice": LinearAggregator()})
    return regime.replace(certainty_equivalent={"alice": LinearExpectation()})


@pytest.mark.parametrize(
    ("slot", "bare"),
    [
        ("koopmans_aggregator", "LinearAggregator"),
        ("certainty_equivalent", "LinearExpectation"),
    ],
)
@pytest.mark.parametrize(
    "solver",
    [
        EGM(savings_grid=_GRID),
        _fues_dcegm(),
        NBEGM(savings_grid=_GRID),
        NEGM(inner=_fues_dcegm(), outer_grid=_GRID),
        NNBEGM(
            inner=NBEGM(savings_grid=_GRID), outer_search=FiniteOuterGrid(grid=_GRID)
        ),
    ],
    ids=["EGM", "DCEGM", "NBEGM", "NEGM", "NNBEGM"],
)
def test_an_endogenous_grid_solver_refuses_a_slot_keyed_by_member(
    *,
    solver: OneMarginSolver | TwoMarginSolver,
    slot: _KeyedSlot,
    bare: str,
) -> None:
    """A solver of one member takes `W` and `CE` as one bare value.

    Every endogenous-grid solver refuses the slot keyed by member with one
    message, which names the solver, the slot and the bare declaration to give.
    """
    regime = _named_member_regime(solver=solver)

    with pytest.raises(
        RegimeInitializationError,
        match=(
            rf"^The {type(solver).__name__} solver takes `{slot}` as one bare "
            rf"value, not keyed by member: declare `{slot}={bare}\(\)` directly"
        ),
    ):
        _key_by_member(regime=regime, slot=slot)
