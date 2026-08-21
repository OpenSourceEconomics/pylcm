"""Direct tests for explicit EGM-regime structural validation.

Two independent mechanisms reject a malformed EGM-family regime: beartype
checks the narrowed annotations on the public constructors, and the regime's
own `_fail_if_*` guards run from `__post_init__`. A dataclass subclass defined
outside the package regenerates an `__init__` beartype never wrapped, so the
guards are reachable in ordinary use and are exercised here directly rather
than through successful construction.

Every stand-in below is a real object of the wrong kind — a solver of the other
margin family, a discrete grid where a continuous one is required. A bare
`object()` would satisfy any predicate that rejects it, so the test would pass
whatever the guard actually checks; each rejection is therefore paired with a
control showing the same guard accepts the valid object.
"""

from dataclasses import dataclass
from types import MappingProxyType

import pytest

from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    categorical,
)
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
)
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import (
    DCEGM,
    NEGM,
    GridSearch,
    OneMarginSolver,
    TwoMarginSolver,
)
from lcm.typing import ScalarInt

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)


@categorical(ordered=False)
class _Employment:
    working: ScalarInt
    retired: ScalarInt


_DISCRETE_GRID = DiscreteGrid(_Employment)


def _one_margin_solver() -> OneMarginSolver:
    return DCEGM(savings_grid=_GRID)


def _two_margin_solver() -> TwoMarginSolver:
    return NEGM(inner=DCEGM(savings_grid=_GRID), outer_grid=_GRID)


def _liquid() -> LiquidMargin:
    return LiquidMargin(
        state="assets",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )


def _outer() -> OuterContinuousMargin:
    return OuterContinuousMargin(
        state="durable",
        action="new_durable",
        post_decision_state="durable_after_choice",
        no_adjustment="keep_durable",
    )


def _next_assets(savings):
    return savings


def _regime_kwargs() -> dict[str, object]:
    return {
        "transition": lambda: 0,
        "states": {"assets": _GRID},
        "actions": {"consumption": _GRID},
        "functions": {
            "utility": lambda consumption: consumption,
            "resources": lambda assets: assets,
            "savings": lambda resources, consumption: resources - consumption,
        },
        "state_transitions": {"assets": _next_assets},
        "liquid": _liquid(),
    }


def test_solver_annotations_remain_narrowed_exactly() -> None:
    assert (
        ConsumptionSavingsRegime.__annotations__["solver"]
        == OneMarginSolver | GridSearch
    )
    assert (
        NestedConsumptionSavingsRegime.__annotations__["solver"]
        == TwoMarginSolver | GridSearch
    )


def test_public_constructor_rejects_the_wrong_solver_family() -> None:
    """The public constructor rejects a two-margin solver on a liquid regime."""
    with pytest.raises(RegimeInitializationError, match="OneMarginSolver"):
        ConsumptionSavingsRegime(solver=_two_margin_solver(), **_regime_kwargs())  # ty: ignore[invalid-argument-type]


def test_an_out_of_package_subclass_rejects_the_wrong_solver_family() -> None:
    """A dataclass subclass reaches the same rejection through its own guard.

    Redeclaring the class as a dataclass regenerates `__init__`, so the
    annotation check that guards the public constructor no longer runs; the
    regime's own guard is what rejects the solver.
    """

    @dataclass(frozen=True, kw_only=True)
    class _Subclassed(ConsumptionSavingsRegime):
        pass

    with pytest.raises(RegimeInitializationError, match="OneMarginSolver"):
        _Subclassed(solver=_two_margin_solver(), **_regime_kwargs())  # ty: ignore[invalid-argument-type]


def test_an_out_of_package_subclass_accepts_the_right_solver_family() -> None:
    """The same subclass constructs when the solver matches the regime."""

    @dataclass(frozen=True, kw_only=True)
    class _Subclassed(ConsumptionSavingsRegime):
        pass

    regime = _Subclassed(solver=_one_margin_solver(), **_regime_kwargs())  # ty: ignore[invalid-argument-type]

    assert isinstance(regime, ConsumptionSavingsRegime)


def test_net_of_adjustment_cost_guard_is_directly_exercised() -> None:
    declaration = object.__new__(NetOfAdjustmentCost)
    object.__setattr__(declaration, "output", "resources")
    object.__setattr__(declaration, "before_cost", "resources")
    object.__setattr__(declaration, "cost", "cost")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_net_of_adjustment_cost_guard_accepts_distinct_names() -> None:
    NetOfAdjustmentCost(
        output="resources", before_cost="gross_resources", cost="cost"
    )._fail_if_names_are_not_pairwise_distinct()


def test_liquid_margin_guard_is_directly_exercised() -> None:
    declaration = object.__new__(LiquidMargin)
    object.__setattr__(declaration, "state", "assets")
    object.__setattr__(declaration, "action", "assets")
    object.__setattr__(declaration, "resources", "resources")
    object.__setattr__(declaration, "post_decision_state", "savings")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_liquid_margin_guard_accepts_distinct_names() -> None:
    _liquid()._fail_if_names_are_not_pairwise_distinct()


def test_outer_margin_guard_is_directly_exercised() -> None:
    declaration = object.__new__(OuterContinuousMargin)
    object.__setattr__(declaration, "state", "durable")
    object.__setattr__(declaration, "action", "new_durable")
    object.__setattr__(declaration, "post_decision_state", "durable")
    object.__setattr__(declaration, "no_adjustment", "keep_durable")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_outer_margin_guard_accepts_distinct_names() -> None:
    _outer()._fail_if_names_are_not_pairwise_distinct()


def test_one_margin_pairing_guard_is_directly_exercised() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", _two_margin_solver())

    with pytest.raises(RegimeInitializationError, match="OneMarginSolver"):
        regime._fail_if_solver_pairing_is_invalid()


def test_one_margin_pairing_guard_accepts_a_one_margin_solver() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", _one_margin_solver())

    regime._fail_if_solver_pairing_is_invalid()


def test_two_margin_pairing_guard_is_directly_exercised() -> None:
    regime = object.__new__(NestedConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", _one_margin_solver())

    with pytest.raises(RegimeInitializationError, match="TwoMarginSolver"):
        regime._fail_if_solver_pairing_is_invalid()


def test_two_margin_pairing_guard_accepts_a_two_margin_solver() -> None:
    regime = object.__new__(NestedConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", _two_margin_solver())

    regime._fail_if_solver_pairing_is_invalid()


def test_liquid_state_guard_rejects_a_discrete_grid() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(regime, "states", MappingProxyType({"assets": _DISCRETE_GRID}))

    with pytest.raises(RegimeInitializationError, match="not a continuous"):
        regime._fail_if_local_liquid_state_is_not_continuous()


def test_liquid_state_guard_accepts_a_continuous_grid() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(regime, "states", MappingProxyType({"assets": _GRID}))

    regime._fail_if_local_liquid_state_is_not_continuous()


def test_liquid_action_guard_rejects_a_discrete_grid() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(
        regime, "actions", MappingProxyType({"consumption": _DISCRETE_GRID})
    )

    with pytest.raises(RegimeInitializationError, match="not a continuous"):
        regime._fail_if_local_liquid_action_is_not_continuous()


def test_liquid_action_guard_accepts_a_continuous_grid() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(regime, "actions", MappingProxyType({"consumption": _GRID}))

    regime._fail_if_local_liquid_action_is_not_continuous()


def test_liquid_function_mask_guard_is_directly_exercised() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(
        regime,
        "functions",
        MappingProxyType({"resources": None, "savings": lambda: None}),
    )

    with pytest.raises(RegimeInitializationError, match="masked by None"):
        regime._fail_if_local_liquid_function_declarations_are_invalid()


def test_liquid_function_mask_guard_accepts_declared_functions() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(
        regime,
        "functions",
        MappingProxyType({"resources": lambda: None, "savings": lambda: None}),
    )

    regime._fail_if_local_liquid_function_declarations_are_invalid()


def test_margin_collision_guard_is_directly_exercised() -> None:
    regime = object.__new__(NestedConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    outer = _outer()
    object.__setattr__(
        regime,
        "outer_continuous",
        OuterContinuousMargin(
            state=outer.state,
            action=outer.action,
            post_decision_state="savings",
            no_adjustment=outer.no_adjustment,
        ),
    )

    with pytest.raises(RegimeInitializationError, match="must not collide"):
        regime._fail_if_liquid_and_outer_names_collide()


def test_margin_collision_guard_accepts_disjoint_names() -> None:
    regime = object.__new__(NestedConsumptionSavingsRegime)
    object.__setattr__(regime, "liquid", _liquid())
    object.__setattr__(regime, "outer_continuous", _outer())

    regime._fail_if_liquid_and_outer_names_collide()
