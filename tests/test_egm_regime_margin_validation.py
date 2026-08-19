"""Direct tests for explicit EGM-regime structural validation.

The staged bundle's beartype stand-in is intentionally inert.  These tests
therefore invoke the project-owned ``_fail_if_*`` guards themselves rather
than treating successful construction as evidence that annotations were
enforced at runtime.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import get_type_hints

import pytest

from lcm.exceptions import RegimeInitializationError
from lcm.regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    NetOfAdjustmentCost,
    OuterContinuousMargin,
)
from lcm.solvers import GridSearch, OneMarginSolver, TwoMarginSolver


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


def test_solver_annotations_remain_narrowed_exactly() -> None:
    assert get_type_hints(ConsumptionSavingsRegime)["solver"] == (
        OneMarginSolver | GridSearch
    )
    assert get_type_hints(NestedConsumptionSavingsRegime)["solver"] == (
        TwoMarginSolver | GridSearch
    )


def test_net_of_adjustment_cost_guard_is_directly_exercised() -> None:
    declaration = object.__new__(NetOfAdjustmentCost)
    object.__setattr__(declaration, "name_in_dag", "resources")
    object.__setattr__(declaration, "before_cost", "resources")
    object.__setattr__(declaration, "cost", "cost")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_liquid_margin_guard_is_directly_exercised() -> None:
    declaration = object.__new__(LiquidMargin)
    object.__setattr__(declaration, "state", "assets")
    object.__setattr__(declaration, "action", "assets")
    object.__setattr__(declaration, "resources", "resources")
    object.__setattr__(declaration, "post_decision_state", "savings")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_outer_margin_guard_is_directly_exercised() -> None:
    declaration = object.__new__(OuterContinuousMargin)
    object.__setattr__(declaration, "state", "durable")
    object.__setattr__(declaration, "action", "new_durable")
    object.__setattr__(declaration, "post_decision_state", "durable")
    object.__setattr__(declaration, "no_adjustment", "keep_durable")

    with pytest.raises(RegimeInitializationError, match="pairwise distinct"):
        declaration._fail_if_names_are_not_pairwise_distinct()


def test_one_margin_pairing_guard_is_directly_exercised() -> None:
    regime = object.__new__(ConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", object())

    with pytest.raises(RegimeInitializationError, match="OneMarginSolver"):
        regime._fail_if_solver_pairing_is_invalid()


def test_two_margin_pairing_guard_is_directly_exercised() -> None:
    regime = object.__new__(NestedConsumptionSavingsRegime)
    object.__setattr__(regime, "solver", object())

    with pytest.raises(RegimeInitializationError, match="TwoMarginSolver"):
        regime._fail_if_solver_pairing_is_invalid()


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
