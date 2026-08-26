"""Regime margins replace the retired NNBEGM inner-name normalizer.

The public NBEGM and NNBEGM dataclasses contain numerical configuration only.
Specialized regime construction binds one shared ``LiquidMargin`` into the
private solver companions, so the nested kernel receives the same explicit
liquid names without dispatching on an inner-solver type.
"""

from dataclasses import fields
from typing import cast

import pytest

from _lcm.solution.nbegm import NBEGM, _BoundNBEGM
from _lcm.solution.nnbegm import (
    NNBEGM,
    _BoundNNBEGM,
    _fail_if_inner_is_not_nbegm,
)
from lcm import LinSpacedGrid
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import FiniteOuterGrid, GridSearch
from lcm.typing import UserFunction

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=8)
_LIQUID = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="cash_on_hand",
    post_decision_state="liquid_savings",
)
_OUTER = OuterContinuousMargin(
    state="illiquid",
    action="illiquid_investment",
    post_decision_state="new_illiquid",
    no_adjustment=outer_unchanged,
)


def _functions() -> dict[str, UserFunction]:
    return {
        "utility": lambda consumption: consumption,
        "cash_on_hand": lambda liquid: liquid,
        "liquid_savings": lambda cash_on_hand, consumption: cash_on_hand - consumption,
        "new_illiquid": lambda illiquid_investment: illiquid_investment,
    }


def _one_margin(*, solver: NBEGM) -> ConsumptionSavingsRegime:
    return ConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"liquid": _GRID},
        actions={"consumption": _GRID},
        functions={
            key: value for key, value in _functions().items() if key != "new_illiquid"
        },
        solver=solver,
        liquid=_LIQUID,
    )


def _two_margin(*, solver: NNBEGM) -> NestedConsumptionSavingsRegime:
    return NestedConsumptionSavingsRegime(
        transition=lambda: 0,
        states={"liquid": _GRID, "illiquid": _GRID},
        actions={"consumption": _GRID, "illiquid_investment": _GRID},
        functions=_functions(),
        solver=solver,
        liquid=_LIQUID,
        outer_continuous=_OUTER,
    )


def test_public_nbegm_contains_numerical_configuration_only() -> None:
    names = {item.name for item in fields(NBEGM)}
    assert {
        "continuous_state",
        "continuous_action",
        "budget_target",
        "post_decision_function",
    }.isdisjoint(names)


def test_one_margin_regime_binds_nbegm_from_liquid_margin() -> None:
    regime = _one_margin(solver=NBEGM(savings_grid=_GRID))
    solver = cast("_BoundNBEGM", regime.solver)

    assert isinstance(solver, _BoundNBEGM)
    assert (
        solver.continuous_state,
        solver.continuous_action,
        solver.budget_target,
        solver.post_decision_function,
    ) == ("liquid", "consumption", "cash_on_hand", "liquid_savings")


def test_public_nnbegm_contains_numerical_configuration_only() -> None:
    names = {item.name for item in fields(NNBEGM)}
    assert names == {"inner", "outer_search"}


def test_nested_regime_binds_both_margins_without_an_inner_spec() -> None:
    """A declared identity no-adjustment map reaches the solver as no candidate."""
    regime = _two_margin(
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_GRID),
            outer_search=FiniteOuterGrid(grid=_GRID),
        )
    )
    solver = cast("_BoundNNBEGM", regime.solver)

    assert isinstance(solver, _BoundNNBEGM)
    assert isinstance(solver.inner, _BoundNBEGM)
    assert (
        solver.inner.continuous_state,
        solver.inner.continuous_action,
        solver.inner.budget_target,
        solver.inner.post_decision_function,
    ) == ("liquid", "consumption", "cash_on_hand", "liquid_savings")
    assert (
        solver.outer_state,
        solver.outer_action,
        solver.outer_post_decision,
        solver.outer_no_adjustment_candidate,
    ) == (
        "illiquid",
        "illiquid_investment",
        "new_illiquid",
        None,
    )


def test_non_nbegm_inner_is_rejected_by_explicit_structural_guard() -> None:
    """A non-NBEGM inner is refused by an explicit check, not only by a type hint.

    The guard is called directly: wherever runtime type checking is active the
    constructor's own annotation refuses the value first, which would leave the
    explicit check unexercised.
    """
    with pytest.raises(
        RegimeInitializationError,
        match=r"NNBEGM\.inner must be an NBEGM",
    ):
        _fail_if_inner_is_not_nbegm(GridSearch())
