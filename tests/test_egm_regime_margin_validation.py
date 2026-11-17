"""An EGM-family regime's own guard rejects the wrong solver family.

Beartype checks the narrowed annotations on the public constructors. A dataclass
subclass defined outside the package regenerates an `__init__` beartype never
wrapped, so the regime's own `_fail_if_*` guards, run from `__post_init__`, are
what reject a malformed regime there. The rejection is paired with a control
showing the same subclass accepts the right solver family.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import pytest

from lcm import LinSpacedGrid
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import ActionEntry, FunctionEntry, StateEntry, StateTransitionEntry
from lcm.solvers import (
    DCEGM,
    NEGM,
    OneMarginSolver,
    RFCEnvelope,
    TwoMarginSolver,
)

_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=11)


def _one_margin_solver() -> OneMarginSolver:
    return DCEGM(savings_grid=_GRID, envelope=RFCEnvelope())


def _two_margin_solver() -> TwoMarginSolver:
    return NEGM(inner=DCEGM(savings_grid=_GRID), outer_grid=_GRID)


def _liquid() -> LiquidMargin:
    return LiquidMargin(
        state="assets",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )


def _next_assets(savings):
    return savings


class _ConsumptionRegimeKwargs(TypedDict):
    """Regime declaration plus its liquid margin."""

    states: Mapping[str, StateEntry]
    actions: Mapping[str, ActionEntry]
    functions: Mapping[str, FunctionEntry]
    state_transitions: Mapping[str, StateTransitionEntry]
    liquid: LiquidMargin


def _regime_kwargs() -> _ConsumptionRegimeKwargs:
    return {
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

    regime = _Subclassed(solver=_one_margin_solver(), **_regime_kwargs())

    assert isinstance(regime, ConsumptionSavingsRegime)
