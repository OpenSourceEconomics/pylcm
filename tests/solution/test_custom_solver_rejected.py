"""A solver outside the shipped set is refused when the model is built.

The engine dispatches on the shipped solver classes at several points — it
synthesizes the simulate-phase budget constraint only for them, and reads a
carry target's inner configuration only from them. A solver that subclasses
`OneMarginSolver` or `TwoMarginSolver` directly would pass regime construction
and then be solved with those steps silently skipped, so it is refused up front
instead.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import pytest

from _lcm.solution.contract import (
    SolutionKernels,
    SolverBuildContext,
)
from _lcm.solution.shipped_solvers import fail_if_solver_is_not_shipped
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import ConsumptionSavingsRegime, LiquidMargin, Regime
from lcm.solvers import (
    DCEGM,
    EGM,
    NBEGM,
    NNBEGM,
    GridSearch,
    OneMarginSolver,
    TwoMarginSolver,
)
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt

_WEALTH_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_ACTION_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=12)


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(savings: FloatND) -> FloatND:
    return savings


def next_regime(_age: float) -> ScalarInt:
    return RegimeId.done


@dataclass(frozen=True, kw_only=True)
class _CustomOneMargin(OneMarginSolver):
    """A one-margin solver written outside the shipped family."""

    continuous_state: str = ""

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        raise NotImplementedError

    def _with_liquid_margin(self, margin: object) -> _CustomOneMargin:
        return replace(self, continuous_state=margin.state)  # ty: ignore[unresolved-attribute]


@dataclass(frozen=True, kw_only=True)
class _CustomTwoMargin(TwoMarginSolver):
    """A two-margin solver written outside the shipped family."""

    margins: tuple[object, object] | None = None

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        raise NotImplementedError

    def _with_margins(self, *, liquid: object, outer: object) -> _CustomTwoMargin:
        return replace(self, margins=(liquid, outer))


@dataclass(frozen=True, kw_only=True)
class _AnnotatedDCEGM(DCEGM):
    """A subclass of a shipped solver, which stays supported."""

    annotation: str = "mine"


def _model(*, solver: OneMarginSolver | GridSearch) -> Model:
    saving_regime = ConsumptionSavingsRegime(
        actions={"consumption": _ACTION_GRID},
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"done": next_wealth}},
        transition=next_regime,
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
        },
        active=lambda age: age == 0,
        solver=solver,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
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


def test_model_build_rejects_a_custom_one_margin_solver() -> None:
    """Building a model on a hand-written one-margin solver fails, naming it."""
    with pytest.raises(ModelInitializationError, match="not supported right now"):
        _model(solver=_CustomOneMargin())


def test_the_rejection_names_the_offending_solver_class() -> None:
    """The message identifies which class was refused."""
    with pytest.raises(ModelInitializationError, match="_CustomOneMargin"):
        _model(solver=_CustomOneMargin())


def test_a_custom_two_margin_solver_is_rejected() -> None:
    """The refusal covers the two-margin marker as well."""
    with pytest.raises(ModelInitializationError, match="not supported right now"):
        fail_if_solver_is_not_shipped(
            solver=_CustomTwoMargin(), regime_name="nested_saving"
        )


def test_a_subclass_of_a_shipped_solver_is_accepted() -> None:
    """Subclassing a shipped solver stays supported, so the guard stays silent."""
    fail_if_solver_is_not_shipped(
        solver=_AnnotatedDCEGM(savings_grid=_SAVINGS_GRID), regime_name="saving"
    )


def test_grid_search_is_accepted() -> None:
    """Grid search consumes no margin, so the guard does not apply to it."""
    fail_if_solver_is_not_shipped(solver=GridSearch(), regime_name="done")


def test_model_build_accepts_a_shipped_solver() -> None:
    """The guard does not fire on the solver the model is meant to use."""
    model = _model(solver=EGM(savings_grid=_SAVINGS_GRID))

    assert "saving" in model.user_regimes


def test_the_case_piece_solver_is_accepted() -> None:
    """`NBEGM` is a shipped one-margin solver, so the guard stays silent on it."""
    fail_if_solver_is_not_shipped(
        solver=NBEGM(savings_grid=_SAVINGS_GRID), regime_name="saving"
    )


def test_the_nested_case_piece_solver_is_accepted() -> None:
    """`NNBEGM` is a shipped two-margin solver, so the guard stays silent on it."""
    fail_if_solver_is_not_shipped(
        solver=NNBEGM(
            inner=NBEGM(savings_grid=_SAVINGS_GRID), outer_grid=_SAVINGS_GRID
        ),
        regime_name="nested_saving",
    )
