"""Construction-time contracts for collective regimes."""

import inspect

import jax.numpy as jnp
import pytest

from _lcm.regime_law import bind_regime_law
from _lcm.user_regime_validation import validate_regime_law
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import DCEGM, EGM, OneMarginSolver
from lcm.typing import (
    ContinuousAction,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

# Shared building blocks: a stripped-down couples problem, in which the two
# members differ only in their disutility of work.


@categorical(ordered=True)
class LaborSupply:
    do_not_work: ScalarInt
    work: ScalarInt


def _utility_f(
    *,
    consumption: ContinuousAction,
    labor_supply_f: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Wife's per-member felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.3 * (labor_supply_f == LaborSupply.work)
        + match_quality
    )


def _utility_m(
    *,
    consumption: ContinuousAction,
    labor_supply_m: DiscreteAction,
    match_quality: float,
) -> FloatND:
    """Husband's per-member felicity (illustrative)."""
    return (
        jnp.log(consumption)
        - 0.5 * (labor_supply_m == LaborSupply.work)
        + match_quality
    )


_WEALTH = LinSpacedGrid(start=1, stop=10, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=1, stop=5, n_points=5)


@categorical(ordered=False)
class _CoupleRegimeId:
    """Regime ids of the minimal two-regime couples model."""

    married: ScalarInt
    widowed: ScalarInt


def test_declaring_non_terminal_members_constructs() -> None:
    """A non-terminal collective regime constructs.

    Declaring `members` and binding a regime transition law is accepted: the
    per-member contract is checked at construction, and the constructed
    regime keeps both its member tuple and its non-terminal status. What a
    collective regime is allowed to route to, and how it solves, is pinned in
    `tests/regime_building/test_nonterminal_collective_solve.py`.
    """

    def _some_transition(_age: float) -> int:
        return 0

    regime = Regime(
        states={"wealth": _WEALTH},
        actions={"labor_supply_f": DiscreteGrid(category_class=LaborSupply)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    law = bind_regime_law(_some_transition)
    validate_regime_law(regime, law=law)
    assert regime.members == ("f", "m")
    assert not law.terminal


def test_terminal_members_without_per_member_utility_is_rejected() -> None:
    """A collective regime must carry a body for every member it names.

    A household may leave a member's body to arrive from the model level,
    so a bare `Regime` cannot tell whether one ever will. Completeness is a
    property of the merged regime and is reported when the model finalizes it.
    """
    married = Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "labor_supply_f": DiscreteGrid(category_class=LaborSupply),
            "labor_supply_m": DiscreteGrid(category_class=LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    widowed = Regime(
        states={"wealth": _WEALTH},
        actions={
            "labor_supply_f": DiscreteGrid(category_class=LaborSupply),
            "consumption": _CONSUMPTION,
        },
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": None})
        },
    )

    with pytest.raises(ModelInitializationError, match="per-member utility"):
        Model(
            regimes={"married": married, "widowed": widowed},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=_CoupleRegimeId,
            initial_nodes={0: "married"},
            edges={"married": {"widowed": 0}},
        )


def test_a_plain_utility_declares_the_one_member_self() -> None:
    """A regime that declares a plain utility has the one member `"self"`."""

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    regime = Regime(
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        functions={"utility": utility},
    )
    assert regime.members == ("self",)


def test_members_is_not_a_constructor_parameter() -> None:
    """A regime's members are declared in its utility, and only there."""
    assert "members" not in inspect.signature(Regime).parameters


@pytest.mark.parametrize(
    "solver",
    [EGM(savings_grid=_WEALTH), DCEGM(savings_grid=_WEALTH)],
    ids=["EGM", "DCEGM"],
)
def test_household_with_a_one_member_solver_is_refused(
    solver: OneMarginSolver,
) -> None:
    """A solver that solves one member only refuses a household of two.

    The message names the solver and the household's members.
    """

    def _utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    with pytest.raises(
        NotImplementedError,
        match=rf"{type(solver).__name__}.*\('f', 'm'\)",
    ):
        ConsumptionSavingsRegime(
            states={"wealth": _WEALTH},
            actions={"consumption": _CONSUMPTION},
            state_transitions={"wealth": lambda wealth: wealth},
            functions={
                "utility": CollectiveUtility(utilities={"f": _utility, "m": _utility})
            },
            solver=solver,
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="wealth",
                post_decision_state="savings",
            ),
        )
