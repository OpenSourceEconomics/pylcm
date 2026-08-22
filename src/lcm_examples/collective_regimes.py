"""Small collective-regime models for the documentation.

The first model isolates the household's shared argmax with stakeholder-specific
values. The second adds same-period outside options, participation constraints, and a
gated dissolution edge while keeping the state and action spaces hand-checkable.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    """Household labor choice."""

    leisure: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class SharedDecisionRegimeId:
    """Regime ids for the shared-decision model."""

    couple: ScalarInt
    couple_terminal: ScalarInt


@categorical(ordered=False)
class DissolutionRegimeId:
    """Regime ids for the participation-and-dissolution model."""

    married: ScalarInt
    married_with_participation: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


_DISCOUNT_FACTOR = 0.95

_SHARED_WAGE_GRID = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)
_DISSOLUTION_WAGE_GRID = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
_MIDDLE_WAGE_LOWER = 1.5
_MIDDLE_WAGE_UPPER = 2.5
_DISSOLUTION_AGE = 2


def get_params() -> dict[str, float]:
    """Return the shared parameter values used by both examples."""
    return {"discount_factor": _DISCOUNT_FACTOR}


def _shared_utility_f(
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _shared_utility_m(
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 2.0 * wage * work


def _next_shared_wage(work: DiscreteAction) -> ContinuousState:
    return 40.0 * work + 8.0 * (1.0 - work)


def _to_shared_terminal() -> ScalarInt:
    return SharedDecisionRegimeId.couple_terminal


def get_shared_decision_model() -> Model:
    """Build a two-period collective model with one shared labor choice.

    Equal stakeholder weights select one household action. The value function keeps
    both stakeholders' values at that common argmax.
    """
    couple = Regime(
        transition=_to_shared_terminal,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _SHARED_WAGE_GRID},
        state_transitions={"wage": _next_shared_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _shared_utility_f,
            "utility_m": _shared_utility_m,
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _SHARED_WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _shared_utility_f,
            "utility_m": _shared_utility_m,
        },
    )
    return Model(
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=SharedDecisionRegimeId,
        description="A shared household labor choice.",
    )


def _probability_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _zero_collective_utility(
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 0.0 * wage * work


def _participation_utility_f(
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _participation_utility_m(
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 0.5 * (1.0 - work) + wage * work


def _single_f_value(wage: ContinuousState) -> FloatND:
    return jnp.where(
        (wage > _MIDDLE_WAGE_LOWER) & (wage < _MIDDLE_WAGE_UPPER),
        5.5,
        1.5,
    )


def _single_m_value(wage: ContinuousState) -> FloatND:
    return 1.0 + 0.0 * wage


def _participation_f(
    Q_f: FloatND,
    V_single_f_ref: FloatND,
) -> BoolND:
    return Q_f >= V_single_f_ref - 0.5


def _participation_m(
    Q_m: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    return Q_m >= V_single_m_ref - 0.2


def _no_dissolution(D_target: BoolND) -> BoolND:
    return ~D_target


def get_dissolution_model() -> Model:
    """Build a three-period collective model with participation and dissolution.

    At wage two, neither household action satisfies both participation constraints.
    The target regime publishes a dissolution flag, and the source's gated edge routes
    each stakeholder to their own singleton fallback.
    """
    single_f = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _DISSOLUTION_WAGE_GRID},
        functions={"utility": _single_f_value},
    )
    single_m = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _DISSOLUTION_WAGE_GRID},
        functions={"utility": _single_m_value},
    )
    married_with_participation = Regime(
        transition={"married_terminal": MarkovTransition(_probability_one)},
        active=lambda age: (age >= 1) & (age < _DISSOLUTION_AGE),
        stakeholders=("f", "m"),
        states={"wage": _DISSOLUTION_WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _participation_utility_f,
            "utility_m": _participation_utility_m,
        },
        value_constraints={
            "participation_f": _participation_f,
            "participation_m": _participation_m,
        },
        same_period_refs={
            "V_single_f_ref": SamePeriodRef(
                regime="single_f",
                projection={"wage": _identity_wage},
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m",
                projection={"wage": _identity_wage},
            ),
        },
    )
    married = Regime(
        transition={"married_with_participation": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _DISSOLUTION_WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _zero_collective_utility,
            "utility_m": _zero_collective_utility,
        },
        gated_edges={
            "married_with_participation": GatedEdge(
                gate=_no_dissolution,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_m",
                            projection={"wage": _identity_wage},
                        ),
                    ),
                },
            )
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= _DISSOLUTION_AGE,
        stakeholders=("f", "m"),
        states={"wage": _DISSOLUTION_WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _zero_collective_utility,
            "utility_m": _zero_collective_utility,
        },
    )
    return Model(
        regimes={
            "married": married,
            "married_with_participation": married_with_participation,
            "married_terminal": married_terminal,
            "single_f": single_f,
            "single_m": single_m,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=DissolutionRegimeId,
        description="Participation constraints and a gated dissolution edge.",
    )
