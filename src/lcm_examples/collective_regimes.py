"""Small collective-regime models for the documentation.

The first model isolates the household's shared argmax with stakeholder-specific
values, declared as a single `CollectiveUtility`. The second adds same-period outside
options through a current `Exit` participation rule, and an `ArrivalDecision`
that reads actual participating-policy support. The participation thresholds
allow slack without deducting it from the single payoff on dissolution. The
state and action spaces remain hand-checkable.

The arrival decision belongs to the incoming married edge. Its admitted branch
is the continuing collective regime; each stakeholder's fallback is that
stakeholder's own singleton regime.
"""

from collections.abc import Mapping
from typing import ClassVar

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    AgeRange,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.execution import ExecutionConfig
from lcm.exits import Exit, ExitTarget, RuleResult
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    IntND,
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


def get_params() -> dict[str, float]:
    """Return the shared parameter values used by both examples."""
    return {"discount_factor": _DISCOUNT_FACTOR}


def _shared_utility_f(
    *,
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _shared_utility_m(
    *,
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 2.0 * wage * work


def _next_shared_wage(work: DiscreteAction) -> ContinuousState:
    return 40.0 * work + 8.0 * (1.0 - work)


def get_shared_decision_model() -> Model:
    """Build a two-period collective model with one shared labor choice.

    Equal stakeholder weights select one household action. The value function keeps
    both stakeholders' values at that common argmax.
    """
    shared_utility = CollectiveUtility(
        utilities={"f": _shared_utility_f, "m": _shared_utility_m}
    )
    couple = Regime(
        states={"wage": _SHARED_WAGE_GRID},
        state_transitions={"wage": _next_shared_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": shared_utility},
    )
    couple_terminal = Regime(
        states={"wage": _SHARED_WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": shared_utility},
    )
    return Model(
        edges={"couple": {"couple_terminal": 0}},
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=SharedDecisionRegimeId,
        initial_nodes={0: "couple"},
        description="A shared household labor choice.",
    )


def _probability_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _zero_collective_utility(
    *,
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 0.0 * wage * work


def _participation_utility_f(
    *,
    wage: ContinuousState,
    work: DiscreteAction,
) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _participation_utility_m(
    *,
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
    *,
    Q_f: FloatND,
    V_single_f_ref: FloatND,
) -> BoolND:
    return Q_f >= V_single_f_ref - 0.5


def _participation_m(
    *,
    Q_m: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    return Q_m >= V_single_m_ref - 0.2


class _ParticipationRule:
    """Choose one common action satisfying both participation thresholds."""

    reads_outside_values: ClassVar[bool] = True

    def __call__(
        self,
        *,
        Q_f: FloatND,
        Q_m: FloatND,
        exit_payoffs: Mapping[str, Mapping[str, FloatND]],
        alternative_ids: IntND,
        pareto_weights: Mapping[str, FloatND],
    ) -> RuleResult:
        q_f, q_m = Q_f.reshape(-1), Q_m.reshape(-1)
        outside = exit_payoffs["dissolve"]
        kept = (
            jnp.isfinite(q_f)
            & jnp.isfinite(q_m)
            & _participation_f(Q_f=q_f, V_single_f_ref=outside["f"])
            & _participation_m(Q_m=q_m, V_single_m_ref=outside["m"])
        )
        # Mask zero-weight members before multiplying ordinary-infeasible -inf.
        objective = pareto_weights["f"] * jnp.where(
            pareto_weights["f"] == 0, 0.0, q_f
        ) + pareto_weights["m"] * jnp.where(pareto_weights["m"] == 0, 0.0, q_m)
        stay = jnp.argmax(jnp.where(kept, objective, -jnp.inf))
        return RuleResult(
            action=alternative_ids[jnp.where(jnp.any(kept), stay, q_f.size)]
        )


def _has_policy_support(target_support: BoolND) -> BoolND:
    return target_support


def get_dissolution_model(
    *,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> Model:
    """Build a three-period collective model with participation and dissolution.

    At wage two, neither household action satisfies both participation constraints.
    The arrival decision reads the couple's participating policy support on
    landing, and routes each stakeholder to their own singleton fallback where it
    is empty.
    """
    single_f = Regime(
        states={"wage": _DISSOLUTION_WAGE_GRID},
        functions={"utility": _single_f_value},
    )
    single_m = Regime(
        states={"wage": _DISSOLUTION_WAGE_GRID},
        functions={"utility": _single_m_value},
    )
    married_with_participation = Regime(
        states={"wage": _DISSOLUTION_WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _participation_utility_f, "m": _participation_utility_m}
            )
        },
        exits={
            "dissolve": Exit(
                targets={
                    "f": ExitTarget(
                        regime="single_f", projection={"wage": _identity_wage}
                    ),
                    "m": ExitTarget(
                        regime="single_m", projection={"wage": _identity_wage}
                    ),
                },
                rule=_ParticipationRule(),
            )
        },
    )
    # The scheduled target is the continuing couple; support is queried only
    # after landing there, before the household or its fallback acts.
    married_law = ByAge(
        cases={
            AgeRange(exclusive_stop=1): {
                "married_with_participation": StochasticTransition(
                    func=_probability_one
                )
            }
        }
    )
    married_arrivals = {
        "married_with_participation": ArrivalDecision(
            predicate=_has_policy_support,
            reads={
                "target_support": ArrivalRead(
                    regime="married_with_participation",
                    stakeholder="f",
                    kind="Participating",
                    field="support",
                    projection={"wage": _identity_wage},
                )
            },
            routes={
                "f": StakeholderRoute(
                    target_stakeholder="f",
                    fallback=ProjectedRegimeValue(
                        regime="single_f",
                        projection={"wage": _identity_wage},
                    ),
                ),
                "m": StakeholderRoute(
                    target_stakeholder="m",
                    fallback=ProjectedRegimeValue(
                        regime="single_m",
                        projection={"wage": _identity_wage},
                    ),
                ),
            },
        )
    }
    married = Regime(
        states={"wage": _DISSOLUTION_WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _zero_collective_utility, "m": _zero_collective_utility}
            )
        },
    )
    married_terminal = Regime(
        states={"wage": _DISSOLUTION_WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _zero_collective_utility, "m": _zero_collective_utility}
            )
        },
    )
    return Model(
        edges={
            "married": Transition(law=married_law, arrivals=married_arrivals),
            "married_with_participation": {"married_terminal": 1},
        },
        regimes={
            "married": married,
            "married_with_participation": married_with_participation,
            "married_terminal": married_terminal,
            "single_f": single_f,
            "single_m": single_m,
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={0: "married"},
        description="Participation constraints and an arrival dissolution decision.",
        execution_config=execution_config,
    )
