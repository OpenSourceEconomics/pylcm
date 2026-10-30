"""The marriage-market example on value grids, so every landing is a grid node.

`lcm_examples.collective_household` lands between wealth nodes, which is where a stored
participation surface and an exact read at the landing differ. This twin keeps the
example's regimes, laws, exit rule and participation check declarations, but declares
wealth and the consumption share as `DiscreteGrid(values=...)` and rounds every next
wealth onto its target's nodes. Couple nodes are exactly twice the single nodes. A value
grid is shared by every regime declaring it, so couples and singles use one wealth grid,
and half of a couple's wealth is rounded onto it too. Every participation read, fallback
and exit projection therefore lands on a node. Shares and nodes are dyadic, so with a
zero interest rate every rounding is decided exactly at either precision.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    Model,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
)
from lcm.exits import Exit, ExitTarget
from lcm.transition import StochasticTransition, Transition
from lcm.typing import DiscreteAction, DiscreteState, FloatND, UserParamsNode
from lcm_examples import collective_household as household

_START = 1.0
_STEP = 2.0
_SHARES = np.array([0.25, 0.5, 1.0])


def _snap(*, value: FloatND, start: float, step: float, n_points: int) -> FloatND:
    """Round a wealth onto the nearest node of `start + step * arange(n_points)`."""
    index = jnp.clip(jnp.rint((value - start) / step), 0, n_points - 1)
    return start + step * index


def _identity(wealth: DiscreteState) -> DiscreteState:
    return wealth


def _consent(
    *, stakeholder: str, fallback: str, half: Callable[..., DiscreteState]
) -> dict[str, ParticipationCheck]:
    """A single's consent to a household, as in the example."""
    route = "her" if stakeholder == "f" else "his"
    return {
        "couple": ParticipationCheck(
            predicate=household._mutual_consent,
            routes={
                route: StakeholderRoute(
                    target_stakeholder=stakeholder,
                    fallback=ProjectedRegimeValue(
                        regime=fallback, projection={"wealth": half}
                    ),
                )
            },
            reads={
                "V_target_f": ParticipationRead(
                    regime="couple",
                    stakeholder="f",
                    kind="Participating",
                    projection={"wealth": _identity},
                ),
                "V_target_m": ParticipationRead(
                    regime="couple",
                    stakeholder="m",
                    kind="Participating",
                    projection={"wealth": _identity},
                ),
                "V_alone_f": ParticipationRead(
                    regime="single_f", kind="Raw", projection={"wealth": half}
                ),
                "V_alone_m": ParticipationRead(
                    regime="single_m", kind="Raw", projection={"wealth": half}
                ),
            },
        )
    }


def _couple_participation(
    *, half: Callable[..., DiscreteState]
) -> dict[str, ParticipationCheck]:
    """A couple's continuation needs a supported household, as in the example."""
    return {
        "couple": ParticipationCheck(
            predicate=household._has_policy_support,
            reads={
                "target_support": ParticipationRead(
                    regime="couple",
                    stakeholder="f",
                    kind="Participating",
                    field="support",
                    projection={"wealth": _identity},
                )
            },
            routes={
                member: StakeholderRoute(
                    target_stakeholder=member,
                    fallback=ProjectedRegimeValue(
                        regime=f"single_{member}", projection={"wealth": half}
                    ),
                )
                for member in ("f", "m")
            },
        )
    }


def get_model(*, n_periods: int, n_points: int = 4) -> Model:
    """Build the value-grid twin of the marriage market.

    Args:
        n_periods: Number of lifecycle periods. The last one is terminal.
        n_points: Number of wealth nodes of every regime.

    Returns:
        The model, solved with `get_params()`.

    """
    nodes = _START + _STEP * np.arange(n_points)

    def snap(value: FloatND) -> FloatND:
        return _snap(value=value, start=_START, step=_STEP, n_points=n_points)

    def consumption(
        *, wealth: DiscreteState, consumption_share: DiscreteAction
    ) -> FloatND:
        return consumption_share * wealth

    def snapped(law: Callable[..., FloatND]) -> Callable[..., DiscreteState]:
        def next_wealth(
            *, wealth: DiscreteState, consumption: FloatND, interest_rate: float
        ) -> DiscreteState:
            return snap(
                law(wealth=wealth, consumption=consumption, interest_rate=interest_rate)
            )

        return next_wealth

    next_single_f = snapped(household._next_single_wealth_f)
    next_single_m = snapped(household._next_single_wealth_m)
    pooled = snapped(household._pooled_wealth)
    next_couple = snapped(household._next_couple_wealth)

    def half(wealth: DiscreteState) -> DiscreteState:
        return snap(0.5 * wealth)

    single_wealth = DiscreteGrid(values=nodes)
    couple_wealth = single_wealth
    share = DiscreteGrid(values=_SHARES)
    last_age = n_periods - 1
    probability = household._transition_probabilities(last_age=last_age)

    def law(*, cells: dict[str, str], terminal: str) -> ByAge:
        return household._until_last_age(
            {
                name: StochasticTransition(func=probability[key])
                for name, key in cells.items()
            },
            terminal=terminal,
            last_age=last_age,
        )

    def single(
        *, next_wealth: Callable[..., DiscreteState], stays: tuple[str, ...]
    ) -> Regime:
        return Regime(
            states={"wealth": single_wealth},
            state_transitions={
                "wealth": {"couple": pooled, **dict.fromkeys(stays, next_wealth)}
            },
            actions={"consumption_share": share},
            functions={
                "consumption": consumption,
                "utility": household._single_utility,
            },
        )

    def single_terminal() -> Regime:
        return Regime(
            states={"wealth": single_wealth},
            actions={"consumption_share": share},
            functions={
                "consumption": consumption,
                "utility": household._terminal_utility,
            },
        )

    couple = Regime(
        states={"wealth": couple_wealth},
        state_transitions={"wealth": next_couple},
        actions={"consumption_share": share},
        functions={
            "consumption": consumption,
            "utility": CollectiveUtility(
                utilities={
                    "f": household._couple_utility_f,
                    "m": household._couple_utility_m,
                }
            ),
        },
        exits={
            "dissolve": Exit(
                targets={
                    member: ExitTarget(
                        regime=f"single_{member}", projection={"wealth": half}
                    )
                    for member in ("f", "m")
                },
                rule=household._ParticipationRule(),
            )
        },
    )
    couple_terminal = Regime(
        states={"wealth": couple_wealth},
        actions={"consumption_share": share},
        functions={
            "consumption": consumption,
            "utility": CollectiveUtility(
                utilities={
                    "f": household._terminal_utility_collective,
                    "m": household._terminal_utility_collective,
                }
            ),
        },
    )
    return Model(
        edges={
            "couple": Transition(
                law=law(
                    cells={
                        "couple": "stays_married",
                        "couple_terminal": "reaches_last_age",
                    },
                    terminal="couple_terminal",
                ),
                participation=_couple_participation(half=half),
            ),
            "single_f": Transition(
                law=law(
                    cells={
                        "couple": "meets_a_partner",
                        "single_f": "meets_nobody",
                        "single_f_terminal": "reaches_last_age",
                    },
                    terminal="single_f_terminal",
                ),
                participation=_consent(stakeholder="f", fallback="single_f", half=half),
            ),
            "single_m": Transition(
                law=law(
                    cells={
                        "couple": "meets_a_partner",
                        "single_m": "meets_nobody",
                        "single_m_terminal": "reaches_last_age",
                    },
                    terminal="single_m_terminal",
                ),
                participation=_consent(stakeholder="m", fallback="single_m", half=half),
            ),
        },
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
            "single_f": single(
                next_wealth=next_single_f, stays=("single_f", "single_f_terminal")
            ),
            "single_m": single(
                next_wealth=next_single_m, stays=("single_m", "single_m_terminal")
            ),
            "single_f_terminal": single_terminal(),
            "single_m_terminal": single_terminal(),
        },
        ages=AgeGrid(start=0, inclusive_stop=last_age, step="Y"),
        regime_id_class=household.RegimeId,
        initial_nodes={0: ("couple", "single_f", "single_m")},
        execution_config=ExecutionConfig(),
    )


def get_params() -> dict[str, UserParamsNode]:
    """Return the example's params with a zero interest rate, so rounding is exact."""
    return household.get_params(interest_rate=0.0)
