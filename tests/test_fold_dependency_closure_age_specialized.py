"""A folded shock may not reach a regime transition, whatever route it takes.

`fold=True` integrates an IID shock out of the stored value by quadrature. A
law of motion may read the shock, since it forms the next state at each node
before the average. The probability of reaching a regime may not, and pylcm
refuses that combination when the model is built.

The refusal is a property of the dependency graph: a helper that reads the
shock is the same read as the shock itself. An `AgeSpecializedFunction` cannot
reach a regime transition at all, so the law of motion is where its per-age
closure may read the shock, and that read builds.
"""

from collections.abc import Hashable

import jax.numpy as jnp
import pytest

from lcm import (
    AgeSpecializedFunction,
    ByAge,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
)
from lcm.exceptions import PyLCMError
from lcm.transition import StochasticTransition
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    UserFunction,
)
from tests.collective_fixtures import AGES, FOLDED_SHOCK, ShockRegimeId, Work

# The two-node liquid state whose law of motion does the offending read.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=2)

# The rejection message, which names the folded state and the conflict.
FOLD_TRANSITION_MESSAGE = (
    r"fold=True on state\(s\) \['wage_shock'\] conflicts with a regime "
    r"transition that reads the shock's realized value"
)


def test_folded_shock_read_by_a_regime_transition_is_rejected() -> None:
    """Building a model whose regime transition reads a folded shock raises.

    The transition probability reads `net_wage`, and `net_wage` reads the
    folded `wage_shock`, so the model is refused and the message names
    `wage_shock`.
    """
    with pytest.raises(PyLCMError, match=FOLD_TRANSITION_MESSAGE):
        _build_model(net_wage=_plain_net_wage, law_reads_wage=True)


def test_age_specialized_helper_may_feed_a_law_of_motion() -> None:
    """A per-age closure reading the folded shock may feed the wealth law.

    The wealth law forms next wealth at each node of the shock, so the model
    builds and stores one value per wealth node, the shock folded out.
    """
    model = _build_model(
        net_wage=AgeSpecializedFunction(
            build=_build_net_wage, signature=_net_wage_signature
        ),
        law_reads_wage=False,
    )
    solution = model.solve(
        params={
            "shocked": {"koopmans_aggregator": {"discount_factor": 0.95}},
            "shocked_terminal": {},
        },
        log_level="off",
    )
    assert solution.values[0]["shocked"].shape == (2,)


def _build_model(
    *, net_wage: UserFunction | AgeSpecializedFunction, law_reads_wage: bool
) -> Model:
    """Build the two-regime model whose regime law or wealth law reads `net_wage`.

    Args:
        net_wage: The wage function, which reads the folded `wage_shock`.
        law_reads_wage: Whether the regime transition reads `net_wage`; the
            wealth law reads it otherwise.

    Returns:
        The built model, for the callers where building succeeds.

    """
    shocked = Regime(
        states={"wealth": WEALTH_GRID, "wage_shock": FOLDED_SHOCK},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility, "net_wage": net_wage},
        state_transitions={"wealth": _hold_wealth if law_reads_wage else _next_wealth},
    )
    shocked_terminal = Regime(
        states={"wealth": WEALTH_GRID},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"shocked": shocked, "shocked_terminal": shocked_terminal},
        ages=AGES,
        regime_id_class=ShockRegimeId,
        initial_nodes={0: "shocked"},
        edges={
            "shocked": Transition(
                targets={"shocked_terminal": 0},
                law=ByAge(
                    cases={
                        0: {
                            "shocked_terminal": StochasticTransition(
                                func=_reach_terminal
                                if law_reads_wage
                                else _reach_terminal_always
                            )
                        }
                    }
                ),
            )
        },
    )


def _plain_net_wage(wage_shock: FloatND) -> FloatND:
    """Return the base wage plus the realized shock."""
    return 10.0 + wage_shock


def _build_net_wage(age: float) -> UserFunction:
    """Return the wage function of `age`, whose base wage grows with age."""
    base_wage = 10.0 + age

    def net_wage(wage_shock: FloatND) -> FloatND:
        return base_wage + wage_shock

    return net_wage


def _net_wage_signature(age: float) -> Hashable:
    """Return the dedup key of `age`'s wage closure: the age it closes over."""
    return float(age)


def _utility(
    *, wealth: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Return wealth plus the shocked wage when working, wealth otherwise."""
    return wealth + work * (10.0 + wage_shock)


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    """Return the terminal payoff: wealth itself."""
    return wealth


def _hold_wealth(wealth: ContinuousState) -> ContinuousState:
    """Return next period's wealth: the wealth the household holds."""
    return wealth


def _next_wealth(*, wealth: ContinuousState, net_wage: FloatND) -> ContinuousState:
    """Return next period's wealth: today's wealth plus the net wage."""
    return wealth + net_wage


def _reach_terminal_always(age: FloatND) -> FloatND:
    """Return probability one."""
    return jnp.ones_like(age, dtype=float)


def _reach_terminal(net_wage: FloatND) -> FloatND:
    """Return probability one, read through the shocked wage."""
    return jnp.ones_like(net_wage)
