"""A folded shock may not reach a law of motion, whatever route it takes.

`fold=True` integrates an IID shock out of the stored value by quadrature, so
the value a period publishes no longer says which node was realized. A law of
motion that prices the shock per node therefore contradicts the value it is
paired with, and pylcm refuses the combination when the model is built.

The refusal is a property of the dependency graph, not of how the reading
function is spelled: a plain helper and an `AgeSpecializedFunction` whose
per-age closure reads the shock are the same read, and both are refused.
"""

from collections.abc import Hashable

import pytest

from lcm import (
    AgeSpecializedFunction,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
)
from lcm.exceptions import PyLCMError
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.collective_fixtures import AGES, FOLDED_SHOCK, ShockRegimeId, Work

# The two-node liquid state whose law of motion does the offending read.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=2)

# The rejection message, which names the folded state and the conflict.
FOLD_TRANSITION_MESSAGE = (
    r"fold=True on state\(s\) \['wage_shock'\] conflicts with a next-period "
    r"transition that reads the shock's realized value"
)


@pytest.mark.parametrize("route", ["plain_helper", "age_specialized_helper"])
def test_folded_shock_read_by_a_law_of_motion_is_rejected(route: str) -> None:
    """Building a model whose law of motion reads a folded shock raises.

    `next_wealth` reads `net_wage`, and `net_wage` reads the folded
    `wage_shock`. Whether `net_wage` is an ordinary function or an
    `AgeSpecializedFunction` whose per-age closure does the reading, the model
    is refused and the message names `wage_shock`.
    """
    net_wage = (
        _plain_net_wage
        if route == "plain_helper"
        else AgeSpecializedFunction(
            build=_build_net_wage, signature=_net_wage_signature
        )
    )
    with pytest.raises(PyLCMError, match=FOLD_TRANSITION_MESSAGE):
        _build_model(net_wage=net_wage)


def _build_model(*, net_wage: UserFunction | AgeSpecializedFunction) -> Model:
    """Build the two-regime model whose wealth law reads `net_wage`.

    Args:
        net_wage: The wage function `next_wealth` reads. Both forms in use here
            read the folded `wage_shock`.

    Returns:
        The built model, for the callers where building succeeds.

    """
    shocked = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wealth": WEALTH_GRID, "wage_shock": FOLDED_SHOCK},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility, "net_wage": net_wage},
        state_transitions={"wealth": _next_wealth},
    )
    shocked_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": WEALTH_GRID},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"shocked": shocked, "shocked_terminal": shocked_terminal},
        ages=AGES,
        regime_id_class=ShockRegimeId,
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
    wealth: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Return wealth plus the shocked wage when working, wealth otherwise."""
    return wealth + work * (10.0 + wage_shock)


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    """Return the terminal payoff: wealth itself."""
    return wealth


def _next_wealth(wealth: ContinuousState, net_wage: FloatND) -> ContinuousState:
    """Return next period's wealth: today's wealth plus the net wage."""
    return wealth + net_wage


def _next_regime() -> ScalarInt:
    """Return the target regime: `shocked` becomes `shocked_terminal`."""
    return ShockRegimeId.shocked_terminal
