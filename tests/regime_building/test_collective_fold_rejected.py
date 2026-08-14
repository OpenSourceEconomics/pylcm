"""A collective regime may not declare a folded state; a singleton still may.

Folding integrates a state's node axis out of the stored value by quadrature,
$V(x) = \\sum_k w_k V(x, \\varepsilon_k)$, so the stored value carries no axis
for that state. A collective regime publishes a different kind of cell
alongside its value: where no action satisfies every stakeholder's
participation constraint it flags the cell as dissolving and writes $-\\infty$,
a sentinel meaning "not sustainable, take the outside option" rather than a
value on the household's own scale.

Quadrature over that sentinel is not an expectation. Because
$w \\cdot -\\infty = -\\infty$ for any positive weight, a single dissolving node
sets the whole sum to $-\\infty$, while the flag itself is set by any dissolving
node: a state that dissolves with small probability is stored as dissolving with
certainty. The combination is therefore rejected when the model is built, with
the regime and the state both named so the author can act on it.

Folding is a property of the shock rather than of the regime, so a singleton
regime declaring the identical process keeps solving.
"""

import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import DiscreteGrid, Model, Regime
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.collective_fixtures import (
    AGES,
    FOLDED_SHOCK,
    FOLDING_SINGLETON_V_PERIOD_0,
    WAGE_GRID,
    CoupleRegimeId,
    Work,
    make_folding_collective_regimes,
    make_folding_singleton_model,
)
from tests.conftest import DECIMAL_PRECISION


@pytest.mark.parametrize("required_name", ["couple", "wage_shock"])
def test_collective_regime_declaring_a_folded_state_is_rejected(
    required_name: str,
) -> None:
    """Building a collective regime over a folded state is refused by name.

    The refusal names both the regime and the folded state, because an author
    whose model carries several regimes and several shocks cannot otherwise
    tell which declaration to drop.
    """
    with pytest.raises(ModelInitializationError, match=required_name):
        Model(
            regimes=make_folding_collective_regimes(),
            ages=AGES,
            regime_id_class=CoupleRegimeId,
        )


def test_collective_fold_under_a_participation_constraint_is_rejected() -> None:
    """A folded collective regime is refused however its dissolution arises.

    A participation constraint is what makes a cell dissolve at some
    quadrature nodes and not others, which is the case the sentinel cannot
    survive being averaged: the household here is sustainable at four of the
    shock's five nodes and would be stored as dissolving at all of them.
    """
    with pytest.raises(ModelInitializationError, match="wage_shock"):
        Model(
            regimes=_folding_collective_regimes_with_participation(),
            ages=AGES,
            regime_id_class=CoupleRegimeId,
        )


def test_singleton_regime_may_still_declare_a_folded_state() -> None:
    """A singleton regime folds its shock into a rank-zero value function.

    Work pays $10 + \\varepsilon$ against a leisure payoff of zero and wins at
    every node of the shock, so the stored value is
    $E[10 + \\varepsilon] + 0.95 \\cdot 4 = 13.8$.
    """
    model, params = make_folding_singleton_model()

    solution = model.solve(params=params, log_level="debug")

    aaae(
        solution[0]["shocked"],
        FOLDING_SINGLETON_V_PERIOD_0,
        decimal=DECIMAL_PRECISION,
    )


def _folding_collective_regimes_with_participation() -> dict[str, Regime]:
    """Return a folded collective source whose participation constraint binds.

    The wife's participation is slack throughout; the husband's demands a
    household consumption of at least `floor_m`, which working delivers at
    every wage node but the pairing of the low wage with the lowest shock
    node, and which leisure never delivers. That one cell of the state-action
    space is the only one with no sustainable action.
    """
    couple = Regime(
        transition=_next_couple_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": WAGE_GRID, "wage_shock": FOLDED_SHOCK},
        state_transitions={"wage": _fixed_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _shocked_wage_utility_f, "utility_m": _consumption_m},
        value_constraints={"participation_m": _participation_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _terminal_utility_f, "utility_m": _terminal_utility_m},
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def _shocked_wage_utility_f(
    wage: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Wife: the shocked wage when working, her leisure value otherwise."""
    return work * (wage + wage_shock) + 30.0 * (1.0 - work)


def _consumption_m(
    wage: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Husband: household consumption, which only the wife's work produces."""
    return 2.0 * work * (wage + wage_shock)


def _participation_m(Q_m: FloatND, floor_m: float) -> BoolND:
    """Husband stays only where his own action value clears his outside option."""
    return Q_m >= floor_m


def _terminal_utility_f(work: DiscreteAction) -> FloatND:
    """Wife's terminal payoff: 10 for work, 0 for leisure."""
    return 10.0 * work


def _terminal_utility_m(work: DiscreteAction) -> FloatND:
    """Husband's terminal payoff: 5 for leisure, 0 for work."""
    return 5.0 * (1.0 - work)


def _fixed_wage(wage: ContinuousState) -> ContinuousState:
    """Wage law: the wage a household starts with is the wage it keeps."""
    return wage


def _next_couple_regime() -> ScalarInt:
    """Regime transition: `couple` becomes `couple_terminal` with probability one."""
    return CoupleRegimeId.couple_terminal
