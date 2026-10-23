"""A collective regime may fold a shock; its value keeps the node axis.

Folding integrates a state's node axis out of the stored value by quadrature,
$V(x) = \\sum_k w_k V(x, \\varepsilon_k)$. A collective regime publishes a
different kind of cell alongside its value: where no action satisfies every
stakeholder's participation constraint it flags the cell as dissolving and
writes $-\\infty$, a sentinel meaning "not sustainable, take the outside option"
rather than a value on the household's own scale.

Quadrature over that sentinel is not an expectation: one dissolving node would
set the whole sum to $-\\infty$ and the flag would be set by any dissolving node.
So a collective regime that folds keeps the node axis in its stored value and
flag, and the average over nodes is taken by whoever reads it, after each node
is routed. Its solution is therefore the unfolded regime's solution.

Folding is a property of the shock rather than of the regime, so a singleton
regime declaring the identical process still folds it into its stored value.
"""

import dataclasses

import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    Model,
    NormalIIDProcess,
    Regime,
    ValueDependentConstraint,
)
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND
from tests.collective_fixtures import (
    AGES,
    DISCOUNT_FACTOR,
    FOLDED_SHOCK,
    FOLDING_SINGLETON_V_PERIOD_0,
    WAGE_GRID,
    CoupleRegimeId,
    Work,
    make_folding_collective_regimes,
    make_folding_singleton_model,
)
from tests.conftest import DECIMAL_PRECISION

_UNFOLDED_SHOCK = dataclasses.replace(FOLDED_SHOCK, fold=False)

_PARTICIPATION_PARAMS = {
    "couple": {
        "koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR},
        "participation_m": {"floor_m": 8.0},
    },
    "couple_terminal": {},
}


def test_folded_collective_value_keeps_the_node_axis() -> None:
    """The stored couple value carries one row per node of the folded shock."""
    regimes = make_folding_collective_regimes()
    model = Model(
        regimes=regimes,
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        initial_nodes={0: "couple"},
        edges={"couple": {"couple_terminal": 0}},
    )
    solution = model.solve(
        params={
            "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
            "couple_terminal": {},
        },
        log_level="off",
    )
    V = np.asarray(solution.values[0]["couple"])
    assert V.shape == (FOLDED_SHOCK.n_points, 2)


def _participation_solution(*, fold: bool):
    regimes = _folding_collective_regimes_with_participation(
        shock=FOLDED_SHOCK if fold else _UNFOLDED_SHOCK
    )
    model = Model(
        regimes=regimes,
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        initial_nodes={0: "couple"},
        edges={"couple": {"couple_terminal": (0, 1)}},
    )
    return model.solve(params=_PARTICIPATION_PARAMS, log_level="off")


def test_folded_collective_value_under_participation_equals_unfolded() -> None:
    """A folded couple stores what the unfolded couple stores, node by node."""
    folded = _participation_solution(fold=True).values
    unfolded = _participation_solution(fold=False).values
    np.testing.assert_array_equal(
        np.asarray(folded[0]["couple"]), np.asarray(unfolded[0]["couple"])
    )


def test_folded_collective_flag_dissolves_one_cell_only() -> None:
    """Exactly one (wage, node) cell dissolves; the flag is not reduced over nodes."""
    flag = np.asarray(
        _participation_solution(fold=True).replay_artifacts.project(DISSOLUTION_FLAG)[
            0
        ]["couple"]
    )
    assert int(flag.sum()) == 1


def test_singleton_regime_may_still_declare_a_folded_state() -> None:
    """A singleton regime folds its shock into a rank-zero value function.

    Work pays $10 + \\varepsilon$ against a leisure payoff of zero and wins at
    every node of the shock, so the stored value is
    $E[10 + \\varepsilon] + 0.95 \\cdot 4 = 13.8$.
    """
    model, params = make_folding_singleton_model()

    solution = model.solve(params=params, log_level="debug").values

    aaae(
        solution[0]["shocked"],
        FOLDING_SINGLETON_V_PERIOD_0,
        decimal=DECIMAL_PRECISION,
    )


def _folding_collective_regimes_with_participation(
    *, shock: NormalIIDProcess
) -> dict[str, Regime]:
    """Return a collective source over `shock` whose participation constraint binds.

    The wife's participation is slack throughout; the husband's demands a
    household consumption of at least `floor_m`, which working delivers at
    every wage node but the pairing of the low wage with the lowest shock
    node, and which leisure never delivers. That one cell of the state-action
    space is the only one with no sustainable action.
    """
    couple = Regime(
        states={"wage": WAGE_GRID, "wage_shock": shock},
        state_transitions={"wage": _fixed_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _shocked_wage_utility_f, "m": _consumption_m}
            )
        },
        constraints={
            "participation_m": ValueDependentConstraint(predicate=_participation_m)
        },
    )
    couple_terminal = Regime(
        states={},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _terminal_utility_f, "m": _terminal_utility_m}
            )
        },
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def _shocked_wage_utility_f(
    *, wage: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Wife: the shocked wage when working, her leisure value otherwise."""
    return work * (wage + wage_shock) + 30.0 * (1.0 - work)


def _consumption_m(
    *, wage: ContinuousState, wage_shock: FloatND, work: DiscreteAction
) -> FloatND:
    """Husband: household consumption, which only the wife's work produces."""
    return 2.0 * work * (wage + wage_shock)


def _participation_m(*, Q_m: FloatND, floor_m: float) -> BoolND:
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
