"""A terminal collective regime may declare value-aware feasibility.

A household's last period is a decision like any other: each partner compares
what the household offers against what leaving offers, and a couple whose final
allocation satisfies neither participation constraint does not stay together.
Refusing `value_constraints` there would force the modeller to invent an extra
non-terminal period to carry the last participation decision.

What a terminal regime publishes is a **flag**, not a resolved outcome. A cell
whose feasible set is empty carries `D = True` and the `-inf` sentinel value:
there is no continuation to route into, so the engine has no outside option to
substitute. Reading the flag — and deciding what a dissolved terminal household
is worth — is the model's business, and a caller consuming the value without it
consumes the sentinel.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    IrregSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

_WAGE = IrregSpacedGrid(points=(1.0, 2.0, 3.0))


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt
    single_f_terminal: ScalarInt


def _zero(wage: ContinuousState) -> FloatND:
    return jnp.zeros_like(wage)


def _wage_for_her(wage: ContinuousState) -> FloatND:
    return wage


def _twice_wage_for_him(wage: ContinuousState) -> FloatND:
    return 2.0 * wage


def _outside_option(*, wage: ContinuousState, outside_option: float) -> FloatND:
    """What leaving is worth to her — a parameter, so a test can move it."""
    return outside_option + jnp.zeros_like(wage)


def _participation_f(*, Q_f: FloatND, V_single_f: FloatND) -> BoolND:
    """She stays only where the household beats what leaving is worth."""
    return Q_f >= V_single_f


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _make_model(*, participation: bool) -> Model:
    couple = Regime(
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        functions={"utility": CollectiveUtility(utilities={"f": _zero, "m": _zero})},
    )
    couple_terminal = Regime(
        states={"wage": _WAGE},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _wage_for_her, "m": _twice_wage_for_him}
            )
        },
        constraints=(
            {
                "participation_f": ValueDependentConstraint(
                    predicate=_participation_f,
                    references={
                        "V_single_f": ProjectedRegimeValue(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        )
                    },
                )
            }
            if participation
            else {}
        ),
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _outside_option},
    )
    return Model(
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
            "single_f_terminal": single_f_terminal,
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        # The outside option is a value dependency, not an admissible start.
        initial_nodes={0: "couple"},
        edges={"couple": {"couple_terminal": 0}},
    )


def _params(*, outside_option: float = 3.0, participation: bool = True) -> dict:
    params = {
        "couple": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "couple_terminal": {},
    }
    if participation:
        params["single_f_terminal"] = {"utility": {"outside_option": outside_option}}
    return params


@pytest.mark.parametrize("participation", [False, True])
def test_a_terminal_value_reference_does_not_become_a_start(
    *, participation: bool
) -> None:
    """Reference demand enlarges solved values only, never admitted/visited pairs."""
    model = _make_model(participation=participation)
    physical = frozenset({(0, "couple"), (1, "couple_terminal")})
    expected_values = physical
    if participation:
        expected_values |= {(1, "single_f_terminal")}

    assert model.graph.initial_nodes == frozenset({(0, "couple")})
    assert model.reachability.visited_nodes == physical
    assert model.reachability.nodes == expected_values


@pytest.mark.parametrize("participation", [False, True])
def test_the_solution_stores_exactly_the_value_demand(*, participation: bool) -> None:
    """Solved values cover the physical pairs plus a referenced outside option only."""
    model = _make_model(participation=participation)
    expected = {(0, "couple"), (1, "couple_terminal")}
    if participation:
        expected |= {(1, "single_f_terminal")}

    solution = model.solve(params=_params(participation=participation), log_level="off")

    assert {
        (period, name) for period in solution.values for name in solution.values[period]
    } == expected


@pytest.mark.parametrize("anchor", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_terminal_flags_and_sentinels_at_adjacent_outside_options(
    *, anchor: float, direction: int
) -> None:
    """At each wage node and its adjacent floats, `D` is exactly `wage < option`.

    The outside option sits on a wage node or one representable step to either
    side in the active precision, so a tolerance or an off-by-one comparison in
    the participation predicate shows up as a flipped flag. Dissolved cells carry
    the `-inf` sentinel pair; intact ones carry `(wage, 2 * wage)`.
    """
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    bound = dtype.type(anchor)
    if direction:
        bound = np.nextafter(bound, dtype.type(np.inf if direction > 0 else -np.inf))
    model = _make_model(participation=True)

    solution = model.solve(params=_params(outside_option=float(bound)), log_level="off")

    wage = np.asarray([1.0, 2.0, 3.0], dtype=dtype)
    expected_flags = wage < bound
    expected_values = np.column_stack((wage, 2 * wage))
    expected_values[expected_flags] = -np.inf
    flags = solution.replay_artifacts.project(DISSOLUTION_FLAG)
    np.testing.assert_array_equal(
        np.asarray(flags[1]["couple_terminal"]), expected_flags
    )
    np.testing.assert_array_equal(
        np.asarray(solution.values[1]["couple_terminal"]), expected_values
    )


def test_a_terminal_participation_constraint_flags_the_cells_it_empties() -> None:
    """Her participation binds at `wage < 3`, and those cells carry `D = True`.

    Her terminal payoff is `wage` and leaving is worth `3`, so the household
    survives its last period only at `wage = 3`. The flag is a property of the
    cell, so it is exact: two dissolved nodes and one intact one.
    """
    model = _make_model(participation=True)

    _solution = model.solve(params=_params(), log_level="off")
    flags = _solution.replay_artifacts.project(DISSOLUTION_FLAG)

    np.testing.assert_array_equal(
        np.asarray(flags[1]["couple_terminal"]), [True, True, False]
    )


def test_an_infeasible_terminal_cell_publishes_the_sentinel_value() -> None:
    """A dissolved terminal cell's value is `-inf`, and an intact one is real.

    pylcm resolves no outside option here: a terminal regime has no
    continuation, so the household's own scale has nothing to fall back to and
    the value is the infeasibility sentinel rather than what either partner
    would get by leaving. The surviving cell carries the ordinary pair
    `(wage, 2 * wage) = (3, 6)`.
    """
    model = _make_model(participation=True)

    solution = model.solve(params=_params(), log_level="off")

    np.testing.assert_array_equal(
        np.asarray(solution.values[1]["couple_terminal"]),
        np.array([[-np.inf, -np.inf], [-np.inf, -np.inf], [3.0, 6.0]]),
    )


def test_the_same_terminal_regime_without_the_constraint_keeps_every_cell() -> None:
    """The control: without the predicate every cell is feasible and finite.

    Without it the flags above could come from the terminal path itself rather
    than from the participation constraint.
    """
    model = _make_model(participation=False)

    solution = model.solve(params=_params(participation=False), log_level="off")
    flags = solution.replay_artifacts.project(DISSOLUTION_FLAG)

    np.testing.assert_array_equal(
        np.asarray(flags[1]["couple_terminal"]), [False, False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(solution.values[1]["couple_terminal"]),
        np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
    )


def test_a_terminal_reference_is_read_at_the_referenced_regimes_own_value() -> None:
    """The predicate reads `single_f_terminal`'s solved value, not a constant.

    Raising the outside option to `4` empties every cell, which it can only do
    if the reference regime's own current-period value is what the predicate
    compares against.
    """
    model = _make_model(participation=True)

    _solution = model.solve(
        params=_params(outside_option=4.0),
        log_level="off",
    )
    flags = _solution.replay_artifacts.project(DISSOLUTION_FLAG)

    np.testing.assert_array_equal(
        np.asarray(flags[1]["couple_terminal"]), [True, True, True]
    )


def test_a_terminal_singleton_regime_still_refuses_value_constraints() -> None:
    """The relaxation is scoped to collective regimes.

    `Q_<s>` is what a value constraint reads, and a singleton regime carries
    none, so the predicate would have no argument to compare.
    """
    with pytest.raises(Exception, match="value_constraints"):
        Regime(
            states={"wage": _WAGE},
            functions={"utility": _wage_for_her},
            constraints={
                "participation_f": ValueDependentConstraint(predicate=_participation_f)
            },
        )
