"""A collective regime publishes ONE household feasibility mask, not one per partner.

A collective regime carries a per-stakeholder `utility_<s>`, so `Q^f` and `Q^m`
genuinely differ. Its feasibility is NOT per stakeholder: the mask is built from
the regime's constraints against the shared function pool, and the household
argmax runs over that single mask. A constraint may name one partner's felicity
node — `utility_f` here — and that names the SAME node for every stakeholder, so
the husband's action set is restricted by the wife's participation condition
exactly as the wife's is.

The model below makes the two readings disagree by construction:

- wage grid $\\{8, 24, 40\\}$, binary `work`;
- $u^f = w\\cdot\\text{work} + 30(1-\\text{work})$ — leisure pays 30, work pays $w$;
- $u^m = 2w\\cdot\\text{work}$ — leisure pays 0, work pays $2w$;
- the constraint is $u^f \\ge 25$.

Reading the constraint against `utility_f` leaves leisure feasible everywhere
and work feasible only at $w = 40$. Reading it against `utility_m` would leave
leisure infeasible everywhere and work feasible from $w = 24$ up — a different
argmax at every wage and a different dissolution flag at $w = 8$. Every value
asserted here is therefore a statement about WHICH node the one shared mask
reads, not merely that some mask was applied.

Hand computation, $\\beta = 0.95$, `next_wage = 40*work + 24*(1-work)`:

Terminal regime, constraints $u^f \\ge 25$ and $w \\ge 10$ (the latter empties the
low-wage action set, so the dissolution flag is exercised too):

- $w = 8$: no feasible action, $D = \\text{True}$, $V = (-\\infty, -\\infty)$;
- $w = 24$: leisure only ($u^f = 30 \\ge 25$; work's $u^f = 24 < 25$), $V = (30, 0)$;
- $w = 40$: both feasible, household objective $60 > 15$ picks work, $V = (40, 80)$.

Non-terminal regime, constraint $u^f \\ge 25$ alone:

- $w = 8$: leisure only, next wage 24, $V = (30, 0) + 0.95 (30, 0) = (58.5, 0)$;
- $w = 24$: leisure only, same continuation, $V = (58.5, 0)$;
- $w = 40$: leisure gives $(58.5, 0)$ with objective 29.25, work gives
  $(40, 80) + 0.95 (40, 80) = (78, 156)$ with objective 117, so $V = (78, 156)$.
"""

import numpy as np
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1


_WAGE_GRID = LinSpacedGrid(start=8.0, stop=40.0, n_points=3)
_BETA = 0.95

# The wife's participation floor; between work's payoff at wage 24 (24) and her
# constant leisure payoff (30), so the constraint bites on the ACTION.
_PARTICIPATION_FLOOR = 25.0

_EXPECTED_V_TERMINAL = np.array([[-np.inf, -np.inf], [30.0, 0.0], [40.0, 80.0]])
_EXPECTED_D_TERMINAL = np.array([True, False, False])
_EXPECTED_V_PERIOD_0 = np.array([[58.5, 0.0], [58.5, 0.0], [78.0, 156.0]])
_EXPECTED_D_PERIOD_0 = np.array([False, False, False])


def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: her leisure is worth a flat 30, working is worth the wage."""
    return wage * work + 30.0 * (1.0 - work)


def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption only, twice the wife's weight on it."""
    return 2.0 * (wage * work)


def _participation_f(utility_f: FloatND) -> BoolND:
    """The household may only choose actions clearing the wife's felicity floor."""
    return utility_f >= _PARTICIPATION_FLOOR


def _viable_wage(wage: ContinuousState) -> BoolND:
    """Terminal-regime floor on the state itself, emptying the low-wage action set."""
    return wage >= 10.0


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Working today yields the top wage tomorrow, leisure the middle one."""
    return 40.0 * work + 24.0 * (1.0 - work)


def _next_regime() -> ScalarInt:
    return RegimeId.couple_terminal


def _make_model() -> Model:
    couple = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
        constraints={"participation_f": _participation_f},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
        constraints={"participation_f": _participation_f, "viable_wage": _viable_wage},
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.fixture(scope="module")
def solved() -> tuple[dict, dict]:
    """Solve the two-regime model once, returning values and dissolution flags."""
    solution, dissolution_flags = _make_model().solve(
        params={"discount_factor": _BETA},
        log_level="debug",
        return_dissolution_flags=True,
    )
    return dict(solution), dict(dissolution_flags)


def test_terminal_collective_values_come_from_the_wife_s_participation_mask(
    solved: tuple[dict, dict],
) -> None:
    """Both partners' terminal values are read at the argmax over `u_f >= 25`."""
    solution, _flags = solved

    np.testing.assert_array_equal(
        np.asarray(solution[1]["couple_terminal"]), _EXPECTED_V_TERMINAL
    )


def test_terminal_collective_dissolution_flags_come_from_the_shared_mask(
    solved: tuple[dict, dict],
) -> None:
    """The terminal regime dissolves exactly where the household mask is empty."""
    _solution, flags = solved

    np.testing.assert_array_equal(
        np.asarray(flags[1]["couple_terminal"]), _EXPECTED_D_TERMINAL
    )


def test_nonterminal_collective_values_come_from_the_wife_s_participation_mask(
    solved: tuple[dict, dict],
) -> None:
    """Both partners' period-0 values are read at the argmax over `u_f >= 25`."""
    solution, _flags = solved

    np.testing.assert_allclose(
        np.asarray(solution[0]["couple"]), _EXPECTED_V_PERIOD_0, rtol=1e-6
    )


def test_nonterminal_collective_dissolution_flags_come_from_the_shared_mask(
    solved: tuple[dict, dict],
) -> None:
    """No period-0 cell dissolves: leisure clears the wife's floor at every wage."""
    _solution, flags = solved

    np.testing.assert_array_equal(np.asarray(flags[0]["couple"]), _EXPECTED_D_PERIOD_0)


def test_the_two_stakeholders_action_values_genuinely_differ(
    solved: tuple[dict, dict],
) -> None:
    """`Q^f` and `Q^m` differ, so a mask taken from one partner would be visible."""
    solution, _flags = solved
    V_0 = np.asarray(solution[0]["couple"])

    assert not np.allclose(V_0[:, 0], V_0[:, 1])


def test_the_husband_is_bound_by_the_wife_s_participation_constraint(
    solved: tuple[dict, dict],
) -> None:
    """At wage 24 the husband's own optimum (work, 48) is ruled out for him too.

    Reading the mask against the husband's own felicity would leave work
    feasible and leisure infeasible there, publishing `(24, 48)` in the terminal
    period instead of the wife-masked `(30, 0)`.
    """
    solution, _flags = solved
    V_terminal_at_middle_wage = np.asarray(solution[1]["couple_terminal"])[1]

    np.testing.assert_array_equal(V_terminal_at_middle_wage, [30.0, 0.0])
    assert not np.allclose(V_terminal_at_middle_wage, [24.0, 48.0])
