"""A participation comparison reads its reference regime's own period grid.

A collective regime's `value_constraints` compare each partner's action value
`Q^s` against the value that partner would have in another regime *in the same
period*, interpolated at the projected state (`same_period_refs`). When the
reference regime declares an `AgeSpecializedGrid`, the coordinates its value
function is tabulated on move with age, so the comparison must read the
reference value on the grid of the period the reading regime is solved in.

The model: `single_f` carries `wealth` on a grid whose ceiling tightens from
100 at age 0 to 10 at age 1 and pays `2 * wealth`, so its period-0 value
function is `[0, 200]` over the nodes `[0, 100]` and its period-1 value
function `[0, 20]` over `[0, 10]`. `couple` is a two-stakeholder regime on the
fixed wealth grid `[1, 5]`, paying the wife `3 + work` and the husband
`2 * (1 - work)`, and admits an action only where the wife's action value
clears her single value at the same wealth (`delta_f = 0`). Every regime is
myopic (`discount_factor = 0`), so each action value is its flow payoff.

The wife's period-0 outside option is therefore `2 * wealth` — 2 at
`wealth = 1`, 10 at `wealth = 5` — against action values of 3 (leisure) and 4
(work):

- `wealth = 1`: both actions clear the threshold, so the household maximizes
  the equally weighted `(Q_f + Q_m) / 2`, which leisure wins with 2.5 against
  work's 2. The couple's value is `(3, 2)` and the cell does not dissolve.
- `wealth = 5`: neither action clears 10, so the action mask is empty, the
  dissolution flag is set, and the value is `(-inf, -inf)`.

Age 1's grid `[0, 10]` places those same two coordinates at 20 and 100 instead,
which dissolves both cells — the separation the mask asserts here.
"""

from functools import cache
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.regime import ProjectedRegimeValue, Regime
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=True)
class Work:
    """The couple's single binary action."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    """The two regime islands: the wife's single regime and the couple."""

    single_f: ScalarInt
    single_f_terminal: ScalarInt
    couple: ScalarInt
    couple_terminal: ScalarInt


# Wealth ceiling of the reference regime at age 0, and at age 1 once it tightens.
INITIAL_CEILING = 100.0
TIGHTENED_CEILING = 10.0

# The couple's own wealth grid, fixed across ages.
COUPLE_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=2)

# Period-0 value of `couple`, indexed (wealth, stakeholder).
EXPECTED_V_COUPLE = ((3.0, 2.0), (-np.inf, -np.inf))

# Period-0 dissolution flag of `couple`, indexed by wealth.
EXPECTED_D_COUPLE = (False, True)


def test_participation_mask_uses_the_reference_regime_period_t_grid():
    """Only the couple whose wife's outside option beats her stake dissolves."""
    _solution, dissolution = _solve(later_ceiling=TIGHTENED_CEILING)
    np.testing.assert_array_equal(
        np.asarray(dissolution[0]["couple"]), np.asarray(EXPECTED_D_COUPLE)
    )


def test_participation_masked_couple_publishes_its_household_argmax_value():
    """The sustained cell publishes `(3, 2)`, the dissolved one `(-inf, -inf)`."""
    solution, _dissolution = _solve(later_ceiling=TIGHTENED_CEILING)
    aaae(
        np.asarray(solution[0]["couple"]),
        np.asarray(EXPECTED_V_COUPLE),
        decimal=DECIMAL_PRECISION,
    )


def test_participation_mask_ignores_the_reference_regime_later_grid():
    """Whether the reference grid tightens later leaves period 0's mask alone."""
    _tightening_solution, tightening = _solve(later_ceiling=TIGHTENED_CEILING)
    _invariant_solution, invariant = _solve(later_ceiling=INITIAL_CEILING)
    np.testing.assert_array_equal(
        np.asarray(tightening[0]["couple"]), np.asarray(invariant[0]["couple"])
    )


@pytest.mark.parametrize(
    ("period", "expected"),
    [(0, (0.0, 200.0)), (1, (0.0, 20.0))],
)
def test_reference_value_function_is_tabulated_on_that_periods_own_nodes(
    period, expected
):
    """The reference regime pays `2 * wealth` on whichever nodes its age has."""
    solution, _dissolution = _solve(later_ceiling=TIGHTENED_CEILING)
    aaae(
        np.asarray(solution[period]["single_f"]),
        np.asarray(expected),
        decimal=DECIMAL_PRECISION,
    )


@cache
def _solve(*, later_ceiling: float) -> tuple[MappingProxyType, MappingProxyType]:
    """Solve the model whose reference grid ends at `later_ceiling` at age 1.

    Args:
        later_ceiling: Upper bound of `single_f`'s wealth grid from age 1 on.

    Returns:
        Tuple of the per-period value functions and the per-period dissolution
        flags.

    """
    return _make_model(later_ceiling=later_ceiling).solve(
        params={
            "single_f": {"koopmans_aggregator": {"discount_factor": 0.0}},
            "single_f_terminal": {},
            "couple": {
                "koopmans_aggregator": {"discount_factor": 0.0},
                "participation_f": {"delta_f": 0.0},
            },
            "couple_terminal": {},
        },
        log_level="debug",
        return_dissolution_flags=True,
    )


def _make_model(*, later_ceiling: float) -> Model:
    """Build the couple-and-single model with an age-specialized reference grid."""

    def _single_wealth_grid(age: float) -> LinSpacedGrid:
        ceiling = INITIAL_CEILING if age < 0.5 else later_ceiling
        return LinSpacedGrid(start=0.0, stop=ceiling, n_points=2)

    single_f = Regime(
        transition={
            "single_f": MarkovTransition(_stays_single),
            "single_f_terminal": MarkovTransition(_leaves_single),
        },
        active=lambda age: age < 2,
        states={
            "wealth": AgeSpecializedGrid(
                build=_single_wealth_grid, signature=lambda age: age < 0.5
            )
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _single_utility},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wealth": LinSpacedGrid(start=0.0, stop=INITIAL_CEILING, n_points=2)},
        functions={"utility": _zero_utility},
    )
    couple = Regime(
        transition={"couple_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wealth": COUPLE_GRID},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _couple_utility_f, "m": _couple_utility_m}
            )
        },
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_participation_f,
                references={
                    "V_single_f": ProjectedRegimeValue(
                        regime="single_f", projection={"wealth": _project_wealth}
                    )
                },
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": COUPLE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _zero_collective_utility, "m": _zero_collective_utility}
            )
        },
    )
    return Model(
        regimes={
            "single_f": single_f,
            "single_f_terminal": single_f_terminal,
            "couple": couple,
            "couple_terminal": couple_terminal,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def _single_utility(wealth: ContinuousState) -> FloatND:
    """The single wife consumes her wealth at a constant marginal value."""
    return 2.0 * wealth


def _zero_utility(wealth: ContinuousState) -> FloatND:
    """Terminal payoff of the single wife: nothing left to gain."""
    return 0.0 * wealth


def _zero_collective_utility(wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    """Terminal payoff of each partner: nothing left to gain."""
    return 0.0 * wealth * work


def _couple_utility_f(work: DiscreteAction) -> FloatND:
    """Wife: a base payoff from the match plus her own earnings when working."""
    return 3.0 + work


def _couple_utility_m(work: DiscreteAction) -> FloatND:
    """Husband: values her time at home and nothing else."""
    return 2.0 * (1.0 - work)


def _participation_f(Q_f: FloatND, V_single_f: FloatND, delta_f: FloatND) -> BoolND:
    """The wife consents to an action only if it beats her single value."""
    return Q_f >= V_single_f - delta_f


def _project_wealth(wealth: ContinuousState) -> ContinuousState:
    """The wife keeps the household's wealth when single."""
    return wealth


def _stays_single(age: FloatND) -> FloatND:
    """The single wife stays single through age 0, and not after."""
    return jnp.asarray(age < 1, dtype=float)


def _leaves_single(age: FloatND) -> FloatND:
    """The single wife enters her terminal regime from age 1 on."""
    return jnp.asarray(age >= 1, dtype=float)


def _prob_one(age: FloatND) -> FloatND:
    """The couple enters its terminal regime with probability one."""
    return jnp.ones_like(age, dtype=float)
