"""A collective regime counts its infeasible households on each period's own axes.

An `AgeSpecializedGrid` is what makes two periods distinguishable. It holds
`n_points` fixed while the bounds move with age, so every array keeps its shape
at every age and no shape check can separate the periods — only the coordinate a
value maps to, and through it the feasibility of a household, which is discrete.

`_build_pair_model` puts the moving grid on a collective regime `pair` whose
constraint empties its feasible set below a floor, so a wrongly-aged count reports
a different number of households with no feasible action.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
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

_DISCOUNT_FACTOR = 0.5

# The grid's ceiling drops once, so its three nodes are `[0, 4, 8]` at age 1 and
# `[0, 2, 4]` at age 2.
_CEILING_EARLY = 8.0
_CEILING_LATE = 4.0

# The amount the couple carries into `pair`: the top node of age 2's grid, and the
# MIDDLE node of age 1's.
_ENTRY = 4.0

_TERMINAL_GRID = LinSpacedGrid(start=0.0, stop=8.0, n_points=3)

# The husband's felicity premium in `pair`, so the two member components differ.
_M_BONUS = 10.0

# The wife's participation floor. Her felicity is `w`, so the household's feasible
# set empties exactly where `w < 3`: the support is `[False, True, True]` on age 1's
# `[0, 4, 8]` and `[False, False, True]` on age 2's `[0, 2, 4]`.
_FLOOR_F = 3.0


@categorical(ordered=True)
class _Effort:
    """A collective regime's discrete action; no felicity depends on it."""

    idle: ScalarInt
    busy: ScalarInt


@categorical(ordered=False)
class _PairRegimeId:
    """Regime ids of the couple / pair model."""

    couple: ScalarInt
    pair: ScalarInt
    pair_terminal: ScalarInt


def test_empty_counts_use_households_and_the_periods_own_grid(
    *,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Infeasible households are counted on each age's current nodes."""
    model = _build_pair_model()
    model.solve(
        params={"discount_factor": _DISCOUNT_FACTOR, "floor_f": _FLOOR_F},
        log_level="warning",
    )
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "no feasible action for regime 'pair'" in record.getMessage()
    ]
    assert warnings == [
        "2 state(s) with no feasible action for regime 'pair', period 2 (age 2.0)",
        "1 state(s) with no feasible action for regime 'pair', period 1 (age 1.0)",
    ]


def _build_pair_model() -> Model:
    r"""Build a couple that moves into a collective `pair` on an age-varying grid.

    Topology over ages 0-3: `couple` is collective, active at ages 0 and 1, and
    leaves for the collective `pair` at age 1 only. `pair` is active at ages 1
    and 2 and holds `w` on the age-varying grid, so its own nodes move with age.
    `pair_terminal` closes the lifecycle at zero felicity.

    `pair` pays `w` to the wife and `10 + w` to the husband, and her participation
    constraint $w \ge 3$ empties the household's feasible set exactly where `w < 3`.

    Returns:
        The model, which `{"discount_factor": 0.5, "floor_f": 3.0}` solves.

    """
    couple = Regime(
        state_transitions={"w": {"pair": _entry_amount}},
        actions={"effort": DiscreteGrid(category_class=_Effort)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _no_felicity, "m": _no_felicity}
            )
        },
    )
    pair = Regime(
        states={"w": AgeSpecializedGrid(build=_moving_grid, signature=_moving_ceiling)},
        state_transitions={"w": fixed_transition("w")},
        actions={"effort": DiscreteGrid(category_class=_Effort)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _pair_felicity_f, "m": _pair_felicity_m}
            )
        },
        constraints={"ir_f": _wife_participates},
    )
    pair_terminal = Regime(
        states={"w": _TERMINAL_GRID},
        actions={"effort": DiscreteGrid(category_class=_Effort)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _no_felicity_on_w, "m": _no_felicity_on_w}
            )
        },
    )
    staying_put = StochasticTransition(func=_probability_of_staying_put)
    leaving = StochasticTransition(func=_probability_of_leaving)
    return Model(
        regimes={"couple": couple, "pair": pair, "pair_terminal": pair_terminal},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_PairRegimeId,
        initial_nodes={0: "couple"},
        edges={
            "couple": Transition(
                targets={"couple": 0, "pair": (0, 1)},
                law=ByAge(
                    cases={
                        0: {"couple": staying_put, "pair": leaving},
                        1: {"pair": leaving},
                    }
                ),
            ),
            "pair": {"pair_terminal": (1, 2)},
        },
    )


def _moving_ceiling(age: float) -> float:
    """The highest value the age-varying grid reaches at this age."""
    return _CEILING_EARLY if age <= 1 else _CEILING_LATE


def _moving_grid(age: float) -> LinSpacedGrid:
    """The age-varying grid: zero to the age's ceiling, on three nodes."""
    return LinSpacedGrid(start=0.0, stop=_moving_ceiling(age), n_points=3)


def _probability_of_staying_put(age: FloatND) -> FloatND:
    """The couple waits one period before leaving."""
    return jnp.where(age < 1.0, 1.0, 0.0)


def _probability_of_leaving(age: FloatND) -> FloatND:
    """The couple leaves at the second age at which it is active."""
    return jnp.where(age < 1.0, 0.0, 1.0)


def _entry_amount() -> ContinuousState:
    """The amount the couple carries into `pair`."""
    return jnp.asarray(_ENTRY)


def _no_felicity(effort: DiscreteAction) -> FloatND:
    """Waiting pays nothing, so the couple's value is its continuation alone."""
    return 0.0 * effort


def _no_felicity_on_w(*, w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """A terminal couple's felicity, zero at every state and action."""
    return 0.0 * w + 0.0 * effort


def _pair_felicity_f(*, w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The pair pays out `w` to the wife."""
    return w + 0.0 * effort


def _pair_felicity_m(*, w: ContinuousState, effort: DiscreteAction) -> FloatND:
    """The pair pays the husband a premium on top of `w`."""
    return _M_BONUS + w + 0.0 * effort


def _wife_participates(*, w: ContinuousState, floor_f: FloatND) -> BoolND:
    """The pair is feasible only where the wife's payout clears her floor."""
    return w >= floor_f
