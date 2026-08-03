"""`EGM` resolves the target's Euler state from the target, not from itself.

A one-dimensional EGM regime and the regime it continues into each have exactly
one continuous state, so the correspondence between them is unambiguous — but
they need not spell it the same way. A working regime holding `wealth` may
continue into a terminal regime holding `estate`; the handoff is declared by an
entry law keyed on the target's name, which is how pylcm expresses a state that
lives in the target rather than the source.

The Euler inversion reads the target's value on the target's grid and reads the
target's transition parameters out of the target's namespace. Both are facts
about the target, so both are resolved there. Requiring the target to reuse the
source's spelling rejects a model the framework can express, and it is the
source's vocabulary leaking into the kernel's view of a different regime.

The oracle is the same model solved by dense `GridSearch`, which never inverts
an Euler equation and so cannot share the defect. The two regimes are given
deliberately different grids, so a solver reading the continuation on its own
grid disagrees rather than coinciding by accident.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

_CRRA = 2.0
_DISCOUNT_FACTOR = 0.98
_RETURN = 0.02
_INCOME = 0.5
_LAST_AGE = 3.0

# Source and target discretize their single continuous state differently, so
# reading the continuation on the source's own nodes is observably wrong rather
# than accidentally right.
_WEALTH_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=12)
_ESTATE_GRID = LinSpacedGrid(start=0.1, stop=26.0, n_points=34)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=200)

# The lowest wealth nodes are borrowing-constrained, where an exact Euler
# inversion and a discrete consumption sweep are not comparable at all. On this
# grid the constraint stops binding above wealth 8, which is index 5.
_UNCONSTRAINED = np.s_[5:]
# What the two solvers still disagree by once neither one's own discretization
# dominates: both interpolate the terminal regime's value, at different points,
# so the gap is set by that grid and shrinks with it. It compounds backward,
# which is why the bound is read off the earliest period rather than the last.
_RELATIVE_TOL = 0.01


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    gone: ScalarInt


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    return consumption ** (1.0 - crra) / (1.0 - crra)


def bequest(estate: ContinuousState, crra: float) -> FloatND:
    """The terminal regime names the state it inherits `estate`."""
    return estate ** (1.0 - crra) / (1.0 - crra)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    return (1.0 + return_liquid) * (wealth - consumption) + retirement_income


def next_estate(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    """The entry law into the terminal regime, keyed on that regime's name."""
    return (1.0 + return_liquid) * (wealth - consumption) + retirement_income


def feasible(wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def prob_survive(age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 < last_age, 1.0, 0.0)


def prob_gone(age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 >= last_age, 1.0, 0.0)


def _model(*, solver, n_consumption=14):
    """A 1-D lifecycle whose terminal regime renames the state it inherits."""
    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=n_consumption)
        },
        states={"wealth": _WEALTH_GRID},
        state_transitions={
            "wealth": {"alive": next_wealth},
            # `estate` is the target's state, not this regime's, so its law is
            # declared here as the entry law into `gone`.
            "estate": {"gone": next_estate},
        },
        constraints={"feasible": feasible},
        transition={
            "alive": MarkovTransition(prob_survive),
            "gone": MarkovTransition(prob_gone),
        },
        functions={"utility": utility},
        active=lambda age: age < _LAST_AGE,
        solver=solver,
    )
    gone = Regime(
        transition=None,
        states={"estate": _ESTATE_GRID},
        functions={"utility": bequest},
        active=lambda age: age >= _LAST_AGE,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=RegimeId,
    )


def _params():
    law = {"return_liquid": _RETURN, "retirement_income": _INCOME}
    return {
        "alive": {
            "utility": {"crra": _CRRA},
            "H": {"discount_factor": _DISCOUNT_FACTOR},
            "alive": {"next_wealth": law, "next_regime": {"last_age": _LAST_AGE}},
            "gone": {"next_estate": law, "next_regime": {"last_age": _LAST_AGE}},
        },
        "gone": {"utility": {"crra": _CRRA}},
    }


def test_egm_accepts_a_target_that_names_the_euler_state_differently():
    """A differently named target state is a valid model, not an ambiguity.

    Both regimes have exactly one continuous state, so which state continues
    into which is fully determined. Only a missing or ambiguous handoff is a
    reason to refuse.
    """
    model = _model(solver=EGM(savings_grid=_SAVINGS_GRID))
    assert model.user_regimes["gone"].states.keys() == {"estate"}


@pytest.mark.parametrize("period", [0, 1, 2])
def test_egm_matches_dense_grid_search_across_a_renamed_terminal_state(period):
    """The solved value agrees with brute force in every period.

    The last working period reads its continuation from `gone` — a different
    regime, a different state name, and a different grid. Reading it on the
    source's own `wealth` nodes instead would move the published value well
    beyond this bound.
    """
    params = _params()
    egm = _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=params, log_level="debug"
    )
    brute = _model(solver=GridSearch(), n_consumption=600).solve(
        params=params, log_level="debug"
    )
    got = np.asarray(egm[period]["alive"])[_UNCONSTRAINED]
    expected = np.asarray(brute[period]["alive"])[_UNCONSTRAINED]
    assert np.isfinite(got).all()
    rel = np.abs(got - expected) / np.abs(expected)
    assert np.max(rel) < _RELATIVE_TOL
