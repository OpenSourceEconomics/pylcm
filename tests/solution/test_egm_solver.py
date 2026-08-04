"""`EGM` solves a 1-D consumption--saving regime against a closed form.

With CRRA utility, a constant gross return `R`, a discount factor `beta`, no
income and no discrete choice, the finite-horizon consumption--saving problem
has an analytical solution. Writing `rho` for relative risk aversion, define the
one-period thrift factor

```{math}
\\kappa = (\\beta R^{1-\\rho})^{1/\\rho},
```

so that with `h` periods of consumption still to come the optimal policy and
value are

```{math}
c_t = \\frac{w_t}{\\sum_{j=0}^{h-1} (\\kappa / R)^{j}},
\\qquad
V_t = \\frac{c_t^{1-\\rho}}{1-\\rho}\\sum_{j=0}^{h-1} \\beta^j \\kappa^{j(1-\\rho)}.
```

The solver is checked against those expressions directly, not against another
implementation that could share its assumptions.
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
_DISCOUNT_FACTOR = 0.95
_RETURN = 0.03
_N_PERIODS = 4
# The lowest wealth nodes are where a CRRA value function curves hardest, so a
# regular grid resolves it worst and the error compounds backward. Above them
# the solve is well inside the unconstrained region the closed form describes.
_UNCONSTRAINED = np.s_[10:]
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=60.0, n_points=200)
_WEALTH_GRID = LinSpacedGrid(start=2.0, stop=60.0, n_points=60)


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    return consumption ** (1.0 - crra) / (1.0 - crra)


def terminal_utility(wealth: ContinuousState, crra: float) -> FloatND:
    """The last period consumes everything on hand."""
    return wealth ** (1.0 - crra) / (1.0 - crra)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    return (1.0 + return_liquid) * (wealth - consumption) + retirement_income


def feasible(wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def prob_continue(age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 < last_age, 1.0, 0.0)


def prob_stop(age: int, last_age: float) -> FloatND:
    return jnp.where(age + 1 >= last_age, 1.0, 0.0)


def _model(*, solver, n_consumption=200):
    """A pure consumption--saving lifecycle with no income and no kinks."""
    wealth_grid = _WEALTH_GRID
    last_age = float(_N_PERIODS - 1)
    saving = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=n_consumption)
        },
        states={"wealth": wealth_grid},
        state_transitions={
            "wealth": {"saving": next_wealth, "done": next_wealth},
        },
        constraints={"feasible": feasible},
        transition={
            "saving": MarkovTransition(prob_continue),
            "done": MarkovTransition(prob_stop),
        },
        functions={"utility": utility},
        active=lambda age, la=last_age: age < la,
        solver=solver,
    )
    done = Regime(
        transition=None,
        states={"wealth": wealth_grid},
        functions={"utility": terminal_utility},
        active=lambda age, la=last_age: age >= la,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _params():
    law = {"return_liquid": _RETURN, "retirement_income": 0.0}
    return {
        "saving": {
            "utility": {"crra": _CRRA},
            "H": {"discount_factor": _DISCOUNT_FACTOR},
            "saving": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
            "done": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
        },
        "done": {"utility": {"crra": _CRRA}},
    }


def _closed_form(wealth, *, periods_of_consumption):
    """Analytical consumption and value with `periods_of_consumption` left."""
    gross_return = 1.0 + _RETURN
    kappa = (_DISCOUNT_FACTOR * gross_return ** (1.0 - _CRRA)) ** (1.0 / _CRRA)
    powers = np.arange(periods_of_consumption)
    consumption = wealth / np.sum((kappa / gross_return) ** powers)
    value = (consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)) * np.sum(
        _DISCOUNT_FACTOR**powers * kappa ** (powers * (1.0 - _CRRA))
    )
    return consumption, value


@pytest.mark.parametrize("period", [0, 1, 2])
def test_egm_value_matches_the_analytical_solution(period):
    """Each period's solved value equals the closed form on the unconstrained set."""
    solution = _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=_params(), log_level="debug"
    )
    wealth = np.asarray(_WEALTH_GRID.to_jax())[_UNCONSTRAINED]
    _, expected = _closed_form(wealth, periods_of_consumption=_N_PERIODS - period)
    got = np.asarray(solution[period]["saving"])[_UNCONSTRAINED]
    np.testing.assert_allclose(got, expected, rtol=1e-2)


def test_egm_agrees_with_dense_grid_search():
    """The EGM value matches a dense brute-force solve of the same model.

    An independent check on the closed form: grid search never inverts an Euler
    equation, so it cannot share a mistake in the inversion.
    """
    params = _params()
    egm = _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=params, log_level="debug"
    )
    brute = _model(solver=GridSearch(), n_consumption=1200).solve(
        params=params, log_level="debug"
    )
    for period in (0, 1, 2):
        got = np.asarray(egm[period]["saving"])[_UNCONSTRAINED]
        expected = np.asarray(brute[period]["saving"])[_UNCONSTRAINED]
        rel = np.abs(got - expected) / np.abs(expected)
        assert np.max(rel) < 1e-2
