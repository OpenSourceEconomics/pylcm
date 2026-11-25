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

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    CollectiveUtility,
    LinSpacedGrid,
    Model,
    categorical,
    post_decision_lower_bound,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime
from lcm.solvers import DCEGM, EGM, NBEGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParamsNode,
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


def utility(*, consumption: ContinuousAction, crra: float) -> FloatND:
    return consumption ** (1.0 - crra) / (1.0 - crra)


def terminal_utility(*, wealth: ContinuousState, crra: float) -> FloatND:
    """The last period consumes everything on hand."""
    return wealth ** (1.0 - crra) / (1.0 - crra)


def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    """The end-of-period balance the law of motion is written through."""
    return wealth - consumption


def next_wealth(
    *,
    savings: FloatND,
    return_liquid: float,
    retirement_income: float,
) -> ContinuousState:
    return (1.0 + return_liquid) * savings + retirement_income


def feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _model(
    *,
    solver: EGM | GridSearch,
    n_consumption: int = 200,
    resources_func: UserFunction | None = None,
    liquid: LiquidMargin | None = None,
    utility_function: UserFunction = utility,
) -> Model:
    """A pure consumption--saving lifecycle with no income and no kinks.

    `resources_func` and `liquid` let a caller vary the two declarations the
    EGM contract constrains. By default, the liquid state itself fills the
    resources role.
    """
    wealth_grid = _WEALTH_GRID
    regime_type = (
        partial(
            ConsumptionSavingsRegime,
            liquid=liquid
            if liquid is not None
            else LiquidMargin(
                state="wealth",
                action="consumption",
                resources=("resources" if resources_func else "wealth"),
                post_decision_state="savings",
            ),
        )
        if isinstance(solver, EGM)
        else Regime
    )
    functions = {
        "utility": utility_function,
        "savings": savings,
    }
    if resources_func is not None:
        functions["resources"] = resources_func

    saving = regime_type(
        actions={
            "consumption": LinSpacedGrid(start=0.05, stop=60.0, n_points=n_consumption)
        },
        states={"wealth": wealth_grid},
        state_transitions={
            "wealth": {"saving": next_wealth, "done": next_wealth},
        },
        constraints={} if isinstance(solver, EGM) else {"feasible": feasible},
        functions=functions,
        solver=solver,
    )
    done = Regime(
        states={"wealth": wealth_grid},
        functions={"utility": terminal_utility},
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving, "done": done},
        ages=AgeGrid(start=0, inclusive_stop=_N_PERIODS - 1, step="Y"),
        edges={
            "saving": {
                "saving": AgeRange(exclusive_stop=_N_PERIODS - 2),
                "done": _N_PERIODS - 2,
            }
        },
        regime_id_class=RegimeId,
        initial_nodes={0: "saving"},
    )


def _params() -> dict[str, UserParamsNode]:
    """The parameters of the closed-form model."""
    law = {"return_liquid": _RETURN, "retirement_income": 0.0}
    return {
        "saving": {
            "utility": {"crra": _CRRA},
            "koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR},
            "saving": {"next_wealth": law},
            "done": {"next_wealth": law},
        },
        "done": {"utility": {"crra": _CRRA}},
    }


def _closed_form(
    *, wealth: np.ndarray, periods_of_consumption: int
) -> tuple[np.ndarray, np.ndarray]:
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
def test_egm_value_matches_the_analytical_solution(period: int) -> None:
    """Each period's solved value equals the closed form on the unconstrained set."""
    solution = (
        _model(solver=EGM(savings_grid=_SAVINGS_GRID))
        .solve(params=_params(), log_level="debug")
        .values
    )
    wealth = np.asarray(_WEALTH_GRID.to_jax())[_UNCONSTRAINED]
    _, expected = _closed_form(
        wealth=wealth, periods_of_consumption=_N_PERIODS - period
    )
    got = np.asarray(solution[period]["saving"])[_UNCONSTRAINED, 0]
    np.testing.assert_allclose(got, expected, rtol=1e-2)


def test_egm_agrees_with_dense_grid_search() -> None:
    """The EGM value matches a dense brute-force solve of the same model.

    An independent check on the closed form: grid search never inverts an Euler
    equation, so it cannot share a mistake in the inversion.
    """
    params = _params()
    egm = (
        _model(solver=EGM(savings_grid=_SAVINGS_GRID))
        .solve(params=params, log_level="debug")
        .values
    )
    brute = (
        _model(solver=GridSearch(), n_consumption=1200)
        .solve(params=params, log_level="debug")
        .values
    )
    for period in (0, 1, 2):
        got = np.asarray(egm[period]["saving"])[_UNCONSTRAINED]
        expected = np.asarray(brute[period]["saving"])[_UNCONSTRAINED]
        rel = np.abs(got - expected) / np.abs(expected)
        assert np.max(rel) < 1e-2


_LIQUID = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="liquid",
    post_decision_state="savings",
)


def _log_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _log_terminal(*, liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _liquid_savings(
    *, liquid: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    return liquid - consumption


def _next_liquid(*, savings: FloatND) -> ContinuousState:
    return savings + 1.0


def _floor(*, age: float) -> FloatND:
    """A floor computed from the age: the solver prepares its query evaluators."""
    return jnp.asarray(0.25 + 0.0 * age)


def _one_member_model(
    *, solver: EGM | DCEGM | NBEGM, member: str | None, callable_floor: bool
) -> Model:
    """A two-age log consumption--saving model, its one member plain or named."""

    def declare(function: UserFunction) -> UserFunction | CollectiveUtility:
        return (
            function
            if member is None
            else CollectiveUtility(utilities={member: function})
        )

    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regimes={
            "saving": ConsumptionSavingsRegime(
                liquid=_LIQUID,
                states={"liquid": LinSpacedGrid(start=2.0, stop=8.0, n_points=17)},
                actions={
                    "consumption": LinSpacedGrid(start=0.05, stop=8.0, n_points=65)
                },
                state_transitions={"liquid": {"done": _next_liquid}},
                functions={
                    "utility": declare(_log_utility),
                    "savings": _liquid_savings,
                },
                constraints={
                    "borrowing": post_decision_lower_bound(
                        margin=_LIQUID, lower=_floor if callable_floor else 0.25
                    )
                },
                solver=solver,
            ),
            "done": Regime(
                states={"liquid": LinSpacedGrid(start=1.0, stop=12.0, n_points=65)},
                functions={"utility": declare(_log_terminal)},
            ),
        },
        edges={"saving": {"done": 0}},
        regime_id_class=RegimeId,
        initial_nodes={0: "saving"},
    )


@pytest.mark.parametrize("callable_floor", [False, True], ids=["literal", "callable"])
@pytest.mark.parametrize("member", ["alice", "bob"])
@pytest.mark.parametrize("solver_type", [EGM, DCEGM, NBEGM])
def test_an_endogenous_grid_solver_solves_a_named_member_as_its_plain_model(
    *,
    solver_type: type[EGM | DCEGM | NBEGM],
    member: str,
    callable_floor: bool,
) -> None:
    """A one-member `CollectiveUtility` regime solves to its plain-utility values.

    Naming the member renames its utility to `utility_<member>`; the solver and
    its query evaluators read it under that name. Both values carry the trailing
    member axis of length one.
    """
    solver = solver_type(
        savings_grid=LinSpacedGrid(start=0.25, stop=10.25, n_points=65)
    )
    plain, named = (
        _one_member_model(solver=solver, member=name, callable_floor=callable_floor)
        .solve(params={"discount_factor": 0.95}, log_level="off")
        .values
        for name in (None, member)
    )
    for period, regimes in plain.items():
        for regime, expected in regimes.items():
            np.testing.assert_array_equal(
                np.asarray(named[period][regime]), np.asarray(expected)
            )
