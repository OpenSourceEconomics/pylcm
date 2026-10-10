"""A random state's draw read toward a target that does not carry the state.

The liquid law `next_liquid = R * savings + s * exp(y')` reads the income draw
toward both targets, where `y'` is the node of a process or the level of a
Markov state's code. `alive` carries `income`, so toward `alive` the draw
persists; `dead` does not, so toward `dead` the draw is taken from the source's
process or Markov law inside the transition and then discarded.

With CRRA utility, a linear bequest `V_dead(l) = l` and resources never binding,
every period's consumption is a constant and every value function is affine in
liquid, so all interpolation is exact and the solution has a closed form. Writing
`E_y = E[exp(y') | y]` (the row of the process's transition matrix),

```{math}
c_1 = (\\beta R)^{-1/\\gamma},\\quad
V_1(l, y) = u(c_1) + \\beta\\,(R (l + b - c_1) + s E_y),
```

and one period earlier, toward `alive` with `V_1` affine of slope `beta * R`,

```{math}
c_0 = (\\beta^2 R^2)^{-1/\\gamma},\\quad
V_0(l, y) = u(c_0) + \\beta\\,E[V_1(R (l + b - c_0) + s e^{y'}, y') \\mid y].
```

`V_1` is reached through the transition-local draw toward `dead`, `V_0` through the
draw `alive` persists. Under a persistent process or Markov state both vary with
the lagged node, so a solver that ignored or misplaced the draw would miss by that
spread.
"""

from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.solvers import DCEGM
from lcm.transition import StochasticTransition
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt
from tests.conftest import X64_ENABLED
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)

type Solver = Literal["brute", "dcegm", "nbegm"]
type Process = Literal["iid", "ar1", "markov"]

INCOME_SCALE = 0.5
BASE_INCOME = 1.0
GROSS_RETURN = 1.02
DISCOUNT_FACTOR = 0.95
CRRA = 2.0
N_LIQUID = 41
LIQUID_MAX = 40.0
LIQUID_GRID = LinSpacedGrid(start=0.0, stop=LIQUID_MAX, n_points=N_LIQUID)

# At `liquid < 3` the borrowing constraint binds and the value function is no
# longer affine; above 30 next liquid can leave the grid. Between, the closed
# form holds exactly.
_COMPARED_LIQUID = (3.0, 30.0)

# EGM inverts the Euler equation exactly, so it meets the closed form to
# rounding: observed 1.4e-14 at float64 and 7.5e-6 at float32. Brute force is
# limited by its 4001-point consumption grid: observed 2.7e-5 at either precision.
_EGM_ATOL = 1e-10 if X64_ENABLED else 5e-5
_BRUTE_ATOL = 1e-4


def _resources(*, liquid: ContinuousState, base_income: float) -> FloatND:
    return liquid + base_income


def _next_liquid(
    *, savings: FloatND, next_income: ContinuousState, return_liquid: float
) -> ContinuousState:
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(next_income)


def _linear_bequest(*, liquid: ContinuousState) -> FloatND:
    return liquid


@categorical(ordered=True)
class IncomeLevel:
    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


MARKOV_LEVELS = np.array([-0.4, 0.0, 0.5])
MARKOV_PROBS = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.25, 0.7]])


def _next_income_probs(*, income: DiscreteState) -> FloatND:
    return jnp.asarray(MARKOV_PROBS)[income]


def _next_liquid_markov(
    *, savings: FloatND, next_income: DiscreteState, return_liquid: float
) -> ContinuousState:
    level = jnp.asarray(MARKOV_LEVELS)[next_income]
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(level)


def _income(
    process: Process,
) -> NormalIIDProcess | RouwenhorstAR1Process | DiscreteGrid:
    if process == "iid":
        return NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=0.3)
    if process == "markov":
        return DiscreteGrid(IncomeLevel)
    return RouwenhorstAR1Process(n_points=5, rho=0.8, sigma=0.3, mu=0.0)


def _income_nodes_and_probs(process: Process) -> tuple[np.ndarray, np.ndarray]:
    """The income draw's node values and its transition matrix."""
    if process == "markov":
        return MARKOV_LEVELS, MARKOV_PROBS
    income = cast("NormalIIDProcess | RouwenhorstAR1Process", _income(process))
    return (
        np.asarray(income.to_jax(), dtype=np.float64),
        np.asarray(income.get_transition_probs(), dtype=np.float64),
    )


def _model(*, solver: Solver, process: Process) -> Model:
    savings_grid = LinSpacedGrid(start=0.0, stop=42.0, n_points=80)
    alive_solver = (
        DCEGM(savings_grid=savings_grid, n_constrained_points=16)
        if solver == "dcegm"
        else resolve_solver(variant=solver, savings_grid=savings_grid)
    )
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=N_LIQUID,
        liquid_max=LIQUID_MAX,
        n_consumption=4001,
        liquid_grid=LIQUID_GRID,
        alive_functions={
            "utility": utility,
            "resources": _resources,
            "savings": savings,
        },
        liquid_law=_next_liquid_markov if process == "markov" else _next_liquid,
        alive_solver=alive_solver,
        constraints={"feasible": feasible} if solver == "brute" else {},
        extra_states={"income": _income(process)},
        extra_state_transitions=(
            {"income": StochasticTransition(func=_next_income_probs)}
            if process == "markov"
            else None
        ),
        dead_functions={"utility": _linear_bequest},
    )


_PARAMS = {
    "crra": CRRA,
    "base_income": BASE_INCOME,
    "return_liquid": GROSS_RETURN - 1.0,
    "discount_factor": DISCOUNT_FACTOR,
}


def _closed_form(process: Process) -> dict[int, np.ndarray]:
    """`alive`'s value per period, indexed `(income, liquid)`."""
    nodes, probs = _income_nodes_and_probs(process)
    expected_shock = probs @ np.exp(nodes)
    liquid = np.asarray(LIQUID_GRID.to_jax(), dtype=np.float64)[None, :]

    def u(consumption: float) -> float:
        return consumption ** (1.0 - CRRA) / (1.0 - CRRA)

    beta, gross = DISCOUNT_FACTOR, GROSS_RETURN
    c_1 = (beta * gross) ** (-1.0 / CRRA)
    v_1 = u(c_1) + beta * (
        gross * (liquid + BASE_INCOME - c_1) + INCOME_SCALE * expected_shock[:, None]
    )
    c_0 = (beta**2 * gross**2) ** (-1.0 / CRRA)
    next_liquid_without_shock = gross * (liquid + BASE_INCOME - c_0)
    expected_v_1 = (
        u(c_1)
        + beta * gross * (next_liquid_without_shock + BASE_INCOME - c_1)
        + beta * gross * INCOME_SCALE * expected_shock[:, None]
        + beta * INCOME_SCALE * (probs @ expected_shock)[:, None]
    )
    return {0: u(c_0) + beta * expected_v_1, 1: v_1}


def _alive_values(model: Model) -> dict[int, np.ndarray]:
    solution = model.solve(params=_PARAMS, log_level="off")
    order = model.state_names(regime_name="alive")
    axes = [order.index(name) for name in ("income", "liquid")]
    return {
        period: np.transpose(np.asarray(values["alive"]), axes)
        for period, values in solution.values.items()
        if "alive" in values
    }


@pytest.mark.parametrize("process", ["iid", "ar1", "markov"])
@pytest.mark.parametrize("solver", ["brute", "dcegm", "nbegm"])
def test_the_non_carrying_target_keeps_only_its_own_states(
    *, solver: Solver, process: Process
) -> None:
    model = _model(solver=solver, process=process)

    assert model.state_names(regime_name="dead") == ("liquid",)


@pytest.mark.parametrize("process", ["iid", "ar1", "markov"])
@pytest.mark.parametrize("solver", ["brute", "dcegm", "nbegm"])
def test_value_matches_the_closed_form(*, solver: Solver, process: Process) -> None:
    values = _alive_values(_model(solver=solver, process=process))
    expected = _closed_form(process)
    liquid = np.asarray(LIQUID_GRID.to_jax())
    compared = (liquid >= _COMPARED_LIQUID[0]) & (liquid <= _COMPARED_LIQUID[1])

    assert sorted(values) == [0, 1]
    for period in (0, 1):
        np.testing.assert_allclose(
            values[period][:, compared],
            expected[period][:, compared],
            rtol=0,
            atol=_BRUTE_ATOL if solver == "brute" else _EGM_ATOL,
            err_msg=f"period={period}",
        )


@pytest.mark.parametrize("process", ["ar1", "markov"])
def test_a_persistent_draw_spreads_the_value_across_lagged_nodes(
    process: Process,
) -> None:
    """The closed form discriminates: the lagged node moves V far beyond tolerance."""
    expected = _closed_form(process)

    assert np.ptp(expected[1][:, 10]) > 0.1
    assert np.ptp(expected[0][:, 10]) > 0.1


@pytest.mark.parametrize("process", ["ar1", "markov"])
@pytest.mark.parametrize("solver", ["brute", "dcegm", "nbegm"])
def test_entering_dead_lands_on_one_of_the_draws_outcomes(
    *, solver: Solver, process: Process
) -> None:
    """Simulated `dead` liquid is `R * savings + s * exp(node)` for a source node."""
    model = _model(solver=solver, process=process)
    n = 6
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "liquid": jnp.linspace(5.0, 25.0, n),
            "income": jnp.zeros(n, dtype=jnp.int32 if process == "markov" else None),
            "age": jnp.zeros(n),
            "regime_id": jnp.zeros(n, dtype=jnp.int32),
        },
        log_level="off",
        seed=0,
    )
    df = result.to_dataframe()
    alive = df.query("regime_name == 'alive' and period == 1").set_index("subject_id")
    dead = df.query("regime_name == 'dead' and period == 2").set_index("subject_id")
    common = alive.index.intersection(dead.index)
    nodes, _ = _income_nodes_and_probs(process)
    saved = (
        alive.loc[common, "liquid"] + BASE_INCOME - alive.loc[common, "consumption"]
    ).to_numpy()
    shock = dead.loc[common, "liquid"].to_numpy() - GROSS_RETURN * saved

    assert len(common) == n
    distance = np.abs(shock[:, None] - INCOME_SCALE * np.exp(nodes)[None, :]).min(1)
    np.testing.assert_allclose(distance, 0.0, atol=1e-4)
