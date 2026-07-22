"""DC-EGM whose own `resources` reads a runtime-param process state.

A process state with runtime-supplied distribution params (a
`RouwenhorstAR1Process` whose `rho` / `sigma` / `mu` arrive in the params dict)
is a node-valued discrete dimension whose grid points are resolved once per
solve. When the regime's own `resources` function reads that node value
directly — the wage level entering the budget as income — the kernel must feed
the *resolved* node value into the resources query, both on the publish grid
and inside the continuation's expectation over next-period nodes.

The oracle is a dense-grid brute-force solve of the same economics (same wealth
and wage grids, dense consumption grid, explicit borrowing constraint). The
process params flow through the same runtime channel in both solvers, so a
correct kernel matches the brute value function up to the brute solver's
consumption-grid resolution.
"""

import functools

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

N_PERIODS = 4
N_WAGE_NODES = 5


@categorical(ordered=False)
class ProcessResourcesRegimeId:
    alive: ScalarInt
    dead: ScalarInt


WEALTH_GRID = LinSpacedGrid(start=1.0, stop=60.0, n_points=120)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=60.0, n_points=600)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(60.0 * (i / 119) ** 3 for i in range(120)))

# The head-to-head lives on the wealth interior, where both solvers are
# well-defined. Below `N_BRUTE_LOW_NODES` the coarse consumption grid makes the
# brute oracle undershoot (log utility curves hardest near the borrowing limit,
# and the wage income shifts that region up the wealth grid), where DC-EGM's
# constrained segment is exact. Above the top `N_BRUTE_HIGH_NODES` the brute
# next-wealth lookup edge-clamps at the wealth grid's ceiling (the budget pushes
# the high-wealth continuation past the grid top), so its value saturates while
# DC-EGM keeps resolving.
N_BRUTE_LOW_NODES = 30
N_BRUTE_HIGH_NODES = 30


def wage_income(wage: ContinuousState) -> FloatND:
    """Labor income is the exponential of the Markov log-wage node."""
    return jnp.exp(wage)


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(
    wealth: ContinuousState, wage_income: FloatND, interest_rate: float
) -> FloatND:
    """Financial resources: return on wealth plus the wage node's income.

    Reads the process state `wage` (through `wage_income`), so the kernel must
    bind the resolved wage node value here rather than a build-time placeholder.
    """
    return (1.0 + interest_rate) * wealth + wage_income


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_brute(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
    wage_income: FloatND,
) -> ContinuousState:
    return (1.0 + interest_rate) * wealth + wage_income - consumption


def borrowing_constraint(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
    wage_income: FloatND,
) -> BoolND:
    return (
        next_wealth_brute(
            wealth=wealth,
            consumption=consumption,
            interest_rate=interest_rate,
            wage_income=wage_income,
        )
        >= 0.0
    )


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        ProcessResourcesRegimeId.dead,
        ProcessResourcesRegimeId.alive,
    )


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


@functools.cache
def _get_model(solver: str) -> Model:
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])

    def active(age: int, la: float = last_age) -> bool:
        return age < la

    states = {
        "wealth": WEALTH_GRID,
        "wage": RouwenhorstAR1Process(n_points=N_WAGE_NODES),
    }
    if solver == "dcegm":
        alive = UserRegime(
            transition=next_regime,
            active=active,
            actions={"consumption": CONSUMPTION_GRID},
            states=states,
            state_transitions={"wealth": next_wealth},
            functions={
                "utility": utility,
                "wage_income": wage_income,
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            },
            solver=DCEGM_SOLVER,
        )
    else:
        alive = UserRegime(
            transition=next_regime,
            active=active,
            actions={"consumption": CONSUMPTION_GRID},
            states=states,
            state_transitions={"wealth": next_wealth_brute},
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={"utility": utility, "wage_income": wage_income},
        )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=ProcessResourcesRegimeId,
    )


def _get_params(solver: str) -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    wage_params = {"mu": 0.0, "sigma": 0.25, "rho": 0.6}
    if solver == "dcegm":
        alive_params = {
            "wage": wage_params,
            "resources": {"interest_rate": 0.0},
        }
    else:
        alive_params = {
            "wage": wage_params,
            "next_wealth": {"interest_rate": 0.0},
            "borrowing_constraint": {"interest_rate": 0.0},
        }
    return {
        "discount_factor": 0.95,
        "final_age_alive": final_age_alive,
        "alive": alive_params,
    }


def test_dcegm_resources_reading_process_matches_dense_brute_force():
    """DC-EGM with a process-reading `resources` matches dense brute force.

    The wage node enters the budget as income through the regime's own
    resources function. A correct kernel binds the resolved wage grid into both
    the publish-grid resources and the continuation's node expectation, so the
    value function matches the brute oracle up to the brute consumption-grid
    resolution, excluding the lowest wealth nodes.
    """
    dcegm_solution = _get_model("dcegm").solve(
        params=_get_params("dcegm"), log_level="debug"
    )
    brute_solution = _get_model("brute").solve(
        params=_get_params("brute"), log_level="debug"
    )

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["alive"])
        dcegm_V = np.asarray(dcegm_solution[period]["alive"])
        assert brute_V.shape == dcegm_V.shape == (N_WAGE_NODES, 120)
        interior = slice(N_BRUTE_LOW_NODES, 120 - N_BRUTE_HIGH_NODES)
        np.testing.assert_allclose(
            dcegm_V[:, interior],
            brute_V[:, interior],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )
