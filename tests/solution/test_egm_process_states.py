"""DC-EGM with stochastic process states (Rouwenhorst AR(1) and IID income).

A process state is a node-valued discrete dimension with intrinsic transition
weights $w(\\text{node}' \\mid \\text{node}, \\text{params})$. In a DC-EGM
regime it rides along exactly like a discrete state on the own side (one carry
row and one V slice per node) while the child side takes an expectation: the
child's node is distributed, so the carry read indexes the child rows at every
node, performs the full read there (resources interpolation, passive blend,
discrete-action aggregation), and weights the resulting per-node values and
marginals — the process expectation sits *outside* the action aggregation,
matching the brute-force solver's `jnp.average` of the already action
aggregated next-period V.

The oracle is a dense-grid brute-force solve of a mathematically equivalent
spec (same wealth and income grids, dense consumption grid, explicit budget
constraint). Process params (`mu`, `sigma`, `rho`) are supplied at runtime, so
the intrinsic weights flow through the same params channel as in brute force.
"""

import functools

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    RouwenhorstAR1Process,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal `dead` regime.
N_PERIODS = 4

# Number of discretization nodes of the income process.
N_INCOME_NODES = 5

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=60.0, n_points=60)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=60.0, n_points=300)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(60.0 * (i / 119) ** 3 for i in range(120)))

# Lowest wealth nodes excluded from the brute-force comparison: there the
# coarse consumption grid makes brute force itself unreliable (the same
# exclusion the discrete and passive DC-EGM tests use).
N_BRUTE_UNSTABLE_NODES = 8


@categorical(ordered=False)
class ProcessRegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class LaborChoice:
    work: ScalarInt
    rest: ScalarInt


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborChoice.work


def labor_income(is_working: BoolND, income: ContinuousState, wage: float) -> FloatND:
    return jnp.where(is_working, wage * jnp.exp(income), 0.0)


def utility_with_labor(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - jnp.where(is_working, disutility_of_work, 0.0)


def utility_consumption_only(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_from_savings_with_labor(
    savings: FloatND, labor_income: FloatND, interest_rate: float
) -> ContinuousState:
    return (1 + interest_rate) * savings + labor_income


def next_wealth_brute_with_labor(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def next_wealth_from_savings_iid(
    savings: FloatND, income: ContinuousState, interest_rate: float
) -> ContinuousState:
    return (1 + interest_rate) * savings + jnp.exp(income)


def next_wealth_brute_iid(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        ProcessRegimeId.dead,
        ProcessRegimeId.alive,
    )


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _income_process(shock_type: str) -> RouwenhorstAR1Process | NormalIIDProcess:
    """Income process with all distribution params supplied at runtime."""
    if shock_type == "rouwenhorst":
        return RouwenhorstAR1Process(n_points=N_INCOME_NODES)
    return NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)


@functools.cache
def _get_model(solver: str, shock_type: str) -> Model:
    """Build one (solver, shock_type) model variant.

    - `shock_type="rouwenhorst"`: persistent AR(1) income scaling the wage of
      a discrete work/rest choice — the income node moves the work margin, so
      the ordering of the process expectation against the discrete-action
      aggregation is observable.
    - `shock_type="iid"`: Gauss-Hermite IID income entering the wealth
      transition additively; consumption is the only action.
    """
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])

    def active(age: int, la: float = last_age) -> bool:
        return age < la

    states = {"wealth": WEALTH_GRID, "income": _income_process(shock_type)}
    if shock_type == "rouwenhorst":
        actions = {
            "labor_supply": DiscreteGrid(LaborChoice),
            "consumption": CONSUMPTION_GRID,
        }
        shared_functions = {
            "utility": utility_with_labor,
            "labor_income": labor_income,
            "is_working": is_working,
        }
        dcegm_next_wealth = next_wealth_from_savings_with_labor
        brute_next_wealth = next_wealth_brute_with_labor
    else:
        actions = {"consumption": CONSUMPTION_GRID}
        shared_functions = {"utility": utility_consumption_only}
        dcegm_next_wealth = next_wealth_from_savings_iid
        brute_next_wealth = next_wealth_brute_iid

    if solver == "dcegm":
        alive = UserRegime(
            transition=next_regime,
            active=active,
            actions=actions,
            states=states,
            state_transitions={"wealth": dcegm_next_wealth},
            functions={
                **shared_functions,
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
            actions=actions,
            states=states,
            state_transitions={"wealth": brute_next_wealth},
            constraints={"borrowing_constraint": borrowing_constraint},
            functions=shared_functions,
        )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=ProcessRegimeId,
    )


def _get_params(shock_type: str) -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    if shock_type == "rouwenhorst":
        income_params = {"mu": 0.0, "sigma": 0.25, "rho": 0.6}
        alive_params = {
            "income": income_params,
            "labor_income": {"wage": 15.0},
            "utility": {"disutility_of_work": 0.5},
        }
    else:
        alive_params = {"income": {"mu": 0.0, "sigma": 0.3}}
    return {
        "discount_factor": 0.95,
        "interest_rate": 0.0,
        "final_age_alive": final_age_alive,
        "alive": alive_params,
    }


def test_rouwenhorst_process_matches_dense_brute_force():
    """DC-EGM with a Rouwenhorst income process matches dense brute force.

    Each income node is an axis of V and the carry; the child's node is
    distributed per the intrinsic AR(1) weights at the parent's node. The
    income node scales the wage, so the work margin flips across nodes at
    high wealth — an expectation taken inside the discrete-action aggregation
    (instead of outside, where brute force takes it) would shift V by the
    value gap between the choices and fail the comparison. Agreement is up to
    the brute solver's consumption-grid resolution, excluding the lowest
    wealth nodes where the coarse consumption grid makes brute force itself
    unreliable.
    """
    params = _get_params("rouwenhorst")
    dcegm_solution = _get_model("dcegm", "rouwenhorst").solve(
        params=params, log_level="debug"
    )
    brute_solution = _get_model("brute", "rouwenhorst").solve(
        params=params, log_level="debug"
    )

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["alive"])
        dcegm_V = np.asarray(dcegm_solution[period]["alive"])
        assert brute_V.shape == dcegm_V.shape == (N_INCOME_NODES, 60)
        np.testing.assert_allclose(
            dcegm_V[:, N_BRUTE_UNSTABLE_NODES:],
            brute_V[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def test_iid_process_matches_dense_brute_force():
    """DC-EGM with a Gauss-Hermite IID income process matches dense brute force.

    IID weights are node-independent rows of the intrinsic transition matrix;
    the child read is averaged over the quadrature nodes with those weights.
    """
    params = _get_params("iid")
    dcegm_solution = _get_model("dcegm", "iid").solve(params=params, log_level="debug")
    brute_solution = _get_model("brute", "iid").solve(params=params, log_level="debug")

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["alive"])
        dcegm_V = np.asarray(dcegm_solution[period]["alive"])
        assert brute_V.shape == dcegm_V.shape == (N_INCOME_NODES, 60)
        np.testing.assert_allclose(
            dcegm_V[:, N_BRUTE_UNSTABLE_NODES:],
            brute_V[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )
