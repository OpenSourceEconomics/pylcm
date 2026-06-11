"""The deterministic retirement model of Iskhakov, Jørgensen, Rust & Schjerning (2017).

A worker chooses consumption and whether to keep working or retire; retirement is
absorbing and death arrives deterministically at a known age. Log utility with a
work disutility. The model replicates "The endogenous grid method for
discrete-continuous dynamic choice models with (or without) taste shocks",
Quantitative Economics 8(2), 317-365, https://doi.org/10.3982/QE643, which provides
a closed-form solution (shipped as test data in `tests/data/analytical_solution/`).

The discrete retirement choice makes the value function non-concave and produces
the paper's signature saw-tooth consumption function: each tooth corresponds to a
different optimal retirement age. As people retire later, their lifetime wealth
increases, which changes the optimal consumption path; the optimal retirement age
moves period by period as current wealth rises, so consumption jumps at the wealth
thresholds in between.
"""

import functools

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class LaborSupply:
    work: ScalarInt
    retire: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def utility_working(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    work_disutility = jnp.where(is_working, disutility_of_work, 0.0)
    return jnp.log(consumption) - work_disutility


def utility_retirement(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def labor_income(is_working: BoolND, wage: float | FloatND) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def resources(wealth: ContinuousState) -> FloatND:
    """Resources out of which consumption is paid; the classic case is wealth."""
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """End-of-period savings (the post-decision state)."""
    return resources - consumption


def next_wealth_from_savings(
    savings: FloatND, labor_income: FloatND, interest_rate: float
) -> ContinuousState:
    """Wealth transition written in terms of the post-decision state.

    Algebraically identical to `next_wealth`'s
    `(1 + interest_rate) * (wealth - consumption) + labor_income`.
    """
    return (1 + interest_rate) * savings + labor_income


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    """Inverse of `u'(c) = 1/c` for log utility (work disutility is additive)."""
    return 1.0 / marginal_continuation


def next_regime_from_working(
    labor_supply: DiscreteAction,
    age: int,
    final_age_alive: float,
) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        jnp.where(
            labor_supply == LaborSupply.retire,
            RegimeId.retirement,
            RegimeId.working_life,
        ),
    )


def next_regime_from_retirement(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retirement,
    )


WEALTH_GRID = LinSpacedGrid(start=1, stop=400, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=400, n_points=500)

_DEFAULT_AGE_GRID = AgeGrid(start=40, stop=70, step="10Y")  # 4 periods
_DEFAULT_LAST_AGE = _DEFAULT_AGE_GRID.exact_values[-1]


working_life = Regime(
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    transition=next_regime_from_working,
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
    },
    active=lambda age: age < _DEFAULT_LAST_AGE,
)

retirement = Regime(
    transition=next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    constraints={"borrowing_constraint": borrowing_constraint},
    functions={"utility": utility_retirement},
    active=lambda age: age < _DEFAULT_LAST_AGE,
)

dead = Regime(
    transition=None,
    functions={"utility": lambda: 0.0},
    active=lambda _age: True,
)


# Exogenous end-of-period savings grid for the DC-EGM solver; the lower bound
# is the borrowing limit (savings >= 0 encodes `consumption <= wealth`). Nodes
# are cubically clustered toward the limit: the value function curves hardest
# where the constraint starts to bind, and the published V is interpolated
# from endogenous points spaced like the savings nodes.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(400.0 * (i / 199) ** 3 for i in range(200)))

DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    # The final decision period consumes everything, so its carry in the
    # queried resources range consists of constrained-segment points only;
    # 64 of them keep the geometric spacing ratio (and hence the carry
    # interpolation error) small.
    n_constrained_points=64,
)

# DC-EGM variants of the two non-terminal regimes. The economic content is
# identical to `working_life` / `retirement`; the spec differs where the
# algorithm requires it:
# - `resources`, `savings`, and `inverse_marginal_utility` are declared
#   regime functions,
# - the wealth transition consumes `savings` (the post-decision state)
#   instead of wealth and consumption directly,
# - the borrowing constraint is dropped — DC-EGM enforces the budget
#   identity and the savings-grid lower bound intrinsically.
dcegm_working_life = Regime(
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth_from_savings},
    transition=next_regime_from_working,
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    solver=DCEGM_SOLVER,
    active=lambda age: age < _DEFAULT_LAST_AGE,
)

dcegm_retirement = Regime(
    transition=next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth_from_savings},
    functions={
        "utility": utility_retirement,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    solver=DCEGM_SOLVER,
    active=lambda age: age < _DEFAULT_LAST_AGE,
)


@functools.cache
def get_model(n_periods: int) -> Model:
    """Create the three-regime retirement model.

    Args:
        n_periods: Number of periods. The last period is spent in the terminal
            `dead` regime; the paper's five-decision-period parametrization
            corresponds to `n_periods=6`.

    Returns:
        A configured Model instance.

    """
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": retirement.replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


@functools.cache
def get_dcegm_model(n_periods: int) -> Model:
    """Create the retirement model with the DC-EGM solver on both regimes.

    Mathematically equivalent to `get_model` (same utility, budget, and
    transitions; `get_params` works unchanged), but solved by Euler-equation
    inversion on the exogenous savings grid instead of grid search — no
    consumption grid enters the solve. Forward simulation works; simulated
    consumption is restricted to the consumption grid (the intrinsic budget
    constraint is applied as a feasibility mask).

    Args:
        n_periods: Number of periods. The last period is spent in the terminal
            `dead` regime; the paper's five-decision-period parametrization
            corresponds to `n_periods=6`.

    Returns:
        A configured Model instance.

    """
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": dcegm_working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": dcegm_retirement.replace(
                active=lambda age, la=last_age: age < la
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=RegimeId,
    )


def get_params(
    n_periods: int,
    discount_factor: float = 0.95,
    disutility_of_work: float = 0.5,
    interest_rate: float = 0.05,
    wage: float = 10.0,
) -> dict:
    """Get parameters for the retirement model.

    The paper's analytical-solution parametrization is `discount_factor=0.98`,
    `disutility_of_work=1.0`, `interest_rate=0.0`, `wage=20.0`.

    Args:
        n_periods: Number of periods (must match `get_model`).
        discount_factor: Discount factor.
        disutility_of_work: Utility cost of working.
        interest_rate: Interest rate on savings.
        wage: Per-period labor income when working.

    Returns:
        Parameter dict ready for `model.solve()`.

    """
    final_age_alive = 40 + (n_periods - 2) * 10
    return {
        "discount_factor": discount_factor,
        "interest_rate": interest_rate,
        "final_age_alive": final_age_alive,
        "working_life": {
            "utility": {"disutility_of_work": disutility_of_work},
            "labor_income": {"wage": wage},
        },
        "retirement": {
            "next_wealth": {"labor_income": 0.0},
        },
    }


__all__ = [
    "CONSUMPTION_GRID",
    "DCEGM_SOLVER",
    "SAVINGS_GRID",
    "WEALTH_GRID",
    "LaborSupply",
    "RegimeId",
    "borrowing_constraint",
    "dcegm_retirement",
    "dcegm_working_life",
    "dead",
    "get_dcegm_model",
    "get_model",
    "get_params",
    "inverse_marginal_utility",
    "is_working",
    "labor_income",
    "next_regime_from_retirement",
    "next_regime_from_working",
    "next_wealth",
    "next_wealth_from_savings",
    "resources",
    "retirement",
    "savings",
    "utility_retirement",
    "utility_working",
    "working_life",
]
