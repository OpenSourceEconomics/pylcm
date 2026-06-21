"""Dobrescu-Shanker housing model, keeper (no-house-trade) regime.

The Dobrescu-Shanker housing model has liquid assets, a durable housing stock,
a proportional house-trade cost, a Markov income state, and a discrete
adjust/keep choice. The *keeper* branch (`obj_noadj`) holds the house fixed
(`h' = h`) and chooses only next-period liquid assets, so it is a plain 1-D
DC-EGM problem:

- the Euler state is liquid assets `liquid_assets` (`a`),
- the continuous action is non-durable consumption `consumption` (`c`),
- housing `housing` (`h`) is a **passive** continuous state: its transition is
  the identity (`fixed_transition("housing")`), decision-independent, so it
  rides along as a value-function grid axis exactly like any other passive
  state,
- income `income` (`z`) is a two-state Markov chain entering resources through
  its value.

The defining feature versus the existing passive-state fixtures is that
utility *reads the passive housing state directly* through the service flow
`alpha * log(housing)`. The DC-EGM envelope condition forbids utility reading
the **Euler** state, but reading a passive state is allowed — so the keeper is
expressible today.

Calibration mirrors the InverseDCDP `housing.py` defaults (`r=0.024`,
`r_H=0.10`, `beta=0.945`, `alpha=0.66`, `delta=0.10`, `gamma_c=1.458`,
`b=0.01`, income states `(0.1, 1.0)` with the Markov matrix
`((0.09, 0.91), (0.06, 0.94))`). Grids are kept tiny — this is a feasibility
probe, not an accuracy benchmark.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.transition import fixed_transition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal `dead` regime.
N_PERIODS = 4

# Income state values (`z_vals` in InverseDCDP `housing.py`); indexed by the
# `Income` code.
INCOME_LOW = 0.1
INCOME_HIGH = 1.0

# Income Markov matrix `Pi`: row `i` is the next-period distribution
# `[P(low), P(high)]` given current income node `i`.
INCOME_PI = ((0.09, 0.91), (0.06, 0.94))

# Tiny probe grids. The liquid-asset grid is the Euler state; the savings grid
# starts at the borrowing limit `b` and is cubically clustered toward it where
# the value function curves hardest, reaching above the Euler grid's top so the
# endogenous grid is not edge-clamped.
BORROWING_LIMIT = 0.01
LIQUID_ASSETS_GRID = LinSpacedGrid(start=BORROWING_LIMIT, stop=20.0, n_points=12)
HOUSING_GRID = LinSpacedGrid(start=0.5, stop=4.0, n_points=4)
CONSUMPTION_GRID = LinSpacedGrid(start=0.05, stop=25.0, n_points=60)
SAVINGS_GRID = IrregSpacedGrid(
    points=tuple(BORROWING_LIMIT + 25.0 * (i / 79) ** 3 for i in range(80))
)


@categorical(ordered=False)
class HousingKeeperRegimeId:
    keeper: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Income:
    low: ScalarInt
    high: ScalarInt


def income_value(income: DiscreteState) -> FloatND:
    """Map the discrete income node to its labor-income value `z`."""
    return jnp.where(income == Income.low, INCOME_LOW, INCOME_HIGH)


def utility(
    consumption: ContinuousAction,
    housing: ContinuousState,
    gamma_c: float,
    alpha: float,
) -> FloatND:
    """CRRA consumption utility plus a log housing-service flow.

    Reads the continuous action `consumption` and the *passive* continuous
    state `housing`; it does not read the Euler state `liquid_assets`.
    """
    return (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c) + alpha * jnp.log(
        housing
    )


def resources(
    liquid_assets: ContinuousState,
    housing: ContinuousState,
    income_value: FloatND,
    r: float,
    r_H: float,
    delta: float,
) -> FloatND:
    """Beginning-of-period cash-on-hand of the keeper.

    `(1 + r) * liquid_assets + (1 + r_H) * housing * (1 - delta) + income`. The
    house is not retraded, so it carries no transaction cost. Strictly
    increasing in `liquid_assets`; independent of the continuous action.
    """
    return (
        (1.0 + r) * liquid_assets + (1.0 + r_H) * housing * (1.0 - delta) + income_value
    )


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision liquid balance `a' = resources - consumption`."""
    return resources - consumption


def next_liquid_assets(savings: FloatND) -> ContinuousState:
    """Euler-state law: next liquid assets equal post-decision savings."""
    return savings


def inverse_marginal_utility(marginal_continuation: FloatND, gamma_c: float) -> FloatND:
    """Invert the consumption marginal utility: `c = (mc) ** (-1 / gamma_c)`."""
    return marginal_continuation ** (-1.0 / gamma_c)


def income_transition(income: DiscreteState) -> FloatND:
    """Markov income law: the row of `Pi` for the current income node.

    Returns the next-period distribution `[P(low), P(high)]`, aligned to the
    `Income` codes (`low=0`, `high=1`).
    """
    pi = jnp.asarray(INCOME_PI)
    return pi[income]


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    """Transition to `dead` at the final living age, else stay a keeper."""
    return jnp.where(
        age >= final_age_alive,
        HousingKeeperRegimeId.dead,
        HousingKeeperRegimeId.keeper,
    )


DCEGM_SOLVER = DCEGM(
    continuous_state="liquid_assets",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=32,
)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


def _active(age: int) -> bool:
    return age < 40 + (N_PERIODS - 1) * 10


def build_working_regime() -> UserRegime:
    """Build the keeper DC-EGM regime (the user-facing `Regime`).

    Liquid assets are the Euler state, consumption the continuous action,
    housing a passive (identity-transition) continuous state read by utility,
    and income a two-state Markov discrete state entering resources.
    """
    return UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "liquid_assets": LIQUID_ASSETS_GRID,
            "housing": HOUSING_GRID,
            "income": DiscreteGrid(Income),
        },
        state_transitions={
            "liquid_assets": next_liquid_assets,
            "housing": fixed_transition("housing"),
            "income": MarkovTransition(income_transition),
        },
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
            "income_value": income_value,
        },
        solver=DCEGM_SOLVER,
    )


def build_model() -> Model:
    """Build the keeper model (the GPU solve target).

    A single non-terminal keeper regime plus the shared terminal `dead`
    regime.
    """
    return Model(
        regimes={"keeper": build_working_regime(), "dead": dead},
        ages=_ages(),
        regime_id_class=HousingKeeperRegimeId,
    )


def build_params() -> dict:
    """Calibration parameters for the keeper model.

    Mirrors the InverseDCDP `housing.py` defaults for the preference and
    return parameters; the income process enters resources through
    `income_value`, so it needs no params here.
    """
    return {
        "discount_factor": 0.945,
        "final_age_alive": 40 + (N_PERIODS - 2) * 10,
        "keeper": {
            "utility": {"gamma_c": 1.458, "alpha": 0.66},
            "resources": {"r": 0.024, "r_H": 0.10, "delta": 0.10},
            "inverse_marginal_utility": {"gamma_c": 1.458},
        },
    }
