"""Dobrescu--Shanker housing benchmark (deterministic-income brute form).

A liquid-asset + durable-housing model with a discrete **adjust / keep** choice and a
proportional transaction cost, the model behind the RFC-vs-NEGM housing comparison. It
is written in dense-grid brute-solvable form -- the oracle the 2-D EGM / NEGM kernels
are validated against.

Each period the agent holds liquid assets `liquid` and a house `housing`, and chooses
to **keep** the house or **adjust** it:

- **Keep:** the house is retained (`next_housing = housing`), no transaction cost, and
  cash-on-hand is `R*liquid + income` (the house is NOT liquidated -- only its service
  flow `alpha*log(housing)` is enjoyed).
- **Adjust:** the old house is sold for `R_H*housing*(1-delta)`, a new house
  `new_housing` is bought for `new_housing*(1 + tau)` (the proportional transaction
  cost), and cash-on-hand is `R*liquid + R_H*housing*(1-delta) + income - new_housing*(1
  + tau)`. The agent lives in and carries forward the new house.

Utility is CRRA in non-durable consumption plus a within-period housing service:
`(c**(1-gamma_c) - 1)/(1-gamma_c) + alpha*log(serviced_housing)`, so utility reads the
housing state/choice directly (unlike the pension model). Budgets, returns, and the
transaction cost are verified against InverseDCDP `housing/housing.py`
(`obj_noadj`/`obj_adj`); calibration from `ConsumerProblem.__init__`.

Income is a deterministic parameter here; the faithful benchmark uses a 2-state Markov
income (`z_vals=(0.1, 1.0)`, transition `Pi`), added as a process-state follow-up. The
terminal bequest is a simple consume-liquid CRRA value (the source's `term_u` scaling
is flagged, not yet replicated).
"""

import jax.numpy as jnp

from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.grids import DiscreteGrid
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class AdjustChoice:
    keep: ScalarInt
    adjust: ScalarInt


def _cash_on_hand(
    liquid: ContinuousState,
    housing: ContinuousState,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
    return_liquid: float,
    return_housing: float,
    depreciation: float,
    income: float,
    transaction_cost: float,
) -> FloatND:
    """Beginning-of-period cash-on-hand, branching on the adjust/keep choice."""
    keep = (1.0 + return_liquid) * liquid + income
    adjust_cash = (
        (1.0 + return_liquid) * liquid
        + (1.0 + return_housing) * housing * (1.0 - depreciation)
        + income
        - new_housing * (1.0 + transaction_cost)
    )
    return jnp.where(adjust == AdjustChoice.adjust, adjust_cash, keep)


def _serviced_housing(
    housing: ContinuousState,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
) -> FloatND:
    """The house lived in this period: the new house if adjusting, else the old one."""
    return jnp.where(adjust == AdjustChoice.adjust, new_housing, housing)


def utility(
    consumption: ContinuousAction,
    housing: ContinuousState,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
    crra: float,
    housing_weight: float,
) -> FloatND:
    """CRRA consumption utility plus the housing-service flow `alpha*log(h)`."""
    serviced = _serviced_housing(housing, adjust, new_housing)
    consumption_utility = (consumption ** (1.0 - crra) - 1.0) / (1.0 - crra)
    return consumption_utility + housing_weight * jnp.log(serviced)


def bequest(liquid: ContinuousState, housing: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume liquid wealth plus the resale value of the house.

    Uses the un-normalized CRRA form, without the `-1` normalization the flow
    utility carries. The `-1` is an additive level constant; omitting it here
    shifts the value function by a constant and leaves the optimal policy
    unchanged, so the two forms need not match term for term.
    """
    return (liquid + housing) ** (1.0 - crra) / (1.0 - crra)


def next_liquid(
    liquid: ContinuousState,
    housing: ContinuousState,
    consumption: ContinuousAction,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
    return_liquid: float,
    return_housing: float,
    depreciation: float,
    income: float,
    transaction_cost: float,
) -> ContinuousState:
    """Next-period liquid assets: cash-on-hand net of consumption."""
    cash = _cash_on_hand(
        liquid=liquid,
        housing=housing,
        adjust=adjust,
        new_housing=new_housing,
        return_liquid=return_liquid,
        return_housing=return_housing,
        depreciation=depreciation,
        income=income,
        transaction_cost=transaction_cost,
    )
    return cash - consumption


def next_housing(
    housing: ContinuousState,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
) -> ContinuousState:
    """Next-period housing stock: the new house if adjusting, else the old one."""
    return _serviced_housing(housing, adjust, new_housing)


def feasible(
    liquid: ContinuousState,
    housing: ContinuousState,
    consumption: ContinuousAction,
    adjust: DiscreteAction,
    new_housing: ContinuousAction,
    return_liquid: float,
    return_housing: float,
    depreciation: float,
    income: float,
    transaction_cost: float,
    borrowing_floor: float,
) -> BoolND:
    """Next-period liquid assets must stay at or above the borrowing floor."""
    cash = _cash_on_hand(
        liquid=liquid,
        housing=housing,
        adjust=adjust,
        new_housing=new_housing,
        return_liquid=return_liquid,
        return_housing=return_housing,
        depreciation=depreciation,
        income=income,
        transaction_cost=transaction_cost,
    )
    return consumption <= cash - borrowing_floor


def next_regime_from_working(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age + 1 >= final_age_alive, RegimeId.dead, RegimeId.working)


def get_model(
    *,
    n_periods: int = 4,
    n_liquid: int = 12,
    n_housing: int = 8,
    n_consumption: int = 16,
    n_new_housing: int = 8,
    borrowing_floor: float = 0.01,
    liquid_max: float = 50.0,
    housing_max: float = 50.0,
) -> Model:
    """Create the two-regime (working, dead) deterministic-income housing model.

    Grid sizes default to a small oracle scale; pass larger values for a finer
    reference solve.
    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(
        start=borrowing_floor, stop=liquid_max, n_points=n_liquid
    )
    housing_grid = LinSpacedGrid(
        start=borrowing_floor, stop=housing_max, n_points=n_housing
    )
    working = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=borrowing_floor, stop=liquid_max, n_points=n_consumption
            ),
            "new_housing": LinSpacedGrid(
                start=borrowing_floor, stop=housing_max, n_points=n_new_housing
            ),
            "adjust": DiscreteGrid(AdjustChoice),
        },
        states={"liquid": liquid_grid, "housing": housing_grid},
        state_transitions={"liquid": next_liquid, "housing": next_housing},
        constraints={"feasible": feasible},
        transition=next_regime_from_working,
        functions={"utility": utility},
        active=lambda age, fa=final_age: age < fa,
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid, "housing": housing_grid},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def get_params(
    *,
    discount_factor: float = 0.945,
    crra: float = 1.458,
    housing_weight: float = 0.66,
    return_liquid: float = 0.024,
    return_housing: float = 0.10,
    depreciation: float = 0.10,
    transaction_cost: float = 0.20,
    income: float = 1.0,
    borrowing_floor: float = 0.01,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the housing model (faithful calibration from `housing.py`)."""
    transition_args = {
        "return_liquid": return_liquid,
        "return_housing": return_housing,
        "depreciation": depreciation,
        "income": income,
        "transaction_cost": transaction_cost,
    }
    return {
        "discount_factor": discount_factor,
        "final_age_alive": final_age_alive,
        "working": {
            "utility": {"crra": crra, "housing_weight": housing_weight},
            "next_liquid": transition_args,
            "next_housing": {},
            "feasible": {**transition_args, "borrowing_floor": borrowing_floor},
        },
        "dead": {"utility": {"crra": crra}},
    }
