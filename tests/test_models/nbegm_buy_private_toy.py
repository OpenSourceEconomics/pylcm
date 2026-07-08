"""One-asset toy with a binary discrete insurance choice (the F-E discrete envelope).

Each period the agent chooses whether to buy private insurance, paying a premium
that lowers cash-on-hand. NBEGM solves the continuous consumption/savings
subproblem inside each insurance branch; the discrete choice is taken by the upper
envelope over the branch values. The brute variant maximises over both the discrete
choice and consumption on a dense grid and is the agreement oracle.
"""

from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid,
    resolve_solver,
    utility,
)


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


def resources(
    liquid: ContinuousState, buy_private: DiscreteAction, premium: float
) -> FloatND:
    """Cash-on-hand: liquid plus base income, less the premium when buying."""
    return liquid + 3.0 - premium * buy_private


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the two-regime (alive, dead) buy-private one-asset toy."""
    alive_functions = {"utility": utility, "resources": resources}
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
    )

    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=next_liquid,
        alive_solver=alive_solver,
        constraints={"feasible": feasible},
        extra_actions={"buy_private": DiscreteGrid(BuyPrivate)},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    premium: float = 1.5,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the buy-private one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "resources": {"premium": premium},
            "alive": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
