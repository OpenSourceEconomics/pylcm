"""Consumption-savings model with income shocks for economic validation tests.

Supports IID shocks (Normal GH) and persistent AR(1) shocks (Rouwenhorst, Tauchen)
with configurable interest rate and wealth grid type.

With FGP-calibrated parameters (LogSpacedGrid for wealth, interest rate = 0.04,
rho = 0.95, sigma = 0.1269, beta = 0.96), this replicates the simplified benchmark
of Fella, Gallipoli & Pan (RED 2019). See `tests/test_fgp_model.py`.

"""

from typing import Literal

from jax import numpy as jnp

import lcm
from lcm import AgeGrid, LinSpacedGrid, LogSpacedGrid, Model, Regime, categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

ShockType = Literal["normal_gh", "rouwenhorst", "tauchen"]

_SHOCK_GRID_CLASSES = {
    "normal_gh": lcm.shocks.iid.Normal,
    "rouwenhorst": lcm.shocks.ar1.Rouwenhorst,
    "tauchen": lcm.shocks.ar1.Tauchen,
}

_SHOCK_GRID_KWARGS: dict[str, dict[str, bool | float]] = {
    "normal_gh": {"gauss_hermite": True},
    "rouwenhorst": {},
    "tauchen": {"gauss_hermite": False, "n_std": 3.0},
}


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> FloatND:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def wealth_constraint(
    consumption: ContinuousAction,
    wealth: ContinuousState,
) -> BoolND:
    return consumption <= wealth


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


@categorical
class RegimeId:
    alive: int
    dead: int


def get_model(
    n_periods: int,
    shock_type: ShockType,
    *,
    wealth_grid_type: Literal["lin", "log"] = "lin",
    wealth_start: float = 1.0,
    wealth_stop: float = 20.0,
    wealth_n_points: int = 7,
    consumption_n_points: int = 7,
    income_n_points: int = 5,
) -> Model:
    final_age_alive = 20 + (n_periods - 2) * 10

    wealth_grid_cls = LogSpacedGrid if wealth_grid_type == "log" else LinSpacedGrid
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": wealth_grid_cls(
                start=wealth_start,
                stop=wealth_stop,
                n_points=wealth_n_points,
            ),
            "income": _SHOCK_GRID_CLASSES[shock_type](
                n_points=income_n_points,
                **_SHOCK_GRID_KWARGS[shock_type],  # ty: ignore[invalid-argument-type]
            ),
        },
        state_transitions={
            "wealth": next_wealth,
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=5, n_points=consumption_n_points
            ),
        },
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )

    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 10, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def get_params(
    shock_type: ShockType,
    *,
    sigma: float,
    mu: float = 0.0,
    rho: float = 0.0,
    interest_rate: float = 0.0,
    discount_factor: float = 0.95,
) -> dict:
    if shock_type == "normal_gh":
        shock_params: dict[str, float] = {"mu": mu, "sigma": sigma}
    else:
        shock_params = {"mu": mu, "sigma": sigma, "rho": rho}

    return {
        "discount_factor": discount_factor,
        "alive": {
            "income": shock_params,
            "next_wealth": {"interest_rate": interest_rate},
        },
    }
