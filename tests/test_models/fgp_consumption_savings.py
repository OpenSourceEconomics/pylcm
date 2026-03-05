"""Simplified FGP-style consumption-savings model.

A single-persistent-shock model designed to compare Rouwenhorst vs Tauchen
discretization quality, following the spirit of Fella, Gallipoli & Pan (2019).

FGP's full benchmark uses EGM with 1000 asset x 10000 income grid points and
Gauss-Hermite quadrature — infeasible with brute-force DP. This simplified version
uses coarser grids and a single persistent shock (no transitory component).

"""

from typing import Literal

import jax.numpy as jnp

import lcm
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt

# FGP reference parameters (Section 4, p. 191)
SIGMA_EPS = 0.1269  # sqrt(0.0161)
RHO = 0.95
R = 0.04
BETA = 0.96
N_PERIODS = 10

_SHOCK_GRID_CLASSES = {
    "rouwenhorst": lcm.shocks.ar1.Rouwenhorst,
    "tauchen": lcm.shocks.ar1.Tauchen,
}

_SHOCK_GRID_KWARGS: dict[str, dict[str, bool | float]] = {
    "rouwenhorst": {},
    "tauchen": {"gauss_hermite": False, "n_std": 3.0},
}


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
) -> FloatND:
    return (1 + R) * (wealth - consumption) + jnp.exp(income)


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.terminal, RegimeId.alive)


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
    terminal: int


def get_model(
    shock_type: Literal["rouwenhorst", "tauchen"],
    n_periods: int = N_PERIODS,
) -> Model:
    final_age_alive = n_periods - 2

    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LogSpacedGrid(
                start=0.5,
                stop=50.0,
                n_points=50,
            ),
            "income": _SHOCK_GRID_CLASSES[shock_type](
                n_points=5,
                **_SHOCK_GRID_KWARGS[shock_type],  # ty: ignore[invalid-argument-type]
            ),
        },
        state_transitions={
            "wealth": next_wealth,
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=10.0, n_points=20),
        },
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )

    terminal = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={"alive": alive, "terminal": terminal},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def get_params(
    shock_type: Literal["rouwenhorst", "tauchen"],  # noqa: ARG001
    *,
    rho: float = RHO,
    sigma: float = SIGMA_EPS,
    mu: float = 0.0,
    discount_factor: float = BETA,
) -> dict:
    shock_params: dict[str, float] = {"mu": mu, "sigma": sigma, "rho": rho}

    return {
        "discount_factor": discount_factor,
        "alive": {
            "income": shock_params,
        },
        "terminal": {},
    }
