"""Minimal consumption-savings model for economic validation tests.

A stripped-down model (no health, no discrete states) designed to test
precautionary savings behavior under different shock parametrizations.

"""

from typing import Literal

from jax import numpy as jnp

import lcm
from lcm import AgeGrid, LinSpacedGrid, Model, Regime, RegimeTransition, categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

_SHOCK_GRID_CLASSES = {
    "normal_gh": lcm.shocks.iid.Normal,
    "rouwenhorst": lcm.shocks.ar1.Rouwenhorst,
}

_SHOCK_GRID_KWARGS: dict[str, dict[str, bool]] = {
    "normal_gh": {"gauss_hermite": True},
    "rouwenhorst": {},
}


def get_model(
    n_periods: int,
    shock_type: Literal["normal_gh", "rouwenhorst"],
) -> Model:
    def next_wealth(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        income: ContinuousState,
    ) -> FloatND:
        return wealth - consumption + jnp.exp(income)

    def next_regime(period: int) -> ScalarInt:
        terminal = period >= n_periods - 1
        return jnp.where(terminal, RegimeId.terminal, RegimeId.alive)

    def wealth_constraint(
        consumption: ContinuousAction,
        wealth: ContinuousState,
    ) -> BoolND:
        return consumption <= wealth

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def alive_active(age):
        return age < n_periods

    def terminal_active(age):
        return age == n_periods

    @categorical
    class RegimeId:
        alive: int
        terminal: int

    alive = Regime(
        active=alive_active,
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=20,
                n_points=7,
                transition=next_wealth,
            ),
            "income": _SHOCK_GRID_CLASSES[shock_type](
                n_points=5,
                **_SHOCK_GRID_KWARGS[shock_type],
            ),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=5, n_points=7),
        },
        transition=RegimeTransition(next_regime),
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )

    terminal = Regime(
        transition=None,
        active=terminal_active,
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={"alive": alive, "terminal": terminal},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, stop=n_periods, step="Y"),
    )


def get_params(
    shock_type: Literal["normal_gh", "rouwenhorst"],
    *,
    sigma: float,
    mu: float = 0.0,
    rho: float = 0.0,
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
        },
        "terminal": {},
    }
