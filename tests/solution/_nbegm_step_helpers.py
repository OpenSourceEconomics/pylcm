"""Shared helpers for the NBEGM unit-step tests.

The step tests compare a NBEGM one-period solve against a dense single-period
Bellman max over a feasible consumption grid — a convention-free oracle that
needs no EGM machinery. The CRRA utility and the dense oracle live here so every
step test uses the identical formulas.
"""

from collections.abc import Callable

import jax.numpy as jnp

from lcm.typing import Float1D, FloatND, ScalarFloat


def crra_utility(consumption: FloatND, crra: ScalarFloat | float) -> FloatND:
    """CRRA utility, log at `crra == 1`."""
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def dense_brute_value(
    *,
    liquid_grid: Float1D,
    coh_of_liquid: Callable[[FloatND], FloatND],
    next_value_of_liquid: Callable[[FloatND], FloatND],
    crra: ScalarFloat | float,
    discount_factor: ScalarFloat | float,
    gross_return: ScalarFloat | float,
    income: ScalarFloat | float,
    n_consumption: int = 4000,
) -> Float1D:
    """Bellman max over a dense feasible consumption grid — convention-free oracle."""
    coh = coh_of_liquid(liquid_grid)
    fractions = jnp.linspace(1e-4, 1.0, n_consumption)
    # consumption = fraction * coh, broadcast over the liquid grid.
    consumption = fractions[:, None] * coh[None, :]
    savings = coh[None, :] - consumption
    next_liquid = gross_return * savings + income
    value = crra_utility(consumption, crra) + discount_factor * next_value_of_liquid(
        next_liquid
    )
    return jnp.max(value, axis=0)
