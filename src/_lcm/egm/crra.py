"""Shared CRRA felicity for the EGM steps."""

import jax.numpy as jnp

from lcm.typing import FloatND, ScalarFloat


def crra_utility(consumption: FloatND, crra: ScalarFloat | float) -> FloatND:
    """Return CRRA felicity, with the log limit at `crra == 1`.

    Args:
        consumption: Consumption level(s); any shape.
        crra: Coefficient of relative risk aversion.

    Returns:
        CRRA utility with the shape of `consumption`.

    """
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )
