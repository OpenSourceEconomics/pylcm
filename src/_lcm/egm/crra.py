"""Shared CRRA felicity for the EGM steps."""

import jax.numpy as jnp

from lcm.typing import FloatND, ScalarFloat


def crra_utility(consumption: FloatND, crra: ScalarFloat | float) -> FloatND:
    """Return CRRA felicity, with the log limit at `crra == 1`.

    The power branch is evaluated at a *safe* exponent so that neither
    argument's derivative is poisoned at the log limit. `jnp.where` evaluates
    both branches, so the raw power form contributes `c ** 0 / 0 == inf` at
    `crra == 1` and the `0 * inf` that flows back through the select turns
    `jax.grad` into NaN in both `consumption` and `crra`. Substituting the
    exponent `1.0` on the log branch keeps the untaken expression finite; its
    derivative with respect to `crra` is zero there, which is the convention
    this function publishes at the (non-differentiable) limit.

    The branch predicate is exact equality rather than a tolerance band on
    purpose: the two expressions differ by the additive constant
    `1 / (1 - crra)`, which diverges as `crra` approaches one, so widening the
    switch would introduce a jump of size `1 / tol` instead of removing one.

    Args:
        consumption: Consumption level(s); any shape.
        crra: Coefficient of relative risk aversion.

    Returns:
        CRRA utility with the shape of `consumption`.

    """
    is_log = crra == 1.0
    safe_exponent = jnp.where(is_log, 1.0, 1.0 - crra)
    return jnp.where(
        is_log,
        jnp.log(consumption),
        consumption**safe_exponent / safe_exponent,
    )
