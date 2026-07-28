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
    # `jnp.where` evaluates both branches, so the unselected one must stay
    # finite: at `crra == 1` the power branch is `c**0 / 0`, and while that
    # infinity is discarded from the primal it reaches the derivative as
    # `0 * inf`, giving a NaN marginal utility for a perfectly ordinary model.
    # Substituting an exponent of one there costs nothing — the branch is not
    # selected — and leaves `c / 1` behind instead.
    #
    # The predicate is exact equality on purpose. This felicity omits the `-1`
    # that would make the power branch tend to `log` as `crra → 1`, so the two
    # branches genuinely differ by `1 / (1 - crra)` nearby. Widening the test to
    # a tolerance band would not remove a discontinuity but introduce one, of
    # size `1 / tol`.
    is_log = crra == 1.0
    safe_exponent = jnp.where(is_log, 1.0, 1.0 - crra)
    return jnp.where(
        is_log,
        jnp.log(consumption),
        consumption**safe_exponent / safe_exponent,
    )
