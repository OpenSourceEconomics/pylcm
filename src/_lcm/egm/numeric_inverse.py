"""Numerical inverse of marginal utility for the iEGM path.

EGM needs $(u')^{-1}$ to recover the optimal action from the Euler equation. When
the model supplies no closed-form `inverse_marginal_utility`, this module inverts
`u'` numerically by a bracketed bisection on the action, so EGM solves any model
whose marginal utility is continuous and strictly decreasing (the concavity the
Euler step already assumes).

The forward solve is a fixed-iteration bisection — bounded, jittable, and
vmap-safe. Its branch comparisons are not differentiated: the root is detached
and the gradient is supplied analytically by the implicit-function theorem
through a single corrector step. At the converged root $c^*$ the correction
$(u'(c^*) - m)/u''(c^*)$ is numerically zero, so the value is the bisection root,
while its derivative is the exact implicit derivative

$$\\frac{\\partial c^*}{\\partial m} = \\frac{1}{u''(c^*)}, \\qquad
\\frac{\\partial c^*}{\\partial \\theta} =
-\\frac{\\partial_\\theta u'(c^*)}{u''(c^*)},$$

independent of the iteration count. Differentiating through the bisection
branches instead would yield a wrong or iteration-dependent gradient, which is a
bug in a JAX solver — hence the detached forward solve plus analytic corrector.

The inverse is defined for an *interior* root: the target must be bracketed by
`[c_lower, c_upper]`. The borrowing-constrained corner is represented separately
(the closed-form constrained candidate), and the Euler step's epsilon clamp keeps
a near-zero marginal continuation from forcing the root to the upper bound.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from lcm.typing import ScalarFloat

_BISECTION_ITERATIONS = 60


def numeric_inverse_marginal_utility(
    *,
    marginal_continuation: ScalarFloat,
    marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    c_lower: ScalarFloat,
    c_upper: ScalarFloat,
    n_iter: int = _BISECTION_ITERATIONS,
) -> ScalarFloat:
    """Solve `marginal_utility(c) = marginal_continuation` for the action `c`.

    `marginal_utility` must be continuous and strictly decreasing on
    `[c_lower, c_upper]` (the marginal utility of a concave utility), so the root
    is unique where it is bracketed. The forward solve is a detached fixed-count
    bisection; the returned value carries the exact implicit-function-theorem
    derivative through a single analytic corrector step, not autodiff through the
    bisection branches.

    Args:
        marginal_continuation: The Euler target $m$ — the discounted expected
            marginal continuation value the action's marginal utility must equal.
        marginal_utility: The marginal utility $u'$ as a callable of the action,
            with every utility parameter already bound. Strictly decreasing.
        c_lower: Lower bracket on the action (a small positive floor).
        c_upper: Upper bracket on the action (from the resources upper bound).
        n_iter: Fixed bisection iteration count; `60` halves a unit bracket to
            below machine epsilon.

    Returns:
        The action `c` at which `marginal_utility(c) == marginal_continuation`.

    """
    m = marginal_continuation

    # Static-count bisection, unrolled at trace time (`n_iter` is a Python int):
    # `u'` is decreasing, so `u'(mid) > m` means `mid` is too small (the root lies
    # to its right) and the lower bound moves up; otherwise the upper bound moves
    # down. The branches are `jnp.where`, so the whole loop is jittable and
    # vmap-safe, and the detached root below keeps autodiff out of it.
    low, high = c_lower, c_upper
    for _ in range(n_iter):
        mid = 0.5 * (low + high)
        too_small = marginal_utility(mid) > m
        low = jnp.where(too_small, mid, low)
        high = jnp.where(too_small, high, mid)
    c_star = jax.lax.stop_gradient(0.5 * (low + high))

    # Implicit-derivative corrector: detached root + one Newton-style step whose
    # value is ~`c_star` (residual ≈ 0 at convergence) but whose gradient is the
    # exact implicit derivative. `u''` is detached so it enters only as the
    # constant `1/u''` Jacobian, never differentiated itself.
    second_derivative = jax.lax.stop_gradient(jax.grad(marginal_utility)(c_star))
    return c_star - (marginal_utility(c_star) - m) / second_derivative
