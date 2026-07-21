"""Numerical inverse of marginal utility for the iEGM path.

EGM needs $(u')^{-1}$ to recover the optimal action from the Euler equation. When
the model supplies no closed-form `inverse_marginal_utility`, this module inverts
`u'` numerically by a bracketed bisection on the action, so EGM solves any model
whose marginal utility is continuous and strictly decreasing (the concavity the
Euler step already assumes).

The forward solve is a fixed-count safeguarded Newton iteration in
log-consumption — a Newton step from the live bracket when it stays inside it, a
bisection step otherwise — bounded, jittable, and vmap-safe. In `log c` the CRRA
family is exactly linear (`log u' = -crra log c`), so Newton converges in a single
step and stays well-conditioned for any power-law-like $u'$ whose steep-near-zero,
flat-at-large-$c$ curvature would make a plain-$c$ Newton overshoot; the bisection
fallback keeps a step that leaves the bracket from diverging. Its branch
comparisons are not differentiated: the root is detached and the gradient is
supplied analytically by the implicit-function theorem through a single corrector
step. At the converged
root $c^*$ the correction $(u'(c^*) - m)/u''(c^*)$ is numerically zero, so the
value is the iterated root, while its derivative is the exact implicit derivative

$$\\frac{\\partial c^*}{\\partial m} = \\frac{1}{u''(c^*)}, \\qquad
\\frac{\\partial c^*}{\\partial \\theta} =
-\\frac{\\partial_\\theta u'(c^*)}{u''(c^*)},$$

independent of the iteration count. Differentiating through the iteration
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

_NEWTON_ITERATIONS = 30


def numeric_inverse_marginal_utility(
    *,
    marginal_continuation: ScalarFloat,
    marginal_utility: Callable[[ScalarFloat], ScalarFloat],
    c_lower: ScalarFloat,
    c_upper: ScalarFloat,
    n_iter: int = _NEWTON_ITERATIONS,
) -> ScalarFloat:
    """Solve `marginal_utility(c) = marginal_continuation` for the action `c`.

    `marginal_utility` must be continuous and strictly decreasing on
    `[c_lower, c_upper]` (the marginal utility of a concave utility), so the root
    is unique where it is bracketed. The forward solve is a detached fixed-count
    safeguarded Newton iteration; the returned value carries the exact
    implicit-function-theorem derivative through a single analytic corrector step,
    not autodiff through the iteration branches.

    Args:
        marginal_continuation: The Euler target $m$ — the discounted expected
            marginal continuation value the action's marginal utility must equal.
        marginal_utility: The marginal utility $u'$ as a callable of the action,
            with every utility parameter already bound. Strictly decreasing.
        c_lower: Lower bracket on the action (a small positive floor).
        c_upper: Upper bracket on the action (from the resources upper bound).
        n_iter: Fixed safeguarded-Newton iteration count; quadratic convergence
            reaches a smooth root well below machine epsilon in a handful of
            steps, and the bisection fallback guarantees progress otherwise.

    Returns:
        The action `c` at which `marginal_utility(c) == marginal_continuation`.

    """
    m = marginal_continuation
    marginal_curvature = jax.grad(marginal_utility)

    # The iteration runs in log-consumption space: in `log_c`, the residual
    # `r(log_c) = log u'(e^{log_c}) - log m` is exactly linear for the CRRA family
    # (`log u' = -crra log_c`), so Newton converges in a single step there and
    # stays well-conditioned for any power-law-like `u'` — the steep-near-zero,
    # flat-at-large-`c` curvature that makes a plain-`c` Newton overshoot. The log
    # is defined only where `u' > 0` and `m > 0` (marginal utility of an increasing
    # utility, a discounted positive continuation) — the iEGM precondition. A
    # utility violating it (a bliss point, a marginal touching zero inside the
    # bracket, a non-finite inactive `where` branch) would make `log u'`/`log m`
    # non-finite; the `log_well_defined` gate below fails such a call *loud* (NaN),
    # so the solve's NaN diagnostics name the offending (regime, period) rather
    # than the log path silently returning a clamped bound.
    mu_lower = marginal_utility(c_lower)
    mu_upper = marginal_utility(c_upper)
    log_m = jnp.log(m)
    # A bracket marginal that overflows to `+inf` (a power-law `u'` at a
    # near-zero `c_lower`) is `> 0.0` but not finite: `log(+inf)` is `+inf`, not
    # NaN, so the positivity checks alone would let the log path proceed on a
    # non-finite endpoint. Require finiteness explicitly so the gate fails loud.
    log_well_defined = (
        (m > 0.0)
        & (mu_lower > 0.0)
        & (mu_upper > 0.0)
        & jnp.isfinite(mu_lower)
        & jnp.isfinite(mu_upper)
        & jnp.isfinite(log_m)
    )

    def log_marginal_utility(log_c: ScalarFloat) -> ScalarFloat:
        return jnp.log(marginal_utility(jnp.exp(log_c)))

    log_marginal_slope = jax.grad(log_marginal_utility)

    # Static-count safeguarded Newton, unrolled at trace time (`n_iter` is a
    # Python int). `r` is decreasing in `log_c`: `r > 0` means `c` is below the
    # root (raise the lower bound), `r < 0` means it is above (lower the upper
    # bound). After that bracket update the root lies in `[log_low, log_high]`;
    # the Newton step is taken only when it lands strictly inside that live
    # bracket (and the slope is non-zero), else the iterate bisects. Branches are
    # `jnp.where`, so the loop is jittable and vmap-safe, and the detached root
    # below keeps autodiff out of it.
    log_low, log_high = jnp.log(c_lower), jnp.log(c_upper)
    log_c = 0.5 * (log_low + log_high)
    for _ in range(n_iter):
        residual = log_marginal_utility(log_c) - log_m
        below_root = residual > 0.0
        log_low = jnp.where(below_root, log_c, log_low)
        log_high = jnp.where(below_root, log_high, log_c)
        slope = log_marginal_slope(log_c)
        safe_slope = jnp.where(slope == 0.0, -1.0, slope)
        log_c_newton = log_c - residual / safe_slope
        take_newton = (
            (log_c_newton > log_low) & (log_c_newton < log_high) & (slope != 0.0)
        )
        log_c = jnp.where(take_newton, log_c_newton, 0.5 * (log_low + log_high))
    c_star = jax.lax.stop_gradient(jnp.exp(log_c))

    # The implicit-derivative corrector below is valid only at an *interior,
    # bracketed* root. `u'` is decreasing, so the target is bracketed iff
    # `u'(c_upper) <= m <= u'(c_lower)`; outside that the iteration clamps to a
    # bound and the interior `1/u''` Jacobian would silently extrapolate a wrong
    # gradient. At a binding bound the active-set derivative is zero, so the
    # unbracketed branch returns the clamped root detached (value at the bound,
    # `dc/dm = 0`) rather than the bogus interior slope.
    bracketed = (mu_upper <= m) & (mu_lower >= m)

    # Implicit-derivative corrector: detached root + one Newton-style step whose
    # value is ~`c_star` (residual ≈ 0 at convergence) but whose gradient is the
    # exact implicit derivative. `u''` is detached so it enters only as the
    # constant `1/u''` Jacobian, never differentiated itself.
    second_derivative = jax.lax.stop_gradient(marginal_curvature(c_star))
    interior = c_star - (marginal_utility(c_star) - m) / second_derivative
    root = jnp.where(bracketed, interior, c_star)
    # Fail loud where the log-space preconditions do not hold: NaN surfaces in the
    # kernel's NaN diagnostics instead of a silently-wrong clamped bound.
    return jnp.where(log_well_defined, root, jnp.nan)
