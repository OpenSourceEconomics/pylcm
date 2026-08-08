"""Numerically inverting marginal utility reproduces the analytic inverse.

The iEGM path lets EGM solve models whose utility has no closed-form `(u')^{-1}`:
it root-finds `u'(c) = marginal_continuation` on a bracket. The numerical inverse
must match the analytic one on CRRA utility — in *value* and in *parameter
derivative* — and converge to a true root for a utility with no closed-form
inverse. A value match with a wrong or zero gradient is the failure mode the
implicit-derivative contract exists to prevent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.numeric_inverse import numeric_inverse_marginal_utility
from tests.conftest import X64_ENABLED

# Root-finding accuracy scales with the float eps of the active precision:
# central finite differences, Newton-iterate residuals, and to-machine-precision
# matches all lose roughly half the mantissa at 32-bit.
_FD_RTOL = 1e-3 if X64_ENABLED else 1e-2
_RESIDUAL_BOUND = 1e-7 if X64_ENABLED else 1e-6
_NEWTON_RTOL = 1e-9 if X64_ENABLED else 1e-7


def _crra_marginal(crra):
    """The CRRA marginal utility `u'(c) = c^{-crra}` as a callable of `c`."""
    return lambda c: c ** (-crra)


@pytest.mark.parametrize("crra", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("m", [0.01, 0.1, 0.5, 1.0, 5.0])
def test_numeric_inverse_matches_crra_analytic_value(crra, m):
    """The root of `c^{-crra} = m` equals the analytic `m^{-1/crra}`."""
    c = numeric_inverse_marginal_utility(
        marginal_continuation=jnp.asarray(m),
        marginal_utility=_crra_marginal(crra),
        c_lower=jnp.asarray(1e-8),
        c_upper=jnp.asarray(1e6),
    )
    analytic = m ** (-1.0 / crra)
    np.testing.assert_allclose(float(c), analytic, rtol=1e-7)


@pytest.mark.parametrize("crra", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("m", [0.05, 0.5, 2.0])
def test_numeric_inverse_derivative_matches_crra_analytic(crra, m):
    """`d c*/d m` from the inverter equals the analytic CRRA derivative.

    For `c* = m^{-1/crra}`, `d c*/d m = -(1/crra) m^{-1/crra - 1}`. The implicit
    derivative must reproduce it; differentiating through the bisection branches
    would not.
    """

    def invert(m_value):
        return numeric_inverse_marginal_utility(
            marginal_continuation=m_value,
            marginal_utility=_crra_marginal(crra),
            c_lower=jnp.asarray(1e-8),
            c_upper=jnp.asarray(1e6),
        )

    grad = float(jax.grad(invert)(jnp.asarray(m)))
    analytic = -(1.0 / crra) * m ** (-1.0 / crra - 1.0)
    np.testing.assert_allclose(grad, analytic, rtol=1e-5)

    finite_diff = float(
        (invert(jnp.asarray(m + 1e-4)) - invert(jnp.asarray(m - 1e-4))) / 2e-4
    )
    np.testing.assert_allclose(grad, finite_diff, rtol=_FD_RTOL)


def test_numeric_inverse_converges_for_non_closed_form_utility():
    """For `u'(c) = c^{-2} + e^{-c}` (no closed-form inverse) the residual vanishes.

    Both terms are strictly decreasing in `c`, so `u'` is invertible; the root
    finder must drive `|u'(c*) - m|` below tolerance across a grid of targets even
    though no algebraic inverse exists.
    """

    def marginal(c):
        return c ** (-2.0) + jnp.exp(-c)

    targets = jnp.linspace(0.05, 3.0, 25)
    solve = jax.vmap(
        lambda m: numeric_inverse_marginal_utility(
            marginal_continuation=m,
            marginal_utility=marginal,
            c_lower=jnp.asarray(1e-6),
            c_upper=jnp.asarray(1e4),
        )
    )
    roots = solve(targets)
    residual = marginal(roots) - targets
    assert float(jnp.max(jnp.abs(residual))) < _RESIDUAL_BOUND


def test_numeric_inverse_clamps_unbracketed_target_with_zero_gradient():
    """A target outside the bracket clamps to the bound with an active-set gradient.

    With `u'(c) = c^{-2}` on `[0.5, 2.0]` the marginal spans `[0.25, 4.0]`; a target
    `m = 100` lies above that, so the root `c = 0.1` is below the bracket. The
    inverter returns the clamped lower bound, and — because the binding-bound
    active-set derivative is zero, not the bogus interior `1/u''` slope — its
    gradient w.r.t. `m` is exactly `0`.
    """

    def invert(m_value):
        return numeric_inverse_marginal_utility(
            marginal_continuation=m_value,
            marginal_utility=_crra_marginal(2.0),
            c_lower=jnp.asarray(0.5),
            c_upper=jnp.asarray(2.0),
        )

    np.testing.assert_allclose(float(invert(jnp.asarray(100.0))), 0.5, atol=1e-6)
    assert float(jax.grad(invert)(jnp.asarray(100.0))) == 0.0


@pytest.mark.parametrize("crra", [1.5, 2.0, 3.0])
def test_numeric_inverse_converges_in_few_iterations_on_a_wide_bracket(crra):
    """A handful of safeguarded-Newton steps suffice where bisection cannot.

    On the wide bracket `[1e-8, 1e6]`, `15` bisection steps shrink it by only
    `2**15` — nowhere near the root — yet the safeguarded Newton iterate hits the
    analytic CRRA inverse to machine precision in that count, because the Newton
    step converges quadratically once it is near the root. This pins the forward
    solve as Newton-driven, not bisection-driven.
    """
    m = 0.5
    c = numeric_inverse_marginal_utility(
        marginal_continuation=jnp.asarray(m),
        marginal_utility=_crra_marginal(crra),
        c_lower=jnp.asarray(1e-8),
        c_upper=jnp.asarray(1e6),
        n_iter=15,
    )
    analytic = m ** (-1.0 / crra)
    np.testing.assert_allclose(float(c), analytic, rtol=_NEWTON_RTOL)


def test_numeric_inverse_fails_loud_on_nonpositive_target():
    """A non-positive Euler target violates the log-space precondition → NaN.

    The log-consumption solve needs `m > 0`. A target `m <= 0` (which a valid
    increasing-utility model never produces) must surface as NaN — caught by the
    kernel's NaN diagnostics — not a silently clamped bound.
    """
    c = numeric_inverse_marginal_utility(
        marginal_continuation=jnp.asarray(-1.0),
        marginal_utility=_crra_marginal(2.0),
        c_lower=jnp.asarray(1e-8),
        c_upper=jnp.asarray(1e6),
    )
    assert bool(jnp.isnan(c))


def test_numeric_inverse_fails_loud_when_marginal_turns_nonpositive():
    """A marginal that goes non-positive on the bracket → NaN, not a wrong root.

    For `u'(c) = 1 - c` on `[0.5, 2.0]` the marginal is negative at the upper
    bound, so `log u'` is undefined there; the solver must fail loud rather than
    return a spurious interior value.
    """
    c = numeric_inverse_marginal_utility(
        marginal_continuation=jnp.asarray(0.25),
        marginal_utility=lambda c: 1.0 - c,
        c_lower=jnp.asarray(0.5),
        c_upper=jnp.asarray(2.0),
    )
    assert bool(jnp.isnan(c))


def test_numeric_inverse_fails_loud_when_bracket_marginal_overflows():
    """A bracket marginal that overflows to `+inf` violates the precondition → NaN.

    A power-law `u'(c) = c^{-crra}` with a steep `crra` and a near-zero `c_lower`
    makes `u'(c_lower)` overflow to `+inf`. That endpoint is `> 0` but not finite,
    so the log-consumption solve cannot proceed on it; the inverter must fail loud
    (NaN, caught by the kernel's NaN diagnostics) rather than run the log path on a
    non-finite bracket endpoint and return a silently clamped bound. The upper
    endpoint stays finite and positive (`2^{-60} > 0`), so only the overflow —
    not a non-positive marginal — can gate this call.
    """
    c = numeric_inverse_marginal_utility(
        marginal_continuation=jnp.asarray(1.0),
        marginal_utility=_crra_marginal(60.0),
        c_lower=jnp.asarray(1e-8),
        c_upper=jnp.asarray(2.0),
    )
    assert bool(jnp.isnan(c))


def test_numeric_inverse_is_vmappable_and_jittable():
    """The inverter vmaps over targets and jit-compiles (kernel requirements)."""
    targets = jnp.linspace(0.1, 4.0, 16)
    fn = jax.jit(
        jax.vmap(
            lambda m: numeric_inverse_marginal_utility(
                marginal_continuation=m,
                marginal_utility=_crra_marginal(2.0),
                c_lower=jnp.asarray(1e-8),
                c_upper=jnp.asarray(1e6),
            )
        )
    )
    roots = fn(targets)
    np.testing.assert_allclose(
        np.asarray(roots), np.asarray(targets ** (-0.5)), rtol=1e-6
    )
