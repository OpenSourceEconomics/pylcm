"""The implicit JVP of the continuous outer optimum (plan section 19.2).

Gate battery on analytic test problems where `f*(theta)` and `df*/dtheta`
have closed forms:

- the primal reproduces the known optimum (mesh + polish, no unimodality
  assumed — including a bimodal objective whose global winner is not the
  first local maximum);
- the custom JVP returns the implicit-function-theorem tangent
  `-Q_{f theta}/Q_{ff}`, not the derivative of the search's control flow;
- the value tangent is the envelope term `Q_theta(f*, theta)`;
- AD agrees with central finite differences of the primal (plan
  section 19.3's cross-check, here at analytic-problem scale);
- diagnostics flag exactly the advertised unresolved cases: optimum at a
  bound, curvature below the floor, tied basins — and nothing else on a
  well-behaved interior problem;
- the primitive composes under `jax.jit` and `jax.grad`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.optimization.implicit_outer_derivative import (
    continuous_outer_optimum,
    implicit_optimum_diagnostics,
)

_BOUNDS = (jnp.array([0.0]), jnp.array([1.0]))


def _quadratic(f, theta):
    """`Q(f, theta) = -(f - theta/2)^2`: f* = theta/2, df*/dtheta = 1/2."""
    return -((f - theta / 2.0) ** 2)


def _scaled_quadratic(f, theta):
    """`Q = -theta * (f - 0.4)^2 + theta`: f* fixed, value moves with theta.

    df*/dtheta = 0 and the envelope value tangent is
    `Q_theta = -(f-0.4)^2 + 1 = 1` at the optimum.
    """
    return -theta * (f - 0.4) ** 2 + theta


def _bimodal(f, theta):
    """Two basins; theta scales the right-hand peak's height.

    Peaks near f=0.2 (height 1) and f=0.8 (height theta). For theta > 1 the
    global winner is the *right* basin — a pure local search seeded low, or
    a naive unimodal assumption, picks the wrong one.
    """
    left = jnp.exp(-200.0 * (f - 0.2) ** 2)
    right = theta * jnp.exp(-200.0 * (f - 0.8) ** 2)
    return left + right


def test_primal_finds_interior_quadratic_optimum() -> None:
    theta = jnp.array(0.6)
    f_star, value, _ = continuous_outer_optimum(_quadratic, theta, _BOUNDS)
    np.testing.assert_allclose(np.asarray(f_star), [0.3], atol=1e-6)
    np.testing.assert_allclose(np.asarray(value), [0.0], atol=1e-10)


def test_primal_finds_global_winner_of_bimodal_objective() -> None:
    theta = jnp.array(1.5)
    f_star, _, _ = continuous_outer_optimum(_bimodal, theta, _BOUNDS)
    np.testing.assert_allclose(np.asarray(f_star), [0.8], atol=1e-4)


def test_jvp_matches_closed_form_implicit_derivative() -> None:
    """df*/dtheta = 1/2 for the quadratic — via the custom JVP, exactly."""

    def f_star_of(theta):
        f_star, _, _ = continuous_outer_optimum(_quadratic, theta, _BOUNDS)
        return f_star[0]

    grad = jax.grad(f_star_of)(jnp.array(0.6))
    np.testing.assert_allclose(np.asarray(grad), 0.5, atol=1e-8)


def test_jvp_is_zero_when_optimum_does_not_move() -> None:
    def f_star_of(theta):
        f_star, _, _ = continuous_outer_optimum(_scaled_quadratic, theta, _BOUNDS)
        return f_star[0]

    grad = jax.grad(f_star_of)(jnp.array(2.0))
    np.testing.assert_allclose(np.asarray(grad), 0.0, atol=1e-8)


def test_value_tangent_is_envelope_term() -> None:
    """d/dtheta max_f Q = Q_theta(f*, theta) = 1 for the scaled quadratic."""

    def value_of(theta):
        _, value, _ = continuous_outer_optimum(_scaled_quadratic, theta, _BOUNDS)
        return value[0]

    grad = jax.grad(value_of)(jnp.array(2.0))
    np.testing.assert_allclose(np.asarray(grad), 1.0, atol=1e-8)


def test_ad_agrees_with_central_finite_differences() -> None:
    """Plan section 19.3: AD must agree with central differences of the primal.

    The FD baseline differentiates the *whole search* (mesh + polish), so
    its own error floor is the polish tolerance; a loose-but-honest atol.
    """
    theta0 = 0.6
    h = 1e-4

    def f_star_of(theta):
        f_star, _, _ = continuous_outer_optimum(_quadratic, jnp.asarray(theta), _BOUNDS)
        return float(f_star[0])

    fd = (f_star_of(theta0 + h) - f_star_of(theta0 - h)) / (2.0 * h)
    ad = jax.grad(lambda t: continuous_outer_optimum(_quadratic, t, _BOUNDS)[0][0])(
        jnp.array(theta0)
    )
    np.testing.assert_allclose(float(ad), fd, atol=1e-3)
    np.testing.assert_allclose(float(ad), 0.5, atol=1e-8)


def test_diagnostics_clean_on_interior_problem() -> None:
    theta = jnp.array(0.6)
    f_star, _, margin = continuous_outer_optimum(_quadratic, theta, _BOUNDS)
    diag = implicit_optimum_diagnostics(
        _quadratic, theta=theta, f_star=f_star, basin_margin=margin, bounds=_BOUNDS
    )
    assert not bool(diag.unresolved[0])


def test_diagnostics_flag_boundary_optimum() -> None:
    """`Q = f` maximizes at the upper bound; the tangent is one-sided."""

    def linear(f, theta):
        return theta * f

    theta = jnp.array(1.0)
    f_star, _, margin = continuous_outer_optimum(linear, theta, _BOUNDS)
    np.testing.assert_allclose(np.asarray(f_star), [1.0], atol=1e-6)
    diag = implicit_optimum_diagnostics(
        linear, theta=theta, f_star=f_star, basin_margin=margin, bounds=_BOUNDS
    )
    assert bool(diag.at_upper_bound[0])
    assert bool(diag.unresolved[0])


def test_diagnostics_flag_flat_curvature() -> None:
    """A constant objective has Q_ff = 0 everywhere: flat-top flag fires."""

    def flat(f, theta):
        return jnp.zeros_like(f) + 0.0 * theta

    theta = jnp.array(1.0)
    f_star, _, margin = continuous_outer_optimum(flat, theta, _BOUNDS)
    diag = implicit_optimum_diagnostics(
        flat, theta=theta, f_star=f_star, basin_margin=margin, bounds=_BOUNDS
    )
    assert bool(diag.flat_curvature[0])
    assert bool(diag.unresolved[0])
    # The guarded tangent must still be finite, not NaN/inf.
    grad = jax.grad(lambda t: continuous_outer_optimum(flat, t, _BOUNDS)[0][0])(theta)
    assert bool(jnp.isfinite(grad))


def test_diagnostics_flag_tied_basins() -> None:
    """theta = 1 makes the bimodal peaks equal-height: basin tie fires."""
    theta = jnp.array(1.0)

    def symmetric(f, theta):
        # Exactly symmetric twin peaks: the mesh (odd-sized, symmetric on
        # [0,1]) sees identical best and runner-up basin values.
        return (
            jnp.exp(-200.0 * (f - 0.25) ** 2)
            + jnp.exp(-200.0 * (f - 0.75) ** 2)
            + 0.0 * theta
        )

    f_star, _, margin = continuous_outer_optimum(symmetric, theta, _BOUNDS)
    diag = implicit_optimum_diagnostics(
        symmetric, theta=theta, f_star=f_star, basin_margin=margin, bounds=_BOUNDS
    )
    assert bool(diag.basin_tie[0])
    assert bool(diag.unresolved[0])


def test_vectorized_cells_get_per_cell_tangents() -> None:
    """Heterogeneous brackets and a shared theta: per-cell implicit tangents."""
    bounds = (jnp.array([0.0, 0.0, 0.2]), jnp.array([1.0, 0.9, 1.0]))

    def per_cell(f, theta):
        # f* = theta * c per cell, c = (0.3, 0.5, 0.7): df*/dtheta = c.
        centers = jnp.array([0.3, 0.5, 0.7])
        return -((f - theta * centers) ** 2)

    theta = jnp.array(1.0)
    f_star, _, _ = continuous_outer_optimum(per_cell, theta, bounds)
    np.testing.assert_allclose(np.asarray(f_star), [0.3, 0.5, 0.7], atol=1e-6)
    jac = jax.jacfwd(lambda t: continuous_outer_optimum(per_cell, t, bounds)[0])(theta)
    np.testing.assert_allclose(np.asarray(jac), [0.3, 0.5, 0.7], atol=1e-8)


def test_composes_under_jit() -> None:
    @jax.jit
    def solve_and_grad(theta):
        f_star, value, _ = continuous_outer_optimum(_quadratic, theta, _BOUNDS)
        grad = jax.grad(
            lambda t: continuous_outer_optimum(_quadratic, t, _BOUNDS)[0][0]
        )(theta)
        return f_star, value, grad

    f_star, _value, grad = solve_and_grad(jnp.array(0.6))
    np.testing.assert_allclose(np.asarray(f_star), [0.3], atol=1e-6)
    np.testing.assert_allclose(np.asarray(grad), 0.5, atol=1e-8)


def test_multidimensional_theta_jacobian() -> None:
    """theta in R^2: f* = (theta_0 + 2 theta_1)/4, closed-form gradient."""

    def q(f, theta):
        return -((f - (theta[0] + 2.0 * theta[1]) / 4.0) ** 2)

    theta = jnp.array([0.4, 0.3])
    f_star, _, _ = continuous_outer_optimum(q, theta, _BOUNDS)
    np.testing.assert_allclose(np.asarray(f_star), [0.25], atol=1e-6)
    jac = jax.jacfwd(lambda t: continuous_outer_optimum(q, t, _BOUNDS)[0][0])(theta)
    np.testing.assert_allclose(np.asarray(jac), [0.25, 0.5], atol=1e-8)
