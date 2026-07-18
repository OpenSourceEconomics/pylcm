"""The padded-row value read differentiates as the interpolant it declares.

`interp_on_padded_grid` / `interp_on_prepared_grid` publish an edge-clamped,
Fritsch-Carlson-limited piecewise interpolant with a right-side bracket
convention. `jax.grad` with respect to the query must return the analytic
derivative of exactly that interpolant — in particular at exact grid nodes,
where plain branch-program autodiff (clip boundaries, zero-weight guards)
returns a value that is neither one-sided slope. Asset-row mode consumes this
derivative in its Euler marginal, and deterministic grid alignments (binding
savings floors, identity or affine resource laws, shared grids) place queries
exactly on carry nodes, so the node case is reachable, not measure-zero.

The published derivative contract, everywhere:

- strictly below the first node, and strictly above the last valid node: `0`
  (the read clamps to a constant),
- inside a bracket: the limited-Hermite (or linear-secant) derivative,
- exactly on an interior node: the *right* piece's derivative (the bracket
  search is right-continuous),
- exactly on the *last* valid node: the node's own limited slope — the
  economics a parent Euler inversion consumes at a reachable top-node
  alignment (zero there would feed `u'(c) = 0` into the inversion). This is
  the one deliberate divergence from the right germ, whose clamp semantics
  (`0` at and above the node) apply to tie *selection* only,
- singleton row: `0` (constant clamp); empty row: NaN (the value carries the
  poison; the derivative is non-authoritative).
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _aggregate_child_choices
from _lcm.egm.interp import (
    interp_on_padded_grid,
    interp_right_germ_on_padded_grid,
    prepare_padded_grid,
)


def test_exact_node_continuation_gradient_matches_carry_marginal():
    """Asset-row AD recovers the declared Hermite node slope exactly.

    A smooth, concave carry row (`V = log`, exact node slopes `1/x`): both
    adjacent Hermite pieces use the node slope `1/2` at `x = 2`, so the
    represented value function is differentiable there and the continuation
    derivative consumed by asset-row mode must be exactly `1/2`.
    """
    xp = jnp.array([1.0, 2.0, 3.0])
    value = jnp.log(xp)
    marginal = 1.0 / xp
    carry = EGMCarry(
        endog_grid=xp[None, :],
        value=value[None, :],
        marginal_utility=marginal[None, :],
        taste_shock_scale=jnp.asarray(0.0),
    )
    search_grid, valid_length = jax.vmap(prepare_padded_grid)(carry.endog_grid)

    def continuation(query):
        continuation_value, _ = _aggregate_child_choices(
            carry=carry,
            prepared_search_grid=search_grid,
            prepared_valid_length=valid_length,
            has_taste_shocks=False,
            child_index=(),
            child_passive_values=(),
            child_passive_grids=(),
            row_queries=query[None],
            row_gradients=jnp.asarray([1.0]),
            n_outer_candidates=0,
        )
        return continuation_value

    query = jnp.asarray(2.0)

    np.testing.assert_allclose(float(continuation(query)), float(jnp.log(2.0)))
    np.testing.assert_allclose(
        float(jax.grad(continuation)(query)), 0.5, rtol=1e-7, atol=1e-7
    )


def test_query_gradient_follows_the_two_object_contract_on_a_node_grid():
    """`jax.grad` equals the germ's first derivative except at the last node.

    On exact-`x²` data the read reproduces the polynomial, so the derivative
    is `2q` off nodes and the right piece's slope at interior nodes — equal to
    the germ everywhere except the exact last valid node, where the published
    derivative is the node's own limited slope while the germ (the tie
    selector) stays at its clamp value zero.
    """
    xp = jnp.array([1.0, 2.0, 3.0, jnp.nan])
    fp = jnp.array([1.0, 4.0, 9.0, jnp.nan])
    slopes = jnp.array([2.0, 4.0, 6.0, jnp.nan])
    queries = [0.5, 1.0, 1.5, 2.0, 2.75, 3.0, 4.0]

    def read(query):
        return interp_on_padded_grid(x_query=query, xp=xp, fp=fp, fp_slopes=slopes)

    got = [float(jax.grad(read)(jnp.asarray(q))) for q in queries]

    np.testing.assert_allclose(got, [0.0, 2.0, 3.0, 4.0, 5.5, 6.0, 0.0], atol=1e-12)
    _, germ_first, _, _ = interp_right_germ_on_padded_grid(
        x_query=jnp.asarray(queries), xp=xp, fp=fp, fp_slopes=slopes
    )
    np.testing.assert_allclose(
        np.asarray(germ_first), [0.0, 2.0, 3.0, 4.0, 5.5, 0.0, 0.0], atol=1e-12
    )


def test_query_gradient_at_a_duplicated_abscissa_uses_the_right_piece():
    """At a duplicated kink abscissa the query derivative is the right piece's.

    The value read at the duplicate publishes the right duplicate's value, so
    its derivative is the right bracket's limited slope — matching the germ,
    never a blend across the kink.
    """
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 1.0, 1.0, 3.0])
    slopes = jnp.array([1.0, 1.0, 2.0, 2.0])

    def read(query):
        return interp_on_padded_grid(x_query=query, xp=xp, fp=fp, fp_slopes=slopes)

    derivative = float(jax.grad(read)(jnp.asarray(1.0)))

    np.testing.assert_allclose(derivative, 2.0, atol=1e-12)


def test_linear_read_query_gradient_is_the_bracket_secant():
    """Without slopes the query derivative is the located bracket's secant.

    Off nodes that is the surrounding secant; exactly on an interior node it
    is the right bracket's secant; exactly on the last node it is the last
    bracket's secant; strictly outside the valid range it is zero.
    """
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])

    def read(query):
        return interp_on_padded_grid(x_query=query, xp=xp, fp=fp)

    got = [
        float(jax.grad(read)(jnp.asarray(q))) for q in (0.5, 1.5, 2.0, 3.0, 4.0, 5.0)
    ]

    np.testing.assert_allclose(got, [0.0, 3.0, 1.0, 1.0, 1.0, 0.0], atol=1e-12)


def test_empty_row_query_gradient_is_nan():
    """An all-NaN (poisoned) row's query derivative is NaN, not a finite zero.

    The read is NaN on a poisoned row; the derivative is non-authoritative and
    must carry the poison rather than offer a finite value that downstream
    consumers could mistake for economics.
    """
    poisoned = jnp.array([jnp.nan, jnp.nan, jnp.nan])

    def read(query):
        return interp_on_padded_grid(
            x_query=query, xp=poisoned, fp=poisoned, fp_slopes=poisoned
        )

    derivative = jax.grad(read)(jnp.asarray(1.0))

    assert bool(jnp.isnan(derivative))


def test_value_tangents_survive_the_query_derivative_contract():
    """Gradients with respect to the node values stay the interpolation weights.

    The analytic query-derivative rule must not disturb the other channels: at
    an interior query the derivative of the read with respect to each node
    value is that node's (Hermite) weight, summing to one on a linear row.
    """
    xp = jnp.array([1.0, 2.0, 4.0])
    slopes = jnp.array([1.0, 1.0, 1.0])

    def read(fp):
        return interp_on_padded_grid(
            x_query=jnp.asarray(1.5), xp=xp, fp=fp, fp_slopes=slopes
        )

    weights = jax.grad(read)(jnp.array([1.0, 2.0, 4.0]))

    assert bool(jnp.isfinite(weights).all())
    np.testing.assert_allclose(float(jnp.sum(weights[:2])), 1.0, rtol=1e-12)
    np.testing.assert_allclose(float(weights[2]), 0.0, atol=1e-12)


def test_singleton_query_gradient_stays_zero_under_the_analytic_rule():
    """A singleton row keeps its zero query derivative under the analytic rule."""
    xp = jnp.array([1.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])
    slopes = jnp.array([2.0, jnp.nan, jnp.nan])

    def read(query):
        return interp_on_padded_grid(x_query=query, xp=xp, fp=fp, fp_slopes=slopes)

    derivative = float(jax.grad(read)(jnp.asarray(1.0)))

    np.testing.assert_allclose(derivative, 0.0, atol=1e-12)
