"""`zero_safe_weighted_term` must stay differentiable in the WEIGHT at ``w == 0``.

The helper neutralizes ``0 * +-inf`` by masking the VALUE before the multiply. A mask
that fires on every zero-weight node also kills the gradient there, because `jnp.where`
is a hard select: the branch taken is a constant, so ``d/dw`` is ``0`` instead of
``value``.

That is invisible while the weight is a CONSTANT of the differentiation -- a Pareto
weight, a regime-transition probability, a quadrature weight -- which is what the helper
was written for. It is wrong the moment the weight is itself a function of the argument
being differentiated. The live instance was `map_coordinates`: an interpolation corner
weight is a function of the coordinate, an exactly-on-node coordinate makes one corner
weight exactly ``0``, and masking there dropped precisely the corner whose weight was
changing. `jax.grad` returned ``-grid[c]`` instead of the segment slope at every on-node
coordinate -- with the VALUES still correct, so nothing that checks levels could see it.

The mask is therefore restricted to NON-FINITE values. These tests pin all three legs of
that: zero-mass safety survives, the derivative is restored, and no value moves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.zero_safe import zero_safe_weighted_term


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
def test_derivative_in_the_weight_is_the_value_including_at_zero(weight):
    """``d/dw [w * v] == v`` for finite ``v``, at ``w == 0`` as much as anywhere."""
    grad = jax.grad(lambda w: zero_safe_weighted_term(w, jnp.asarray(7.0)))
    assert float(grad(jnp.asarray(weight))) == pytest.approx(7.0)


@pytest.mark.parametrize("value", [-jnp.inf, jnp.inf, jnp.nan])
def test_zero_weight_still_annihilates_a_nonfinite_value(value):
    """The load-bearing property: a zero weight kills any value, never yielding nan."""
    got = zero_safe_weighted_term(jnp.asarray(0.0), jnp.asarray(value))
    assert float(got) == 0.0


def test_a_nonzero_weight_still_propagates_an_infinite_value():
    """Only the ZERO-weight case is neutralized; -inf must survive a live weight."""
    got = zero_safe_weighted_term(jnp.asarray(1.0), jnp.asarray(-jnp.inf))
    assert float(got) == -jnp.inf


def test_values_are_numerically_equal_to_the_unrestricted_mask():
    """Restricting the mask to non-finite values moves no finite result.

    `0 * v == 0` exactly for finite `v`, so masking it or not cannot change the
    product -- which is why this fix is gradient-only and carries no numerical risk.
    """
    weights = jax.random.normal(jax.random.PRNGKey(0), (10_000,)).at[::7].set(0.0)
    values = jax.random.normal(jax.random.PRNGKey(1), (10_000,)) * 1e3

    unrestricted = weights * jnp.where(weights == 0, 0.0, values)
    assert jnp.array_equal(zero_safe_weighted_term(weights, values), unrestricted)

    # ...and a -inf parked on every zero-weight node still produces no nan.
    with_infs = values.at[::7].set(-jnp.inf)
    assert not bool(jnp.isnan(zero_safe_weighted_term(weights, with_infs)).any())


def test_the_equality_above_is_numerical_and_not_bitwise_at_signed_zero():
    """Pin the ONE case where the two forms differ in bits: `+0` weight, negative value.

    This test exists because the assertion above was originally named "bit_identical"
    -- and could never have checked that, since `jnp.array_equal` treats `+0.0 == -0.0`
    as True. An outside review found the gap (round-3 audit H2). The restricted mask no
    longer fires at a finite zero-weight node, so the product carries the sign of the
    value: `-0.0` where the old form gave `+0.0`.

    Nothing downstream can observe it -- equal by comparison, same derivative, and any
    reduction consuming the term is byte-for-byte unchanged -- but the DOCUMENTATION now
    says "numerically equal", and this pins the exception so the stronger wording cannot
    creep back in unchecked.
    """
    weight, value = jnp.asarray(0.0), jnp.asarray(-7.0)

    restricted = np.asarray(zero_safe_weighted_term(weight, value))
    unrestricted = np.asarray(weight * jnp.where(weight == 0, 0.0, value))

    assert restricted == unrestricted == 0.0  # numerically equal ...
    assert bool(np.signbit(restricted))  # ... but -0.0 ...
    assert not bool(np.signbit(unrestricted))  # ... vs +0.0 ...
    assert restricted.tobytes() != unrestricted.tobytes()  # ... so NOT bit-identical.

    # Invisible once reduced, which is why this is documentation-only.
    summed_new = np.asarray(jnp.sum(jnp.asarray([float(restricted), 3.0])))
    summed_old = np.asarray(jnp.sum(jnp.asarray([float(unrestricted), 3.0])))
    assert summed_new.tobytes() == summed_old.tobytes()


def test_map_coordinates_gradient_is_correct_on_node():
    """The regression this came from: interpolating a quadratic on its own nodes.

    With ``grid[i] = i**2`` the correct slope on ``[i, i+1]`` is
    ``grid[i+1] - grid[i]``, and linear interpolation reproduces it EXACTLY on the
    open segment. The on-node coordinates are the ones the unrestricted mask got
    wrong -- it returned ``-grid[c]``.
    """
    grid = jnp.arange(10.0) ** 2

    def interpolate(coordinate):
        return map_coordinates(grid, [jnp.atleast_1d(coordinate)])[0]

    for node in (1.0, 2.0, 3.0):
        slope = float(grid[int(node) + 1] - grid[int(node)])
        got = float(jax.grad(interpolate)(jnp.asarray(node)))
        assert got == pytest.approx(slope), f"on-node gradient wrong at c={node}"

    # The off-node case always worked; keep it so the test cannot pass by
    # accidentally breaking the interior.
    for midpoint in (1.5, 2.5):
        slope = float(grid[int(midpoint) + 1] - grid[int(midpoint)])
        assert float(jax.grad(interpolate)(jnp.asarray(midpoint))) == pytest.approx(
            slope
        )
