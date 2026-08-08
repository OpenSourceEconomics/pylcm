"""An EGM child node that can occur is priced, however small its probability.

The continuation expectation runs over the product of the child regime's
stochastic node axes, and a node's probability is the product of one factor per
axis. Each factor can sit comfortably inside the dtype's normal range while
their product falls below it: `sqrt(tiny)/2` squared is `tiny/4`. Multiplied as
plain floats the product is flushed to zero on a backend that cannot hold it,
and an exactly-zero weight is how this engine spells "this cannot happen" — so a
node with strictly positive probability would be dropped, and a `-inf` standing
at it (the ordinary value of a state where no action is feasible) would never
reach the answer.

The nodes therefore travel as weights the dtype can multiply plus one common
base-two scale, and the scale is undone on the expectation rather than on any
weight. A node's probability is small, but the quantity it multiplies need not
be: against a value near the top of the range, `tiny/4` times `1/tiny` is a
quarter, which no tolerance absorbs.

A factor of exactly zero is a different thing entirely — a genuine null event —
and still contributes nothing, whatever value stands at it.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.continuation import _joint_node_weights, _on_node_scale
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import FloatND, IntND


def _active_dtype() -> np.dtype:
    """The precision the suite is running at."""
    return np.dtype(jnp.zeros(()).dtype)


def _underflowing_factor() -> float:
    """A normal probability whose square the dtype cannot hold as a normal one."""
    return float(np.sqrt(np.finfo(_active_dtype()).tiny) / 2)


def _two_axis_weights(
    factor: float,
) -> tuple[tuple[FloatND, FloatND], tuple[IntND, ...]]:
    """A two-axis node mesh whose leading node carries `factor` on both axes."""
    dtype = _active_dtype()
    vec = jnp.asarray([factor, 1.0 - factor], dtype=dtype)
    mesh = jnp.meshgrid(
        jnp.arange(2, dtype=jnp.int32), jnp.arange(2, dtype=jnp.int32), indexing="ij"
    )
    return (vec, vec), tuple(axis.ravel() for axis in mesh)


def _expectation(*, values: FloatND, weights: FloatND, shift: IntND) -> float:
    """The weighted node sum the continuation forms, back on the model's scale."""
    scaled_sum = jnp.sum(
        zero_safe_weighted_term(
            weight=weights, value=values, subnormal_is_accounted_for=True
        )
    )
    return float(_on_node_scale(values=scaled_sum, shift=shift))


def test_a_joint_node_below_the_normal_range_stays_distinguishable() -> None:
    """A node the dtype cannot hold as a normal number keeps a nonzero weight."""
    weight_vecs, node_indices = _two_axis_weights(_underflowing_factor())

    weights, shift = _joint_node_weights(
        weight_vecs=weight_vecs, node_indices=node_indices
    )

    assert float(weights[0]) != 0.0
    assert int(shift) > 0


def test_a_joint_node_below_the_normal_range_is_priced_at_its_share() -> None:
    """`tiny/4` against `1/tiny` contributes a quarter, not nothing.

    The three other nodes of the mesh stand at zero, so the expectation is that
    one node's contribution alone and is known exactly.
    """
    dtype = _active_dtype()
    weight_vecs, node_indices = _two_axis_weights(_underflowing_factor())
    values = jnp.asarray([1.0 / np.finfo(dtype).tiny, 0.0, 0.0, 0.0], dtype=dtype)

    weights, shift = _joint_node_weights(
        weight_vecs=weight_vecs, node_indices=node_indices
    )

    assert _expectation(values=values, weights=weights, shift=shift) == 0.25


def test_an_infeasible_joint_node_below_the_normal_range_keeps_its_infinity() -> None:
    """A reachable state at which no action is feasible is worth `-inf`."""
    dtype = _active_dtype()
    weight_vecs, node_indices = _two_axis_weights(_underflowing_factor())
    values = jnp.asarray([-jnp.inf, 1.0, 1.0, 1.0], dtype=dtype)

    weights, shift = _joint_node_weights(
        weight_vecs=weight_vecs, node_indices=node_indices
    )

    assert _expectation(values=values, weights=weights, shift=shift) == -np.inf


def test_a_node_that_cannot_occur_contributes_nothing() -> None:
    """A zero factor is a null event, so the value standing at it never matters."""
    dtype = _active_dtype()
    weight_vecs, node_indices = _two_axis_weights(0.0)
    values = jnp.asarray([-jnp.inf, 1.0, 1.0, 1.0], dtype=dtype)

    weights, shift = _joint_node_weights(
        weight_vecs=weight_vecs, node_indices=node_indices
    )

    assert float(weights[0]) == 0.0
    assert _expectation(values=values, weights=weights, shift=shift) == 1.0


def test_an_unlifted_mesh_returns_the_reduction_it_was_handed() -> None:
    """With no scale to undo, even a quantity below the normal range stands.

    A weighted sum can be legitimately tiny — the Epstein-Zin deviation sum
    approaches zero as risk aversion approaches one, and what it still carries
    there is the first-order generator mean.
    """
    dtype = _active_dtype()
    tiny = jnp.asarray(np.finfo(dtype).tiny, dtype=dtype)
    subnormal = jnp.nextafter(tiny, jnp.zeros((), dtype=dtype))

    unlifted = _on_node_scale(values=subnormal, shift=jnp.zeros((), jnp.int32))

    assert float(unlifted) == float(subnormal)


def test_ordinary_weights_keep_the_plain_product_and_need_no_scale() -> None:
    """A mesh the dtype holds throughout is the product it always was."""
    dtype = _active_dtype()
    first = jnp.asarray([0.25, 0.75], dtype=dtype)
    second = jnp.asarray([0.5, 0.5], dtype=dtype)
    mesh = jnp.meshgrid(
        jnp.arange(2, dtype=jnp.int32), jnp.arange(2, dtype=jnp.int32), indexing="ij"
    )
    node_indices = tuple(axis.ravel() for axis in mesh)

    weights, shift = _joint_node_weights(
        weight_vecs=(first, second), node_indices=node_indices
    )

    assert int(shift) == 0
    np.testing.assert_array_equal(
        np.asarray(weights), np.asarray([0.125, 0.125, 0.375, 0.375], dtype=dtype)
    )
