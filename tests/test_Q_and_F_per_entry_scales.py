"""A node's scale travels with its coefficient all the way to the aggregate.

A joint probability that the format cannot hold as a plain number travels as a
normal coefficient beside a base-two shift. The pair is one number, and every
consumer between the transition and the certainty equivalent has to read it as
one: a target's own reduction, the flattening that turns a target into a
lottery, and the collector that lays the targets end to end.

What goes wrong when a consumer reads the coefficient alone is not a rounding
error. The coefficient of a probability of `2**-2800` is a number near one, so a
consumer that takes it for the probability values a near-impossible node as an
even chance — and when that node carries a large payoff, the action ranked best
against it is not the one the model specifies.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.probability import scaled_exact_product
from _lcm.regime_building.Q_and_F import (
    _aggregate_joint_lottery,
    _as_lottery,
    _expectation_over_stochastic_nodes,
)
from _lcm.typing import Float1D, Int1D
from _lcm.zero_safe import scaled_joint_weight
from tests.conftest import DECIMAL_PRECISION


class _InheritedLinearExpectation(LinearExpectation):
    """A user subclass, which the engine routes through the joint lottery.

    `Q_and_F` selects the per-target route on the exact type, so a subclass
    reaches `aggregate` even when it overrides nothing — that is the point of
    testing against this class rather than `LinearExpectation` itself.
    """


def _dtype() -> jnp.dtype:
    """Return the float type the suite is running at."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _wide_target() -> tuple[Float1D, Int1D]:
    """Return a two-node law whose nodes sit far enough apart to flush.

    The rare node's probability is below the smallest normal of the running
    format, so it exists only as a `(coefficient, shift)` pair. Both precisions
    get a spread wide enough that reading the coefficient alone reverses the
    decision rather than merely perturbing it.
    """
    dtype = _dtype()
    exponent, count = (-700, 4) if jax.config.jax_enable_x64 else (-100, 3)
    factors = jnp.stack(
        [jnp.asarray([1.0, 2.0**exponent], dtype=dtype) for _ in range(count)]
    )
    return jax.vmap(scaled_exact_product, in_axes=1, out_axes=0)(factors)


def _smallest_safe_value() -> Float1D:
    """Return the smallest normal of the running format."""
    dtype = _dtype()
    return jnp.ldexp(
        jnp.asarray(1.0, dtype=dtype), -1021 if jax.config.jax_enable_x64 else -125
    )


def test_per_target_reduction_reads_each_node_on_its_own_scale() -> None:
    """A near-impossible node paired with the largest finite value stays rare.

    Its coefficient is of order one; only the shift says the node cannot
    happen. A reduction that drops the shift values the target near the payoff
    itself, which is above the smallest normal rather than far below it.
    """
    coefficients, shifts = _wide_target()
    values = jnp.asarray([0.0, jnp.finfo(_dtype()).max], dtype=_dtype())

    got = _expectation_over_stochastic_nodes(
        values=values, weights=coefficients, shifts=shifts
    )

    assert int(got > _smallest_safe_value()) == 0


def test_per_target_reduction_reads_each_node_on_its_own_scale_under_jit() -> None:
    """The scale-aware reduction survives compilation."""
    coefficients, shifts = _wide_target()
    values = jnp.asarray([0.0, jnp.finfo(_dtype()).max], dtype=_dtype())

    got = jax.jit(_expectation_over_stochastic_nodes)(
        values=values, weights=coefficients, shifts=shifts
    )

    assert int(got > _smallest_safe_value()) == 0


def test_flattening_a_target_keeps_the_scale_its_coefficients_need() -> None:
    """A target worth one everywhere is worth one, whatever its internal law.

    Its nodes are normalized to unit mass on the way into the joint lottery.
    Dividing the coefficients by their plain sum is the step that loses the
    rare node: the sum is the size of the likeliest node, so the ratio the
    division has to represent for the rarest one is exactly the quantity that
    does not fit. Reached with probability one half beside a target worth
    nothing, the continuation is one half.
    """
    dtype = _dtype()
    coefficients, shifts = _wide_target()

    values, node_coefficients, node_shifts = _as_lottery(
        values=jnp.ones((2,), dtype=dtype),
        weights=coefficients,
        shifts=shifts,
        has_stochastic_states=True,
    )
    half = jnp.asarray(0.5, dtype=dtype)
    weighted, product_shifts = scaled_joint_weight(
        jnp.stack(jnp.broadcast_arrays(half, node_coefficients))
    )

    got = _aggregate_joint_lottery(
        certainty_equivalent=_InheritedLinearExpectation(),
        lottery_values=[values, jnp.asarray([0.0], dtype=dtype)],
        lottery_weights=[weighted, jnp.asarray([0.5], dtype=dtype)],
        lottery_shifts=[
            product_shifts + node_shifts,
            jnp.asarray([0], dtype=jnp.int32),
        ],
        ce_flat_param_names={},
        states_actions_params={},
    )

    np.testing.assert_array_almost_equal(got, 0.5, decimal=DECIMAL_PRECISION)
