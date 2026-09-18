"""RT3: native JAX regression tests. NOT RUN by the reviewer.

Run in the declared Python >=3.14 / JAX >=0.11.1 project environment. This is a
unit regression for a legitimate stateless/scalar gate call. The additional
public Model.simulate acceptance cases are specified in TEST-STRATEGY.md; this
unit test is not represented as a completed public-model regression campaign.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.gated_routing import population_call


def scalar_gate(*, threshold):
    return threshold > 0


def state_gate(*, x, threshold):
    return x > threshold


def empty_projection():
    return {}


@pytest.mark.parametrize("n", [1, 3, 5, 9])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_stateless_gate_keeps_population_extent(*, n, width):
    call = population_call(func=scalar_gate, axis_size=n, subject_width=width)
    for threshold, expected in [(1.0, True), (-1.0, False)]:
        out = call({}, {"threshold": jnp.asarray(threshold)})
        assert out.shape == (n,)
        np.testing.assert_array_equal(np.asarray(out), np.full(n, expected))
    assert call is population_call(func=scalar_gate, axis_size=n, subject_width=width)


@pytest.mark.parametrize("n", [1, 3, 5, 9])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_stateful_remainders_match_literal_rows(*, n, width):
    xs = jnp.arange(n, dtype=jnp.float32)
    call = population_call(func=state_gate, axis_size=n, subject_width=width)
    out = call({"x": xs}, {"threshold": jnp.asarray(1.5)})
    np.testing.assert_array_equal(np.asarray(out), np.arange(n) > 1.5)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_empty_stateless_projection_is_a_legal_pytree(n):
    call = population_call(func=empty_projection, axis_size=n, subject_width=2)
    assert jax.tree.leaves(call({}, {})) == []
