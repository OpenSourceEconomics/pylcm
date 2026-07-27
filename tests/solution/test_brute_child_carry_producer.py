"""Unit contract for the living-brute child carry producer.

A brute (`GridSearch`) regime an endogenous-grid parent transitions into has its
carry built from its solved value array. The period kernel hands the producer the
regime's *whole* flat-param payload alongside the state grids, so the producer must
absorb arbitrary param types it never reads — a `MappingLeaf` tax schedule, a scalar
rate — and publish only the value array and its Euler-state gradient.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.terminal import get_brute_child_carry_producer
from _lcm.params.mapping_leaf import MappingLeaf


def test_brute_child_carry_publishes_value_and_ignores_non_array_params():
    """The producer returns the value array unchanged and tolerates a `MappingLeaf`
    schedule param passed through with the state grids."""
    producer = get_brute_child_carry_producer(state_name="liquid")
    liquid = jnp.linspace(1.0, 5.0, 4)
    v_arr = jnp.log(liquid)
    carry = producer(
        V_arr=v_arr,
        liquid=liquid,
        income_tax_schedule=MappingLeaf({"brackets_upper": jnp.array([1.0, 2.0])}),
        return_rate=jnp.asarray(0.03),
    )
    np.testing.assert_allclose(np.asarray(carry.value), np.asarray(v_arr), atol=1e-12)
