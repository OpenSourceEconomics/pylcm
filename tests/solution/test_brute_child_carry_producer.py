"""Unit contract for the living-brute child carry producer.

A brute (`GridSearch`) regime an endogenous-grid parent transitions into has its
carry built from its solved value array. The period kernel hands the producer the
regime's *whole* flat-param payload alongside the state grids, so the producer must
absorb arbitrary param types it never reads — a `MappingLeaf` tax schedule, a scalar
rate — and publish only the value array and its Euler-state gradient.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.egm.terminal import get_brute_child_carry_producer
from _lcm.params.mapping_leaf import MappingLeaf
from tests.conftest import DECIMAL_PRECISION


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


def test_marginal_at_a_feasibility_boundary_uses_the_one_sided_slope():
    """A feasible grid point next to an infeasible one carries its one-sided slope.

    The point just above a child regime's feasibility boundary has a well-defined
    derivative from the feasible side, so it must not inherit the zero reserved for
    infeasible states.
    """
    producer = get_brute_child_carry_producer(state_name="liquid")
    liquid = jnp.arange(1.0, 6.0)
    value = jnp.array([-jnp.inf, 2.0, 3.5, 4.0, 4.25])
    carry = producer(V_arr=value, liquid=liquid)
    aaae(
        np.asarray(carry.marginal_utility),
        [0.0, 1.5, 1.0, 0.375, 0.25],
        decimal=DECIMAL_PRECISION,
    )


def test_marginal_around_an_infeasible_interior_point_stays_one_sided():
    """Both neighbours of an infeasible interior state keep their own-side slope."""
    producer = get_brute_child_carry_producer(state_name="liquid")
    liquid = jnp.arange(1.0, 6.0)
    value = jnp.array([1.0, 2.0, -jnp.inf, 4.0, 5.0])
    carry = producer(V_arr=value, liquid=liquid)
    aaae(
        np.asarray(carry.marginal_utility),
        [1.0, 1.0, 0.0, 1.0, 1.0],
        decimal=DECIMAL_PRECISION,
    )
