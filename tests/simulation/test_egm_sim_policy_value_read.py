"""The branch re-decision value read follows the solve's Hermite convention.

The solve publishes conditional values by reading the refined row with the
cubic Hermite interpolant, using the marginal-utility row as exact node slopes
(envelope theorem). Simulation compares conditional branch values with the
same interpolant, so the ranking the re-decision sees is the ranking the solve
convention implies — a linear read of the same row can rank two close branches
differently.
"""

import jax.numpy as jnp
import pytest

from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.simulation.simulate import _interp_rows_with_support


def test_value_read_is_cubic_hermite_with_the_marginal_slopes():
    """On nodes `(1, 0)`, `(2, log 2)` with slopes `(1, 1/2)`, the read at `1.5`
    is the cubic Hermite value `0.40907`, not the linear chord `0.34657`."""
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([0.0, jnp.log(2.0)]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    value, in_support = _interp_rows_with_support(
        sim_policy=sim_policy,
        field="value",
        index=(),
        resources=jnp.array([1.5]),
        n_subjects=1,
    )
    assert bool(in_support[0])
    assert float(value[0]) == pytest.approx(0.4090736, abs=1e-6)


def test_policy_read_stays_piecewise_linear():
    """The policy read is the linear chord: `0.75` midway between `0.5` and `1.0`.

    Only the value row carries exact node slopes (the marginal-utility row via
    the envelope theorem); the policy row has no slope data, so its read is
    piecewise linear.
    """
    sim_policy = EGMSimPolicy(
        endog_grid=jnp.array([1.0, 2.0]),
        policy=jnp.array([0.5, 1.0]),
        value=jnp.array([0.0, jnp.log(2.0)]),
        marginal_utility=jnp.array([1.0, 0.5]),
    )
    policy, _ = _interp_rows_with_support(
        sim_policy=sim_policy,
        field="policy",
        index=(),
        resources=jnp.array([1.5]),
        n_subjects=1,
    )
    assert float(policy[0]) == pytest.approx(0.75, abs=1e-9)
