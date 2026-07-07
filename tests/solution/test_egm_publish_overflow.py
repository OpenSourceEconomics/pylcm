"""Envelope overflow poisons every published row so the NaN diagnostics fire.

When the refined upper envelope keeps more points than the padded carry length
(`n_kept > n_pad`), the published rows are unreliable. The solve loop's NaN
diagnostics are the mechanism that surfaces the offending (regime, period), so
overflow must NaN-poison all three outputs — the value row on the exogenous grid,
the carry value row, and the carry marginal-utility row. A finite marginal-utility
row on overflow would silently feed the parent period a wrong Hermite slope.
"""

import jax.numpy as jnp

from _lcm.egm.step_core import _publish_V_and_carry_rows


def test_envelope_overflow_nan_poisons_all_published_rows():
    """`n_kept > n_pad` returns NaN for the V, carry-value, and marginal rows."""
    n_pad = 4
    refined_grid = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    refined_policy = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    refined_value = jnp.asarray([-1.0, -0.5, -0.2, -0.1])

    v_row, value_row, marginal_row = _publish_V_and_carry_rows(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=jnp.asarray(n_pad + 1, dtype=jnp.int32),
        n_pad=n_pad,
        publish_resources=jnp.asarray([1.5, 2.5, 3.5]),
        borrowing_limit=jnp.asarray(0.0),
        utility_of_action=jnp.log,
        discounted_expected_value_at_limit=jnp.asarray(-2.0),
    )

    assert bool(jnp.all(jnp.isnan(v_row)))
    assert bool(jnp.all(jnp.isnan(value_row)))
    assert bool(jnp.all(jnp.isnan(marginal_row)))


def test_no_overflow_keeps_published_rows_finite():
    """`n_kept <= n_pad` leaves the published rows finite (the marginal row too).

    Guards against over-poisoning: the overflow gate must fire only on overflow,
    not on every publish.
    """
    n_pad = 4
    refined_grid = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    refined_policy = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    refined_value = jnp.asarray([-1.0, -0.5, -0.2, -0.1])

    v_row, value_row, marginal_row = _publish_V_and_carry_rows(
        refined_grid=refined_grid,
        refined_policy=refined_policy,
        refined_value=refined_value,
        n_kept=jnp.asarray(n_pad, dtype=jnp.int32),
        n_pad=n_pad,
        publish_resources=jnp.asarray([1.5, 2.5, 3.5]),
        borrowing_limit=jnp.asarray(0.0),
        utility_of_action=jnp.log,
        discounted_expected_value_at_limit=jnp.asarray(-2.0),
    )

    assert not bool(jnp.any(jnp.isnan(v_row)))
    assert not bool(jnp.any(jnp.isnan(value_row)))
    assert not bool(jnp.any(jnp.isnan(marginal_row)))
