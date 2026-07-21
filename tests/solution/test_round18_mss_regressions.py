import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope


def test_an_unresolved_chord_cannot_be_anchored_below_the_true_envelope():
    """A point anchored below a covering unresolved chord never understates it.

    A covering chord is strictly above the stored point in exact arithmetic but
    by less than the conservative certification margin, while a second branch is
    exactly tied to the point at its endpoint. The point may not be anchored on
    its stored value (below the true envelope) merely because the endpoint tie
    makes it representable: the read at the node reaches the covering chord or
    the node fails loudly.
    """
    # Branch B spans the point from the left and is strictly above it in exact
    # arithmetic, but by less than the certificate's unresolved band. Branch A
    # begins exactly at the stored point, so it makes the point representable.
    lower = np.float32(119.9279556274414)
    upper = np.float32(148.13192749023438)
    query = np.float32(145.0667724609375)
    value_lower = np.float32(-192.93545532226562)
    value_upper = np.float32(23.524169921875)
    point_value = np.float32(-0.00025820426526479423)

    span = float(upper) - float(lower)
    slope_b = (float(value_upper) - float(value_lower)) / span
    utility_scale = 100.0
    policy_b = np.float32(utility_scale / slope_b)
    slope_a = 20.0
    policy_a = np.float32(utility_scale / slope_a)
    right_a = np.float32(float(query) + 5.0)
    value_right_a = np.float32(
        float(point_value) + slope_a * (float(right_a) - float(query))
    )

    grid = jnp.asarray([lower, upper, query, right_a], dtype=jnp.float32)
    policy = jnp.asarray([policy_b, policy_b, policy_a, policy_a], dtype=jnp.float32)
    value = jnp.asarray(
        [value_lower, value_upper, point_value, value_right_a],
        dtype=jnp.float32,
    )

    savings = np.asarray(grid - policy)
    assert np.all(policy > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    exact_chord = float(value_lower) + slope_b * (float(query) - float(lower))
    competitor = float(np.nextafter(point_value, np.float32(np.inf), dtype=np.float32))
    assert float(point_value) < competitor < exact_chord

    refined_grid, _refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=16,
    )
    if int(n_kept) > 16:
        return  # fail-loud is a valid conservative repair

    got_value = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_value
        )
    )

    assert got_value >= exact_chord
    assert not (got_value < competitor < exact_chord)


def test_a_rounded_node_landing_cannot_hide_an_observable_switch():
    """A switch just inside a candidate node is not lost to abscissa rounding.

    Two branches cross strictly inside the interval, one representable step
    before the right candidate node, so the incoming branch already wins at the
    last float below the node. The refined row must carry that switch — as an
    interior record or a loud overflow — so the read at the predecessor float
    returns the incoming branch, not the outgoing branch extrapolated across the
    node.
    """
    # Two scaled-log-utility-consistent EGM runs. Within each run,
    # V_R = A / c with A ≈ 14.8201843 and constant positive policies.
    left = np.float32(23630.662109375)
    right = np.float32(25235.28515625)
    value_scale = np.float32(2**20)
    values_a = value_scale * np.array(
        [-0.04547542333602905, 1.0055551528930664], dtype=np.float32
    )
    values_b = value_scale * np.array(
        [-0.1297287493944168, 1.005555272102356], dtype=np.float32
    )
    policy_a = np.float32(22626.18217095353)
    policy_b = np.float32(20947.013112761735)

    grid = jnp.asarray([left, right, left, right], dtype=jnp.float32)
    policy = jnp.asarray([policy_a, policy_a, policy_b, policy_b], dtype=jnp.float32)
    value = jnp.asarray(
        [values_a[0], values_a[1], values_b[0], values_b[1]],
        dtype=jnp.float32,
    )

    # This is a valid ascending savings chain.
    savings = np.asarray(grid - policy)
    assert np.all(policy > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    span = float(right) - float(left)
    slope_a = (float(values_a[1]) - float(values_a[0])) / span
    slope_b = (float(values_b[1]) - float(values_b[0])) / span
    crossing = float(left) + (float(values_a[0]) - float(values_b[0])) / (
        slope_b - slope_a
    )

    query = np.nextafter(right, np.float32(-np.inf), dtype=np.float32)
    assert crossing < float(query) < float(right)

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=16,
    )

    live_grid = np.asarray(refined_grid)[: int(n_kept)]

    overflow = int(n_kept) > 16
    has_interior_switch = bool(np.any((live_grid > left) & (live_grid < right)))

    got_value = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_value
        )
    )
    got_policy = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_policy
        )
    )

    value_a = float(values_a[0]) + slope_a * (float(query) - float(left))
    value_b = float(values_b[0]) + slope_b * (float(query) - float(left))
    true_value = max(value_a, value_b)
    true_policy = float(policy_a if value_a >= value_b else policy_b)

    competitor = float(np.nextafter(np.float32(true_value), np.float32(np.inf)))

    assert overflow or has_interior_switch
    if not overflow:
        assert not (true_value < competitor < got_value)
        np.testing.assert_allclose(got_policy, true_policy, rtol=0.0, atol=0.0)


def test_an_interior_crossing_cannot_publish_a_value_below_both_branches():
    """An emitted crossing carries the envelope value, never one below both branches.

    Two steep branches cross inside an interval; the emitted crossing abscissa
    rounds off the true intersection. The published crossing value must be the
    envelope there — at or above both branch values — not the outgoing line's
    far-anchored product, which can round below both; when the branches cannot
    be certified to meet at the rounded abscissa the interval fails loudly.
    """
    left = np.float32(0.014666654169559479)
    right = np.float32(0.023388933390378952)
    value_scale = np.float32(2**30)
    values_a = value_scale * np.array(
        [-7.139721674320754e-06, 2.3968204914126545e-05],
        dtype=np.float32,
    )
    values_b = value_scale * np.array(
        [-0.000663454644382, 0.002228219760581851],
        dtype=np.float32,
    )

    # The branches satisfy V_R = A/c for positive constant policies, and the
    # original candidate order is an increasing savings chain.
    policy_a = np.float32(0.01)
    span = float(right) - float(left)
    slope_a = (float(values_a[1]) - float(values_a[0])) / span
    slope_b = (float(values_b[1]) - float(values_b[0])) / span
    utility_scale = float(policy_a) * slope_a
    policy_b = np.float32(utility_scale / slope_b)

    grid = jnp.asarray([left, right, left, right], dtype=jnp.float32)
    policy = jnp.asarray([policy_a, policy_a, policy_b, policy_b], dtype=jnp.float32)
    value = jnp.asarray(
        [values_a[0], values_a[1], values_b[0], values_b[1]],
        dtype=jnp.float32,
    )
    savings = np.asarray(grid - policy)
    assert np.all(policy > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    refined_grid, _, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=16,
    )
    overflow = int(n_kept) > 16
    live_grid = np.asarray(refined_grid)[: min(int(n_kept), 16)]
    live_value = np.asarray(refined_value)[: min(int(n_kept), 16)]
    if overflow:
        return
    duplicate = np.flatnonzero(live_grid[1:] == live_grid[:-1])
    assert duplicate.size == 1
    i = int(duplicate[0])
    crossing = float(live_grid[i])
    published = float(live_value[i])

    value_a = float(values_a[0]) + slope_a * (crossing - float(left))
    value_b = float(values_b[0]) + slope_b * (crossing - float(left))
    true_envelope = max(value_a, value_b)
    competitor = -2.4

    np.testing.assert_allclose(published, true_envelope, rtol=0.0, atol=0.02)
    assert published < competitor < true_envelope
