from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.mss import refine_envelope


def _f32_fraction(value):
    return Fraction.from_float(float(np.float32(value)))


def _stored_chord(*, x, lower, upper, lower_value, upper_value):
    return _f32_fraction(lower_value) + (
        _f32_fraction(upper_value) - _f32_fraction(lower_value)
    ) * (_f32_fraction(x) - _f32_fraction(lower)) / (
        _f32_fraction(upper) - _f32_fraction(lower)
    )


def test_exact_zero_interior_tie_does_not_fail_loudly():
    lower = np.float32(36496.36)
    upper = np.float32(156616.2)
    query = np.float32(126586.24)
    lower_value = np.float32(-0.06288506)
    upper_value = np.float32(0.02212576)
    point_value = np.float32(0.00087305484)

    exact_cross_product = (_f32_fraction(lower_value) - _f32_fraction(point_value)) * (
        _f32_fraction(upper) - _f32_fraction(lower)
    ) + (_f32_fraction(upper_value) - _f32_fraction(lower_value)) * (
        _f32_fraction(query) - _f32_fraction(lower)
    )
    assert exact_cross_product == 0

    chord_slope = (float(upper_value) - float(lower_value)) / (
        float(upper) - float(lower)
    )
    chord_policy = np.float32(10000.0)
    utility_scale = chord_slope * float(chord_policy)
    point_policy = np.float32(110000.0)
    point_right = np.nextafter(query, np.float32(np.inf), dtype=np.float32)
    point_value_right = np.float32(
        float(point_value)
        + utility_scale / float(point_policy) * (float(point_right) - float(query))
    )

    grid = jnp.asarray(
        [query, point_right, lower, upper],
        dtype=jnp.float32,
    )
    policy = jnp.asarray(
        [point_policy, point_policy, chord_policy, chord_policy],
        dtype=jnp.float32,
    )
    value = jnp.asarray(
        [point_value, point_value_right, lower_value, upper_value],
        dtype=jnp.float32,
    )
    segment_id = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(np.asarray(grid - policy)) > 0.0)

    *_, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=16,
        segment_id=segment_id,
    )

    assert int(n_kept) <= 16


def test_endpoint_faithful_crossing_value_cannot_reverse_a_competitor():
    lower = np.float32(4.193303993815789e-06)
    upper = np.float32(0.15926940739154816)
    a_lower = np.float32(4.0003429603530094e-05)
    a_upper = np.float32(0.0017246000934392214)
    b_lower = np.float32(-0.3707933)
    b_upper = np.float32(0.04315591)

    span = float(upper) - float(lower)
    slope_a = (float(a_upper) - float(a_lower)) / span
    slope_b = (float(b_upper) - float(b_lower)) / span
    utility_scale = 0.002
    policy_a = np.float32(utility_scale / slope_a)
    policy_b = np.float32(utility_scale / slope_b)

    grid = jnp.asarray([lower, upper, lower, upper], dtype=jnp.float32)
    policy = jnp.asarray(
        [policy_a, policy_a, policy_b, policy_b],
        dtype=jnp.float32,
    )
    value = jnp.asarray(
        [a_lower, a_upper, b_lower, b_upper],
        dtype=jnp.float32,
    )
    segment_id = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(np.asarray(grid - policy)) > 0.0)

    refined_grid, _, refined_value, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=16,
        segment_id=segment_id,
    )
    if int(n_kept) > 16:
        return

    live_grid = np.asarray(refined_grid)[: int(n_kept)]
    duplicate = np.flatnonzero(live_grid[1:] == live_grid[:-1])
    assert duplicate.size == 1
    index = int(duplicate[0])
    crossing = np.float32(live_grid[index])

    branch_a = _stored_chord(
        x=crossing,
        lower=lower,
        upper=upper,
        lower_value=a_lower,
        upper_value=a_upper,
    )
    branch_b = _stored_chord(
        x=crossing,
        lower=lower,
        upper=upper,
        lower_value=b_lower,
        upper_value=b_upper,
    )
    true_envelope = max(branch_a, branch_b)
    published = _f32_fraction(refined_value[index])
    competitor = _f32_fraction(
        np.nextafter(
            np.float32(float(published)),
            np.float32(np.inf),
            dtype=np.float32,
        )
    )

    assert not (published < competitor < true_envelope)


def test_rounded_crossing_does_not_flip_policy_before_the_owner_changes():
    lower = np.float32(721.72509765625)
    upper = np.float32(721.9190063476562)
    policy_a = np.float32(0.293766587972641)
    policy_b = np.float32(0.0028656991198658943)
    a_lower = np.float32(4.816302379140325e-08)
    a_upper = np.float32(81.56422424316406)
    b_lower = np.float32(-500.957275390625)
    b_upper = np.float32(7860.298828125)

    grid = jnp.asarray([lower, upper, lower, upper], dtype=jnp.float32)
    policy = jnp.asarray(
        [policy_a, policy_a, policy_b, policy_b],
        dtype=jnp.float32,
    )
    value = jnp.asarray(
        [a_lower, a_upper, b_lower, b_upper],
        dtype=jnp.float32,
    )
    segment_id = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(np.asarray(grid - policy)) > 0.0)

    refined_grid, refined_policy, _, n_kept = refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        n_refined=8,
        segment_id=segment_id,
    )
    if int(n_kept) > 8:
        return

    live_grid = np.asarray(refined_grid)[: int(n_kept)]
    duplicate = np.flatnonzero(live_grid[1:] == live_grid[:-1])
    assert duplicate.size == 1
    index = int(duplicate[0])
    crossing = np.float32(live_grid[index])

    branch_a = _stored_chord(
        x=crossing,
        lower=lower,
        upper=upper,
        lower_value=a_lower,
        upper_value=a_upper,
    )
    branch_b = _stored_chord(
        x=crossing,
        lower=lower,
        upper=upper,
        lower_value=b_lower,
        upper_value=b_upper,
    )
    assert branch_a > branch_b

    np.testing.assert_allclose(
        float(refined_policy[index + 1]),
        float(policy_a),
        rtol=0.0,
        atol=0.0,
    )
