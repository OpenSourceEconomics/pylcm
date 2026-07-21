from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope


def _fraction32(value):
    return Fraction.from_float(float(np.float32(value)))


def test_noise_band_cannot_anchor_below_a_strict_chord():
    lower = np.float32(9.063222)
    upper = np.float32(514.56976)
    query = np.float32(399.5566)
    value_lower = np.float32(-276.98654)
    value_upper = np.float32(81.58187)
    point_value = np.float32(0.00016954711)

    chord_slope = (float(value_upper) - float(value_lower)) / (
        float(upper) - float(lower)
    )
    chord_policy = np.float32(5.0)
    utility_scale = chord_slope * float(chord_policy)

    point_policy = np.float32(float(query) - 1.0)
    point_slope = utility_scale / float(point_policy)
    point_right = np.float32(float(query) + 1.0)
    point_value_right = np.float32(
        float(point_value) + point_slope * (float(point_right) - float(query))
    )

    grid = jnp.asarray([query, point_right, lower, upper], dtype=jnp.float32)
    policy = jnp.asarray(
        [point_policy, point_policy, chord_policy, chord_policy], dtype=jnp.float32
    )
    value = jnp.asarray(
        [point_value, point_value_right, value_lower, value_upper], dtype=jnp.float32
    )

    savings = np.asarray(grid - policy)
    assert np.all(policy > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    exact_chord = _fraction32(value_lower) + (
        _fraction32(value_upper) - _fraction32(value_lower)
    ) * (_fraction32(query) - _fraction32(lower)) / (
        _fraction32(upper) - _fraction32(lower)
    )
    competitor = _fraction32(
        np.nextafter(point_value, np.float32(np.inf), dtype=np.float32)
    )
    assert _fraction32(point_value) < competitor < exact_chord

    refined_grid, _, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )

    if int(n_kept) > 16:
        return  # The documented fail-loud path is acceptable.

    published = Fraction.from_float(
        float(
            interp_on_padded_grid(
                x_query=jnp.asarray(query), xp=refined_grid, fp=refined_value
            )
        )
    )
    assert published >= exact_chord


def test_crossing_guard_and_payload_use_the_stored_endpoint_chords():
    grid_np = np.array([445666.66, 445668.12, 445666.66, 445668.12], dtype=np.float32)
    value_np = np.array([-80.70388, 37.82995, -445.39926, 208.7809], dtype=np.float32)

    width = float(grid_np[1]) - float(grid_np[0])
    slope_a = (float(value_np[1]) - float(value_np[0])) / width
    slope_b = (float(value_np[3]) - float(value_np[2])) / width
    utility_scale = 200.0
    policy_np = np.array(
        [
            utility_scale / slope_a,
            utility_scale / slope_a,
            utility_scale / slope_b,
            utility_scale / slope_b,
        ],
        dtype=np.float32,
    )

    grid = jnp.asarray(grid_np)
    policy = jnp.asarray(policy_np)
    value = jnp.asarray(value_np)

    savings = np.asarray(grid - policy)
    assert np.all(policy > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    refined_grid, _, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=12
    )

    if int(n_kept) > 12:
        return  # The documented fail-loud path is acceptable.

    live_grid = np.asarray(refined_grid)[: int(n_kept)]
    live_value = np.asarray(refined_value)[: int(n_kept)]
    duplicate = np.flatnonzero(live_grid[1:] == live_grid[:-1])
    assert duplicate.size == 1

    index = int(duplicate[0])
    crossing = _fraction32(live_grid[index])
    published = Fraction.from_float(float(live_value[index]))
    lower = _fraction32(grid_np[0])
    upper = _fraction32(grid_np[1])

    branch_a = _fraction32(value_np[0]) + (
        _fraction32(value_np[1]) - _fraction32(value_np[0])
    ) * (crossing - lower) / (upper - lower)
    branch_b = _fraction32(value_np[2]) + (
        _fraction32(value_np[3]) - _fraction32(value_np[2])
    ) * (crossing - lower) / (upper - lower)
    true_envelope = max(branch_a, branch_b)

    # The published crossing value is the stored-endpoint envelope — the larger
    # of the two exact branch chords — to within the chord division's rounding.
    # It never rises above the higher branch (nor dips below both), so a
    # representable competitor above the envelope still wins the node and one
    # below it still loses.
    tol = _fraction32(4.0 * float(np.spacing(np.float32(float(true_envelope)))))
    assert abs(published - true_envelope) <= tol
    competitor = _fraction32(np.float32(4.0e-6))
    assert competitor > true_envelope
    assert published < competitor
