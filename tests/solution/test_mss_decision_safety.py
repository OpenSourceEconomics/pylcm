"""Structural orderings the MSS envelope decides, and the readings they publish.

Which branch owns a query, whether the winning branch switched at all, and where
the switch happened are discrete facts about the candidate geometry, not
quantities carried to some number of digits. Each test below fixes a witness in
which the deciding margin is one representable step wide or exactly zero, and
asserts the decision — the owning branch, the presence of the switch record, the
side a read lands on — rather than the value that expresses it.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.mss import refine_envelope


def _fraction32(value: float | np.floating) -> Fraction:
    """Read a float32 as the exact rational it stores."""
    return Fraction(float(np.float32(value)))


def _live(*, array: jnp.ndarray, n_kept: int) -> np.ndarray:
    """Return the published prefix of a NaN-padded refined row."""
    return np.asarray(array)[:n_kept]


def test_a_crossing_landing_exactly_on_a_query_node_is_published() -> None:
    """A branch switch whose abscissa is a candidate node still carries both owners.

    Two branches meet exactly at a candidate abscissa. The envelope has a policy
    jump there, so the refined row must hold two records at that abscissa — the
    outgoing owner's policy first, the incoming owner's second — and a read just
    above the node must return the incoming owner.
    """
    # Branch A spans [9, 10] and branch B spans [9.5, 10.5]; both pass through
    # (10, 5), so the kink sits exactly on the candidate node at 10.
    grid = jnp.asarray([9.0, 10.0, 9.5, 10.5])
    policy = jnp.asarray([8.0, 8.0, 2.0, 2.0])
    value = jnp.asarray([4.875, 5.0, 4.75, 5.25])

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )
    kept = int(n_kept)
    assert kept <= 16

    live_grid = _live(array=refined_grid, n_kept=kept)
    live_policy = _live(array=refined_policy, n_kept=kept)
    live_value = _live(array=refined_value, n_kept=kept)

    assert np.all(np.diff(live_grid) >= 0.0)
    at_node = np.flatnonzero(live_grid == 10.0)
    assert at_node.size == 2
    np.testing.assert_array_equal(live_policy[at_node], np.array([8.0, 2.0]))
    np.testing.assert_array_equal(live_value[at_node], np.array([5.0, 5.0]))


def test_a_crossing_publishes_the_envelope_value_at_its_own_abscissa() -> None:
    """An inserted crossing carries a value at or above both branches there.

    Two steep branches meet far from the endpoints their chords are stored by.
    The crossing's published value is the envelope at the emitted abscissa, so a
    representable competitor below the envelope cannot win the node from it.
    """
    left = np.float32(0.014666654169559479)
    right = np.float32(0.023388933390378952)
    value_scale = np.float32(2**30)
    values_a = value_scale * np.array(
        [-7.139721674320754e-06, 2.3968204914126545e-05], dtype=np.float32
    )
    values_b = value_scale * np.array(
        [-0.000663454644382, 0.002228219760581851], dtype=np.float32
    )

    span = float(right) - float(left)
    slope_a = (float(values_a[1]) - float(values_a[0])) / span
    slope_b = (float(values_b[1]) - float(values_b[0])) / span
    policy_a = np.float32(0.01)
    utility_scale = float(policy_a) * slope_a
    policy_b = np.float32(utility_scale / slope_b)

    grid = jnp.asarray([left, right, left, right], dtype=jnp.float32)
    policy = jnp.asarray([policy_a, policy_a, policy_b, policy_b], dtype=jnp.float32)
    value = jnp.asarray(
        [values_a[0], values_a[1], values_b[0], values_b[1]], dtype=jnp.float32
    )
    # The witness is a valid EGM candidate cloud: positive policies on an
    # ascending savings chain.
    savings = np.asarray(grid - policy)
    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    refined_grid, _refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )
    kept = int(n_kept)
    assert kept <= 16

    live_grid = _live(array=refined_grid, n_kept=kept)
    live_value = _live(array=refined_value, n_kept=kept)
    interior = np.flatnonzero((live_grid > left) & (live_grid < right))
    assert interior.size == 2

    crossing = _fraction32(live_grid[interior[0]])
    lower = _fraction32(left)
    upper = _fraction32(right)
    branch_a = _fraction32(values_a[0]) + (
        _fraction32(values_a[1]) - _fraction32(values_a[0])
    ) * (crossing - lower) / (upper - lower)
    branch_b = _fraction32(values_b[0]) + (
        _fraction32(values_b[1]) - _fraction32(values_b[0])
    ) * (crossing - lower) / (upper - lower)
    envelope = max(branch_a, branch_b)

    for index in interior:
        published = Fraction(float(live_value[index]))
        assert published >= min(branch_a, branch_b)
        tolerance = _fraction32(
            8.0 * float(np.spacing(np.float32(float(abs(envelope)))))
        )
        assert abs(published - envelope) <= tolerance


def test_a_one_ulp_value_ordering_still_publishes_the_branch_switch() -> None:
    """A switch decided by one representable step is still published.

    Two branches span one interval and the incoming branch is above the outgoing
    one at the right endpoint by a single representable step. The read at the
    last float below that endpoint returns the incoming branch's policy, so the
    switch is in the refined row rather than smeared across the node.
    """
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
        [values_a[0], values_a[1], values_b[0], values_b[1]], dtype=jnp.float32
    )
    savings = np.asarray(grid - policy)
    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    span = float(right) - float(left)
    slope_a = (float(values_a[1]) - float(values_a[0])) / span
    slope_b = (float(values_b[1]) - float(values_b[0])) / span
    query = np.nextafter(right, np.float32(-np.inf), dtype=np.float32)
    value_a = float(values_a[0]) + slope_a * (float(query) - float(left))
    value_b = float(values_b[0]) + slope_b * (float(query) - float(left))
    # The witness only bites where the incoming branch genuinely leads at the
    # queried float; assert that rather than assume it.
    assert value_b > value_a

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )
    kept = int(n_kept)
    assert kept <= 16

    live_grid = _live(array=refined_grid, n_kept=kept)
    assert np.any((live_grid > left) & (live_grid < right))

    got_policy = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_policy
        )
    )
    assert got_policy == float(policy_b)

    got_value = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_value
        )
    )
    assert got_value >= value_a


def test_a_node_under_a_covering_chord_publishes_the_chord() -> None:
    """A node whose stored point sits under a covering chord publishes the chord.

    A branch's stored point at the node is exactly the float the covering chord
    rounds to when it is reached by extrapolating from a distant anchor, while
    the chord's own value there is thousands of representable steps higher. The
    node belongs to the chord — its value and its policy both — rather than to
    the point that ties it in one rounded reading.
    """
    lower = np.float32(9.063222)
    upper = np.float32(514.56976)
    query = np.float32(399.64185)
    value_lower = np.float32(-276.98654)
    value_upper = np.float32(81.58187)
    # The stored point is the covering chord read by anchoring on its lower
    # endpoint, which is what makes the two tie in the working precision.
    relative = np.float32(np.float32(query - lower) / np.float32(upper - lower))
    point_value = np.float32(
        value_lower + np.float32(relative * np.float32(value_upper - value_lower))
    )

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
    assert np.all(np.asarray(policy) > 0.0)
    assert np.all(np.diff(savings) > 0.0)

    exact_chord = _fraction32(value_lower) + (
        _fraction32(value_upper) - _fraction32(value_lower)
    ) * (_fraction32(query) - _fraction32(lower)) / (
        _fraction32(upper) - _fraction32(lower)
    )
    step = _fraction32(np.spacing(point_value))
    # The witness only bites while the chord is genuinely above the stored point
    # and the format can tell them apart.
    assert exact_chord - _fraction32(point_value) > step

    refined_grid, refined_policy, refined_value, n_kept = refine_envelope(
        endog_grid=grid, policy=policy, value=value, n_refined=16
    )
    kept = int(n_kept)
    assert kept <= 16

    published_policy = float(
        interp_on_padded_grid(
            x_query=jnp.asarray(query), xp=refined_grid, fp=refined_policy
        )
    )
    assert published_policy == float(chord_policy)

    published = Fraction(
        float(
            interp_on_padded_grid(
                x_query=jnp.asarray(query), xp=refined_grid, fp=refined_value
            )
        )
    )
    assert published > _fraction32(point_value)
    assert abs(published - exact_chord) <= 2 * step
