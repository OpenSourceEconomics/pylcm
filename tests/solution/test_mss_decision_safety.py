"""Structural orderings the MSS envelope decides, and the readings they publish.

Which branch owns a query, whether the winning branch switched at all, and where
the switch happened are discrete facts about the candidate geometry, not
quantities carried to some number of digits. Each test below fixes a witness in
which the deciding margin is one representable step wide or exactly zero, and
asserts the decision — the owning branch, the presence of the switch record, the
side a read lands on — rather than the value that expresses it.
"""

from fractions import Fraction
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import mss
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


@jax.jit
def _r4_refined_row(
    *, grid: jax.Array, policy: jax.Array, value: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Share one compilation per shape across the stored-operand mutations."""
    return refine_envelope(
        endog_grid=grid,
        policy=policy,
        value=value,
        segment_id=labels,
        n_refined=32,
    )


def _r4_dtype() -> type[np.float32 | np.float64]:
    """Use the active test precision rather than silently narrowing to fp32."""
    return np.asarray(jnp.asarray(0.0)).dtype.type


def _r4_fraction(value: float | np.floating) -> Fraction:
    """Decode an already-cast stored operand, without production arithmetic."""
    return Fraction(float(value))


def _r4_exact_chord(
    *,
    x0: float | np.floating,
    x1: float | np.floating,
    v0: float | np.floating,
    v1: float | np.floating,
    query: float | np.floating,
) -> Fraction:
    """Independent unbounded rational affine evaluation for one stored link."""
    a, b, c, d, q = map(_r4_fraction, (x0, x1, v0, v1, query))
    return c + (d - c) * (q - a) / (b - a)


def test_r4_scaled_publication_preserves_owner() -> None:
    """Independent x/value scales and source order cannot remove a finite owner."""
    dtype = _r4_dtype()
    exponents = (
        (-120, -80, 0, 80, 120) if dtype == np.float32 else (-1018, -600, 0, 600, 1022)
    )
    for x_exponent in exponents:
        z = 2.0**x_exponent
        for value_exponent in exponents:
            level = 2.0**value_exponent
            grid = np.asarray([z, 2 * z, 1.5 * z, 1.75 * z], dtype=dtype)
            policy = np.asarray([z, z, z / 4, z / 4], dtype=dtype)
            value = np.asarray([level, 2 * level, level / 16, level / 16], dtype=dtype)
            labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
            exact_value = _r4_exact_chord(
                x0=grid[0], x1=grid[1], v0=value[0], v1=value[1], query=grid[2]
            )
            for order in ([0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2]):
                out = _r4_refined_row(
                    grid=jnp.asarray(grid[order]),
                    policy=jnp.asarray(policy[order]),
                    value=jnp.asarray(value[order]),
                    labels=jnp.asarray(labels[order]),
                )
                kept = int(out[3])
                assert kept == 4
                x, p, v = (np.asarray(channel)[:kept] for channel in out[:3])
                assert np.all(np.diff(x) >= 0)
                assert all(np.isfinite(channel).all() for channel in (x, p, v))
                ids = np.flatnonzero(x == grid[2])
                assert len(ids) == 1
                assert p[ids[0]] == policy[0], (x_exponent, value_exponent, order)
                # Every answer in this dyadic family is exactly representable.
                assert _r4_fraction(v[ids[0]]) == exact_value
                assert all(
                    np.isnan(np.asarray(channel)[kept:]).all() for channel in out[:3]
                )


def test_r4_crossing_publication_is_directed_and_node_consistent() -> None:
    """Both records bound the exact chords, also when one reuses a query node."""
    dtype = _r4_dtype()
    for origin in (0, 50, 64):
        for exponent in (-8, 0, 8):
            factor = 2.0**exponent
            # Nonrepresentable ordinates of both signs, and an exact common level.
            for values in ([0, 1, -1, 3], [-1, 1, -2, 3], [1, 1, 0, 3]):
                for node_aligned in (False, True):
                    grid = [origin, origin + 3, origin, origin + 3]
                    policy = [8, 8, 2, 2]
                    value = [v * factor for v in values]
                    labels = [0, 0, 1, 1]
                    if node_aligned:
                        # This dominated branch supplies a query at the crossing;
                        # neither crossing chord has a stored endpoint there.
                        grid += [origin + 1, origin + 2]
                        policy += [1, 1]
                        value += [-4 * factor, -4 * factor]
                        labels += [2, 2]
                    arrays = tuple(
                        jnp.asarray(a, dtype=dtype) for a in (grid, policy, value)
                    )
                    out = _r4_refined_row(
                        grid=arrays[0],
                        policy=arrays[1],
                        value=arrays[2],
                        labels=jnp.asarray(labels, dtype=jnp.int32),
                    )
                    kept = int(out[3])
                    assert kept <= 32
                    x, p, v = (np.asarray(channel)[:kept] for channel in out[:3])
                    ids = np.flatnonzero(x == origin + 1)
                    assert len(ids) == 2, (origin, exponent, values, node_aligned, x)
                    np.testing.assert_array_equal(p[ids], [8, 2])
                    exact_level = _r4_exact_chord(
                        x0=grid[0],
                        x1=grid[1],
                        v0=value[0],
                        v1=value[1],
                        query=origin + 1,
                    )
                    other_level = _r4_exact_chord(
                        x0=grid[2],
                        x1=grid[3],
                        v0=value[2],
                        v1=value[3],
                        query=origin + 1,
                    )
                    assert exact_level == other_level
                    for published in v[ids]:
                        assert _r4_fraction(published) >= exact_level
                        previous = np.nextafter(published, dtype(-np.inf))
                        assert _r4_fraction(previous) < exact_level
                    assert v[ids[0]].tobytes() == v[ids[1]].tobytes()
                    assert np.isfinite(v).all()


@jax.jit
def _r4_read_channels(
    *, x0: jax.Array, x1: jax.Array, v0: jax.Array, v1: jax.Array, query: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    reading, status = mss._chord_reading(x=query, x0=x0, x1=x1, v0=v0, v1=v1)
    upper = mss._chord_upper_value(x=query, x0=x0, x1=x1, v0=v0, v1=v1)
    return reading, status, upper


def test_r4_reader_boundaries_and_generated() -> None:
    """Fraction certificates cover cancellation, binades, subnormals and one ULP."""
    dtype = _r4_dtype()
    maximum = np.finfo(dtype).max
    tiny = np.finfo(dtype).tiny
    subnormal = np.nextafter(dtype(0), dtype(1))
    cases = [
        (0, 3, 0, 1, 1),
        (0, 3, 0, -1, 1),
        (-maximum, maximum, -maximum, maximum, 0),
        (-maximum, maximum, 1, 1, 0),
        (0, 1, -0.0, -0.0, 0.5),
        (0, 1, -0.0, 0.0, 0),
        (0, 1, -0.0, 0.0, 1),
        (0, 1, tiny, subnormal, 0.5),
        (0, 1, subnormal, 2 * float(subnormal), 0.5),
        (0, 1, -subnormal, -2 * float(subnormal), 0.5),
        (subnormal, 4 * float(subnormal), 1, 4, 2 * float(subnormal)),
        (tiny, np.nextafter(tiny, dtype(np.inf)), 1, 2, tiny),
        (1, 2, maximum, maximum, 1.5),
    ]
    exponents = (
        (-120, -80, 0, 80, 120) if dtype == np.float32 else (-1018, -600, 0, 600, 1016)
    )
    rng = np.random.default_rng(48271)
    for exponent in exponents:
        scale = 2.0**exponent
        for _ in range(12):
            a, b = rng.integers(-16, 17, size=2)
            q = dtype(rng.uniform(-3, 7))
            cases.append((-3, 7, float(a) * scale, float(b) * scale, q))
        level = dtype(scale)
        neighbour = np.nextafter(level, dtype(np.inf))
        cases.extend([(0, 3, level, neighbour, q) for q in (0, 1, 2, 3)])
    stored = np.asarray(cases, dtype=dtype)
    out = _r4_read_channels(
        **dict(
            zip(
                ("x0", "x1", "v0", "v1", "query"),
                (jnp.asarray(column) for column in stored.T),
                strict=True,
            )
        )
    )
    readings, statuses, uppers = map(np.asarray, out)
    np.testing.assert_array_equal(statuses, np.zeros(len(cases), dtype=np.int32))
    for operands, reading, upper in zip(stored, readings, uppers, strict=True):
        x0, x1, v0, v1, query = operands
        exact_level = _r4_exact_chord(x0=x0, x1=x1, v0=v0, v1=v1, query=query)
        assert np.isfinite(reading)
        assert np.isfinite(upper)
        step = abs(
            _r4_fraction(reading) - _r4_fraction(np.nextafter(reading, dtype(0)))
        )
        step = step or _r4_fraction(subnormal)
        assert abs(_r4_fraction(reading) - exact_level) <= 2 * step, operands
        assert _r4_fraction(upper) >= exact_level, operands
        previous = np.nextafter(upper, dtype(-np.inf))
        assert not np.isfinite(previous) or _r4_fraction(previous) < exact_level, (
            operands
        )
        if query == x0:
            assert reading.tobytes() == v0.tobytes()
        elif query == x1:
            assert reading.tobytes() == v1.tobytes()
        elif v0.tobytes() == v1.tobytes():
            assert reading.tobytes() == v0.tobytes()


def test_r4_invalid_read_status_is_explicit() -> None:
    """Invalid geometry, nonfinite operands and overflowing answers stay NaN."""
    dtype = _r4_dtype()
    maximum = np.finfo(dtype).max
    cases = np.asarray(
        [
            (0, 0, 1, 1, 0),
            (1, 0, 1, 1, 0.5),
            (0, 1, np.nan, 1, 0),
            (0, 1, 1, np.inf, 0.5),
            (0, 1, 1, 1, np.inf),
            (0, 1, 0, maximum, 2),
        ],
        dtype=dtype,
    )
    readings, statuses, uppers = map(
        np.asarray,
        _r4_read_channels(
            **dict(
                zip(
                    ("x0", "x1", "v0", "v1", "query"),
                    (jnp.asarray(column) for column in cases.T),
                    strict=True,
                )
            )
        ),
    )
    assert np.all(statuses != 0)
    assert np.isnan(readings).all()
    assert np.isnan(uppers).all()


def test_r4_failed_read_does_not_remove_owner() -> None:
    """Fault-inject a nonzero reader status, not the exact owner/comparator."""
    links = mss._comparable_links(
        left_grid=jnp.asarray([0.0, 0.0]),
        right_grid=jnp.asarray([2.0, 2.0]),
        left_policy=jnp.asarray([8.0, 2.0]),
        right_policy=jnp.asarray([8.0, 2.0]),
        left_value=jnp.asarray([4.0, 1.0]),
        right_value=jnp.asarray([4.0, 1.0]),
        segment_live=jnp.asarray([True, True]),
    )
    native_read = mss.exact_affine_read

    def refused_read(**operands: jax.Array) -> tuple[jax.Array, jax.Array]:
        reading, status = native_read(**operands)
        # Deliberately finite garbage with a refused status: finiteness must not
        # override the explicit status, including at endpoints/common levels.
        return jnp.full_like(reading, 123), jnp.full_like(status, 2)

    with patch.object(mss, "exact_affine_read", refused_read):
        value, policy, owner, segment = mss._evaluate_envelope(
            query_grid=jnp.asarray([0.0, 1.0, 3.0]),
            links=links,
            segment_id=jnp.asarray([10, 20], dtype=jnp.int32),
        )
    np.testing.assert_array_equal(np.asarray(owner)[:2], [0, 0])
    np.testing.assert_array_equal(np.asarray(segment), [10, 10, -1])
    assert np.isnan(np.asarray(value)[:2]).all()
    assert np.isnan(np.asarray(policy)).all()
    assert np.isneginf(np.asarray(value)[2])
