"""Structural orderings the MSS envelope decides, and the readings they publish.

Which branch owns a query, whether the winning branch switched at all, and where
the switch happened are discrete facts about the candidate geometry, not
quantities carried to some number of digits. Each test below fixes a witness in
which the deciding margin is one representable step wide or exactly zero, and
asserts the decision — the owning branch, the presence of the switch record, the
side a read lands on — rather than the value that expresses it.
"""

from fractions import Fraction
from itertools import product
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


def test_scaled_publication_preserves_owner() -> None:
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
            labels = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype)
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


def test_crossing_publication_is_directed_and_node_consistent() -> None:
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
                    labels = [0.0, 0.0, 1.0, 1.0]
                    if node_aligned:
                        # This dominated branch supplies a query at the crossing;
                        # neither crossing chord has a stored endpoint there.
                        grid += [origin + 1, origin + 2]
                        policy += [1, 1]
                        value += [-4 * factor, -4 * factor]
                        labels += [2.0, 2.0]
                    arrays = tuple(
                        jnp.asarray(a, dtype=dtype) for a in (grid, policy, value)
                    )
                    out = _r4_refined_row(
                        grid=arrays[0],
                        policy=arrays[1],
                        value=arrays[2],
                        labels=jnp.asarray(labels, dtype=dtype),
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
    reading, status = mss._chord_reading(
        x=query, x0=x0, x1=x1, v0=v0, v1=v1, arithmetic="certified"
    )
    upper = mss._chord_upper_value(
        x=query, x0=x0, x1=x1, v0=v0, v1=v1, arithmetic="certified"
    )
    return reading, status, upper


def test_reader_boundaries_and_generated() -> None:
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


def test_invalid_read_status_is_explicit() -> None:
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


def test_failed_read_does_not_remove_owner() -> None:
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
            arithmetic="certified",
        )
    np.testing.assert_array_equal(np.asarray(owner)[:2], [0, 0])
    np.testing.assert_array_equal(np.asarray(segment), [10, 10, -1])
    assert np.isnan(np.asarray(value)[:2]).all()
    assert np.isnan(np.asarray(policy)).all()
    assert np.isneginf(np.asarray(value)[2])


@jax.jit
def _r2_refined_row(
    *, grid: jax.Array, policy: jax.Array, value: jax.Array, labels: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Use real production refinement with a shared compilation per shape."""
    return refine_envelope(
        endog_grid=grid, policy=policy, value=value, segment_id=labels, n_refined=32
    )


def _r2_exact_root(*, grid: np.ndarray, value: np.ndarray) -> Fraction:
    """Direct rational line intersection, independent of native localization."""
    x0, x1, x2, x3 = (Fraction(float(x)) for x in grid[:4])
    v0, v1, v2, v3 = (Fraction(float(v)) for v in value[:4])
    slope_a, slope_b = (v1 - v0) / (x1 - x0), (v3 - v2) / (x3 - x2)
    assert slope_a < slope_b
    return (v0 - slope_a * x0 - v2 + slope_b * x2) / (slope_b - slope_a)


def _r2_ceil(*, root: Fraction, dtype: type) -> np.floating:
    """Find the least representable state at/above a rational (oracle only)."""
    candidate = dtype(float(root))
    while Fraction(float(candidate)) < root:
        candidate = np.nextafter(candidate, dtype(np.inf))
    previous = np.nextafter(candidate, dtype(-np.inf))
    while np.isfinite(previous) and Fraction(float(previous)) >= root:
        candidate = previous
        previous = np.nextafter(candidate, dtype(-np.inf))
    return candidate


def _r2_assert_event(
    *, out: tuple, root: Fraction, dtype: type, policies: tuple = (8.0, 2.0)
) -> None:
    """Assert counts/order and actual policy reads at adjacent stored states."""
    kept = int(out[3])
    assert kept <= len(out[0])
    grid, policy, value = (np.asarray(a)[:kept] for a in out[:3])
    assert all(np.isfinite(a).all() for a in (grid, policy, value))
    assert np.all(grid[1:] >= grid[:-1])
    state = _r2_ceil(root=root, dtype=dtype)
    ids = np.flatnonzero(grid == state)
    assert len(ids) == 2, (root, state, grid, policy)
    np.testing.assert_array_equal(policy[ids], policies)
    duplicates = grid[:-1][grid[:-1] == grid[1:]]
    np.testing.assert_array_equal(duplicates, [state])
    queries = np.asarray(
        [
            np.nextafter(state, dtype(-np.inf)),
            state,
            np.nextafter(state, dtype(np.inf)),
        ],
        dtype=dtype,
    )
    assert Fraction(float(queries[0])) < root <= Fraction(float(state))
    readings = np.asarray(
        interp_on_padded_grid(x_query=jnp.asarray(queries), xp=out[0], fp=out[1])
    )
    np.testing.assert_array_equal(readings, [policies[0], policies[1], policies[1]])
    for array in out[:3]:
        assert np.isnan(np.asarray(array)[kept:]).all()


def test_rounded_away_gap_family() -> None:
    """Translations, independent scales, subnormal values, labels and orientation."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype.type
    x_exponents = (
        (-120, -80, 0, 80, 120) if dtype == np.float32 else (-1017, -600, 0, 600, 1017)
    )
    v_exponents = (
        (-149, -126, 0, 80, 127)
        if dtype == np.float32
        else (-1074, -1022, 0, 600, 1023)
    )
    for origin in (-64, 50, 64):
        for x_exponent in x_exponents:
            scale = 2.0**x_exponent
            for v_exponent in v_exponents:
                level = dtype(2.0**v_exponent)
                above = np.nextafter(level, dtype(np.inf))
                grid = np.asarray(
                    [origin, origin + 8, origin + 1, origin + 5], dtype=dtype
                ) * dtype(scale)
                value = np.asarray([level, above, level, above], dtype=dtype)
                policy = np.asarray([8, 8, 2, 2], dtype=dtype)
                labels = np.asarray([7.0, 7.0, 13.0, 13.0], dtype=dtype)
                root = _r2_exact_root(grid=grid, value=value)
                assert root == Fraction(float(dtype((origin + 2) * scale)))
                for order in ([0, 1, 2, 3], [2, 3, 0, 1], [1, 0, 3, 2]):
                    out = _r2_refined_row(
                        grid=jnp.asarray(grid[order]),
                        policy=jnp.asarray(policy[order]),
                        value=jnp.asarray(value[order]),
                        labels=jnp.asarray(labels[order]),
                    )
                    _r2_assert_event(out=out, root=root, dtype=dtype)


def test_generated_exact_handover_and_policy_sides() -> None:
    """Rational roots of either sign, not a rounded secant or a tolerance check."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype.type
    for seed in (731, 48271, 99217):
        rng = np.random.default_rng(seed)
        for _ in range(24):
            left = int(rng.integers(-64, 64))
            right = left + int(rng.integers(2, 17))
            a0, a1 = rng.integers(-16, 17, size=2)
            gap0, gap1 = rng.integers(1, 17, size=2)
            grid = np.asarray([left, right, left, right], dtype=dtype)
            value = np.asarray([a0, a1, a0 - gap0, a1 + gap1], dtype=dtype)
            root = _r2_exact_root(grid=grid, value=value)
            out = _r2_refined_row(
                grid=jnp.asarray(grid),
                policy=jnp.asarray([8, 8, 2, 2], dtype=dtype),
                value=jnp.asarray(value),
                labels=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
            )
            _r2_assert_event(out=out, root=root, dtype=dtype)


def test_representable_handover_coalesces_with_query_nodes() -> None:
    """Ceiling a nonrepresentable root to an existing node must not add a third row."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype.type
    for root in (Fraction(1, 3), Fraction(-1, 3), Fraction(1, 2)):
        state = _r2_ceil(root=root, dtype=dtype)
        left, right = (-1, 1) if root < 0 else (0, 1)
        # A constant line at the numerator, B(x)=denominator*x.
        for copies in (1, 2, 4):
            grid = np.asarray(
                [left, right, left, right] + [state] * copies + [right], dtype=dtype
            )
            value = np.asarray(
                [
                    root.numerator,
                    root.numerator,
                    root.denominator * left,
                    root.denominator * right,
                ]
                + [-8] * (copies + 1),
                dtype=dtype,
            )
            policy = np.asarray([8, 8, 2, 2] + [1] * (copies + 1), dtype=dtype)
            labels = np.asarray(
                [0.0, 0.0, 1.0, 1.0] + [2.0] * (copies + 1), dtype=dtype
            )
            assert _r2_exact_root(grid=grid, value=value) == root
            out = _r2_refined_row(
                grid=jnp.asarray(grid),
                policy=jnp.asarray(policy),
                value=jnp.asarray(value),
                labels=jnp.asarray(labels),
            )
            _r2_assert_event(out=out, root=root, dtype=dtype)
            assert int(out[3]) == len(np.unique(grid)) + 1
    # There is no representable state strictly inside this terminal cell.
    left = dtype(1)
    right = np.nextafter(left, dtype(np.inf))
    grid = np.asarray([left, right, left, right], dtype=dtype)
    value = np.asarray([2, 2, 1, 3], dtype=dtype)
    out = _r2_refined_row(
        grid=jnp.asarray(grid),
        policy=jnp.asarray([8, 8, 2, 2], dtype=dtype),
        value=jnp.asarray(value),
        labels=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
    )
    _r2_assert_event(out=out, root=_r2_exact_root(grid=grid, value=value), dtype=dtype)
    assert int(out[3]) == 3


def _r2_intersection(*, x0, x1, v0, v1, lower, upper, left, right):
    """Expose geometry/status without borrowing production location as an oracle."""
    links = mss._Links(
        x0=x0,
        x1=x1,
        v0=v0,
        v1=v1,
        lower=lower,
        upper=upper,
        upper_value=v1,
        p0=jnp.asarray([8, 2], dtype=x0.dtype),
        p1=jnp.asarray([8, 2], dtype=x0.dtype),
        live=jnp.asarray([True, True]),
    )
    row = mss._crossing_in_interval(
        seg_a=jnp.asarray(0, dtype=jnp.int32),
        seg_b=jnp.asarray(1, dtype=jnp.int32),
        prev_grid=left,
        this_grid=right,
        links=links,
        arithmetic="certified",
    )
    return row.grid, row.resolved, row.at_left, row.at_right, row.unresolved, row.value


_r2_compiled_intersection = jax.jit(_r2_intersection)


def test_oriented_endpoint_coverage_and_collinear_cases() -> None:
    """A touching, parallel or unsupported pair is not an interior branch switch."""
    cases = [
        # v0, v1, lower support, upper support, resolved, left node, right node
        ([0, 0], [0, 1], [0, 0], [1, 1], True, True, False),
        ([0, -1], [0, 0], [0, 0], [1, 1], True, False, True),
        ([0, -1], [0, 1], [0, 0], [1, 1], True, False, False),
        ([0, 0], [1, 1], [0, 0], [1, 1], False, False, False),  # collinear
        ([1, 0], [2, 1], [0, 0], [1, 1], False, False, False),  # parallel
        # Wrong orientation at either endpoint, then a reverse crossing.
        ([0, 0], [1, 0], [0, 0], [1, 1], False, False, False),
        ([0, 1], [0, 0], [0, 0], [1, 1], False, False, False),
        ([0, 1], [0, -1], [0, 0], [1, 1], False, False, False),
        # A root in a coverage gap, then an artificially widened point.
        ([0, -1], [0, 1], [0, 0.75], [0.25, 1], False, False, False),
        ([0, -1], [0, 1], [0, 0], [0, 1], False, False, False),
    ]
    for v0, v1, lower, upper, resolved, at_left, at_right in cases:
        arrays = {
            k: jnp.asarray(v)
            for k, v in {
                "x0": [0.0, 0.0],
                "x1": [1.0, 1.0],
                "v0": v0,
                "v1": v1,
                "lower": lower,
                "upper": upper,
                "left": 0.0,
                "right": 1.0,
            }.items()
        }
        # All affine channels must share the working floating dtype.
        arrays = {k: v.astype(arrays["x0"].dtype) for k, v in arrays.items()}
        out = _r2_compiled_intersection(**arrays)
        assert bool(out[1]) == resolved, (v0, v1, lower, upper, out)
        assert bool(out[2]) == at_left
        assert bool(out[3]) == at_right
        assert not bool(out[4])
        if resolved:
            expected = 0 if at_left else 1 if at_right else 0.5
            assert float(out[0]) == expected


def test_refused_location_poison_is_not_a_missing_event() -> None:
    """Even finite native payloads cannot override a refused localization status."""
    native = mss.exact_affine_handover

    def refused(**operands):
        location, status = native(**operands)
        return location, jnp.full_like(status, 2)

    dtype = np.asarray(jnp.asarray(0.0)).dtype.type
    above = np.nextafter(dtype(1), dtype(np.inf))
    with patch.object(mss, "exact_affine_handover", refused):
        out = refine_envelope(
            endog_grid=jnp.asarray([50, 58, 51, 55], dtype=dtype),
            policy=jnp.asarray([49, 49, 39, 39], dtype=dtype),
            value=jnp.asarray([1, above, 1, above], dtype=dtype),
            n_refined=16,
        )
    kept = int(out[3])
    assert kept >= 4
    assert np.isnan(np.asarray(out[2])[:kept]).any()
    assert np.isnan(np.asarray(out[1])[:kept]).any()
    live_grid = np.asarray(out[0])[:kept]
    assert np.isfinite(live_grid).all()
    assert len(np.unique(live_grid)) == kept  # no fabricated endpoint event


@jax.jit
def _geometry_node(
    *,
    grid: jax.Array,
    policy: jax.Array,
    value: jax.Array,
    labels: jax.Array,
    query: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exercise the production node boundary without the publication sweep."""
    links = mss._comparable_links(
        left_grid=grid[:-1],
        right_grid=grid[1:],
        left_policy=policy[:-1],
        right_policy=policy[1:],
        left_value=value[:-1],
        right_value=value[1:],
        segment_live=labels[:-1] == labels[1:],
    )
    return mss._evaluate_envelope(
        query_grid=query,
        links=links,
        segment_id=labels[:-1].astype(jnp.int32),
        arithmetic="certified",
    )


@jax.jit
def _geometry_inferred_row(
    *, grid: jax.Array, policy: jax.Array, value: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reuse one compilation for inferred-topology coordinate mutations."""
    return refine_envelope(endog_grid=grid, policy=policy, value=value, n_refined=32)


def _geometry_oracle(
    *,
    grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    labels: np.ndarray,
    query: np.floating,
) -> tuple[Fraction, Fraction, int]:
    """Independent scalar Fraction enumeration of ORIGINAL represented links."""
    q = Fraction(float(query))
    candidates = []
    for index in range(len(grid) - 1):
        if labels[index] != labels[index + 1]:
            continue
        x0, x1 = (Fraction(float(x)) for x in grid[index : index + 2])
        v0, v1 = (Fraction(float(v)) for v in value[index : index + 2])
        p0, p1 = (Fraction(float(p)) for p in policy[index : index + 2])
        if x1 < x0:
            x0, x1, v0, v1, p0, p1 = x1, x0, v1, v0, p1, p0
        if not x0 <= q <= x1:
            continue
        slope = (v1 - v0) / (x1 - x0) if x1 != x0 else Fraction(0)
        reading = v0 + slope * (q - x0)
        action = p0 + (p1 - p0) * (q - x0) / (x1 - x0) if x1 != x0 else p0
        candidates.append(((reading, x1 > q, slope, -index), action, index))
    key, action, index = max(candidates, key=lambda candidate: candidate[0])
    return key[0], action, index


def _geometry_assert_node(
    *,
    grid: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    labels: np.ndarray,
    query: np.floating,
) -> None:
    expected_value, expected_policy, expected_index = _geometry_oracle(
        grid=grid, policy=policy, value=value, labels=labels, query=query
    )
    got = _geometry_node(
        grid=jnp.asarray(grid),
        policy=jnp.asarray(policy),
        value=jnp.asarray(value),
        labels=jnp.asarray(labels),
        query=jnp.asarray([query], dtype=grid.dtype),
    )
    assert int(got[2][0]) == expected_index
    assert Fraction(float(got[0][0])) == expected_value
    assert Fraction(float(got[1][0])) == expected_policy


def test_original_geometry_singleton_tie_family() -> None:
    """Translation, binary scaling and endpoint/link order preserve true support."""
    dtype = _r4_dtype()
    for origin in (2.0, 10.0, 64.0):
        for scale in (0.5, 1.0, 16.0):
            for slope in (-0.25, 0.0, 0.25):
                for value_scale in (2.0**-20, 1.0, 2.0**20):
                    for permutation in (
                        [0, 1, 2, 3],
                        [2, 3, 0, 1],
                        [1, 0, 3, 2],
                        [3, 2, 1, 0],
                    ):
                        grid = np.asarray(
                            [origin, origin, origin - scale, origin + scale],
                            dtype=dtype,
                        )[permutation]
                        policy = np.asarray([8, 8, 2, 2], dtype=dtype)[permutation]
                        value = np.asarray(
                            [value_scale * v for v in (1, 1, 1 - slope, 1 + slope)],
                            dtype=dtype,
                        )[permutation]
                        labels = np.asarray([0, 0, 1, 1], dtype=dtype)[permutation]
                        _geometry_assert_node(
                            grid=grid,
                            policy=policy,
                            value=value,
                            labels=labels,
                            query=dtype(origin),
                        )
                        out = _r4_refined_row(
                            grid=jnp.asarray(grid),
                            policy=jnp.asarray(policy),
                            value=jnp.asarray(value),
                            labels=jnp.asarray(labels),
                        )
                        reading = interp_on_padded_grid(
                            x_query=jnp.asarray(origin, dtype=dtype),
                            xp=out[0],
                            fp=out[1],
                        )
                        assert float(reading) == 2.0


def test_original_geometry_subnormal_support_and_orientation() -> None:
    """A positive stored width never becomes a point, in either stored order."""
    dtype = _r4_dtype()
    small = float(np.nextafter(dtype(0), dtype(1)))
    for count in (1, 2, 7, 1024):
        for sign in (-1, 1):
            d = sign * count * small
            for grid, value, query in (
                ([d, 2 * d, -1, 1], [1, 3, 2, 2], 2 * d),
                ([0, 0, -1, 1], [3, 3, 2, 2], d),
                ([d, d, -1, 1], [3, 3, 2, 2], 0),
            ):
                for permutation in ([0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1]):
                    _geometry_assert_node(
                        grid=np.asarray(grid, dtype=dtype)[permutation],
                        policy=np.asarray([8, 8, 2, 2], dtype=dtype)[permutation],
                        value=np.asarray(value, dtype=dtype)[permutation],
                        labels=np.asarray([0, 0, 1, 1], dtype=dtype)[permutation],
                        query=dtype(query),
                    )


def test_original_geometry_distinct_query_nodes() -> None:
    """All stored nodes survive; the ordinary downstream query still reads 2."""
    dtype = _r4_dtype()
    small = float(np.nextafter(dtype(0), dtype(1)))
    widths = [count * small for count in (1, 2, 7, 1024)]
    widths.append(float(np.finfo(dtype).tiny))
    for width in widths:
        for tail in (1.0, 2.0, 4.0):
            for shift in (0, -2 * width):
                grid = np.asarray(
                    [shift, width + shift, 2 * width + shift, tail], dtype=dtype
                )
                for explicit in (False, True):
                    arguments = {
                        "grid": jnp.asarray(grid),
                        "policy": jnp.asarray([8, 8, 2, 2], dtype=dtype),
                        "value": jnp.asarray([1, 2, 3, 4], dtype=dtype),
                    }
                    out = (
                        _r4_refined_row(**arguments, labels=jnp.zeros(4, dtype=dtype))
                        if explicit
                        else _geometry_inferred_row(**arguments)
                    )
                    assert int(out[3]) == 4
                    assert np.asarray(out[0])[:4].tobytes() == grid.tobytes()
                    reading = interp_on_padded_grid(
                        x_query=jnp.asarray(tail / 2, dtype=dtype),
                        xp=out[0],
                        fp=out[1],
                    )
                    assert float(reading) == 2.0


def test_original_geometry_signed_zeros_and_extreme_singletons() -> None:
    """One zero location, and readable singleton channels even at max finite."""
    dtype = _r4_dtype()
    small = float(np.nextafter(dtype(0), dtype(1)))
    maximum = float(np.finfo(dtype).max)
    for coordinate in (-maximum, -1, -small, -0.0, 0.0, small, 1, maximum):
        grid = np.asarray([coordinate, coordinate], dtype=dtype)
        out = _r4_refined_row(
            grid=jnp.asarray(grid),
            policy=jnp.asarray([8, 9], dtype=dtype),
            value=jnp.asarray([1, 99], dtype=dtype),
            labels=jnp.zeros(2, dtype=dtype),
        )
        assert int(out[3]) == 1
        assert np.asarray(out[0])[0].tobytes() == grid[0].tobytes()
        assert float(out[1][0]) == 8
        assert float(out[2][0]) == 1
    for zeros in ([0.0, -0.0], [-0.0, 0.0]):
        grid = np.asarray([*zeros, -1, 1], dtype=dtype)
        out = _r4_refined_row(
            grid=jnp.asarray(grid),
            policy=jnp.asarray([8, 9, 2, 2], dtype=dtype),
            value=jnp.asarray([1, 99, 1, 1], dtype=dtype),
            labels=jnp.asarray([0, 0, 1, 1], dtype=dtype),
        )
        assert int(out[3]) == 3
        np.testing.assert_array_equal(np.asarray(out[0])[:3], [-1, 0, 1])
        np.testing.assert_array_equal(np.asarray(out[1])[:3], [2, 2, 2])


def test_original_geometry_predicates_share_exact_order() -> None:
    """A tiny exhaustive domain checks all pairs, including negative subnormals."""
    dtype = _r4_dtype()
    small = float(np.nextafter(dtype(0), dtype(1)))
    tiny = float(np.finfo(dtype).tiny)
    maximum = float(np.finfo(dtype).max)
    points = np.asarray(
        [
            -maximum,
            -1,
            -tiny,
            -1024 * small,
            -7 * small,
            -2 * small,
            -small,
            -0.0,
            0.0,
            small,
            2 * small,
            7 * small,
            1024 * small,
            tiny,
            1,
            maximum,
        ],
        dtype=dtype,
    )
    exact_points = [Fraction(float(x)) for x in points]
    expected_equal = np.asarray([[a == b for b in exact_points] for a in exact_points])
    expected_less = np.asarray([[a < b for b in exact_points] for a in exact_points])

    def predicates(x):
        return (
            mss._stored_equal(left=x[:, None], right=x[None, :]),
            mss._stored_less(left=x[:, None], right=x[None, :]),
            mss._stored_in_span(query=x[:, None], lower=x[None, :], upper=x[None, :]),
            jnp.argsort(mss._stored_key(value=x[::-1]), stable=True),
        )

    for function in (predicates, jax.jit(predicates)):
        equal, less, singleton, order = function(jnp.asarray(points))
        np.testing.assert_array_equal(equal, expected_equal)
        np.testing.assert_array_equal(less, expected_less)
        np.testing.assert_array_equal(singleton, expected_equal)
        expected_order = sorted(range(len(points)), key=lambda i: exact_points[::-1][i])
        np.testing.assert_array_equal(order, expected_order)
    # Return each predicate alone as well: extra diagnostic outputs can prevent
    # a compiler rewrite and make a combined-output test pass spuriously.
    for predicate, expected in (
        (mss._stored_equal, expected_equal),
        (mss._stored_less, expected_less),
    ):
        actual = jax.jit(predicate)(
            left=jnp.asarray(points)[:, None], right=jnp.asarray(points)[None, :]
        )
        np.testing.assert_array_equal(actual, expected)
    batched = jax.jit(jax.vmap(predicates))(jnp.stack([jnp.asarray(points)] * 2))
    np.testing.assert_array_equal(batched[0], np.stack([expected_equal] * 2))
    np.testing.assert_array_equal(batched[1], np.stack([expected_less] * 2))
    nan = jnp.asarray([np.nan], dtype=dtype)
    assert not bool(mss._stored_equal(left=nan, right=nan)[0])
    assert not bool(mss._stored_less(left=nan, right=jnp.zeros_like(nan))[0])


def test_original_geometry_crossing_coalescence_uses_emitted_bits() -> None:
    """Subnormal event/query identity obeys the same rule as node identity."""
    dtype = _r4_dtype()
    small = float(np.nextafter(dtype(0), dtype(1)))
    for count in (1, 2, 7, 1024):
        d = count * small
        for grid, value, labels, expected in (
            ([0, 2 * d, 0, 2 * d], [2, 2, 1, 3], [0, 0, 1, 1], [0, d, d, 2 * d]),
            (
                [0, d, 2 * d, 0, d, 2 * d],
                [2, 2, 2, 1, 2, 3],
                [0, 0, 0, 1, 1, 1],
                [0, d, d, 2 * d],
            ),
        ):
            out = _r4_refined_row(
                grid=jnp.asarray(np.asarray(grid, dtype=dtype)),
                policy=jnp.asarray(
                    [8 if label == 0 else 2 for label in labels], dtype=dtype
                ),
                value=jnp.asarray(value, dtype=dtype),
                labels=jnp.asarray(labels, dtype=dtype),
            )
            assert int(out[3]) == 4
            assert (
                np.asarray(out[0])[:4].tobytes()
                == np.asarray(expected, dtype=dtype).tobytes()
            )
            np.testing.assert_array_equal(np.asarray(out[1])[:4], [8, 8, 2, 2])
            np.testing.assert_array_equal(np.asarray(out[2])[:4], [2, 2, 2, 3])
    # Half the smallest subnormal is not representable. Its handover is the
    # right query, not zero, and only one extra record is needed at that query.
    out = _r4_refined_row(
        grid=jnp.asarray(np.asarray([0, small, 0, small], dtype=dtype)),
        policy=jnp.asarray([8, 8, 2, 2], dtype=dtype),
        value=jnp.asarray([2, 2, 1, 3], dtype=dtype),
        labels=jnp.asarray([0, 0, 1, 1], dtype=dtype),
    )
    assert int(out[3]) == 3
    assert (
        np.asarray(out[0])[:3].tobytes()
        == np.asarray([0, small, small], dtype=dtype).tobytes()
    )
    np.testing.assert_array_equal(np.asarray(out[1])[:3], [8, 8, 2])
    np.testing.assert_array_equal(np.asarray(out[2])[:3], [2, 3, 3])


def test_original_geometry_selector_receives_original_operands() -> None:
    """Even the singleton's unused endpoint value reaches the real selector."""
    links = mss._comparable_links(
        left_grid=jnp.asarray([2.0, 3.0]),
        right_grid=jnp.asarray([2.0, 1.0]),
        left_value=jnp.asarray([1.0, 1.0]),
        right_value=jnp.asarray([99.0, 1.0]),
        left_policy=jnp.asarray([8.0, 2.0]),
        right_policy=jnp.asarray([9.0, 2.0]),
        segment_live=jnp.asarray([True, True]),
    )
    calls = []
    native_selector = mss.exact_query_winner_batched

    def capture(**operands):
        calls.append(operands)
        return native_selector(**operands)

    with patch.object(mss, "exact_query_winner_batched", capture):
        out = mss._evaluate_envelope(
            query_grid=jnp.asarray([2.0]),
            links=links,
            segment_id=jnp.asarray([10, 20], dtype=jnp.int32),
            arithmetic="certified",
        )
    assert float(out[1][0]) == 2.0
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["left_grid"], [[2, 1]])
    np.testing.assert_array_equal(calls[0]["right_grid"], [[2, 3]])
    np.testing.assert_array_equal(calls[0]["left_value"], [[1, 1]])
    np.testing.assert_array_equal(calls[0]["right_value"], [[99, 1]])
    np.testing.assert_array_equal(calls[0]["stable_index"], [[0, 1]])


def test_original_geometry_batched_selector_refusal_stays_explicit() -> None:
    """Fault injection targets the BATCHED selector actually called in production."""
    links = mss._comparable_links(
        left_grid=jnp.asarray([0.0, 0.0]),
        right_grid=jnp.asarray([2.0, 2.0]),
        left_value=jnp.asarray([4.0, 1.0]),
        right_value=jnp.asarray([4.0, 1.0]),
        left_policy=jnp.asarray([8.0, 2.0]),
        right_policy=jnp.asarray([8.0, 2.0]),
        segment_live=jnp.asarray([True, True]),
    )
    native_selector = mss.exact_query_winner_batched

    def refused(**operands):
        owner, status = native_selector(**operands)
        return owner, jnp.full_like(status, mss.UNRESOLVED_STATUS)

    def evaluate(query):
        return mss._evaluate_envelope(
            query_grid=query,
            links=links,
            segment_id=jnp.asarray([10, 20], dtype=jnp.int32),
            arithmetic="certified",
        )

    query = jnp.asarray([0.0, 1.0, 3.0])
    with patch.object(mss, "exact_query_winner_batched", refused):
        outputs = [evaluate(query), jax.jit(evaluate)(query)]
        batched = jax.jit(jax.vmap(evaluate))(jnp.stack([query, query]))
        outputs.extend(tuple(channel[i] for channel in batched) for i in range(2))
        for value, policy, owner, segment in outputs:
            np.testing.assert_array_equal(np.asarray(owner)[:2], [0, 0])
            np.testing.assert_array_equal(np.asarray(segment), [10, 10, -1])
            assert np.isnan(np.asarray(value)[:2]).all()
            assert np.isnan(np.asarray(policy)).all()
            assert np.isneginf(np.asarray(value)[2])


_KNOT_CHANNEL_NAMES = ("x", "p", "v", "labels", "query")


def _as_knot_kwargs(channels):
    """Name the five aligned candidate channels a knot check varies."""
    return dict(zip(_KNOT_CHANNEL_NAMES, channels, strict=True))


def _interval_knot_row(*, x, p, v, labels, query, explicit):
    """Exercise the published row and its actual policy/value consumer."""
    out = mss.refine_envelope(
        endog_grid=x,
        policy=p,
        value=v,
        segment_id=labels if explicit else None,
        n_refined=32,
    )
    return (
        out,
        interp_on_padded_grid(x_query=query, xp=out[0], fp=out[1]),
        interp_on_padded_grid(x_query=query, xp=out[0], fp=out[2]),
    )


_interval_knot_compiled = jax.jit(_interval_knot_row, static_argnames=("explicit",))


def _interval_knot_vmap(*, x, p, v, labels, query, explicit):
    return jax.vmap(
        lambda xx, pp, vv, ll, qq: _interval_knot_row(
            x=xx, p=pp, v=vv, labels=ll, query=qq, explicit=explicit
        )
    )(x, p, v, labels, query)


_interval_knot_batched = jax.jit(_interval_knot_vmap, static_argnames=("explicit",))


def _assert_interval_knot(
    *, result, root, value_scale, expected_policy, expected_value
):
    """Check exact root/policy identities against rational expected channels."""
    out, policy_read, value_read = result
    jax.block_until_ready(result)
    count = int(out[3])
    assert count <= 32
    x, p, v = (np.asarray(z)[:count] for z in out[:3])
    assert all(np.isfinite(z).all() for z in (x, p, v, policy_read, value_read))
    assert np.all(x[1:] >= x[:-1])
    duplicated = x[:-1][x[:-1] == x[1:]]
    assert len(duplicated) == 1
    assert Fraction(float(duplicated[0])) == root
    ids = np.flatnonzero(x == float(root))
    assert len(ids) == 2
    np.testing.assert_array_equal(p[ids], [8, 2])
    assert all(Fraction(float(z)) == value_scale * Fraction(23, 2) for z in v[ids])
    assert [Fraction(float(z)) for z in np.asarray(policy_read)] == expected_policy
    assert [Fraction(float(z)) for z in np.asarray(value_read)] == expected_value


def test_interval_piece_canonical_transforms_and_topology() -> None:
    """The incoming node owner is NOT the incoming interval chord at its knot."""
    dtype = _r4_dtype()
    arrays = tuple(
        jnp.asarray(z, dtype=dtype)
        for z in (
            [10, 12, 10, 11, 12],
            [8, 8, 2, 2, 2],
            [11, 13, 10, 13, 18],
            [0, 0, 1, 1, 1],
            [85 / 8],
        )
    )
    for explicit in (False, True):
        results = [
            _interval_knot_row(**_as_knot_kwargs(arrays), explicit=explicit),
            _interval_knot_compiled(**_as_knot_kwargs(arrays), explicit=explicit),
        ]
        batched = _interval_knot_batched(
            **_as_knot_kwargs([jnp.stack([z, z]) for z in arrays]),
            explicit=explicit,
        )
        results.extend(jax.tree.map(lambda z, row=i: z[row], batched) for i in range(2))
        for result in results:
            _assert_interval_knot(
                result=result,
                root=Fraction(21, 2),
                value_scale=Fraction(1),
                expected_policy=[Fraction(2)],
                expected_value=[Fraction(95, 8)],
            )


def test_interval_piece_future_piece_invariance_family() -> None:
    """648 knot mutations and 108 linear controls share the same earlier root.

    Expected values are direct Fraction evaluations of the two fixed LEFT
    pieces: A(t)=11+t and B(t)=10+3t. Only B's following piece is varied.
    No production selection, crossing or comparison supplies the reference.
    """
    dtype = _r4_dtype()
    fractions = (Fraction(1, 4), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8))
    permutations = ((0, 1, 2, 3, 4), (2, 3, 4, 0, 1), (1, 0, 4, 3, 2))
    checked = controls = 0
    for rate, origin, scale, value_scale, order in product(
        (0.5, 1, 1.5, 2, 3, 4, 8), (-16, 0, 10), (0.5, 1, 8), (0.5, 1, 16), range(3)
    ):
        x = np.asarray(origin + scale * np.asarray([0, 2, 0, 1, 2]), dtype=dtype)
        p = np.asarray([8, 8, 2, 2, 2], dtype=dtype)
        v = np.asarray(
            value_scale * np.asarray([11, 13, 10, 13, 14 + rate]), dtype=dtype
        )
        labels = np.asarray([0, 0, 1, 1, 1], dtype=dtype)
        permutation = list(permutations[order])
        channels = [jnp.asarray(z[permutation]) for z in (x, p, v, labels)]
        query = jnp.asarray([origin + scale * float(t) for t in fractions], dtype=dtype)
        expected_p = [Fraction(8 if t < Fraction(1, 2) else 2) for t in fractions]
        expected_v = [
            Fraction(value_scale) * max(11 + t, 10 + 3 * t) for t in fractions
        ]
        for explicit in (True, False) if order == 0 else (True,):
            _assert_interval_knot(
                result=_interval_knot_compiled(
                    **_as_knot_kwargs([*channels, query]), explicit=explicit
                ),
                root=Fraction(origin) + Fraction(scale) / 2,
                value_scale=Fraction(value_scale),
                expected_policy=expected_p,
                expected_value=expected_v,
            )
            checked += 1
            controls += rate == 2
    assert (checked, controls) == (756, 108)


def test_interval_piece_policy_channels_follow_the_left_trace() -> None:
    """Future policy slopes cannot leak into an earlier event or policy read."""
    dtype = _r4_dtype()
    for future_policy in (6, 9, 30):
        out, pr, vr = _interval_knot_compiled(
            x=jnp.asarray([10, 12, 10, 11, 12], dtype=dtype),
            p=jnp.asarray([8, 8, 1, 3, future_policy], dtype=dtype),
            v=jnp.asarray([11, 13, 10, 13, 18], dtype=dtype),
            labels=jnp.asarray([0, 0, 1, 1, 1], dtype=dtype),
            query=jnp.asarray([85 / 8], dtype=dtype),
            explicit=True,
        )
        count = int(out[3])
        ids = np.flatnonzero(np.asarray(out[0])[:count] == 10.5)
        assert len(ids) == 2
        np.testing.assert_array_equal(np.asarray(out[1])[ids], [8, 2])
        assert Fraction(float(pr[0])) == Fraction(9, 4)
        assert Fraction(float(vr[0])) == Fraction(95, 8)


def test_interval_piece_boundary_only_handover() -> None:
    """A supported touching endpoint needs no backward extrapolation at all."""
    dtype = _r4_dtype()
    out, pr, vr = _interval_knot_compiled(
        x=jnp.asarray([0, 1, 1, 2], dtype=dtype),
        p=jnp.asarray([8, 8, 2, 2], dtype=dtype),
        v=jnp.asarray([0, 1, 1, 3], dtype=dtype),
        labels=jnp.asarray([0, 0, 1, 1], dtype=dtype),
        query=jnp.asarray([1.25], dtype=dtype),
        explicit=True,
    )
    count = int(out[3])
    x = np.asarray(out[0])[:count]
    ids = np.flatnonzero(x == 1)
    assert len(ids) == 2
    np.testing.assert_array_equal(np.asarray(out[1])[ids], [8, 2])
    np.testing.assert_array_equal(np.asarray(out[2])[ids], [1, 1])
    assert float(pr[0]) == 2
    assert float(vr[0]) == 1.5


def test_interval_piece_disconnected_same_label_is_unresolved() -> None:
    """A label match cannot connect unequal knot values into one affine trace."""
    dtype = _r4_dtype()
    out, _, _ = _interval_knot_compiled(
        x=jnp.asarray([0, 2, 0, 1, 1, 2], dtype=dtype),
        p=jnp.asarray([8, 8, 2, 2, 2, 2], dtype=dtype),
        v=jnp.asarray([1, 3, 0, 1.5, 3, 6], dtype=dtype),
        labels=jnp.asarray([0, 0, 1, 1, 1, 1], dtype=dtype),
        query=jnp.asarray([0.5], dtype=dtype),
        explicit=True,
    )
    count = int(out[3])
    x = np.asarray(out[0])[:count]
    assert np.isfinite(x).all()
    assert len(np.unique(x)) == count  # no event from a following-piece extrapolation
    at_knot = np.flatnonzero(x == 1)
    assert len(at_knot) == 1
    assert np.isnan(np.asarray(out[1])[at_knot]).all()
    assert np.isnan(np.asarray(out[2])[at_knot]).all()


def test_interval_piece_refused_location_stays_explicit() -> None:
    """A refused trace cannot quietly remove a kink from an otherwise finite row."""
    original = mss._interval_piece

    def refused(**operands):
        piece, resolved = original(**operands)
        return piece, jnp.zeros_like(resolved)

    dtype = _r4_dtype()
    with patch.object(mss, "_interval_piece", refused):
        out = mss.refine_envelope(
            endog_grid=jnp.asarray([10, 12, 10, 11, 12], dtype=dtype),
            policy=jnp.asarray([8, 8, 2, 2, 2], dtype=dtype),
            value=jnp.asarray([11, 13, 10, 13, 18], dtype=dtype),
            segment_id=jnp.asarray([0, 0, 1, 1, 1], dtype=dtype),
            n_refined=32,
        )
    count = int(out[3])
    x = np.asarray(out[0])[:count]
    assert np.isfinite(x).all()
    assert len(np.unique(x)) == count
    assert np.isnan(np.asarray(out[1])[:count]).any()
    assert np.isnan(np.asarray(out[2])[:count]).any()


def test_interval_piece_missing_common_support_emits_no_event() -> None:
    """Owners separated by a gap hand over without a kink, keeping their own values.

    Branches supported on `[0, 1]` and `[2, 3]` never meet, so the envelope jumps
    from one to the other instead of crossing. No event may be manufactured in the
    empty interval between them, and neither node may be poisoned: at `x = 2` only
    the second branch is defined, so it reads its own policy and value there.
    """
    dtype = _r4_dtype()
    out, _, _ = _interval_knot_compiled(
        x=jnp.asarray([0, 1, 2, 3], dtype=dtype),
        p=jnp.asarray([8, 8, 2, 2], dtype=dtype),
        v=jnp.asarray([0, 1, 2, 3], dtype=dtype),
        labels=jnp.asarray([0, 0, 1, 1], dtype=dtype),
        query=jnp.asarray([1.5], dtype=dtype),
        explicit=True,
    )
    count = int(out[3])
    assert count == 4
    np.testing.assert_array_equal(np.asarray(out[0])[:count], [0, 1, 2, 3])
    np.testing.assert_array_equal(np.asarray(out[1])[:count], [8, 8, 2, 2])
    np.testing.assert_array_equal(np.asarray(out[2])[:count], [0, 1, 2, 3])


def test_interval_piece_overlapping_traces_use_one_sided_exact_ties() -> None:
    """Left traces choose the smaller slope at a right-node value tie.

    The first pair has identical affine values (stable link 0 wins). In the
    second pair the right-node owner is still link 0 by its steeper slope, but
    the branch's left trace is link 1. This is an exact limit, not a probe at
    the previous floating number, and remains valid for adjacent coordinates.
    """
    dtype = _r4_dtype()
    for lower_values, expected in (([0, 0], 0), ([0, 1], 1)):
        links = mss._comparable_links(
            left_grid=jnp.asarray([0, 0], dtype=dtype),
            right_grid=jnp.asarray([1, 1], dtype=dtype),
            left_policy=jnp.asarray([2, 3], dtype=dtype),
            right_policy=jnp.asarray([2, 3], dtype=dtype),
            left_value=jnp.asarray(lower_values, dtype=dtype),
            right_value=jnp.asarray([2, 2], dtype=dtype),
            segment_live=jnp.asarray([True, True]),
        )
        piece, resolved = mss._interval_piece(
            node_link=jnp.int32(0),
            branch=jnp.int32(1),
            prev_grid=jnp.asarray(0, dtype=dtype),
            this_grid=jnp.asarray(1, dtype=dtype),
            incoming=True,
            links=links,
            link_segment=jnp.asarray([1, 1], dtype=jnp.int32),
            arithmetic="certified",
        )
        assert bool(resolved)
        assert int(piece) == expected
