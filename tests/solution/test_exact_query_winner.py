"""The certified query path has one exact owner and three selected reads."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope._exact_affine import (
    exact_affine_read,
    exact_query_winner,
)
from _lcm.egm.upper_envelope.query import envelope_at_query
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _dtype() -> np.dtype:
    return np.dtype(np.float64 if X64_ENABLED else np.float32)


def _jdtype():
    return jnp.float64 if X64_ENABLED else jnp.float32


def _uint() -> np.dtype:
    return np.dtype(np.uint64 if X64_ENABLED else np.uint32)


def _fraction(value) -> Fraction:
    return Fraction(float(value))


def _oracle_winner(*, left, right, v_left, v_right, live, query) -> int | None:
    """Literal exact total order, independent of the native implementation."""
    held_key = None
    held_index = None
    q = _fraction(query)
    for index, (x0, x1, y0, y1, is_live) in enumerate(
        zip(left, right, v_left, v_right, live, strict=True)
    ):
        if not is_live:
            continue
        x0_f, x1_f = _fraction(x0), _fraction(x1)
        lower, upper = min(x0_f, x1_f), max(x0_f, x1_f)
        if not lower <= q <= upper:
            continue
        if x0_f == x1_f:
            value = _fraction(y0)
            extends_right = False
            slope = Fraction(0)
        else:
            # Read the link left-to-right whichever way it happens to be stored.
            descending = x0_f > x1_f
            start, stop = (x1_f, x0_f) if descending else (x0_f, x1_f)
            at_start, at_stop = (y1, y0) if descending else (y0, y1)
            slope = (_fraction(at_stop) - _fraction(at_start)) / (stop - start)
            value = _fraction(at_start) + (q - start) * slope
            extends_right = q < stop
        key = (value, extends_right, slope)
        if held_key is None or key > held_key:
            held_key = key
            held_index = index
    return held_index


def test_round27_cancellation_family_publishes_exactly() -> None:
    """Normal stored operands no longer publish the cancellation surrogate."""
    dtype, jdtype, uint = _dtype(), _jdtype(), _uint()
    tiny = dtype.type(np.finfo(dtype).tiny)
    tiny_bits = int(np.asarray(tiny).view(uint))
    values = np.arange(tiny_bits + 1, tiny_bits + 65, dtype=uint).view(dtype)
    query = np.nextafter(dtype.type(0.5), dtype.type(0.0), dtype=dtype)

    def publish(v):
        channel = jnp.stack([v, -v])
        return envelope_at_query(
            endog_grid=jnp.asarray([0.0, 1.0], dtype=jdtype),
            policy=channel,
            value=channel,
            marginal=channel,
            segment_id=jnp.asarray([0.0, 0.0], dtype=jdtype),
            x_query=jnp.asarray([query], dtype=jdtype),
        )

    got = jax.jit(jax.vmap(publish))(jnp.asarray(values, dtype=jdtype))
    expected, status = exact_affine_read(
        x0=jnp.zeros(values.shape, dtype=jdtype),
        x1=jnp.ones(values.shape, dtype=jdtype),
        v0=jnp.asarray(values, dtype=jdtype),
        v1=jnp.asarray(-values, dtype=jdtype),
        x_query=jnp.full(values.shape, query, dtype=jdtype),
    )
    assert bool(np.all(np.asarray(status) == 0))
    expected_bits = np.asarray(expected).view(uint)
    for channel in got:
        np.testing.assert_array_equal(
            np.asarray(channel)[:, 0].view(uint), expected_bits
        )


def test_round27_exact_slope_family_selects_policy_22() -> None:
    """An exact endpoint tie is broken by the exact, not rounded, value slope."""
    dtype, jdtype, uint = _dtype(), _jdtype(), _uint()
    tiny = dtype.type(np.finfo(dtype).tiny)
    tiny_bits = int(np.asarray(tiny).view(uint))
    left_values = np.arange(tiny_bits + 1, tiny_bits + 65, dtype=uint).view(dtype)

    def owner(v0):
        return envelope_at_query(
            endog_grid=jnp.asarray([tiny, 1.0, tiny, 1.0], dtype=jdtype),
            policy=jnp.asarray([11.0, 11.0, 22.0, 22.0], dtype=jdtype),
            value=jnp.stack(
                [
                    v0,
                    jnp.asarray(1.0, jdtype),
                    jnp.asarray(tiny, jdtype),
                    jnp.asarray(1.0, jdtype),
                ]
            ),
            marginal=jnp.zeros((4,), dtype=jdtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jdtype),
            x_query=jnp.asarray([1.0], dtype=jdtype),
        )[1][0]

    got = np.asarray(jax.jit(jax.vmap(owner))(jnp.asarray(left_values, jdtype)))
    np.testing.assert_array_equal(got, np.full(got.shape, dtype.type(22)))


def test_exact_winner_matches_fraction_mutations() -> None:
    """Endpoint order, right extension, exact slope and stable index compose exactly."""
    dtype, jdtype = _dtype(), _jdtype()
    rng = np.random.default_rng(20260815 + dtype.itemsize)
    n_batch, n_segment = 64, 7
    left = rng.uniform(-4.0, 4.0, (n_batch, n_segment)).astype(dtype)
    width = np.exp2(rng.integers(-12, 3, (n_batch, n_segment))).astype(dtype)
    right = (left + width).astype(dtype)
    reverse = rng.random((n_batch, n_segment)) < 0.35
    old_left = left.copy()
    left[reverse], right[reverse] = right[reverse], old_left[reverse]
    zero_width = rng.random((n_batch, n_segment)) < 0.15
    right[zero_width] = left[zero_width]
    v_left = rng.normal(size=(n_batch, n_segment)).astype(dtype)
    v_right = rng.normal(size=(n_batch, n_segment)).astype(dtype)
    live = rng.random((n_batch, n_segment)) > 0.15
    query = left[:, 0].copy()  # every row has at least one bracket
    live[:, 0] = True

    def resolve(lg, rg, lv, rv, mask, q):
        return exact_query_winner(
            left_grid=lg,
            right_grid=rg,
            left_value=lv,
            right_value=rv,
            live=mask,
            x_query=q,
        )

    winner, status = jax.jit(jax.vmap(resolve))(
        jnp.asarray(left, jdtype),
        jnp.asarray(right, jdtype),
        jnp.asarray(v_left, jdtype),
        jnp.asarray(v_right, jdtype),
        jnp.asarray(live),
        jnp.asarray(query, jdtype),
    )
    expected = np.asarray(
        [
            _oracle_winner(
                left=left[i],
                right=right[i],
                v_left=v_left[i],
                v_right=v_right[i],
                live=live[i],
                query=query[i],
            )
            for i in range(n_batch)
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(np.asarray(status), 0)
    np.testing.assert_array_equal(np.asarray(winner), expected)


def test_zero_width_descending_and_stable_tie_contract() -> None:
    """Stored orientation and self-brackets do not alter the documented order."""
    jdtype = _jdtype()
    # Segment 0 is descending, segment 1 is the same line ascending, and segment
    # 2 is a higher zero-width point at q=0.5. The point wins by exact value.
    winner, status = exact_query_winner(
        left_grid=jnp.asarray([1.0, 0.0, 0.5], dtype=jdtype),
        right_grid=jnp.asarray([0.0, 1.0, 0.5], dtype=jdtype),
        left_value=jnp.asarray([1.0, 0.0, 2.0], dtype=jdtype),
        right_value=jnp.asarray([0.0, 1.0, 2.0], dtype=jdtype),
        live=jnp.asarray([True, True, True]),
        x_query=jnp.asarray([0.5], dtype=jdtype),
    )
    assert int(np.asarray(status)[0]) == 0
    assert int(np.asarray(winner)[0]) == 2

    # Remove the point. The two exact lines tie on every field, so the stable
    # lower stored index wins despite opposite endpoint orientation.
    winner, status = exact_query_winner(
        left_grid=jnp.asarray([1.0, 0.0], dtype=jdtype),
        right_grid=jnp.asarray([0.0, 1.0], dtype=jdtype),
        left_value=jnp.asarray([1.0, 0.0], dtype=jdtype),
        right_value=jnp.asarray([0.0, 1.0], dtype=jdtype),
        live=jnp.asarray([True, True]),
        x_query=jnp.asarray([0.5], dtype=jdtype),
    )
    assert int(np.asarray(status)[0]) == 0
    assert int(np.asarray(winner)[0]) == 0


def test_right_extension_precedes_exact_slope() -> None:
    """At a shared node, the link defined immediately right owns the read."""
    jdtype = _jdtype()
    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([0.0, 1.0, 1.0, 2.0], dtype=jdtype),
        policy=jnp.asarray([10.0, 10.0, 20.0, 20.0], dtype=jdtype),
        value=jnp.asarray([0.0, 10.0, 10.0, 11.0], dtype=jdtype),
        marginal=jnp.asarray([10.0, 10.0, 1.0, 1.0], dtype=jdtype),
        segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jdtype),
        x_query=jnp.asarray([1.0], dtype=jdtype),
    )
    assert float(value[0]) == 10.0
    assert float(policy[0]) == 20.0
    assert float(marginal[0]) == 1.0


def test_selected_descending_and_zero_width_links_are_read_exactly() -> None:
    """Winner-only publication canonicalizes orientations and self-brackets."""
    jdtype = _jdtype()
    descending = envelope_at_query(
        endog_grid=jnp.asarray([1.0, 0.0], dtype=jdtype),
        policy=jnp.asarray([3.0, 1.0], dtype=jdtype),
        value=jnp.asarray([3.0, 1.0], dtype=jdtype),
        marginal=jnp.asarray([3.0, 1.0], dtype=jdtype),
        segment_id=jnp.asarray([0.0, 0.0], dtype=jdtype),
        x_query=jnp.asarray([0.5], dtype=jdtype),
    )
    assert all(float(channel[0]) == 2.0 for channel in descending)

    point = envelope_at_query(
        endog_grid=jnp.asarray([0.5], dtype=jdtype),
        policy=jnp.asarray([7.0], dtype=jdtype),
        value=jnp.asarray([7.0], dtype=jdtype),
        marginal=jnp.asarray([7.0], dtype=jdtype),
        segment_id=jnp.asarray([0.0], dtype=jdtype),
        x_query=jnp.asarray([0.5], dtype=jdtype),
    )
    assert all(float(channel[0]) == 7.0 for channel in point)


def test_all_channels_publish_or_all_are_nan() -> None:
    """One unreadable selected channel poisons value, policy and marginal together."""
    jdtype = _jdtype()
    got = envelope_at_query(
        endog_grid=jnp.asarray([0.0, 1.0], dtype=jdtype),
        policy=jnp.asarray([jnp.inf, jnp.inf], dtype=jdtype),
        value=jnp.asarray([0.0, 1.0], dtype=jdtype),
        marginal=jnp.asarray([3.0, 3.0], dtype=jdtype),
        segment_id=jnp.asarray([0.0, 0.0], dtype=jdtype),
        x_query=jnp.asarray([0.5], dtype=jdtype),
    )
    assert all(bool(np.isnan(np.asarray(channel)).all()) for channel in got)


def test_dense_and_every_block_partition_are_bit_identical() -> None:
    """The certified block argument cannot alter any discrete or published result."""
    jdtype, uint = _jdtype(), _uint()

    def publish(block_size: int) -> tuple[FloatND, FloatND, FloatND]:
        return envelope_at_query(
            endog_grid=jnp.asarray([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=jdtype),
            policy=jnp.asarray([10.0, 10.0, 10.0, 20.0, 20.0, 20.0], dtype=jdtype),
            value=jnp.asarray([0.0, 1.0, 2.0, -0.25, 1.25, 2.75], dtype=jdtype),
            marginal=jnp.asarray([1.0, 1.0, 1.0, 1.5, 1.5, 1.5], dtype=jdtype),
            segment_id=jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jdtype),
            x_query=jnp.linspace(-0.25, 2.25, 37, dtype=jdtype),
            segment_block_size=block_size,
        )

    dense = publish(0)
    for block_size in range(1, 9):
        blocked = publish(block_size)
        for expected, got in zip(dense, blocked, strict=True):
            expected_arr, got_arr = np.asarray(expected), np.asarray(got)
            np.testing.assert_array_equal(np.isnan(got_arr), np.isnan(expected_arr))
            finite = ~np.isnan(expected_arr)
            np.testing.assert_array_equal(
                got_arr[finite].view(uint), expected_arr[finite].view(uint)
            )


def test_lowering_has_one_winner_and_three_selected_reads() -> None:
    """Representation width never appears as per-candidate traced arithmetic."""
    jdtype = _jdtype()

    def evaluate(query):
        return envelope_at_query(
            endog_grid=jnp.asarray([0.0, 1.0, 0.0, 1.0], dtype=jdtype),
            policy=jnp.asarray([1.0, 1.0, 2.0, 2.0], dtype=jdtype),
            value=jnp.asarray([0.0, 1.0, 0.0, 2.0], dtype=jdtype),
            marginal=jnp.asarray([1.0, 1.0, 2.0, 2.0], dtype=jdtype),
            segment_id=jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jdtype),
            x_query=query,
        )

    query = jnp.asarray([0.25, 0.75], dtype=jdtype)
    text = jax.jit(evaluate).lower(query).as_text()
    suffix = "F64" if X64_ENABLED else "F32"
    assert text.count("stablehlo.custom_call") == 4
    assert text.count(f"ExactQueryWinner{suffix}") == 1
    assert text.count(f"ExactAffineRead{suffix}") == 3

    vmapped = jax.jit(jax.vmap(evaluate))
    vmapped_text = vmapped.lower(jnp.stack([query, query])).as_text()
    assert vmapped_text.count("stablehlo.custom_call") == 4
    assert vmapped_text.count(f"ExactQueryWinner{suffix}") == 1
    assert vmapped_text.count(f"ExactAffineRead{suffix}") == 3
    assert vmapped_text.count("stablehlo.while") == 1
