"""Certified sign of an affine value difference.

The envelope's structural decisions — which branch wins, whether two links cross
inside a cell, whether a crossing sits exactly on a node — are settled by the
sign of the difference of two affine value lines. That sign must be exact with
respect to the represented float inputs: invariant to a common additive value
level, never masking a strict winner behind a magnitude-scaled tolerance, and
reporting `UNRESOLVED_SIGN` (fail loud) rather than guessing when the arithmetic
cannot certify it.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)


def _exact_line_margin(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    x: float,
) -> int:
    """Return the exact sign of `A(x) - B(x)` for the two extended affine lines.

    The inputs are read as the active precision represents them, which is the
    contract the certified predicate is held to.
    """

    def to_q(value: float) -> Fraction:
        return Fraction.from_float(_as_represented(value))

    def line(seg: tuple[float, float, float, float]) -> Fraction:
        x0, x1, v0, v1 = (to_q(value) for value in seg)
        return v0 + (to_q(x) - x0) * (v1 - v0) / (x1 - x0)

    margin = line(a) - line(b)
    return (margin > 0) - (margin < 0)


def _sign_at(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    x: float,
) -> int:
    """Evaluate the production primitive on one scalar configuration."""
    a_x0, a_x1, a_v0, a_v1 = a
    b_x0, b_x1, b_v0, b_v1 = b
    return int(
        certified_margin_sign(
            a_x0=jnp.asarray(a_x0),
            a_x1=jnp.asarray(a_x1),
            a_v0=jnp.asarray(a_v0),
            a_v1=jnp.asarray(a_v1),
            b_x0=jnp.asarray(b_x0),
            b_x1=jnp.asarray(b_x1),
            b_v0=jnp.asarray(b_v0),
            b_v1=jnp.asarray(b_v1),
            x_query=jnp.asarray(x),
        )
    )


F3_A = (9.0, 10.0, 4.875, 5.0)
F3_B = (9.5, 10.5, 4.75, 5.25)


def test_exact_node_aligned_crossing_is_a_certified_tie():
    """Two links meeting exactly at a node have margin sign 0 there."""
    assert _sign_at(F3_A, F3_B, 10.0) == 0


@pytest.mark.parametrize(("x_query", "expected"), [(9.9, 1), (10.1, -1)])
def test_margin_sign_matches_the_exact_rational_sign_around_the_crossing(
    x_query: float, expected: int
):
    """The certified sign matches the exact rational sign on both sides."""
    assert _exact_line_margin(F3_A, F3_B, x_query) == expected
    assert _sign_at(F3_A, F3_B, x_query) == expected


def _as_represented(x: float) -> float:
    """Return `x` as it is actually represented at the active precision."""
    return float(jnp.asarray(x))


def _few_ulp_gap_at_a_large_level() -> tuple[float, float]:
    """Return a large value level and a gap of a few ULPs representable there.

    The witness has to be built from the active precision: at `1e12` a float32
    has an ULP of 65536, so a `0.001` gap would not survive rounding and the two
    lines would collapse into one.
    """
    level = 1.0e12 if jnp.zeros(()).dtype == jnp.float64 else 1.0e5
    return level, 2.0 * float(jnp.spacing(jnp.asarray(level)))


def test_strict_dominance_survives_a_large_common_value_level():
    """A few-ULP gap at a large value level is a strict winner, not a tie."""
    level, gap = _few_ulp_gap_at_a_large_level()
    a = (0.0, 1.0, level, level + gap)
    b = (0.0, 1.0, level + 2.0 * gap, level + 3.0 * gap)
    # The gap must survive rounding, or the witness is vacuous.
    assert _as_represented(a[2]) != _as_represented(b[2])
    assert _exact_line_margin(a, b, 0.5) == -1
    assert _sign_at(a, b, 0.5) == -1


def test_margin_sign_is_invariant_to_a_common_additive_shift():
    """Adding a common constant to every value cannot change the sign."""
    shift, gap = _few_ulp_gap_at_a_large_level()
    a = (0.0, 1.0, 0.0, gap)
    b = (0.0, 1.0, 2.0 * gap, 3.0 * gap)
    shifted_a = (a[0], a[1], a[2] + shift, a[3] + shift)
    shifted_b = (b[0], b[1], b[2] + shift, b[3] + shift)
    assert _as_represented(shifted_a[2]) != _as_represented(shifted_b[2])
    assert _sign_at(a, b, 0.5) == _sign_at(shifted_a, shifted_b, 0.5) == -1


def test_links_that_round_to_the_same_representation_are_a_certified_tie():
    """When rounding collapses two lines into one, the tie is exact, not a guess."""
    level = 1.0e12 if jnp.zeros(()).dtype == jnp.float64 else 1.0e5
    below_one_ulp = 0.1 * float(jnp.spacing(jnp.asarray(level)))
    a = (0.0, 1.0, level, level)
    b = (0.0, 1.0, level + below_one_ulp, level + below_one_ulp)
    assert _as_represented(a[2]) == _as_represented(b[2])
    assert _sign_at(a, b, 0.5) == 0


def test_identical_links_are_a_certified_tie():
    """A link compared with itself has an exactly zero margin."""
    assert _sign_at(F3_A, F3_A, 9.5) == 0


def test_non_finite_input_is_unresolved_rather_than_a_silent_sign():
    """A dead (NaN) endpoint cannot certify a sign."""
    dead = (9.0, 10.0, float("nan"), 5.0)
    assert _sign_at(dead, F3_B, 9.75) == UNRESOLVED_SIGN


def test_certified_sign_matches_the_rational_oracle_on_randomized_links():
    """Whenever the primitive certifies a sign it equals the exact rational sign."""
    key = jax.random.key(20260727)
    keys = jax.random.split(key, 400)
    unresolved = 0
    for k in keys:
        raw = jax.random.uniform(k, (5,), minval=-5.0, maxval=5.0)
        a_x0 = float(raw[0])
        a_x1 = a_x0 + float(jnp.abs(raw[1])) + 0.25
        b_x0 = float(raw[2])
        b_x1 = b_x0 + float(jnp.abs(raw[3])) + 0.25
        a = (a_x0, a_x1, float(raw[4]), float(raw[4]) + 0.5)
        b = (b_x0, b_x1, float(raw[0]), float(raw[2]) - 0.25)
        x_query = float(raw[3])
        got = _sign_at(a, b, x_query)
        if got == UNRESOLVED_SIGN:
            unresolved += 1
            continue
        assert got == _exact_line_margin(a, b, x_query)
    assert unresolved == 0


def test_certified_sign_is_jit_and_vmap_compatible():
    """The primitive runs under jit and vmap with static shapes."""
    a_x0 = jnp.array([9.0, 0.0])
    a_x1 = jnp.array([10.0, 1.0])
    a_v0 = jnp.array([4.875, 0.0])
    a_v1 = jnp.array([5.0, 1.0])
    b_x0 = jnp.array([9.5, 0.0])
    b_x1 = jnp.array([10.5, 1.0])
    b_v0 = jnp.array([4.75, 1.0])
    b_v1 = jnp.array([5.25, 0.0])
    x_query = jnp.array([10.0, 0.25])

    batched = jax.jit(
        jax.vmap(
            lambda *args: certified_margin_sign(
                a_x0=args[0],
                a_x1=args[1],
                a_v0=args[2],
                a_v1=args[3],
                b_x0=args[4],
                b_x1=args[5],
                b_v0=args[6],
                b_v1=args[7],
                x_query=args[8],
            )
        )
    )
    got = batched(a_x0, a_x1, a_v0, a_v1, b_x0, b_x1, b_v0, b_v1, x_query)
    # Second cell: A(x)=x, B(x)=1-x, so A-B<0 at x=0.25.
    assert got.tolist() == [0, -1]
