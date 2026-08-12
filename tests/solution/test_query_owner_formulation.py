"""Ownership at a query is decided by the exact margin, never by row order.

`envelope_at_query` publishes the value, policy and marginal of whichever
candidate segment owns a query abscissa. Owning is a structural decision: the
candidate whose affine line is exactly higher at the query owns it, and that is
a property of the two lines alone. Nothing about how the candidates happen to
be laid out in the input arrays may reach it.

Two ways of breaking that rule are pinned here. The first is treating "the
arithmetic could not separate these two lines" as "these two lines are level":
a difference below the certificate's own resolution is an unresolved
comparison, not a tie, and sending it to the tie-break lets a deterministic
preference decide a question the geometry has already answered. The second is
ranking genuine ties through a single packed scalar, which brings distinct
slopes onto the same key and so hands the tie-break a collision to resolve
positionally.

Where a comparison genuinely cannot be settled the published answer is NaN.
Abstaining is a defined outcome; publishing the loser is not.
"""

from dataclasses import dataclass
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from _lcm.egm.upper_envelope.query import (
    _slope_words,
    _tie_break_key,
    _TieBreakKey,
    envelope_at_query,
)


def _key_at(key: _TieBreakKey, column: int) -> tuple[float, ...]:
    """Return one segment's ordered tie-break fields as plain floats."""
    return tuple(float(field[0, column]) for field in key)


def _exact(value: float | np.floating) -> Fraction:
    """Return the float's exact value as a rational."""
    return Fraction(*float(value).as_integer_ratio())


def _exact_line_at(
    *,
    x0: float | np.floating,
    x1: float | np.floating,
    v0: float | np.floating,
    v1: float | np.floating,
    query: float | np.floating,
) -> Fraction:
    """Return the affine line's exact value at `query`, free of any rounding."""
    return _exact(v0) + (_exact(query) - _exact(x0)) * (_exact(v1) - _exact(v0)) / (
        _exact(x1) - _exact(x0)
    )


@dataclass(frozen=True, kw_only=True)
class _RaisedWitness:
    """Two candidate branches whose exact margin at the query is positive but tiny."""

    endog_grid: np.ndarray
    """Abscissae of the four candidates, two per branch."""
    value: np.ndarray
    """Candidate values, sitting on a level far above the gap between the lines."""
    policy: np.ndarray
    """Candidate policies, distinct per branch so the published owner is visible."""
    marginal: np.ndarray
    """Candidate marginals, zero throughout."""
    segment_id: np.ndarray
    """Branch label, so the two lines never join into one segment."""
    query: np.floating
    """The abscissa at which ownership is contested."""
    winning_policy: float
    """The policy of the branch that is exactly above at `query`."""


def _raised_witness() -> _RaisedWitness:
    """Two lines whose exact margin at the query is positive but tiny.

    Both value lines sit on a common additive level many orders of magnitude
    above the gap between them, which is the regime the certified comparison
    exists to survive. The first line is exactly above the second at the query,
    so its policy is the only correct answer.
    """
    dtype = np.float32 if jnp.zeros(()).dtype.itemsize == 4 else np.float64
    cast = dtype
    if dtype is np.float32:
        width = cast(2**24)
        offset = cast(1_355_734.625)
        magnitude = cast(2**20)
        low_gap, high_gap = cast(1.0), cast(-11.375)
    else:
        width = cast(2**53)
        offset = cast(825_086_954_632_762.6)
        magnitude = cast(2**49)
        low_gap, high_gap = cast(1.5), cast(-14.875)

    shift = cast(1024.0)
    level = cast(2.0**18)
    x0, x1, query = cast(shift), cast(shift + width), cast(shift + offset)
    a0, a1 = cast(-magnitude + level), cast(magnitude + level)
    b0, b1 = cast(a0 - low_gap), cast(a1 - high_gap)

    margin = _exact_line_at(x0=x0, x1=x1, v0=a0, v1=a1, query=query) - _exact_line_at(
        x0=x0, x1=x1, v0=b0, v1=b1, query=query
    )
    assert margin > 0, "the witness only says anything while the first line wins"

    return _RaisedWitness(
        endog_grid=np.asarray([x0, x1, x0, x1], dtype=dtype),
        value=np.asarray([a0, a1, b0, b1], dtype=dtype),
        policy=np.asarray([0.5, 0.5, 0.25, 0.25], dtype=dtype),
        marginal=np.zeros(4, dtype=dtype),
        segment_id=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=dtype),
        query=query,
        winning_policy=float(dtype(0.5)),
    )


def _published_policy(*, witness: _RaisedWitness, order: np.ndarray) -> float:
    """Return the policy published for the witness with its rows in `order`."""
    _value, policy, _marginal = envelope_at_query(
        endog_grid=jnp.asarray(witness.endog_grid[order]),
        policy=jnp.asarray(witness.policy[order]),
        value=jnp.asarray(witness.value[order]),
        marginal=jnp.asarray(witness.marginal[order]),
        segment_id=jnp.asarray(witness.segment_id[order]),
        x_query=jnp.asarray([witness.query]),
    )
    return float(policy[0])


_STORED = np.arange(4)
_SWAPPED = np.asarray([2, 3, 0, 1])


@pytest.mark.parametrize("order", [_STORED, _SWAPPED])
def test_a_strictly_positive_exact_margin_never_publishes_the_losing_policy(
    order: np.ndarray,
) -> None:
    """A candidate exactly above the others is published, or nothing is."""
    witness = _raised_witness()
    policy = _published_policy(witness=witness, order=order)
    assert np.isnan(policy) or policy == witness.winning_policy


def test_row_order_does_not_change_the_owner_of_a_query() -> None:
    """The same two lines publish the same policy however the rows are laid out."""
    witness = _raised_witness()
    stored = _published_policy(witness=witness, order=_STORED)
    swapped = _published_policy(witness=witness, order=_SWAPPED)
    assert (np.isnan(stored) and np.isnan(swapped)) or stored == swapped


@pytest.mark.parametrize("magnitude", [1.0, 1.0e3, 1.0e6])
def test_distinct_slopes_are_not_collapsed_onto_one_tie_break_key(
    magnitude: float,
) -> None:
    """Two links whose slopes differ order strictly, at every slope magnitude.

    The right-continuous tie-break prefers the larger value-slope among
    candidates that are genuinely level. Slopes hundreds of representable steps
    apart are not level, and a ranking that maps both onto one key has lost the
    ordering it exists to express. A steep link is where that bites: any
    bounded reparametrisation of the slope flattens as it saturates, so the
    ordering survives near zero and is lost exactly where EGM candidates are
    steepest.
    """
    dtype = jnp.zeros(()).dtype
    lower = jnp.asarray(magnitude, dtype=dtype)
    upper = lower
    for _ in range(798):
        upper = jnp.nextafter(upper, jnp.asarray(jnp.inf, dtype=dtype))
    assert float(upper) > float(lower)

    grid = jnp.zeros((1, 2), dtype=dtype)
    slope_high, slope_low = _slope_words(
        left_value=grid,
        right_value=jnp.stack([lower, upper])[None, :],
        left_grid=grid,
        right_grid=jnp.ones((1, 2), dtype=dtype),
    )
    eligible = jnp.ones((1, 2), dtype=bool)

    key = _tie_break_key(
        level_with=eligible,
        right_available=eligible,
        slope_high=slope_high,
        slope_low=slope_low,
    )

    assert _key_at(key, 0) != _key_at(key, 1)


def test_a_subnormal_affine_reading_is_never_published_as_zero() -> None:
    """A query whose exact affine reading is subnormal-sourced is not published as 0.

    With `x0 = 0`, `x1` the smallest normal and the query the smallest
    subnormal, the exact reading of a `[0, 1]` link is `2**-23` at float32 and
    `2**-52` at float64 — ordinary numbers, far from zero. Publishing zero
    instead is a wrong answer that a rival at half the exact value then beats,
    so the policy reverses. Either the exact reading or a loud NaN is
    acceptable; a silent zero is not.
    """
    dtype = np.float32 if jnp.zeros(()).dtype.itemsize == 4 else np.float64
    x1 = dtype(np.finfo(dtype).tiny)
    query = dtype(np.finfo(dtype).smallest_subnormal)
    expected = float(_exact(query) / _exact(x1))

    value, policy, marginal = envelope_at_query(
        endog_grid=jnp.asarray([dtype(0.0), x1]),
        policy=jnp.asarray([dtype(0.0), dtype(1.0)]),
        value=jnp.asarray([dtype(0.0), dtype(1.0)]),
        marginal=jnp.asarray([dtype(0.0), dtype(1.0)]),
        segment_id=jnp.asarray([dtype(0.0), dtype(0.0)]),
        x_query=jnp.asarray([query]),
    )

    for channel in (value, policy, marginal):
        published = float(channel[0])
        assert np.isnan(published) or published == expected


# How far below the smallest normal the query sits. One halving is the worst
# case of the family, not merely a case: the exact gap is 0.25 there and halves
# with every further step down, so the damage is greatest at the *top* edge of
# the subnormal band. The float32 band is 23 halvings deep, so these stay
# strictly inside it at both precisions — below that the query is not
# unreadable, it is genuinely zero, which is a different thing and not a defect.
@pytest.mark.parametrize("halvings", [1, 2, 20])
def test_a_subnormal_operand_is_refused_rather_than_certified(
    halvings: int,
) -> None:
    """A comparison whose operands the backend cannot read reports unresolved.

    The backend flushes subnormals to zero in comparison as well as in
    arithmetic, so a difference formed from one is taken against zero rather
    than against the operand. With `x0 = 0`, `x1` the smallest normal and a
    subnormal query, both affine numerators collapse to exactly zero and
    nothing is discarded on the way — so the determinant is exactly zero,
    carries no error at all, and would certify two strictly ordered lines as
    exactly level. That is the one outcome that lets a caller choose freely,
    which is why the operands are refused before the determinant is read.

    Only the *abscissa* is unreadable, and the values it is read against are
    ordinary. One halving puts the query at `tiny/2`, where the two lines read
    `0.5` and `0.25` — so certifying them level is an error of a quarter, not
    of a rounding. Nothing here rests on the difference being too small to
    represent; it rests on the abscissa being unreadable, and the error that
    follows is the size of the values.

    The construction is scale-free — the lines span `[0, tiny]` and the query is
    a fraction of `tiny`, so the readings are `0.5` and `0.25` wherever `tiny`
    happens to sit. Raising the precision does not shrink the error, it only
    moves the band it lives in, which is why the same figures hold at float32
    and at float64.
    """
    dtype = np.float32 if jnp.zeros(()).dtype.itemsize == 4 else np.float64
    x0, x1 = dtype(0.0), dtype(np.finfo(dtype).tiny)
    query = dtype(np.ldexp(np.float64(x1), -halvings))
    above, below = dtype(1.0), dtype(0.5)

    at_above = _exact_line_at(x0=x0, x1=x1, v0=dtype(0.0), v1=above, query=query)
    at_below = _exact_line_at(x0=x0, x1=x1, v0=dtype(0.0), v1=below, query=query)
    assert at_above > at_below

    sign = certified_margin_sign(
        a_x0=jnp.asarray(x0),
        a_x1=jnp.asarray(x1),
        a_v0=jnp.asarray(dtype(0.0)),
        a_v1=jnp.asarray(above),
        b_x0=jnp.asarray(x0),
        b_x1=jnp.asarray(x1),
        b_v0=jnp.asarray(dtype(0.0)),
        b_v1=jnp.asarray(below),
        x_query=jnp.asarray(query),
    )

    assert int(sign) == UNRESOLVED_SIGN
