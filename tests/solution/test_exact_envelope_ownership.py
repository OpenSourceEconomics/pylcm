"""Ownership of the exact envelope where several links meet at one place.

Inside a node cell every live link spans the whole cell, so the envelope there is
the pointwise maximum of full lines. These tests pin the three ways that maximum
is easy to get wrong: several links reading the same published value at a
boundary, several links crossing at one abscissa, and a change of resource units
that pushes the comparison arithmetic out of its exact range. In every case the
row must publish the owner the exact rational envelope selects — failing loud is
only allowed when a decision genuinely cannot be certified.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact


def _active_dtype() -> type:
    """Return the numpy dtype matching the precision the suite runs at."""
    return np.float64 if jnp.zeros(()).dtype == jnp.float64 else np.float32


def _refine(grid, policy, value, *, n_refined: int, max_runs: int):
    """Refine one candidate chain under jit and drop the NaN padding."""
    out_grid, out_policy, out_value, n_kept = jax.jit(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g, policy=p, value=v, n_refined=n_refined, max_runs=max_runs
        )
    )(jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value))
    keep = int(n_kept)
    return (
        np.asarray(out_grid),
        np.asarray(out_policy),
        np.asarray(out_value),
        keep,
    )


def _read(grid: np.ndarray, ordinate: np.ndarray, query: float) -> float:
    """Read the published row at `query`, taking the right branch at a node."""
    upper = int(np.clip(np.searchsorted(grid, query, side="right"), 1, len(grid) - 1))
    lower = upper - 1
    width = grid[upper] - grid[lower]
    if width == 0:
        return float(ordinate[upper])
    weight = (query - grid[lower]) / width
    return float((1 - weight) * ordinate[lower] + weight * ordinate[upper])


def _reciprocal_slope(x0, x1, v0, v1, dtype) -> float:
    """Return `1 / slope` of the link, as the policy the chain carries."""
    slope = (Fraction(float(v1)) - Fraction(float(v0))) / (
        Fraction(float(x1)) - Fraction(float(x0))
    )
    return dtype(float(1 / slope))


def _same_reading_chain(dtype, *, scale_exponent: int = 0):
    """Build four runs whose boundary readings collapse onto one published value.

    The fourth run owns the envelope at `delta / 2` by a margin far below the
    spacing of the values, so a judge that reads only the published doubles
    cannot tell it from the first run. `scale_exponent` rescales the resource
    axis by an exact power of two, which changes no ownership.
    """
    if dtype is np.float32:
        level, delta, offset = dtype(1e5), dtype(1e-3), dtype(99999.81)
    else:
        level, delta, offset = dtype(1e12), dtype(1e-5), dtype(999999999999.8976)
    ulp = np.spacing(level).astype(dtype)

    grid = np.array([-2, delta, 0, 3, -0.6, delta, -1, 3], dtype=dtype)
    value = np.array(
        [
            level - 100,
            dtype((level - 100) + (delta + 2) / 100),
            level,
            dtype(level + 0.3),
            offset,
            level,
            dtype(level - 0.5),
            dtype(level + 1.5 + ulp),
        ],
        dtype=dtype,
    )
    policy = np.array(
        [
            _reciprocal_slope(grid[i], grid[j], value[i], value[j], dtype)
            for i, j in ((0, 1), (2, 3), (4, 5), (6, 7))
            for _ in range(2)
        ],
        dtype=dtype,
    )
    query = dtype(delta / 2)
    if scale_exponent:
        grid = np.ldexp(grid, scale_exponent).astype(dtype)
        policy = np.ldexp(policy, scale_exponent).astype(dtype)
        query = np.ldexp(query, scale_exponent).astype(dtype)
    return grid, policy, value, query


def test_equal_published_readings_do_not_decide_ownership():
    """The link certified above the others owns the interval beside a boundary.

    Four links read the same double at the boundary, so the published value
    cannot separate them; the policy still can, and it must be the one the exact
    envelope selects rather than whichever link the rounded comparison favours.
    """
    dtype = _active_dtype()
    grid, policy, value, query = _same_reading_chain(dtype)
    assert np.all(np.diff(grid - policy) > 0), "the chain must be savings-ordered"

    out_grid, out_policy, _out_value, keep = _refine(
        grid, policy, value, n_refined=30, max_runs=8
    )
    assert keep <= 30, "a fully certifiable row must publish rather than fail loud"
    assert _read(out_grid[:keep], out_policy[:keep], float(query)) == pytest.approx(
        float(policy[6]), abs=0.0
    )


def test_equal_published_readings_survive_a_change_of_resource_units():
    """Rescaling resources by an exact power of two changes no ownership.

    The comparison is a determinant in the abscissae and values, so a pure change
    of units may not move the envelope — even when the rescaled determinant is
    small enough that a naive expansion would underflow to zero.
    """
    dtype = _active_dtype()
    exponent = -100 if dtype is np.float32 else -530
    grid, policy, value, query = _same_reading_chain(dtype, scale_exponent=exponent)

    out_grid, out_policy, _out_value, keep = _refine(
        grid, policy, value, n_refined=30, max_runs=8
    )
    assert keep <= 30, "a fully certifiable row must publish rather than fail loud"
    assert _read(out_grid[:keep], out_policy[:keep], float(query)) == pytest.approx(
        float(policy[6]), abs=0.0
    )


@pytest.mark.parametrize(
    "policies",
    [(10.0, 5.0, 1.0), (10.0, 4.0, 1.0), (12.0, 6.0, 2.0), (20.0, 8.0, 1.0)],
)
def test_three_links_crossing_at_one_abscissa_emit_only_the_two_owners(policies):
    """A link owning no interval leaves no record at a shared crossing.

    Three links meet at a single abscissa; the outer two own the ground on either
    side and the middle one owns nothing. The event is certifiable, so the row
    publishes exactly two records there — the outgoing owner and the incoming
    one — instead of reporting overflow.
    """
    dtype = _active_dtype()
    low, high, middle = policies
    grid = np.array([0, 2, 0, 2, 0, 2], dtype=dtype)
    policy = np.array([low, low, high, high, middle, middle], dtype=dtype)
    value = np.array([-0.1, 0.1, -0.2, 0.2, -1.0, 1.0], dtype=dtype)
    assert np.all(np.diff(grid - policy) > 0), "the chain must be savings-ordered"

    out_grid, out_policy, _out_value, keep = _refine(
        grid, policy, value, n_refined=20, max_runs=8
    )
    assert keep <= 20, "a certifiable simultaneous event must not report overflow"
    at_event = out_grid[:keep] == dtype(1)
    assert int(at_event.sum()) == 2, "the event carries the two one-sided owners"
    np.testing.assert_array_equal(out_policy[:keep][at_event], [low, middle])
