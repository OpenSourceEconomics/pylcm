"""Spec for interpolation on NaN-padded, weakly-ascending carry rows.

Contract under test — `_lcm.egm.interp.interp_on_padded_grid`:

    interp_on_padded_grid(
        *, x_query: FloatND, xp: Float1D, fp: Float1D,
        fp_slopes: Float1D | None = None,
    ) -> FloatND

`xp` is a NaN-padded, weakly ascending grid row (NaNs only in the tail).
Behavior:

- linear interpolation between neighboring non-NaN nodes,
- below the first node: linear extrapolation along the first bracket's secant;
  at or above the last non-NaN node: clamp to the boundary value,
- tie-safe at duplicated abscissae (zero-width brackets from envelope kinks):
  queries strictly below the duplicate interpolate toward the left value, queries
  at or above it use the right value — never a division by the zero bracket,
- `-inf` endpoints (infeasible values) force their bracket interior to `-inf`
  instead of NaN; a query exactly on a finite neighbor returns that value,
- with `fp_slopes` (the exact node derivatives), brackets whose endpoint
  values and slopes are all finite get a monotone cubic Hermite correction;
  all other brackets — and the contracts above — behave exactly as in the
  linear case.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import interp


def test_matches_numpy_interp_on_clean_grid():
    """On an unpadded strictly ascending grid, results equal `np.interp`."""
    xp = jnp.array([1.0, 2.0, 4.0, 8.0])
    fp = jnp.array([0.0, 3.0, 5.0, 6.0])
    x_query = jnp.array([1.0, 1.5, 3.0, 7.9, 8.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    expected = np.interp(np.asarray(x_query), np.asarray(xp), np.asarray(fp))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_nan_tail_is_ignored():
    """NaN padding in the tail does not affect interpolation on the prefix."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan, jnp.nan])
    x_query = jnp.array([1.5, 3.0, 4.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    expected = np.interp(np.asarray(x_query), [1.0, 2.0, 4.0], [0.0, 3.0, 5.0])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_below_support_extrapolates_and_above_support_clamps():
    """Below the first node the first bracket's secant continues; above clamps.

    Below support the carry read matches the canonical state-grid read
    (`map_coordinates` extrapolates linearly), so a transition landing below
    the child grid (a borrowing corner whose savings undershoot the grid
    start) is priced on the edge slope rather than credited with the boundary
    value — which no feasible action attains there. At or above the last node
    the boundary value applies: a refined envelope row can end in a
    crossing-inserted near-duplicate bracket whose secant slope is arbitrarily
    steep, so extending it would poison every above-support read.
    """
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    x_query = jnp.array([0.0, 100.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    below = 0.0 + (0.0 - 1.0) * (3.0 - 0.0) / (2.0 - 1.0)
    above = 5.0
    np.testing.assert_allclose(got, jnp.array([below, above]), atol=1e-12)


def test_out_of_range_reads_ignore_the_hermite_correction():
    """With slopes, out-of-range queries follow the slope-free convention.

    The Hermite correction vanishes at the bracket endpoints, so extending it
    outside the bracket would leave the cubic's tail; the read instead
    continues the first bracket's secant below support and clamps to the
    boundary value above, matching the slope-free convention.
    """
    xp = jnp.array([1.0, 2.0, 4.0])
    fp = jnp.array([0.0, 3.0, 5.0])
    fp_slopes = jnp.array([0.0, 0.0, 0.0])

    got = interp.interp_on_padded_grid(
        x_query=jnp.array([0.5, 5.0]), xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, jnp.array([-1.5, 5.0]), atol=1e-12)


def test_duplicated_abscissa_is_tie_safe():
    """A zero-width bracket (envelope kink) yields finite one-sided values."""
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 10.0, 20.0, 30.0])
    x_query = jnp.array([0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    np.testing.assert_allclose(got[0], 5.0, atol=1e-12)
    np.testing.assert_allclose(got[2], 25.0, atol=1e-12)
    np.testing.assert_allclose(got[1], 20.0, atol=1e-12)


def test_neg_inf_node_yields_neg_inf_inside_brackets():
    """A `-inf` node forces its bracket interiors to `-inf`, never NaN.

    Value rows may carry `-inf` at a node (e.g. a terminal bequest at zero
    wealth); a query exactly on the finite neighbor returns that neighbor's
    value.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-jnp.inf, 3.0, 5.0])
    x_query = jnp.array([0.0, 0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    np.testing.assert_array_equal(np.asarray(got), [-np.inf, -np.inf, 3.0, 4.0])


def test_all_neg_inf_row_yields_neg_inf_everywhere():
    """An all-`-inf` row (infeasible combo) interpolates to `-inf`, never NaN."""
    xp = jnp.array([0.0, 1.0, 2.0, jnp.nan])
    fp = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, jnp.nan])
    x_query = jnp.array([-1.0, 0.5, 1.0, 3.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    assert bool(jnp.isneginf(got).all())


def test_exact_slopes_reproduce_a_monotone_cubic():
    """With exact node slopes, a monotone cubic is interpolated exactly.

    Per bracket the interpolant matches endpoint values and derivatives — four
    degrees of freedom — so a cubic whose slopes pass the monotonicity limiter
    untouched is reproduced to round-off, where linear interpolation is biased.
    """
    xp = jnp.array([1.0, 1.5, 2.0])
    fp = xp**3
    fp_slopes = 3.0 * xp**2
    x_query = jnp.array([1.1, 1.25, 1.7, 1.95])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, np.asarray(x_query) ** 3, atol=1e-12)


def test_slopes_keep_nan_tail_boundary_reads_and_node_values():
    """The Hermite path preserves NaN-tail handling, boundary reads, and nodes."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    fp_slopes = jnp.array([3.0, 2.0, 0.5, jnp.nan])
    x_query = jnp.array([0.0, 1.0, 2.0, 4.0, 100.0])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    below = 0.0 + (0.0 - 1.0) * 3.0
    above = 5.0
    np.testing.assert_allclose(
        got, jnp.array([below, 0.0, 3.0, 5.0, above]), atol=1e-12
    )


def test_slopes_keep_duplicated_abscissa_tie_safe():
    """Zero-width kink brackets yield the same one-sided values as linear."""
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 10.0, 20.0, 30.0])
    fp_slopes = jnp.array([10.0, 10.0, 10.0, 10.0])
    x_query = jnp.array([1.0])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, jnp.array([20.0]), atol=1e-12)


def test_slopes_on_neg_inf_bracket_fall_back_to_linear_rule():
    """A `-inf` endpoint forces its bracket interior to `-inf` with slopes given.

    Carry rows store slope `0.0` at `-inf` value nodes; the bracket must
    behave exactly as in the linear case — `-inf` inside, the finite
    neighbor's value on it.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-jnp.inf, 3.0, 5.0])
    fp_slopes = jnp.array([0.0, 2.0, 2.0])
    x_query = jnp.array([0.0, 0.5, 1.0, 1.5])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_array_equal(np.asarray(got), [-np.inf, -np.inf, 3.0, 4.0])


def test_steep_slopes_are_limited_to_keep_monotonicity():
    """Node slopes far above the secant cannot produce non-monotone output.

    An unlimited cubic Hermite with endpoint slopes ten times the secant
    overshoots inside the bracket; the limiter caps slopes at three times the
    secant, so the interpolant stays within the bracket's value range and
    weakly increasing on increasing data.
    """
    xp = jnp.array([0.0, 1.0])
    fp = jnp.array([0.0, 1.0])
    fp_slopes = jnp.array([10.0, 10.0])
    x_query = jnp.linspace(0.0, 1.0, 101)

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    assert bool(jnp.all(jnp.diff(got) >= 0.0))
    assert bool(jnp.all((got >= 0.0) & (got <= 1.0)))


# Prepared-row interpolation: `prepare_padded_grid` builds the row's +inf search
# key and valid length once; `interp_on_prepared_grid` does only searchsorted +
# gather. `interp_on_padded_grid` is the thin wrapper that prepares the row and
# delegates, so the prepared path must be bit-identical to the padded path.

_PREPARED_GRID_CASES = {
    "clean": (
        jnp.array([1.0, 2.0, 4.0, 8.0]),
        jnp.array([0.0, 3.0, 5.0, 6.0]),
        None,
    ),
    "nan_tail": (
        jnp.array([1.0, 2.0, 4.0, jnp.nan, jnp.nan]),
        jnp.array([0.0, 3.0, 5.0, jnp.nan, jnp.nan]),
        None,
    ),
    "duplicated_abscissa": (
        jnp.array([0.0, 1.0, 1.0, 2.0]),
        jnp.array([0.0, 10.0, 20.0, 30.0]),
        None,
    ),
    "neg_inf_node": (
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([-jnp.inf, 3.0, 5.0]),
        None,
    ),
    "all_neg_inf": (
        jnp.array([0.0, 1.0, 2.0, jnp.nan]),
        jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, jnp.nan]),
        None,
    ),
    "hermite_cubic": (
        jnp.array([1.0, 1.5, 2.0]),
        jnp.array([1.0, 1.5, 2.0]) ** 3,
        3.0 * jnp.array([1.0, 1.5, 2.0]) ** 2,
    ),
    "hermite_nan_tail": (
        jnp.array([1.0, 2.0, 4.0, jnp.nan]),
        jnp.array([0.0, 3.0, 5.0, jnp.nan]),
        jnp.array([3.0, 2.0, 0.5, jnp.nan]),
    ),
    "hermite_neg_inf": (
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([-jnp.inf, 3.0, 5.0]),
        jnp.array([0.0, 2.0, 2.0]),
    ),
}


@pytest.mark.parametrize("case", list(_PREPARED_GRID_CASES))
def test_prepared_grid_interpolation_matches_padded_grid(case):
    """Preparing the row then interpolating equals interpolating the padded row.

    The query mesh spans the non-NaN range plus out-of-range clamps, kink
    abscissae, and on-node points, so the comparison covers the linear, edge
    clamp, tie-safe, `-inf`, and Hermite branches.
    """
    xp, fp, fp_slopes = _PREPARED_GRID_CASES[case]
    x_query = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 100.0])

    padded = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )
    search_grid, valid_length = interp.prepare_padded_grid(xp)
    prepared = interp.interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )

    np.testing.assert_array_equal(np.asarray(prepared), np.asarray(padded))


def test_prepare_padded_grid_reports_inf_key_and_valid_length():
    """The search key replaces the NaN tail with `+inf`; valid length counts the
    non-NaN prefix."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan, jnp.nan])

    search_grid, valid_length = interp.prepare_padded_grid(xp)

    np.testing.assert_array_equal(
        np.asarray(search_grid), [1.0, 2.0, 4.0, np.inf, np.inf]
    )
    assert int(valid_length) == 3


def test_prepared_interpolation_holds_no_grid_by_query_intermediate():
    """A row prepared once and read at many queries holds no grid-by-query mesh.

    This is the transient-memory contract: with the search key and valid length
    prepared above the query fan-out, the per-query path does only a scalar-carry
    `searchsorted` plus gathers, so no operation carries both the query axis and
    the grid (`n_pad`) axis. A regression here would reintroduce the
    `O(queries * n_pad)` working buffer the NaN preamble used to create.
    """
    n_grid, n_query = 64, 11
    xp = jnp.arange(n_grid, dtype=float)
    fp = jnp.sqrt(xp)
    queries = jnp.linspace(0.0, n_grid, n_query)

    def read_many(search_grid, valid_length, xp, fp, queries):
        return interp.interp_on_prepared_grid(
            x_query=queries,
            search_grid=search_grid,
            valid_length=valid_length,
            xp=xp,
            fp=fp,
        )

    search_grid, valid_length = interp.prepare_padded_grid(xp)
    hlo = jax.jit(read_many).lower(search_grid, valid_length, xp, fp, queries).compile()
    forbidden = {(n_query, n_grid), (n_grid, n_query)}
    offenders = [
        line.strip()
        for line in (hlo.as_text() or "").splitlines()
        if any(f"[{a},{b}]" in line for a, b in forbidden)
    ]
    assert not offenders, "grid-by-query intermediate present:\n" + "\n".join(offenders)


def test_paired_read_derivative_at_an_interior_node_is_the_right_side_slope():
    """At an exact interior node the paired derivative is the right piece's slope.

    The value read at a node selects the right bracket (`side="right"`), whose
    Hermite derivative at its left endpoint is that node's limited slope. RT1
    of the external audit: with zero node slopes both adjacent pieces have
    derivative zero at the node, so anything else (an autodiff subgradient of
    the clip/where representation) is wrong.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([1.0, 2.0, 4.0])
    fp_slopes = jnp.array([0.0, 0.0, 0.0])

    value, derivative = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(1.0), xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(np.asarray(value), 2.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(derivative), 0.0, atol=1e-12)


def test_paired_read_derivative_uses_node_slopes_at_first_and_last_node():
    """The first node reads its right-side slope, the last its left-side slope.

    With exact node slopes that pass the monotonicity limiter untouched, the
    selected piece's endpoint derivative is the node's own slope on both
    boundaries — the first node starts the first bracket, the last node ends
    the last bracket (the read clamps to it).
    """
    xp = jnp.array([1.0, 1.5, 2.0])
    fp = xp**3
    fp_slopes = 3.0 * xp**2

    _, at_first = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(1.0), xp=xp, fp=fp, fp_slopes=fp_slopes
    )
    _, at_last = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(2.0), xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(np.asarray(at_first), 3.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(at_last), 12.0, rtol=1e-12)


def test_paired_read_derivative_matches_finite_differences_off_node():
    """Between nodes the paired derivative is the value read's exact slope."""
    xp = jnp.array([1.0, 1.5, 2.0])
    fp = xp**3
    fp_slopes = 3.0 * xp**2
    step = 1e-6

    for query in (1.1, 1.25, 1.7, 1.95):
        _, derivative = interp.interp_and_derivative_on_padded_grid(
            x_query=jnp.asarray(query), xp=xp, fp=fp, fp_slopes=fp_slopes
        )
        left = interp.interp_on_padded_grid(
            x_query=jnp.asarray(query - step), xp=xp, fp=fp, fp_slopes=fp_slopes
        )
        right = interp.interp_on_padded_grid(
            x_query=jnp.asarray(query + step), xp=xp, fp=fp, fp_slopes=fp_slopes
        )
        np.testing.assert_allclose(
            np.asarray(derivative),
            (np.asarray(right) - np.asarray(left)) / (2.0 * step),
            rtol=1e-5,
        )


def test_paired_read_derivative_out_of_support_matches_the_value_conventions():
    """Below support the derivative is the first bracket's secant; above it is zero.

    The value read extends the first bracket's secant below support and clamps
    to the boundary value above; the paired derivative is the exact slope of
    that value read on both sides.
    """
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    fp_slopes = jnp.array([3.0, 2.0, 0.5, jnp.nan])

    _, below = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(0.0), xp=xp, fp=fp, fp_slopes=fp_slopes
    )
    _, above = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(100.0), xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(np.asarray(below), 3.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(above), 0.0, atol=1e-12)


def test_paired_read_derivative_at_a_duplicated_abscissa_reads_the_right_piece():
    """At a duplicated kink abscissa the derivative belongs to the right piece.

    The value read at the duplicate returns the right value (`side="right"`);
    the paired derivative is the right piece's left-endpoint slope, so the
    (value, derivative) pair describes one consistent branch of the kink.
    """
    xp = jnp.array([0.0, 1.0, 1.0, 2.0])
    fp = jnp.array([0.0, 10.0, 2.0, 3.0])

    value, derivative = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(1.0), xp=xp, fp=fp
    )

    np.testing.assert_allclose(np.asarray(value), 2.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(derivative), 1.0, atol=1e-12)


def test_paired_read_derivative_is_zero_on_neg_inf_brackets():
    """A `-inf` endpoint pins the paired derivative to zero, matching the value.

    Infeasible (`-inf`) values read as `-inf` with exactly zero marginal, so
    the pair stays consistent and the transform space never sees `inf * 0`.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([-jnp.inf, 3.0, 5.0])

    _, derivative = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(0.5), xp=xp, fp=fp
    )

    np.testing.assert_allclose(np.asarray(derivative), 0.0, atol=1e-12)


def test_paired_read_linear_derivative_at_a_node_is_the_right_secant():
    """Without slopes, an exact-node read pairs with the right bracket's secant."""
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([1.0, 2.0, 4.0])

    _, at_interior = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(1.0), xp=xp, fp=fp
    )
    _, at_last = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(2.0), xp=xp, fp=fp
    )

    np.testing.assert_allclose(np.asarray(at_interior), 2.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(at_last), 2.0, atol=1e-12)


def test_value_read_on_a_singleton_row_is_the_constant_node_value():
    """A one-node row reads as the constant `fp[0]` at any query."""
    xp = jnp.array([0.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])

    value = interp.interp_on_padded_grid(x_query=jnp.asarray(0.25), xp=xp, fp=fp)

    np.testing.assert_allclose(np.asarray(value), 5.0, atol=1e-12)


def test_paired_read_on_a_singleton_row_is_the_node_value_with_zero_slope():
    """A one-node row's paired read is the constant pair `(fp[0], 0)`."""
    xp = jnp.array([0.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])
    fp_slopes = jnp.array([2.0, jnp.nan, jnp.nan])

    value, derivative = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(0.25), xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(np.asarray(value), 5.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(derivative), 0.0, atol=1e-12)


def test_reads_on_an_empty_row_propagate_nan():
    """An all-NaN (poisoned) row reads as NaN, never as a finite constant.

    A poisoned carry row must surface in the runtime NaN diagnostics; a read
    that converts it to a finite value would let a poisoned candidate lose a
    downstream maximum silently instead of failing loudly.
    """
    empty = jnp.full(3, jnp.nan)

    value = interp.interp_on_padded_grid(x_query=jnp.asarray(0.25), xp=empty, fp=empty)
    paired_value, paired_derivative = interp.interp_and_derivative_on_padded_grid(
        x_query=jnp.asarray(0.25), xp=empty, fp=empty, fp_slopes=empty
    )

    assert bool(jnp.isnan(value))
    assert bool(jnp.isnan(paired_value))
    assert bool(jnp.isnan(paired_derivative))
