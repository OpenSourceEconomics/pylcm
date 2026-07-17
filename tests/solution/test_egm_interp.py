"""Spec for interpolation on NaN-padded, weakly-ascending carry rows.

Contract under test — `_lcm.egm.interp.interp_on_padded_grid`:

    interp_on_padded_grid(
        *, x_query: FloatND, xp: Float1D, fp: Float1D,
        fp_slopes: Float1D | None = None,
    ) -> FloatND

`xp` is a NaN-padded, weakly ascending grid row (NaNs only in the tail).
Behavior:

- linear interpolation between neighboring non-NaN nodes,
- edge clamp outside the non-NaN range,
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
from tests.conftest import X64_ENABLED

# The germ read and its host reference rerun identical interpolation
# arithmetic, so the two sides agree to the active float precision's
# roundoff, not better. The k-th germ component divides by the bracket
# width to the k-th power, amplifying that roundoff by one power of `1/h`
# per derivative order — so the higher components carry correspondingly
# looser float32 bounds.
_GERM_ATOL = 1e-9 if X64_ENABLED else 2e-5
if X64_ENABLED:
    _GERM_COMPONENT_RTOL_ATOL = dict.fromkeys((1, 2, 3), (1e-07, 1e-09))
else:
    _GERM_COMPONENT_RTOL_ATOL = {
        1: (1e-3, 3e-4),
        2: (1e-3, 1e-2),
        3: (1e-3, 3e-1),
    }


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


def test_edge_clamp_below_and_above():
    """Queries outside the non-NaN range return the boundary values."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    x_query = jnp.array([0.0, 100.0])

    got = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=fp)

    np.testing.assert_allclose(got, jnp.array([0.0, 5.0]), atol=1e-12)


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


def test_singleton_row_clamps_to_its_node():
    """A row with one valid node reads that node's value at every query.

    The edge clamp applies on both sides of a single-node row: the value read
    (with slopes) and the slope-free marginal read alike return the node's
    value below, at, and above the node — a valid one-point carry is never
    silently replaced by zero.
    """
    xp = jnp.array([1.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])
    slopes = jnp.array([2.0, jnp.nan, jnp.nan])
    x_query = jnp.array([0.5, 1.0, 2.0])

    value = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=slopes
    )
    marginal = interp.interp_on_padded_grid(x_query=x_query, xp=xp, fp=slopes)

    np.testing.assert_array_equal(np.asarray(value), [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(np.asarray(marginal), [2.0, 2.0, 2.0])


def test_singleton_row_is_constant_under_autodiff():
    """A singleton row's constant clamp differentiates as a constant.

    `jax.grad` with respect to the query must be exactly zero (finite, not
    NaN): asset-row mode differentiates the continuation read in the Euler
    slot, so a NaN tangent leaking from the padded bracket would poison the
    published Euler marginal even though the primal value is correct.
    """
    xp = jnp.array([1.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])
    slopes = jnp.array([2.0, jnp.nan, jnp.nan])

    def read(query):
        return interp.interp_on_padded_grid(
            x_query=query, xp=xp, fp=fp, fp_slopes=slopes
        )

    value, derivative = jax.value_and_grad(read)(jnp.asarray(1.0))

    np.testing.assert_allclose(float(value), 5.0, atol=1e-12)
    np.testing.assert_allclose(float(derivative), 0.0, atol=1e-12)


def test_singleton_row_passes_its_node_values_tangent_through():
    """The singleton clamp is the node's value: unit derivative in that node.

    Differentiating the read with respect to the row's single valid value must
    give exactly one — the clamp publishes that value verbatim — with no NaN
    contamination from the padded slots.
    """
    xp = jnp.array([1.0, jnp.nan, jnp.nan])
    slopes = jnp.array([2.0, jnp.nan, jnp.nan])

    def read(node_value):
        fp = jnp.array([jnp.nan, jnp.nan, jnp.nan]).at[0].set(node_value)
        return interp.interp_on_padded_grid(
            x_query=jnp.asarray(2.0), xp=xp, fp=fp, fp_slopes=slopes
        )

    derivative = jax.grad(read)(jnp.asarray(5.0))

    np.testing.assert_allclose(float(derivative), 1.0, atol=1e-12)


def test_singleton_row_stays_constant_under_vmap_and_autodiff():
    """The zero query derivative survives per-row `vmap` batching.

    The production readers map the row read over a stacked candidate axis, so
    the degenerate-row handling must stay AD-safe inside `vmap` — a construct
    that re-evaluates both sides of a branch (as `lax.cond` lowered under
    `vmap` would) leaks the padded bracket's NaN tangent back in.
    """
    xp = jnp.array([[1.0, 2.0, 3.0], [1.0, jnp.nan, jnp.nan]])
    fp = jnp.array([[1.0, 4.0, 9.0], [5.0, jnp.nan, jnp.nan]])
    slopes = jnp.array([[2.0, 4.0, 6.0], [2.0, jnp.nan, jnp.nan]])

    def summed_read(query):
        def read_row(xp_row, fp_row, slopes_row):
            return interp.interp_on_padded_grid(
                x_query=query, xp=xp_row, fp=fp_row, fp_slopes=slopes_row
            )

        return jnp.sum(jax.vmap(read_row)(xp, fp, slopes))

    derivative = jax.grad(summed_read)(jnp.asarray(2.5))

    # The regular row holds `x²` data, which the Hermite read reproduces
    # (derivative `2q = 5`); the singleton row contributes exactly zero.
    np.testing.assert_allclose(float(derivative), 5.0, atol=1e-12)


def test_nan_query_reads_nan_on_a_singleton_row():
    """A NaN query stays NaN even where the singleton clamp is constant.

    A NaN query marks an upstream failure; the constant clamp must not convert
    it into a finite value — regular rows already propagate NaN queries through
    the bracket arithmetic, and singleton rows must fail just as loudly.
    """
    xp = jnp.array([1.0, jnp.nan, jnp.nan])
    fp = jnp.array([5.0, jnp.nan, jnp.nan])

    got = interp.interp_on_padded_grid(x_query=jnp.array([jnp.nan, 1.0]), xp=xp, fp=fp)

    assert bool(jnp.isnan(got[0]))
    np.testing.assert_allclose(float(got[1]), 5.0, atol=1e-12)


def test_empty_row_reads_nan():
    """An all-NaN (poisoned) row reads NaN at every query, never a finite value.

    An empty valid prefix only arises from an already-poisoned carry; the read
    must preserve the NaN so the runtime diagnostics surface the poison instead
    of masking it with a finite constant.
    """
    poisoned = jnp.array([jnp.nan, jnp.nan, jnp.nan])
    x_query = jnp.array([0.0, 1.0])

    linear = interp.interp_on_padded_grid(x_query=x_query, xp=poisoned, fp=poisoned)
    hermite = interp.interp_on_padded_grid(
        x_query=x_query, xp=poisoned, fp=poisoned, fp_slopes=poisoned
    )

    assert bool(jnp.isnan(linear).all())
    assert bool(jnp.isnan(hermite).all())


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


def test_slopes_keep_nan_tail_edge_clamp_and_node_values():
    """The Hermite path preserves NaN-tail handling, edge clamps, and nodes."""
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    fp_slopes = jnp.array([3.0, 2.0, 0.5, jnp.nan])
    x_query = jnp.array([0.0, 1.0, 2.0, 4.0, 100.0])

    got = interp.interp_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=fp_slopes
    )

    np.testing.assert_allclose(got, jnp.array([0.0, 0.0, 3.0, 5.0, 5.0]), atol=1e-12)


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


def _host_right_germ(
    xp: np.ndarray, fp: np.ndarray, slopes: np.ndarray, x_query: float
) -> tuple[bool, float, float, float]:
    """Scalar NumPy reference for the value read's right germ.

    Independent of the JAX implementation: plain `searchsorted` bracket lookup
    (right-continuous at nodes), the Fritsch-Carlson limiter written out
    scalar-wise, and the bracket cubic differentiated analytically — with `t`
    the relative position, `h` the width, and `c_l`, `c_u` the Hermite
    coefficients, the piece is `p(t) = f_l + Δf t + c_l t + (c_u - 2 c_l) t² +
    (c_l - c_u) t³`. Right-finite with all derivatives zero on both clamp rays
    (strictly below the first node; at or above the last).
    """
    valid = ~np.isnan(xp)
    x_valid, f_valid, s_valid = xp[valid], fp[valid], slopes[valid]
    if x_query < x_valid[0] or x_query >= x_valid[-1]:
        return True, 0.0, 0.0, 0.0
    lower = int(np.searchsorted(x_valid, x_query, side="right")) - 1
    width = x_valid[lower + 1] - x_valid[lower]
    relative_position = (x_query - x_valid[lower]) / width
    right_finite = bool(np.isfinite(f_valid[lower]) and np.isfinite(f_valid[lower + 1]))
    df = f_valid[lower + 1] - f_valid[lower]
    if not np.isfinite(df):
        return right_finite, 0.0, 0.0, 0.0
    secant = df / width
    if not (np.isfinite(s_valid[lower]) and np.isfinite(s_valid[lower + 1])):
        return right_finite, float(secant), 0.0, 0.0

    def limit(slope: float) -> float:
        if slope * secant <= 0.0:
            return 0.0
        return float(np.sign(secant) * min(abs(slope), 3.0 * abs(secant)))

    coeff_lower = width * limit(s_valid[lower]) - df
    coeff_upper = df - width * limit(s_valid[lower + 1])
    t = relative_position
    first = (
        df
        + (1.0 - 2.0 * t) * ((1.0 - t) * coeff_lower + t * coeff_upper)
        + t * (1.0 - t) * (coeff_upper - coeff_lower)
    ) / width
    second = (
        2.0 * (coeff_upper - 2.0 * coeff_lower) + 6.0 * (coeff_lower - coeff_upper) * t
    ) / width**2
    third = 6.0 * (coeff_lower - coeff_upper) / width**3
    return right_finite, float(first), float(second), float(third)


def test_right_germ_reproduces_a_cubic_interior_derivatives():
    """With exact node slopes the germ of `x**3` is `(3x², 6x, 6)`.

    A cubic with exact endpoint values and slopes is reproduced exactly by the
    Hermite bracket (four constraints determine the cubic), so all three of its
    derivative reads are exact wherever the limiter is inactive — and the read
    is right-finite.
    """
    xp = jnp.array([1.0, 1.5, 2.0, 3.0])
    fp = xp**3
    slopes = 3.0 * xp**2
    x_query = jnp.array([1.2, 1.5, 1.9, 2.4])

    right_finite, first, second, third = interp.interp_right_germ_on_padded_grid(
        x_query=x_query, xp=xp, fp=fp, fp_slopes=slopes
    )

    assert bool(right_finite.all())
    np.testing.assert_allclose(first, 3.0 * np.asarray(x_query) ** 2, rtol=1e-9)
    np.testing.assert_allclose(second, 6.0 * np.asarray(x_query), rtol=1e-9)
    np.testing.assert_allclose(third, np.full(4, 6.0), rtol=1e-9)


def test_right_germ_at_a_node_uses_the_right_brackets_limited_slope():
    """At a node the first right derivative is the right bracket's limited slope.

    The value row rises by `0.1` per bracket while the node carries a raw slope
    of `100`; the Fritsch-Carlson limiter caps the value read's slope at three
    times the secant, so the first right derivative at the node is `0.3`, not
    `100`.
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([0.9, 1.0, 1.1])
    slopes = jnp.array([0.1, 100.0, 0.1])

    _, first, _, _ = interp.interp_right_germ_on_padded_grid(
        x_query=jnp.array([1.0]), xp=xp, fp=fp, fp_slopes=slopes
    )

    np.testing.assert_allclose(float(first[0]), 0.3, atol=_GERM_ATOL)


def test_right_germ_is_flat_and_finite_on_the_clamp_rays():
    """The read clamps outside the valid range: right-finite, all derivatives zero.

    Strictly below the first node and at or above the last valid node the value
    read is constant; a query exactly on the first node is interior (its right
    bracket exists) and keeps the bracket's slope.
    """
    xp = jnp.array([1.0, 2.0, 4.0, jnp.nan])
    fp = jnp.array([0.0, 3.0, 5.0, jnp.nan])
    slopes = jnp.array([3.0, 3.0, 1.0, jnp.nan])

    right_finite, first, second, third = interp.interp_right_germ_on_padded_grid(
        x_query=jnp.array([0.5, 1.0, 4.0, 7.0]), xp=xp, fp=fp, fp_slopes=slopes
    )

    assert bool(right_finite.all())
    np.testing.assert_allclose(np.asarray(first), [0.0, 3.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(float(second[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(third[0]), 0.0, atol=1e-12)


def test_right_germ_flags_a_neg_inf_bracket_as_not_right_finite():
    """A `-inf` bracket endpoint is not right-finite; derivatives are zero.

    The read is `-inf` on the bracket's interior, so a candidate tied at a
    finite node whose right bracket dies must be distinguishable from a finite
    constant clamp — the flag carries that distinction, the derivatives stay
    zero (never NaN).
    """
    xp = jnp.array([0.0, 1.0, 2.0])
    fp = jnp.array([0.0, 1.0, -jnp.inf])
    slopes = jnp.array([1.0, 1.0, 1.0])

    right_finite, first, second, third = interp.interp_right_germ_on_padded_grid(
        x_query=jnp.array([1.0, 1.5]), xp=xp, fp=fp, fp_slopes=slopes
    )

    assert not bool(right_finite.any())
    np.testing.assert_allclose(np.asarray(first), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(second), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(third), [0.0, 0.0], atol=1e-12)


def test_right_germ_matches_scalar_host_on_random_rows():
    """The vectorized right germ equals an independent scalar reference.

    Random strictly-ascending rows with arbitrary finite values and slopes are
    read at interior points, exactly at nodes, and on both clamp rays; the JAX
    implementation must agree with the scalar NumPy host at every query, in
    every germ component.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(5):
        xp = np.cumsum(rng.uniform(0.1, 1.0, size=7))
        fp = rng.normal(size=7)
        slopes = rng.normal(scale=5.0, size=7)
        nodes = xp.tolist()
        interior = ((xp[:-1] + xp[1:]) / 2.0).tolist()
        rays = [xp[0] - 0.5, xp[-1] + 0.5]
        queries = np.array(nodes + interior + rays)

        got = interp.interp_right_germ_on_padded_grid(
            x_query=jnp.asarray(queries),
            xp=jnp.asarray(xp),
            fp=jnp.asarray(fp),
            fp_slopes=jnp.asarray(slopes),
        )

        expected = [_host_right_germ(xp, fp, slopes, q) for q in queries]
        np.testing.assert_array_equal(np.asarray(got[0]), [e[0] for e in expected])
        for component in (1, 2, 3):
            rtol, atol = _GERM_COMPONENT_RTOL_ATOL[component]
            np.testing.assert_allclose(
                np.asarray(got[component]),
                [e[component] for e in expected],
                rtol=rtol,
                atol=atol,
            )
