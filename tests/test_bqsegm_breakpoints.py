"""Breakpoint IR: the mixed-kind solver representation of a case boundary.

BQSEGM merges every institutional boundary on the liquid axis into one sorted
interval partition. The first step is a uniform representation: each declared
case-boundary surface (a jump, a continuous kink, a hard constraint) becomes a
`BreakpointSource` carrying the monotone boundary variable, its threshold, the
discontinuity kind, and the side that owns the exact boundary point — regardless
of which `@lcm.case`/`@lcm.piece` form the author used.
"""

import jax.numpy as jnp
import numpy as np

import lcm
from _lcm.egm.bqsegm import collect_bqsegm_metadata
from _lcm.egm.bqsegm_breakpoints import (
    BreakpointSource,
    affine_coefficients,
    breakpoint_sources_from_registry,
    interval_index,
    interval_midpoints,
    interval_segment_coefficients,
    linear_asset_preimage,
    n_intervals,
)


def _medicaid_pool():
    """A one-boundary Medicaid asset-test pool with both subsidy pieces."""

    @lcm.case_boundary(
        lcm.boundary(
            "assets", "medicaid_asset_limit", equality="otherwise", kind="jump"
        )
    )
    def medicaid_eligible(assets, medicaid_asset_limit):
        return assets < medicaid_asset_limit

    @lcm.piece("subsidy", when=medicaid_eligible)
    def subsidy_medicaid(subsidy_amount):
        return subsidy_amount

    @lcm.piece("subsidy", otherwise=medicaid_eligible)
    def subsidy_none(subsidy_amount):
        return 0.0 * subsidy_amount

    return {
        "medicaid_eligible": medicaid_eligible,
        "subsidy_medicaid": subsidy_medicaid,
        "subsidy_none": subsidy_none,
    }


def test_breakpoint_source_records_the_boundary_variable_and_threshold():
    """A breakpoint source names the monotone variable and its threshold."""
    registry = collect_bqsegm_metadata(functions=_medicaid_pool())
    sources = breakpoint_sources_from_registry(registry)
    assert len(sources) == 1
    assert sources[0].variable == "assets"
    assert sources[0].threshold == "medicaid_asset_limit"


def test_breakpoint_sources_lift_a_piecewise_affine_schedule():
    """A declared multi-bracket schedule contributes one breakpoint per threshold."""

    @lcm.piecewise_affine(
        "tax",
        variable="capital_income",
        breakpoints=(
            lcm.affine_breakpoint("bracket_low", kind="continuous_kink"),
            lcm.affine_breakpoint("bracket_high", kind="continuous_kink"),
        ),
    )
    def tax_schedule(capital_income, rate):
        return rate * capital_income

    registry = collect_bqsegm_metadata(functions={"tax": tax_schedule})
    sources = breakpoint_sources_from_registry(registry)
    assert [(s.variable, s.threshold, s.kind) for s in sources] == [
        ("capital_income", "bracket_low", "continuous_kink"),
        ("capital_income", "bracket_high", "continuous_kink"),
    ]


def test_breakpoint_source_records_the_kind_and_equality_owner():
    """A breakpoint source carries the discontinuity kind and equality owner."""
    registry = collect_bqsegm_metadata(functions=_medicaid_pool())
    source = breakpoint_sources_from_registry(registry)[0]
    assert source.kind == "jump"
    assert source.equality_owner == "otherwise"


def test_breakpoint_source_admits_an_open_boundary_kind():
    """The IR represents a solver-synthesized open one-sided limit breakpoint."""
    source = BreakpointSource(
        variable="savings",
        threshold="asset_limit",
        kind="open_boundary",
        equality_owner="when",
    )
    assert source.kind == "open_boundary"


def test_affine_coefficients_recover_slope_and_intercept():
    """An affine boundary variable's slope and intercept are recovered exactly."""

    def gross_income(assets):
        return 0.04 * assets + 12_000.0

    slope, intercept = affine_coefficients(gross_income)
    np.testing.assert_allclose(slope, 0.04, atol=1e-9)
    np.testing.assert_allclose(intercept, 12_000.0, atol=1e-6)


def test_linear_asset_preimage_inverts_an_increasing_boundary_variable():
    """A threshold in an increasing affine variable maps to its asset value."""

    def gross_income(assets):
        return 0.04 * assets + 12_000.0

    asset_value = linear_asset_preimage(gross_income, threshold=jnp.asarray(20_000.0))
    np.testing.assert_allclose(asset_value, 200_000.0, atol=1e-3)


def test_linear_asset_preimage_inverts_a_decreasing_boundary_variable():
    """A threshold in a decreasing affine variable maps to its asset value."""

    def countable(assets):
        return -0.5 * assets + 3_000.0

    asset_value = linear_asset_preimage(countable, threshold=jnp.asarray(1_000.0))
    np.testing.assert_allclose(asset_value, 4_000.0, atol=1e-6)


def test_linear_asset_preimage_traces_a_runtime_threshold_and_slope():
    """The preimage is a traced quantity in the runtime threshold and slope."""

    def make_gross_income(rate_of_return):
        def gross_income(assets):
            return rate_of_return * assets + 12_000.0

        return gross_income

    def preimage(rate_of_return, threshold):
        return linear_asset_preimage(
            make_gross_income(rate_of_return), threshold=threshold
        )

    jitted = jnp.vectorize(preimage)
    # z = 0.05 a + 12000, threshold 17000 -> a = 5000 / 0.05 = 100000.
    np.testing.assert_allclose(
        jitted(jnp.asarray(0.05), jnp.asarray(17_000.0)), 100_000.0, atol=1e-3
    )


def test_n_intervals_is_one_more_than_the_breakpoint_count():
    """Merging N breakpoints on the asset axis yields N+1 ordered intervals."""
    assert n_intervals(n_breakpoints=3) == 4
    assert n_intervals(n_breakpoints=0) == 1


def test_interval_index_sorts_unordered_breakpoints():
    """Each liquid grid point lands in the interval bounded by sorted breakpoints."""
    breakpoints = jnp.asarray([200_000.0, 50_000.0])
    grid = jnp.asarray([0.0, 60_000.0, 250_000.0])
    np.testing.assert_array_equal(
        np.asarray(interval_index(liquid_grid=grid, breakpoints=breakpoints)),
        np.asarray([0, 1, 2]),
    )


def test_interval_index_assigns_a_point_on_a_breakpoint_to_the_upper_interval():
    """A liquid point exactly on a breakpoint joins the interval above it."""
    breakpoints = jnp.asarray([50_000.0, 200_000.0])
    grid = jnp.asarray([50_000.0, 200_000.0])
    np.testing.assert_array_equal(
        np.asarray(interval_index(liquid_grid=grid, breakpoints=breakpoints)),
        np.asarray([1, 2]),
    )


def test_interval_index_with_no_breakpoints_is_a_single_interval():
    """With no breakpoints every liquid point shares interval zero."""
    grid = jnp.asarray([0.0, 1_000.0, 500_000.0])
    np.testing.assert_array_equal(
        np.asarray(interval_index(liquid_grid=grid, breakpoints=jnp.zeros((0,)))),
        np.asarray([0, 0, 0]),
    )


def test_interval_midpoints_pick_a_representative_point_per_interval():
    """Each interval gets an interior point between its bounding breakpoints."""
    grid = jnp.asarray([0.0, 30.0])
    breakpoints = jnp.asarray([8.0, 20.0])
    np.testing.assert_allclose(
        np.asarray(interval_midpoints(liquid_grid=grid, breakpoints=breakpoints)),
        np.asarray([4.0, 14.0, 25.0]),
        atol=1e-6,
    )


def test_interval_segment_coefficients_recover_the_active_affine_segment():
    """Per interval, a piecewise-affine schedule's active slope and intercept hold."""

    def tax(liquid):
        return (
            0.1 * jnp.minimum(liquid, 8.0)
            + 0.2 * jnp.clip(liquid - 8.0, 0.0, 12.0)
            + 0.3 * jnp.maximum(liquid - 20.0, 0.0)
        )

    midpoints = jnp.asarray([4.0, 14.0, 25.0])
    slopes, intercepts = interval_segment_coefficients(
        schedule=tax, interval_midpoints=midpoints
    )
    np.testing.assert_allclose(np.asarray(slopes), [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(np.asarray(intercepts), [0.0, -0.8, -2.8], atol=1e-5)
