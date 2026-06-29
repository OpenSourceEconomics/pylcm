"""Breakpoint IR: the mixed-kind solver representation of a case boundary.

BQSEGM merges every institutional boundary on the liquid axis into one sorted
interval partition. The first step is a uniform representation: each declared
case-boundary surface (a jump, a continuous kink, a hard constraint) becomes a
`BreakpointSource` carrying the monotone boundary variable, its threshold, the
discontinuity kind, and the side that owns the exact boundary point — regardless
of which `@lcm.case`/`@lcm.piece` form the author used.
"""

import lcm
from _lcm.egm.bqsegm import collect_bqsegm_metadata
from _lcm.egm.bqsegm_breakpoints import (
    BreakpointSource,
    breakpoint_sources_from_registry,
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
