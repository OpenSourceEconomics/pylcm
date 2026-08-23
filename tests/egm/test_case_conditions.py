"""Piecewise-affine schedule thresholds normalize onto the condition IR."""

import pytest

from _lcm.constraints.ir import Compare
from _lcm.egm.case_conditions import condition_of_breakpoint
from lcm.case_piece import AffineBreakpoint
from lcm.exceptions import ModelInitializationError


def test_a_scalar_bracket_edge_normalizes_to_a_structured_comparison():
    """A named schedule threshold becomes an inspectable condition."""
    condition = condition_of_breakpoint(
        edge=AffineBreakpoint(threshold="bracket_top", kind="continuous_kink"),
        variable="taxable_income",
    )

    assert str(condition) == "taxable_income < bracket_top"


@pytest.mark.parametrize("equality_owner", ["below", "above"])
def test_a_bracket_edge_operator_carries_its_equality_ownership(equality_owner):
    """The condition operator records which schedule segment owns equality."""
    condition = condition_of_breakpoint(
        edge=AffineBreakpoint(
            threshold="bracket_top",
            kind="continuous_kink",
            equality_owner=equality_owner,
        ),
        variable="taxable_income",
    )

    assert isinstance(condition.expression, Compare)
    assert condition.expression.admits_equality is (equality_owner == "below")


def test_an_indexed_threshold_is_refused_rather_than_normalized():
    """A table-valued threshold has no named-reference spelling."""
    with pytest.raises(ModelInitializationError, match="indexed_by"):
        condition_of_breakpoint(
            edge=AffineBreakpoint(
                threshold="bracket_table",
                kind="continuous_kink",
                indexed_by="health",
            ),
            variable="taxable_income",
        )
