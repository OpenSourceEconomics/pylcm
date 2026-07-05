"""Behavior of the NBEGM case-boundary and formula-piece decorators.

The decorators only attach metadata to a user's DAG function and return the
same object unchanged; they never wrap or alter runtime behavior.
"""

import pytest

import lcm
from lcm.case_piece import (
    AffineBreakpoint,
    BoundarySurface,
    CaseBoundaryMeta,
    PieceMeta,
    PiecewiseAffineMeta,
    affine_breakpoint,
    boundary,
    case_boundary,
    piece,
    piecewise_affine,
)
from lcm.exceptions import NBEGMCaseError


def medicaid_eligible(assets, medicaid_asset_limit):
    return assets < medicaid_asset_limit


def test_boundary_captures_variable_threshold_equality_and_kind():
    """`boundary(...)` records the equality surface, its owner, and its kind."""
    surface = boundary(
        "assets", "medicaid_asset_limit", equality="otherwise", kind="jump"
    )
    assert surface == BoundarySurface(
        variable="assets",
        threshold="medicaid_asset_limit",
        equality_owner="otherwise",
        kind="jump",
    )


def test_case_boundary_attaches_one_boundary_surface():
    """`case_boundary` stores each declared surface in `__lcm_case_boundary__`."""

    @case_boundary(
        boundary("assets", "medicaid_asset_limit", equality="otherwise", kind="jump")
    )
    def predicate(assets, medicaid_asset_limit):
        return assets < medicaid_asset_limit

    assert predicate.__lcm_case_boundary__ == CaseBoundaryMeta(  # ty: ignore[unresolved-attribute]
        boundaries=(
            BoundarySurface(
                variable="assets",
                threshold="medicaid_asset_limit",
                equality_owner="otherwise",
                kind="jump",
            ),
        )
    )


def test_case_boundary_returns_the_same_function_object():
    """The decorator returns the original function, never a wrapper (claw-safe)."""

    def predicate(assets, medicaid_asset_limit):
        return assets < medicaid_asset_limit

    decorated = case_boundary(
        boundary("assets", "medicaid_asset_limit", equality="otherwise", kind="jump")
    )(predicate)
    assert decorated is predicate


def test_case_boundary_rejects_bare_two_tuple_without_explicit_ownership():
    """A bare `(variable, threshold)` tuple cannot declare equality ownership."""
    with pytest.raises(NBEGMCaseError, match=r"lcm\.boundary"):
        case_boundary(("assets", "medicaid_asset_limit"))


def test_piece_records_output_predicate_and_when_side():
    """`piece(output, when=pred)` labels the `when` branch of the split."""

    @piece("oop", when=medicaid_eligible)
    def oop_medicaid(medical_expense):
        return medical_expense

    assert oop_medicaid.__lcm_piece__ == PieceMeta(  # ty: ignore[unresolved-attribute]
        output="oop", predicate_name="medicaid_eligible", side="when"
    )


def test_piece_records_otherwise_side():
    """`piece(output, otherwise=pred)` labels the complementary branch."""

    @piece("oop", otherwise=medicaid_eligible)
    def oop_private(medical_expense):
        return medical_expense

    assert oop_private.__lcm_piece__ == PieceMeta(  # ty: ignore[unresolved-attribute]
        output="oop", predicate_name="medicaid_eligible", side="otherwise"
    )


def test_piece_returns_the_same_function_object():
    """The piece decorator returns the original function, never a wrapper."""

    def oop_medicaid(medical_expense):
        return medical_expense

    decorated = piece("oop", when=medicaid_eligible)(oop_medicaid)
    assert decorated is oop_medicaid


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"when": medicaid_eligible, "otherwise": medicaid_eligible},
    ],
)
def test_piece_requires_exactly_one_side(kwargs):
    """Neither or both of `when=`/`otherwise=` is rejected — exactly one is required."""
    with pytest.raises(ValueError, match="exactly one"):
        piece("oop", **kwargs)


def test_affine_breakpoint_captures_threshold_and_kind():
    """`affine_breakpoint(...)` records one threshold of a schedule and its kind."""
    bracket = affine_breakpoint("bracket_top", kind="continuous_kink")
    assert bracket == AffineBreakpoint(threshold="bracket_top", kind="continuous_kink")


def test_piecewise_affine_attaches_schedule_metadata():
    """`piecewise_affine` records the output, monotone variable, and breakpoints."""

    @piecewise_affine(
        "tax",
        variable="capital_income",
        breakpoints=(
            affine_breakpoint("bracket_low", kind="continuous_kink"),
            affine_breakpoint("bracket_high", kind="continuous_kink"),
        ),
    )
    def tax_schedule(capital_income, rate):
        return rate * capital_income

    assert tax_schedule.__lcm_piecewise_affine__ == PiecewiseAffineMeta(  # ty: ignore[unresolved-attribute]
        output="tax",
        variable="capital_income",
        breakpoints=(
            AffineBreakpoint(threshold="bracket_low", kind="continuous_kink"),
            AffineBreakpoint(threshold="bracket_high", kind="continuous_kink"),
        ),
    )


def test_piecewise_affine_returns_the_same_function_object():
    """The schedule decorator returns the original function, never a wrapper."""

    def tax_schedule(capital_income, rate):
        return rate * capital_income

    decorated = piecewise_affine(
        "tax",
        variable="capital_income",
        breakpoints=(affine_breakpoint("bracket_low", kind="continuous_kink"),),
    )(tax_schedule)
    assert decorated is tax_schedule


def test_decorators_are_reexported_from_lcm():
    """The user-facing entry points live on the top-level `lcm` namespace."""
    assert lcm.case_boundary is case_boundary
    assert lcm.piece is piece
    assert lcm.boundary is boundary
    assert lcm.piecewise_affine is piecewise_affine
    assert lcm.affine_breakpoint is affine_breakpoint
