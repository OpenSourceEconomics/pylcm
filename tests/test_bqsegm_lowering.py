"""Lowering a case-piece model into per-case smooth DAG variants.

The lowering reads the decorator metadata off a regime's function pool, checks
that every split output is fully covered, and produces one producer-swap map per
(predicate, side) case — the specialized smooth DAG BQSEGM runs EGM on.
"""

import pytest

from _lcm.egm.bqsegm import PieceSet, collect_bqsegm_metadata
from lcm.case_piece import boundary, case_boundary, piece
from lcm.exceptions import BQSEGMCaseError


@case_boundary(
    boundary("assets", "medicaid_asset_limit", equality="otherwise", kind="jump")
)
def medicaid_eligible(assets, medicaid_asset_limit):
    return assets < medicaid_asset_limit


@piece("oop", when=medicaid_eligible)
def oop_medicaid(medical_expense):
    return 0.1 * medical_expense


@piece("oop", otherwise=medicaid_eligible)
def oop_private(medical_expense):
    return 0.9 * medical_expense


def utility(consumption, oop):
    return consumption - oop


def _covered_pool():
    return {
        "medicaid_eligible": medicaid_eligible,
        "oop_medicaid": oop_medicaid,
        "oop_private": oop_private,
        "utility": utility,
    }


def test_collect_records_the_piece_set_and_its_boundary():
    """A fully-covered split yields one `PieceSet` and registers its boundary."""
    registry = collect_bqsegm_metadata(functions=_covered_pool())
    assert registry.piece_sets == (
        PieceSet(
            output="oop",
            predicate_name="medicaid_eligible",
            when_func="oop_medicaid",
            otherwise_func="oop_private",
        ),
    )
    assert set(registry.boundaries) == {"medicaid_eligible"}


def test_collect_rejects_a_split_missing_its_otherwise_side():
    """A `when` piece without its `otherwise` counterpart fails coverage."""
    pool = {
        "medicaid_eligible": medicaid_eligible,
        "oop_medicaid": oop_medicaid,
        "utility": utility,
    }
    with pytest.raises(BQSEGMCaseError, match="otherwise"):
        collect_bqsegm_metadata(functions=pool)


def test_collect_rejects_a_boundary_with_no_declared_surface():
    """A `case_boundary()` with no surfaces cannot guide endpoint eligibility."""

    @case_boundary()
    def empty_predicate(assets):
        return assets < 0.0

    @piece("oop", when=empty_predicate)
    def oop_a(medical_expense):
        return medical_expense

    @piece("oop", otherwise=empty_predicate)
    def oop_b(medical_expense):
        return medical_expense

    pool = {
        "empty_predicate": empty_predicate,
        "oop_a": oop_a,
        "oop_b": oop_b,
    }
    with pytest.raises(BQSEGMCaseError, match="surface"):
        collect_bqsegm_metadata(functions=pool)


def test_collect_rejects_a_piece_referencing_an_undeclared_boundary():
    """A piece whose predicate is not a `case_boundary` cannot be lowered."""

    def not_a_boundary(assets):
        return assets < 0.0

    @piece("oop", when=not_a_boundary)
    def oop_a(medical_expense):
        return medical_expense

    @piece("oop", otherwise=not_a_boundary)
    def oop_b(medical_expense):
        return medical_expense

    pool = {"oop_a": oop_a, "oop_b": oop_b}
    with pytest.raises(BQSEGMCaseError, match="case_boundary"):
        collect_bqsegm_metadata(functions=pool)
