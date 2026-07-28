"""Declaration errors surface as `NBEGMCaseError`, with the offending name.

Every declaration failure a user can cause belongs to pylcm's exception
hierarchy, so `except PyLCMError` catches it, and names what was declared.
"""

import pytest

import lcm
from lcm.exceptions import NBEGMCaseError, PyLCMError


def predicate(liquid, limit):
    """Boundary predicate for the toy split."""
    return liquid < limit


def test_a_piece_naming_neither_side_is_an_lcm_error() -> None:
    """`lcm.piece` with no `when=`/`otherwise=` raises inside the hierarchy."""
    with pytest.raises(PyLCMError, match=r"'subsidy'.*exactly one"):
        lcm.piece("subsidy")


def test_a_piece_naming_both_sides_is_rejected() -> None:
    """`lcm.piece` with both `when=` and `otherwise=` is ambiguous."""
    with pytest.raises(NBEGMCaseError, match="exactly one"):
        lcm.piece("subsidy", when=predicate, otherwise=predicate)


def test_a_piece_keyed_by_a_lambda_predicate_is_rejected() -> None:
    """Pieces are keyed by the predicate's name, so lambdas would all collide."""
    with pytest.raises(NBEGMCaseError, match="lambda"):
        lcm.piece("subsidy", when=lambda liquid, limit: liquid < limit)


def test_a_multi_dotted_breakpoint_threshold_is_rejected() -> None:
    """A `MappingLeaf` holds a flat `.data`, so `leaf.subkey` is the deepest name."""
    with pytest.raises(NBEGMCaseError, match="more than one dot"):
        lcm.affine_breakpoint("schedule.bracket.upper")


def test_a_single_dotted_breakpoint_threshold_splits_leaf_and_subkey() -> None:
    """`leaf.subkey` reads `subkey` out of the `MappingLeaf` param `leaf`."""
    declared = lcm.affine_breakpoint("schedule.upper")
    assert (declared.threshold, declared.threshold_subkey) == ("schedule", "upper")
