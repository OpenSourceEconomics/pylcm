"""Declaration errors surface as `NBEGMCaseError`, with the offending name.

Every declaration failure a user can cause belongs to pylcm's exception
hierarchy, so `except PyLCMError` catches it, and names what was declared.
"""

import pytest

import lcm
from lcm.exceptions import NBEGMCaseError, PyLCMError

predicate = lcm.case_boundary(
    condition=lcm.ref("liquid") < lcm.ref("limit"), kind="jump"
)


def test_a_piece_naming_neither_side_is_an_lcm_error() -> None:
    """`lcm.piece` with no `when=`/`otherwise=` raises inside the hierarchy."""
    with pytest.raises(PyLCMError, match=r"'subsidy'.*exactly one"):
        lcm.piece(output="subsidy")


def test_a_piece_naming_both_sides_is_rejected() -> None:
    """`lcm.piece` with both `when=` and `otherwise=` is ambiguous."""
    with pytest.raises(NBEGMCaseError, match="exactly one"):
        lcm.piece(output="subsidy", when=predicate, otherwise=predicate)


def test_an_equality_condition_cannot_declare_a_binary_case_split() -> None:
    """A case boundary must order the liquid coordinate around one threshold."""
    with pytest.raises(NBEGMCaseError, match="exactly one"):
        lcm.case_boundary(condition=lcm.ref("liquid") == lcm.ref("limit"), kind="jump")


def test_a_multi_dotted_breakpoint_threshold_is_rejected() -> None:
    """A `MappingLeaf` holds a flat `.data`, so `leaf.subkey` is the deepest name."""
    with pytest.raises(NBEGMCaseError, match="more than one dot"):
        lcm.affine_breakpoint(threshold="schedule.bracket.upper")


def test_an_affine_breakpoint_records_which_coordinate_side_owns_equality() -> None:
    """Schedule equality ownership is explicit in the schedule coordinate."""
    declared = lcm.affine_breakpoint(threshold="limit", equality="below")
    assert declared.equality_owner == "below"


def test_a_single_dotted_breakpoint_threshold_splits_leaf_and_subkey() -> None:
    """`leaf.subkey` reads `subkey` out of the `MappingLeaf` param `leaf`."""
    declared = lcm.affine_breakpoint(threshold="schedule.upper")
    assert (declared.threshold, declared.threshold_subkey) == ("schedule", "upper")
