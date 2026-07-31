"""`assert_agrees_to_ulp` compares non-finite entries exactly.

A ULP distance is defined only between finite numbers — the spacing at infinity
is NaN, so every comparison against it is false. The helper therefore settles the
non-finite entries by exact equality first, and only measures ULP distance where
both sides are finite.
"""

import numpy as np
import pytest

from tests.conftest import assert_agrees_to_ulp


def test_a_finite_value_does_not_agree_with_an_infinity():
    """Infinity where a finite value is expected is never within tolerance."""
    with pytest.raises(AssertionError):
        assert_agrees_to_ulp(np.array([np.inf]), np.array([1.0]), n_ulp=16)


def test_opposite_sign_infinities_do_not_agree():
    """`+inf` and `-inf` are different answers, not neighbouring floats."""
    with pytest.raises(AssertionError):
        assert_agrees_to_ulp(np.array([np.inf]), np.array([-np.inf]), n_ulp=16)


def test_a_nan_does_not_agree_with_a_finite_value():
    """A NaN where a finite value is expected is a failure, not a rounding gap."""
    with pytest.raises(AssertionError):
        assert_agrees_to_ulp(np.array([np.nan]), np.array([1.0]), n_ulp=16)


def test_matching_infinities_agree():
    """Two identical infinities are the same answer."""
    assert_agrees_to_ulp(
        np.array([np.inf, -np.inf]), np.array([np.inf, -np.inf]), n_ulp=0
    )


def test_finite_entries_are_still_compared_in_ulp():
    """A one-ULP gap between finite entries passes at a bound of one ULP."""
    expected = np.array([1.0])
    assert_agrees_to_ulp(np.nextafter(expected, np.inf), expected, n_ulp=1)
