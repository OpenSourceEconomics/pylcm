"""Knot interpolation refuses to extrapolate past the knots it was given.

The calibration's age-keyed knots pin the spline over one age range. A period
outside that range has no knot on one side, so the cubic continues on its
leading term alone and the value it reports is an artifact of the fit rather
than a calibrated number. Feeding that into the model's disutility silently
changes what the model says.

Periods the caller has explicitly declared flat are not extrapolation: they
take the last knot's value by construction.
"""

import numpy as np
import pytest

from lcm_examples.mahler_yum_2024 import _age_keys_to_periods, _interpolate_knots

_KNOTS = {"25": 0.0, "35": 1.0, "45": 1.5, "55": 1.2}
_KNOT_PERIODS, _ = _age_keys_to_periods(age_keyed_dict=_KNOTS)
_FIRST, _LAST = int(_KNOT_PERIODS[0]), int(_KNOT_PERIODS[-1])


def test_periods_inside_the_knot_range_interpolate():
    """A period range covered by the knots returns finite interpolated values."""
    period_range = np.arange(_FIRST, _LAST + 1)

    values = _interpolate_knots(
        age_keyed_dict=_KNOTS, period_range=period_range, flat_after=None
    )

    assert np.isfinite(values).all()
    assert len(values) == len(period_range)


def test_a_period_past_the_last_knot_is_rejected():
    """Querying beyond the last knot raises instead of extrapolating."""
    with pytest.raises(ValueError, match="outside the knot range"):
        _interpolate_knots(
            age_keyed_dict=_KNOTS,
            period_range=np.arange(_FIRST, _LAST + 5),
            flat_after=None,
        )


def test_a_period_before_the_first_knot_is_rejected():
    """Querying below the first knot raises instead of extrapolating."""
    with pytest.raises(ValueError, match="outside the knot range"):
        _interpolate_knots(
            age_keyed_dict=_KNOTS,
            period_range=np.arange(_FIRST - 3, _LAST + 1),
            flat_after=None,
        )


def test_periods_declared_flat_are_not_extrapolation():
    """`flat_after` periods take the last knot's value and are accepted."""
    values = _interpolate_knots(
        age_keyed_dict=_KNOTS,
        period_range=np.arange(_FIRST, _LAST + 5),
        flat_after=_LAST,
    )

    np.testing.assert_allclose(values[_LAST - _FIRST :], _KNOTS["55"])
