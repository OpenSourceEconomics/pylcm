"""Per-period state grids equal the identity value function's coordinates."""

import numpy as np
import pytest

from tests.simulation.test_state_grid_period_axes import _model


@pytest.mark.parametrize("n_points", [2, 3, 5])
@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("shift", [0.0, 0.5])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_period_grid_equals_identity_value_coordinates(
    *, n_points, scale, shift, enable_jit
):
    model = _model(n_points=n_points, scale=scale, shift=shift, enable_jit=enable_jit)
    values = model.solve(params={}, log_level="off").values
    # Repeat/reorder queries on the same model; the representative must not change.
    for period in (1, 0, 1, 0):
        factor = 1.0 if period == 0 else scale
        expected = np.linspace(shift + factor, shift + 2.0 * factor, n_points)
        got = model.state_grid(
            params={}, regime_name="end", state_name="wealth", period=period
        )
        np.testing.assert_array_equal(np.asarray(got), expected)
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(values[period]["end"])
        )
