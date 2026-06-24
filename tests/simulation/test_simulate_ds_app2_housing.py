"""DS-2026 App.2 housing keeps the next-housing choice inside the grid bounds.

The NEGM solve searches the next-housing choice on an outer grid floored at a
small positive stock and capped at the top housing level, but the forward
simulation re-optimises over the symmetric `housing_investment` action grid
`[-housing_max, housing_max]`. A delta that drives
`next_housing = housing + housing_investment` below the floor lands where the CES
service flow `H^{1-gamma_H}` is NaN, and a delta above the cap extrapolates off
the solved outer grid; the budget feasibility mask catches neither. The
`housing_stays_in_bounds` constraint masks both out-of-range deltas, mirroring
the solve's floored, capped outer grid, so the simulated policy never steps onto
an out-of-bounds house.
"""

import jax.numpy as jnp

from lcm import LinSpacedGrid
from tests.test_models import ds_app2_housing as m


def _bounds_constraint():
    """Build the model and return its housing-bounds constraint plus the bounds."""
    model = m.build_model(n_grid=8, n_periods=3, n_consumption=60)
    working = model.user_regimes["working"]
    constraint = working.constraints["housing_stays_in_bounds"]
    housing_state = working.states["housing"]
    assert isinstance(housing_state, LinSpacedGrid)
    return constraint, housing_state.start, housing_state.stop


def test_next_housing_below_the_floor_is_infeasible():
    """A next house below `housing_min` is masked out.

    The symmetric investment grid can drive `next_housing` below the floor, into
    the NaN region of the CES service flow; the constraint rejects it.
    """
    constraint, housing_min, _ = _bounds_constraint()
    below_floor = jnp.asarray(housing_min - 1.0)
    assert bool(constraint(next_housing=below_floor)) is False


def test_next_housing_above_the_cap_is_infeasible():
    """A next house above `housing_max` is masked out.

    The symmetric investment grid can drive `next_housing` above the top outer
    node, off the solved grid; the constraint rejects it.
    """
    constraint, _, housing_max = _bounds_constraint()
    above_cap = jnp.asarray(housing_max + 1.0)
    assert bool(constraint(next_housing=above_cap)) is False


def test_next_housing_inside_the_grid_is_feasible():
    """A next house strictly inside `[housing_min, housing_max]` is feasible."""
    constraint, housing_min, housing_max = _bounds_constraint()
    interior = jnp.asarray(0.5 * (housing_min + housing_max))
    assert bool(constraint(next_housing=interior)) is True
