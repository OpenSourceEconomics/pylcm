"""The kink-aware reader never bridges the jump, including at an on-node limit.

A grid node landing exactly on the case boundary carries the value of whichever
side owns equality there. The reader's one-sided stencils must exclude that node
from the *other* side, or the far-side limit collapses onto the owning side and
the continuation just past the boundary is wrong by the whole jump.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import _kink_aware_interp
from lcm.case_piece import EqualityOwner

_GRID = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
_LIMIT = 2.0
_JUMP = 10.0


@pytest.mark.parametrize(
    ("equality_owner", "value_at_limit"),
    [("when", _LIMIT), ("otherwise", _LIMIT + _JUMP)],
)
def test_kink_aware_interp_reads_the_owning_side_at_the_limit(
    equality_owner: EqualityOwner, value_at_limit: float
) -> None:
    """A query exactly at the boundary reads the side that owns equality."""
    values = jnp.where(
        _GRID <= _LIMIT if equality_owner == "when" else _GRID < _LIMIT,
        _GRID,
        _GRID + _JUMP,
    )
    read = _kink_aware_interp(jnp.array(_LIMIT), _GRID, values, _LIMIT, equality_owner)
    np.testing.assert_allclose(read, value_at_limit)


@pytest.mark.parametrize("equality_owner", ["when", "otherwise"])
def test_kink_aware_interp_reads_the_far_side_just_past_an_on_node_limit(
    equality_owner: EqualityOwner,
) -> None:
    """Just above an on-node boundary the reader returns the otherwise value."""
    values = jnp.where(
        _GRID <= _LIMIT if equality_owner == "when" else _GRID < _LIMIT,
        _GRID,
        _GRID + _JUMP,
    )
    query = jnp.nextafter(jnp.array(_LIMIT), jnp.array(jnp.inf))
    read = _kink_aware_interp(query, _GRID, values, _LIMIT, equality_owner)
    np.testing.assert_allclose(read, _LIMIT + _JUMP, rtol=1e-12)


def test_kink_aware_interp_interpolates_within_the_lower_branch() -> None:
    """Strictly below the boundary the reader stays on the when branch."""
    values = jnp.where(_GRID <= _LIMIT, _GRID, _GRID + _JUMP)
    read = _kink_aware_interp(jnp.array(1.5), _GRID, values, _LIMIT, "otherwise")
    np.testing.assert_allclose(read, 1.5, rtol=1e-12)


def test_kink_aware_interp_interpolates_within_the_upper_branch() -> None:
    """Strictly above the boundary the reader stays on the otherwise branch."""
    values = jnp.where(_GRID <= _LIMIT, _GRID, _GRID + _JUMP)
    read = _kink_aware_interp(jnp.array(3.5), _GRID, values, _LIMIT, "otherwise")
    np.testing.assert_allclose(read, 13.5, rtol=1e-12)
