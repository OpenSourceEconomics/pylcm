"""Adaptive outer interpolation keeps each inner action's conditional profile."""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    SafeguardedSearchResult,
    safeguarded_continuous_argmax,
)
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION


def _search(*, profiles: FloatND) -> SafeguardedSearchResult:
    """Optimize every trailing conditional cell on the same four exact nodes."""
    nodes = jnp.asarray((0.0, 1.0, 2.0, 3.0))
    interpolant = LocalCubicOuterInterpolant()
    return safeguarded_continuous_argmax(
        objective=lambda query: interpolant.evaluate(
            nodes=nodes, values=profiles, query=query
        ),
        nodes=nodes,
        node_values=profiles,
        golden_iterations=40,
    )


def _profiles() -> FloatND:
    """Two quadratic action profiles have different continuous outer maxima."""
    return jnp.asarray(((0.0, 2.0), (2.0, 0.0), (2.0, 0.0), (0.0, 2.0)))


def test_each_action_profile_reaches_its_own_continuous_maximum() -> None:
    """An interior action peak exceeds the interpolated meshwise action maximum."""
    profiles = _profiles()
    np.testing.assert_array_equal(jnp.max(profiles, axis=1), (2.0,) * 4)
    result = _search(profiles=profiles)
    np.testing.assert_array_almost_equal(
        result.value, (2.25, 2.0), decimal=DECIMAL_PRECISION
    )
    # Smooth optimizer abscissae have square-root-epsilon error, as in the
    # established continuous outer-carry conditional-marginal controls.
    eps = float(np.finfo(np.asarray(profiles).dtype).eps)
    np.testing.assert_allclose(
        result.x[0], 1.5, rtol=0.0, atol=max(1e-4, 16.0 * eps**0.5)
    )
    assert int(jnp.argmax(result.value)) == 0
    assert int(jnp.argmax(profiles[0])) == 1
    early = _search(profiles=jnp.max(profiles, axis=1))
    np.testing.assert_array_almost_equal(early.value, 2.0, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_almost_equal(
        jnp.max(result.value) - early.value, 0.25, decimal=DECIMAL_PRECISION
    )
