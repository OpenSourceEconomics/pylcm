"""A handover whose crossing is exactly representable lands on that state.

Two links crossing at an abscissa the float format can hold exactly is the
ordinary case, not a corner: EGM lines routinely meet at a node. The state the
crossing lands on belongs to the incoming link under the right-continuous
convention, so the refined row must publish the duplicate *there* — one state
later is a different policy, and its marginal is what the parent's Euler
inversion reads.

The witness is a shifted-support float32 pair on a large common value level,
where the located root's residual is smaller than the residual's own error
bound. A repair that reads such a residual as a side decision moves the
handover one float right.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact

_GRID = np.array([-1.3316007, -0.003239952, -0.53161657, 0.48618442], np.float32)
_POLICY = np.array([1.545463, 1.545463, 0.19281395, 0.19281395], np.float32)
_VALUE = np.array(
    [-47_838_256.0, -47_838_256.0, -47_838_260.0, -47_838_252.0], np.float32
)


def _as_fraction(value: np.floating) -> Fraction:
    return Fraction(float(value))


def _exact_crossing(grid: np.ndarray, value: np.ndarray) -> Fraction:
    """Return the crossing of the two stored lines in exact rational arithmetic."""

    def line(i: int, j: int) -> tuple[Fraction, Fraction]:
        slope = (_as_fraction(value[j]) - _as_fraction(value[i])) / (
            _as_fraction(grid[j]) - _as_fraction(grid[i])
        )
        return slope, _as_fraction(value[i]) - slope * _as_fraction(grid[i])

    slope_a, intercept_a = line(0, 1)
    slope_b, intercept_b = line(2, 3)
    return (intercept_a - intercept_b) / (slope_b - slope_a)


def _refine(grid: np.ndarray, policy: np.ndarray, value: np.ndarray):
    refined = jax.jit(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g, policy=p, value=v, n_refined=12, max_runs=4
        )
    )(jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value))
    out_grid, out_policy, _out_value, n_kept = refined
    kept = int(n_kept)
    return np.asarray(out_grid[:kept]), np.asarray(out_policy[:kept])


def _read_row(grid: np.ndarray, payload: np.ndarray, query: float) -> float:
    """Read a refined row the way a consumer does — linearly, right-continuously."""
    upper = int(np.clip(np.searchsorted(grid, query, side="right"), 1, len(grid) - 1))
    lower = upper - 1
    width = grid[upper] - grid[lower]
    if width == 0:
        return float(payload[upper])
    weight = (query - grid[lower]) / width
    return float((1 - weight) * payload[lower] + weight * payload[upper])


def test_the_handover_lands_on_the_exactly_representable_crossing():
    """The published duplicate is the crossing itself, not its successor."""
    expected = np.float32(float(_exact_crossing(_GRID, _VALUE)))
    grid, _policy = _refine(_GRID, _POLICY, _VALUE)
    duplicate = np.flatnonzero(grid[1:] == grid[:-1])
    assert len(duplicate) == 1
    assert grid[duplicate[0]] == expected


def test_the_incoming_policy_is_published_at_the_crossing():
    """At the handover state the row reads the incoming link's policy."""
    expected = np.float32(float(_exact_crossing(_GRID, _VALUE)))
    grid, policy = _refine(_GRID, _POLICY, _VALUE)
    assert _read_row(grid, policy, float(expected)) == float(_POLICY[2])


@pytest.mark.parametrize("exponent", range(-30, 31))
def test_the_exact_tie_survives_a_power_of_two_rescaling(exponent: int):
    """Scaling the resource axis by a power of two moves the event, not its identity.

    A power of two is exact in binary floating point, so the rescaled problem has
    a rescaled exact crossing and the handover must still land on it.
    """
    scale = np.float32(2.0**exponent)
    grid = (_GRID * scale).astype(np.float32)
    if not np.all(np.isfinite(grid)) or len(np.unique(grid)) != len(grid):
        pytest.skip("the rescaled grid is not representable as four distinct states")
    expected = np.float32(float(_exact_crossing(grid, _VALUE)))
    refined_grid, _policy = _refine(grid, _POLICY, _VALUE)
    duplicate = np.flatnonzero(refined_grid[1:] == refined_grid[:-1])
    assert len(duplicate) == 1
    assert refined_grid[duplicate[0]] == expected
