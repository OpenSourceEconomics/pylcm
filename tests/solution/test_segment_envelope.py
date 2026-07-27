"""Exactness of the segment-envelope construction.

The refined row must be the pointwise maximum of the candidate links: a branch
that owns only an interior subinterval has to survive, a crossing that falls
exactly on a node has to separate the two policies, and the published value and
policy must agree with the exact rational envelope everywhere — not only at the
candidate abscissae.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact
from tests.solution._mss_segment_oracle import Branch, exact_envelope

F = Fraction


def _refine(
    grid: list[float],
    policy: list[float],
    value: list[float],
    n_refined: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    out_grid, out_policy, out_value, n_kept = jax.jit(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g, policy=p, value=v, n_refined=n_refined
        )
    )(jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value))
    keep = int(n_kept)
    return (
        np.asarray(out_grid)[:keep],
        np.asarray(out_policy)[:keep],
        np.asarray(out_value)[:keep],
        keep,
    )


def _read(grid: np.ndarray, ordinate: np.ndarray, x: float) -> float:
    """Read the refined row at `x`, taking the right branch at a duplicated node."""
    return float(np.interp(x, grid, ordinate))


def _tolerance() -> float:
    """Return a value tolerance for the active precision.

    Ownership is exact in both precisions; only the interpolated level carries
    representation error, which is eps-scale.
    """
    return 1e-12 if jnp.zeros(()).dtype == jnp.float64 else 1e-6


F1_GRID = [0.0, 1.0, 0.0, 1.0, 0.6, 1.2]
F1_POLICY = [10.0, 10.0, 3.0, 3.0, 1.0, 1.0]
F1_VALUE = [1.0, 1.1, 0.84, 0.84 + 1.0 / 3.0, 0.8, 1.4]


def test_a_middle_branch_owning_an_interior_interval_is_published():
    """The round-15 F1 witness: branch B owns R in [0.686, 0.96] and must survive."""
    grid, policy, value, _ = _refine(F1_GRID, F1_POLICY, F1_VALUE)
    # Exact envelope at R=0.8 is B: value 0.84 + 0.8/3, policy 3.
    assert _read(grid, value, 0.8) == pytest.approx(0.84 + 0.8 / 3.0, abs=_tolerance())
    assert _read(grid, policy, 0.8) == pytest.approx(3.0, abs=_tolerance())


def test_an_exact_node_aligned_crossing_separates_the_two_policies():
    """The round-15 F3 witness: A and B meet exactly at R=10, B owns the right."""
    grid, policy, value, _ = _refine(
        [9.0, 10.0, 9.5, 10.5], [8.0, 8.0, 2.0, 2.0], [4.875, 5.0, 4.75, 5.25]
    )
    assert _read(grid, policy, 10.1) == pytest.approx(2.0, abs=_tolerance())
    assert _read(grid, value, 10.1) == pytest.approx(5.05, abs=_tolerance())


def test_a_strictly_dominant_branch_is_not_masked_by_a_large_value_level():
    """The round-15 F2 witness: a few-ULP gap at a large level still decides."""
    level = 1.0e12 if jnp.zeros(()).dtype == jnp.float64 else 1.0e5
    gap = 2.0 * float(jnp.spacing(jnp.asarray(level)))
    grid, policy, _value, _ = _refine(
        [0.0, 1.0, 0.0, 1.0],
        [1000.0, 1000.0, 500.0, 500.0],
        [level, level + gap, level + 2.0 * gap, level + 3.0 * gap],
    )
    assert _read(grid, policy, 0.5) == pytest.approx(500.0, abs=_tolerance())


def test_a_single_concave_branch_passes_through_unchanged():
    """With nothing to refine the envelope reproduces the candidate chain."""
    grid, policy, value, n_kept = _refine(
        [1.0, 2.0, 3.0], [0.5, 1.0, 1.5], [0.0, 1.0, 1.5]
    )
    assert n_kept == 3
    np.testing.assert_allclose(grid, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(policy, [0.5, 1.0, 1.5])
    np.testing.assert_allclose(value, [0.0, 1.0, 1.5])


def test_the_refined_row_is_weakly_ascending_and_nan_padded():
    """The published row keeps the envelope's grid order and pads the tail."""
    out_grid, _, _, n_kept = jax.jit(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g, policy=p, value=v, n_refined=64
        )
    )(jnp.asarray(F1_GRID), jnp.asarray(F1_POLICY), jnp.asarray(F1_VALUE))
    keep = int(n_kept)
    kept = np.asarray(out_grid)[:keep]
    assert np.all(np.diff(kept) >= 0)
    assert np.all(np.isnan(np.asarray(out_grid)[keep:]))


def test_dead_candidates_are_excluded_from_the_envelope():
    """NaN-padded candidates neither contribute nor bridge a branch."""
    grid, _policy, value, _ = _refine(
        [1.0, 2.0, float("nan"), 3.0],
        [0.5, 1.0, float("nan"), 1.5],
        [0.0, 1.0, float("nan"), 1.5],
    )
    assert np.all(np.isfinite(grid))
    assert _read(grid, value, 1.5) == pytest.approx(0.5, abs=_tolerance())


def _oracle_branches() -> tuple[Branch, ...]:
    """The F1 witness expressed for the exact rational oracle."""
    return (
        Branch("A", (F(0), F(1)), (F(1), F(11, 10)), (F(10), F(10))),
        Branch("B", (F(0), F(1)), (F(21, 25), F(88, 75)), (F(3), F(3))),
        Branch("C", (F(3, 5), F(6, 5)), (F(4, 5), F(7, 5)), (F(1), F(1))),
    )


@pytest.mark.parametrize(
    "query", [0.05, 0.2, 0.35, 0.5, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1.05, 1.15]
)
def test_published_value_matches_the_exact_rational_envelope(query: float):
    """The refined row agrees with the exact envelope away from the nodes too."""
    grid, _policy, value, _ = _refine(F1_GRID, F1_POLICY, F1_VALUE)
    expected, _owners = exact_envelope(_oracle_branches(), Fraction(query))
    assert _read(grid, value, query) == pytest.approx(float(expected), abs=_tolerance())


def test_refinement_is_vmap_compatible():
    """A batch of candidate rows refines with static shapes."""
    grids = jnp.array([F1_GRID, F1_GRID])
    policies = jnp.array([F1_POLICY, F1_POLICY])
    values = jnp.array([F1_VALUE, F1_VALUE])
    batched = jax.jit(
        jax.vmap(
            lambda g, p, v: refine_envelope_exact(
                endog_grid=g, policy=p, value=v, n_refined=64
            )
        )
    )
    _grid, _policy, _value, n_kept = batched(grids, policies, values)
    assert n_kept.tolist()[0] == n_kept.tolist()[1]
