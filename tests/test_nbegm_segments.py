"""Masking and segmentation of a NBEGM case's EGM candidate path.

An invalid candidate is NaN-dead in *every* channel before the envelope, never a
finite abscissa with `-inf` value (which the query envelope would treat as a live
link). A case's candidate path is then split into maximal ascending, hole-free
subsegments so the upper envelope never bridges a fold or a masked gap.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_segments import mask_dead_candidates, segment_ids_from_folds


def test_mask_dead_sets_every_channel_to_nan():
    """An invalid candidate is NaN in grid, value, policy, and marginal."""
    endog_grid = jnp.array([0.0, 1.0, 2.0])
    value = jnp.array([10.0, 11.0, 12.0])
    policy = jnp.array([0.5, 0.6, 0.7])
    marginal = jnp.array([1.0, 1.0, 1.0])
    valid = jnp.array([True, False, True])

    masked = mask_dead_candidates(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        marginal=marginal,
        valid=valid,
    )
    for channel in masked:
        assert bool(jnp.isnan(channel[1]))
    np.testing.assert_allclose(np.asarray(masked[0]), [0.0, np.nan, 2.0])
    np.testing.assert_allclose(np.asarray(masked[1]), [10.0, np.nan, 12.0])


def test_mask_dead_leaves_valid_candidates_untouched():
    """Valid candidates keep their exact values; no `-inf` is introduced."""
    endog_grid = jnp.array([0.0, 1.0])
    value = jnp.array([10.0, 11.0])
    policy = jnp.array([0.5, 0.6])
    marginal = jnp.array([1.0, 2.0])
    valid = jnp.array([True, True])

    grid, val, pol, marg = mask_dead_candidates(
        endog_grid=endog_grid,
        value=value,
        policy=policy,
        marginal=marginal,
        valid=valid,
    )
    np.testing.assert_allclose(np.asarray(grid), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(val), [10.0, 11.0])
    np.testing.assert_allclose(np.asarray(pol), [0.5, 0.6])
    np.testing.assert_allclose(np.asarray(marg), [1.0, 2.0])


def test_monotone_path_is_a_single_segment():
    """A strictly ascending candidate path carries one segment id."""
    segment_id = segment_ids_from_folds(endog_grid=jnp.array([0.0, 1.0, 2.0, 3.0]))
    assert len(np.unique(np.asarray(segment_id))) == 1


def test_fold_starts_a_new_segment():
    """A fold (the endogenous grid stops ascending) starts a new segment."""
    # The grid ascends to 2.0 then folds back to 1.5: index 3 opens a new branch.
    segment_id = segment_ids_from_folds(endog_grid=jnp.array([0.0, 1.0, 2.0, 1.5, 2.5]))
    ids = np.asarray(segment_id)
    assert ids[2] != ids[3]
    assert len(np.unique(ids)) == 2


def test_interior_hole_splits_the_path_and_is_dead():
    """A NaN-dead interior candidate breaks the path and carries no live id."""
    segment_id = segment_ids_from_folds(
        endog_grid=jnp.array([0.0, 1.0, jnp.nan, 2.0, 3.0])
    )
    ids = np.asarray(segment_id)
    assert bool(np.isnan(ids[2]))
    assert ids[1] != ids[3]
