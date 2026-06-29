"""Mask and segment a BQSEGM case's EGM candidate path before the envelope.

EGM emits one candidate `(endog_grid, value, policy, marginal)` per savings node
within a case. Two transforms prepare those candidates for the branch-aware upper
envelope:

- NaN-dead masking: an invalid candidate is set to NaN in *every* channel, never
  left as a finite abscissa with `-inf` value. The query envelope treats only NaN
  endpoints as dead, so a finite-`-inf` candidate would stay a live link and can
  emit NaN through `0 * -inf`. `-inf` / `0` are reserved for published
  infeasible-choice rows, after the envelope.
- Fold/hole segmentation: the path is split into maximal ascending, hole-free
  subsegments, each carrying its own `segment_id`. The envelope links only
  same-id consecutive candidates, so it never bridges a fold or a masked gap.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D


def mask_dead_candidates(
    *,
    endog_grid: Float1D,
    value: Float1D,
    policy: Float1D,
    marginal: Float1D,
    valid: BoolND,
) -> tuple[Float1D, Float1D, Float1D, Float1D]:
    """Set every channel of an invalid candidate to NaN (NaN-dead masking).

    Args:
        endog_grid: Candidate endogenous grid points (resources).
        value: Candidate value-correspondence points.
        policy: Candidate policy values.
        marginal: Candidate marginal values.
        valid: Per-candidate validity mask; `False` marks an invalid candidate.

    Returns:
        Tuple of the four channels with every invalid candidate NaN in all of
        them. Valid candidates are unchanged — no `-inf` is introduced.

    """
    return (
        jnp.where(valid, endog_grid, jnp.nan),
        jnp.where(valid, value, jnp.nan),
        jnp.where(valid, policy, jnp.nan),
        jnp.where(valid, marginal, jnp.nan),
    )


def segment_ids_from_folds(*, endog_grid: Float1D) -> Float1D:
    """Label maximal ascending, hole-free runs of a candidate path.

    A new segment starts where the endogenous grid stops strictly ascending (a
    fold) or at a NaN-dead candidate (a hole). NaN-dead candidates carry a NaN id
    so the envelope, which links only equal finite ids, never bridges across
    them.

    Args:
        endog_grid: Candidate endogenous grid points in EGM-traversal order; a
            NaN entry is a dead candidate.

    Returns:
        Per-candidate segment id, NaN where the candidate is dead.

    """
    dead = jnp.isnan(endog_grid)
    ascending = endog_grid[1:] > endog_grid[:-1]
    starts_new = ~ascending | dead[1:] | dead[:-1]
    raw_id = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(starts_new.astype(jnp.int32))]
    )
    return jnp.where(dead, jnp.nan, raw_id.astype(endog_grid.dtype))
