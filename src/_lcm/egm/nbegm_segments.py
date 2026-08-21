"""Mask and segment a NBEGM case's EGM candidate path before the envelope.

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

Feasibility itself is decided by `affords_an_action`, pointwise and exactly, so no
attainable action is ever masked away and no point's candidate depends on another
point's budget.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, FloatND


def affords_an_action(budget: FloatND) -> BoolND:
    """Whether `budget` affords an action, decided exactly and pointwise.

    A residual budget — cash-on-hand, or cash-on-hand minus a savings node — affords
    consuming it whenever it is positive, and its sign is exactly decidable. IEEE
    subtraction is correctly rounded, so for the floats actually held, `a - b`
    evaluates positive exactly when `a > b`: cancellation costs the difference its
    significant digits, never its sign. The comparison is therefore a certificate,
    not an estimate, and needs no tolerance.

    Two properties follow, and both are contracts:

    - **Every exactly represented positive budget is feasible.** A grid may place a
      node anywhere, and a node whose budget is one ULP affords an action just as a
      node whose budget is a thousand does. Deleting it would remove an attainable —
      possibly optimal — candidate from the envelope.
    - **The decision is pointwise.** It reads one point's own budget, so extending a
      grid with a far-away node cannot change it. The signature admits no array-wide
      magnitude for exactly that reason: any threshold scaled to a whole grid makes a
      local action's existence depend on unrelated states.

    What this does *not* certify is a budget the author meant to be zero and a grid
    could not represent — `jnp.linspace(-1.0, 5.0, 13)` places its third node at
    `+6e-8` in 32-bit and `-1e-16` in 64-bit. That is a question about the grid, not
    about the sign of the number on it, and it belongs upstream where the intent is
    still known. Downstream, the tiny positive budget is answered honestly: an action
    exists, and it is a very bad one.

    Args:
        budget: Residual budget whose positivity decides feasibility.

    Returns:
        Boolean mask, `True` where an action exists.

    """
    return budget > 0.0


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
