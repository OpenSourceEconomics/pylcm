"""Exact query-side upper envelope of an EGM candidate correspondence.

The query-side counterpart of the full-row refiners (`fues`, `rfc`, `ltm`,
`mss`). Those materialise the whole refined envelope row and the caller then
reads it at a query; this evaluates the envelope *directly* at a set of query
abscissae without ever building the row.

For one query the value is the maximum, over every live branch segment that
brackets it, of the segment's linear value; the policy and marginal are the
winning segment's. A folded branch contributes several bracketing segments, so
the maximum is exact for the piecewise-linear correspondence. Topology is
explicit: a segment is the link between two consecutive candidates carrying the
same `segment_id`, so unrelated branches are never bridged — the contract the
host oracle enforces.

The evaluation is a fixed-shape `(n_query, n_segment)` bracket-and-reduce: no
sequential scan, no NaN-padded refined row, branch-parallel and reduction-heavy,
which is the shape an accelerator runs fastest. This is the backend asset-row
mode wants — one query per Euler node, no full envelope to refine.
"""

import jax.numpy as jnp

from lcm.typing import Float1D, FloatND


def envelope_at_query(
    *,
    endog_grid: Float1D,
    policy: Float1D,
    value: Float1D,
    marginal: Float1D,
    segment_id: Float1D,
    x_query: FloatND,
) -> tuple[FloatND, FloatND, FloatND]:
    """Evaluate the branch-aware upper envelope at each query abscissa.

    Args:
        endog_grid: Candidate endogenous grid points (resources), any order
            within a branch; a NaN entry is a dead/padding candidate.
        policy: Candidate policy values at `endog_grid`.
        value: Candidate value-correspondence points at `endog_grid`.
        marginal: Candidate marginal values (the supgradient) at `endog_grid`.
        segment_id: Per-candidate branch label. A segment is a consecutive-pair
            link whose endpoints share a label, so unrelated branches never join.
        x_query: Abscissae at which to evaluate the envelope.

    Returns:
        Tuple of the envelope value, the winning segment's policy, and the
        winning segment's marginal at each query, each shaped like `x_query`. A
        query no live segment brackets yields NaN in all three.
    """
    dead = jnp.isnan(endog_grid) | jnp.isnan(value)
    left_grid, right_grid = endog_grid[:-1], endog_grid[1:]
    left_value, right_value = value[:-1], value[1:]
    left_policy, right_policy = policy[:-1], policy[1:]
    left_marginal, right_marginal = marginal[:-1], marginal[1:]
    # A link is a real segment only within one branch: both endpoints live and
    # carrying the same label.
    segment_live = ~dead[:-1] & ~dead[1:] & (segment_id[:-1] == segment_id[1:])

    query = jnp.asarray(x_query)
    flat = query.reshape(-1)[:, None]
    lower = jnp.minimum(left_grid, right_grid)[None, :]
    upper = jnp.maximum(left_grid, right_grid)[None, :]
    brackets = segment_live[None, :] & (flat >= lower) & (flat <= upper)

    width = (right_grid - left_grid)[None, :]
    safe_width = jnp.where(width == 0.0, 1.0, width)
    relative = jnp.where(width == 0.0, 0.0, (flat - left_grid[None, :]) / safe_width)
    value_interp = left_value[None, :] + relative * (right_value - left_value)[None, :]
    policy_interp = (
        left_policy[None, :] + relative * (right_policy - left_policy)[None, :]
    )
    marginal_interp = (
        left_marginal[None, :] + relative * (right_marginal - left_marginal)[None, :]
    )

    masked_value = jnp.where(brackets, value_interp, -jnp.inf)
    any_bracket = jnp.any(brackets, axis=1)
    max_value = jnp.max(masked_value, axis=1, keepdims=True)
    # Break a value tie right-continuously, matching the kernel's `side="right"`
    # read: among the bracketing segments attaining the maximum, the one with the
    # larger value-slope is higher just to the right, so it carries the policy.
    slope = (right_value - left_value)[None, :] / safe_width
    near_max = brackets & (masked_value >= max_value - 1e-12)
    best = jnp.argmax(jnp.where(near_max, slope, -jnp.inf), axis=1)
    env_value = jnp.where(any_bracket, max_value[:, 0], jnp.nan)
    env_policy = jnp.where(
        any_bracket,
        jnp.take_along_axis(policy_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    env_marginal = jnp.where(
        any_bracket,
        jnp.take_along_axis(marginal_interp, best[:, None], axis=1)[:, 0],
        jnp.nan,
    )
    return (
        env_value.reshape(query.shape),
        env_policy.reshape(query.shape),
        env_marginal.reshape(query.shape),
    )
