"""The 2-D G2EGM upper envelope over triangular segment meshes.

Each constraint segment's endogenous candidate cloud is imaged into the current-state
$(m, n)$ plane and triangulated (`mesh_geometry`). The upper envelope selects, at every
common-grid target, the best feasible policy across all segments:

1. **First (within-segment) envelope.** For one segment and one target, consider every
   triangle the target is *admissible* to (barycentric weights above a negative
   threshold — covering and mildly extrapolated triangles alike), interpolate the
   segment's policy there, recompute the Bellman objective, drop infeasible candidates,
   and take the maximum. Admissible-not-only-covering matters: an extrapolated triangle
   can hold the true maximizer even when another triangle covers the target.
2. **Second (across-segment) envelope.** Take the per-target maximum over the segments'
   first-envelope values, gathering the winning segment's policy.

The objective is supplied as a callable `(state, policy) -> (value, feasible)` so the
envelope is independent of any one model: it recomputes the Bellman value from the
interpolated policy (never transports an interpolated value) and masks economically
infeasible candidates to minus infinity before either maximum.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from _lcm.egm.mesh_geometry import (
    barycentric_weights,
    interpolate_on_triangle,
    is_admissible,
)
from lcm.typing import BoolND, Float1D, Float2D, FloatND, Int2D, IntND

# Maps `(state, policy)` to `(value, feasible)`; infeasible candidates are masked.
ObjectiveEvaluator = Callable[[Float1D, Float1D], tuple[FloatND, BoolND]]


class SegmentMesh(NamedTuple):
    """One constraint segment's triangulated candidate cloud in the state plane.

    Each node is a candidate: the endogenous current state it is optimal at and the
    policy chosen there. The triangles connect the nodes; a node flagged invalid (e.g.
    an off-grid NaN-inverse) drops every triangle that touches it.
    """

    region_label: int
    """Which KKT segment (`ucon`/`dcon`/`acon`/`con`) produced the mesh."""
    node_state: Float2D
    """Endogenous current state `(m, n)` per node, shape `(n_node, 2)`."""
    node_policy: Float2D
    """Policy `(c, d)` per node, shape `(n_node, n_policy)`."""
    simplices: Int2D
    """Triangle node-index triples, shape `(n_simplex, 3)`."""
    valid_node: BoolND
    """Per-node validity mask, shape `(n_node,)`."""


class EnvelopeResult(NamedTuple):
    """The published upper-envelope value and policy on the common target grid."""

    value: FloatND
    """Envelope value per target, shape `(n_target,)`."""
    policy: Float2D
    """Winning policy per target, shape `(n_target, n_policy)`."""
    segment: IntND
    """Index of the winning segment per target, shape `(n_target,)`."""


def first_envelope(
    *,
    mesh: SegmentMesh,
    targets: Float2D,
    objective: ObjectiveEvaluator,
    threshold: float,
) -> tuple[FloatND, Float2D]:
    """Maximize the recomputed objective over admissible triangles, per target.

    For each target, every triangle is scored: its barycentric weights locate the
    target, the segment policy is interpolated, the objective and feasibility are
    recomputed, and the candidate is kept only if the triangle is admissible (all
    weights above `-threshold`, all three nodes valid) and the policy feasible.

    Args:
        mesh: The segment's triangulated candidate cloud.
        targets: Common-grid query states, shape `(n_target, 2)`.
        objective: Recomputes `(value, feasible)` from `(state, policy)`.
        threshold: Non-negative barycentric extrapolation tolerance.

    Returns:
        Tuple of the segment's per-target value (`-inf` where no admissible feasible
        candidate exists), shape `(n_target,)`, and its winning policy, shape
        `(n_target, n_policy)`.

    """
    triangle_states = mesh.node_state[mesh.simplices]
    triangle_policies = mesh.node_policy[mesh.simplices]
    triangle_valid = jnp.all(mesh.valid_node[mesh.simplices], axis=1)

    def at_target(target: Float1D) -> tuple[FloatND, Float1D]:
        def per_triangle(
            triangle: Float2D, node_policy: Float2D, valid: BoolND
        ) -> tuple[FloatND, Float1D]:
            weights = barycentric_weights(triangle=triangle, query=target)
            policy = interpolate_on_triangle(node_values=node_policy, weights=weights)
            value, feasible = objective(target, policy)
            admissible = is_admissible(weights=weights, threshold=threshold) & valid
            # Mask non-finite value/policy to -inf so a NaN candidate (from a
            # degenerate triangle or clamped continuation) cannot be selected by argmax.
            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(policy))
            candidate = jnp.where(admissible & feasible & finite, value, -jnp.inf)
            return candidate, policy

        candidates, policies = jax.vmap(per_triangle)(
            triangle_states, triangle_policies, triangle_valid
        )
        best = jnp.argmax(candidates)
        return candidates[best], policies[best]

    return jax.vmap(at_target)(targets)


def second_envelope(
    *, segment_values: Float2D, segment_policies: FloatND
) -> EnvelopeResult:
    """Take the per-target maximum across segments and gather the winning policy.

    Args:
        segment_values: First-envelope values, shape `(n_segment, n_target)`.
        segment_policies: First-envelope policies, shape
            `(n_segment, n_target, n_policy)`.

    Returns:
        The envelope value, winning policy, and winning segment index per target.

    """
    winning_segment = jnp.argmax(segment_values, axis=0).astype(jnp.int32)
    value = jnp.max(segment_values, axis=0)
    target_index = jnp.arange(segment_values.shape[1])
    policy = segment_policies[winning_segment, target_index]
    return EnvelopeResult(value=value, policy=policy, segment=winning_segment)
