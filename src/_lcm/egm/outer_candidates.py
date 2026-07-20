"""Stacked exact inner solves conditional on shared outer candidate nodes.

A nested outer-search solver (`NNBEGM`) solves one exact inner 1-D problem per
outer candidate node. The finite solver only ever needs the running maximum of
those solves, but the continuous-outer extension needs *all* of them at once:
the outer interpolant reads value/marginal/policy across the candidate axis,
and the adaptive mesh decides refinement from the full bank. This module holds
the bank data structures; the finite collapse that consumes them lives beside
the fold it replaces in `_lcm.solution.nnbegm` (the `egm` layer never imports
the `solution` layer).

The bank stores the stacked payloads as pytrees (`EGMCarry`, `EGMSimPolicy`)
with a leading candidate axis on every leaf, rather than one flat array field
per row kind: the pytree form keeps every payload leaf — including the
taste-shock scale and any later-added field — in sync with the payload's own
definition, with `None` subtrees (e.g. `breakpoints`, which NNBEGM validation
already forbids) dropping out naturally.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from lcm.typing import BoolND, FloatND, ScalarFloat


@dataclass(frozen=True, kw_only=True)
class OuterCandidateResult:
    """One adjuster node's exact conditional inner solve."""

    outer_node: ScalarFloat
    """The outer post-decision node this solve was conditioned on."""

    V_arr: FloatND
    """The conditional value array on the regime's exogenous state grid."""

    carry: EGMCarry
    """The conditional continuation rows on the shared liquid state grid."""

    sim_policy: EGMSimPolicy | None
    """The conditional published simulation policy, or `None`."""


@dataclass(frozen=True, kw_only=True)
class OuterCandidateBank:
    """Every adjuster node's exact solve, stacked along a leading candidate axis.

    The keeper is *not* an entry: its outer action is state-dependent (one
    no-adjustment value per current durable state), so it cannot share the
    bank's one-node-per-candidate layout and enters only at collapse time.
    """

    outer_nodes: FloatND
    """The shared outer candidate nodes, shape `(C,)`, in grid order."""

    V_arr: FloatND
    """Stacked conditional value arrays, shape `(C, *V_shape)`."""

    carry: EGMCarry
    """Stacked conditional carries: every leaf gains a leading `(C,)` axis."""

    sim_policy: EGMSimPolicy | None
    """Stacked conditional simulation policies (leading `(C,)` axis on every
    leaf), or `None` when the inner solver publishes none."""

    candidate_valid: BoolND
    """Per-candidate validity mask, shape `(C,)`.

    All-`True` for a finite exogenous sweep; the adaptive mesh marks nodes
    whose solves are unusable (e.g. everywhere-infeasible) instead of
    silently bridging across them.
    """

    @property
    def n_candidates(self) -> int:
        """Number of stacked candidates `C` (static)."""
        return int(self.outer_nodes.shape[0])

    def candidate_v_arr(self, index: int) -> FloatND:
        """The `index`-th candidate's conditional value array."""
        return self.V_arr[index]

    def candidate_carry(self, index: int) -> EGMCarry:
        """The `index`-th candidate's conditional carry, leading axis removed."""
        return jax.tree.map(lambda leaf: leaf[index], self.carry)


def build_outer_candidate_bank(
    *,
    outer_nodes: FloatND,
    results: Sequence[OuterCandidateResult],
) -> OuterCandidateBank:
    """Stack per-node exact solves into an `OuterCandidateBank`.

    Args:
        outer_nodes: The shared outer candidate nodes, shape `(C,)`, in grid
            order; must match the order of `results`.
        results: One `OuterCandidateResult` per node, in the same order.

    Returns:
        The bank, with every payload leaf stacked along a new leading axis.

    Raises:
        ValueError: If the node/result counts disagree, or the inner solver
            published a simulation policy for some nodes but not others (a
            partially-published bank has no consistent stacked layout).

    """
    if len(results) != int(outer_nodes.shape[0]):
        msg = (
            f"Expected one candidate result per outer node, got "
            f"{len(results)} results for {int(outer_nodes.shape[0])} nodes."
        )
        raise ValueError(msg)
    published = [result.sim_policy is not None for result in results]
    if any(published) and not all(published):
        msg = (
            "The inner solver published a simulation policy for "
            f"{sum(published)} of {len(results)} outer candidates; a bank "
            "requires all candidates or none to publish."
        )
        raise ValueError(msg)
    stacked_carry = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[result.carry for result in results],
    )
    stacked_policy = (
        jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *[result.sim_policy for result in results],
        )
        if all(published)
        else None
    )
    return OuterCandidateBank(
        outer_nodes=outer_nodes,
        V_arr=jnp.stack([result.V_arr for result in results]),
        carry=stacked_carry,
        sim_policy=stacked_policy,
        candidate_valid=jnp.ones(len(results), dtype=bool),
    )
