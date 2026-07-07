"""Collective-regime (E1) readout: the stakeholder value gather at the household argmax.

The mathematical heart of the "collective regimes" extension (design doc
`pylcm-extension-collective-regimes.md`, §2 E1). A collective regime carries one
per-stakeholder action-value array ``Q^s`` each, chooses the action that maximizes
a household *scalarization* ``O = Σ_s λ_s Q^s`` over the feasible set, and then reads
off *each stakeholder's own* ``Q^s`` at that common argmax — NOT the scalarized
value ``O`` (paper eqs. 10-12: the couple maximizes the weighted objective, but the
individual married values are each partner's own utility stream under that joint
choice).

This module is a pure, engine-topology-free building block: it takes already-computed
per-stakeholder ``Q`` arrays and the feasibility mask and returns the per-stakeholder
``V`` plus the all-infeasible flag ``D`` (the divorce / empty-feasible-set marker,
kept distinct from a numeric ``-inf`` that can arise on-path). The terminal and
non-terminal solve kernels call it after building their ``Q^s``; wiring it into the
kernels and threading the stakeholder axis through the V-array topology is the
remaining part of slice 1 (see `pylcm-extension-implementation-plan.md`).
"""

from collections.abc import Mapping

import jax.numpy as jnp

from _lcm.regime_building.argmax import (
    _flatten_last_n_axes,
    _move_axes_to_back,
    argmax_and_max,
)
from lcm.typing import BoolND, FloatND, IntND


def collective_readout(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    feasibility: BoolND,
    weights: Mapping[str, FloatND | float],
    action_axes: tuple[int, ...],
) -> tuple[dict[str, FloatND], BoolND]:
    r"""Household argmax of the scalarization, then per-stakeholder value readout.

    Implements the E1 readout (eqs. 10-12):

    .. math::
        a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_s \lambda_s\, Q^s(x, a),
        \qquad V^s(x) = Q^s(x, a^*(x)).

    All stakeholders share the same argmax ``a*`` (the joint household choice), so
    ties are broken identically for every stakeholder — ``argmax_and_max`` selects
    the first maximizer, and the gather uses that same flattened index for each
    ``Q^s``. A cell with no feasible action yields ``D = True`` (the divorce /
    empty-``F`` marker); the returned ``V^s`` in such a cell is the ``argmax_and_max``
    ``initial`` (``-inf``) and must be routed by the caller through the divorce
    fallback, never read as a value.

    Args:
        stakeholder_Q: Mapping stakeholder name -> its action-value array, each of
            shape ``(*state_axes, *action_axes)`` (identical shape across stakeholders).
        feasibility: Boolean mask of the same shape as each ``Q^s``; ``True`` where
            the (state, action) is feasible.
        weights: Mapping stakeholder name -> Pareto weight ``λ_s`` (scalar or an array
            broadcastable to the state axes). Should sum to 1 across stakeholders,
            though this is not enforced (a caller may pass unnormalized weights).
        action_axes: The axes of ``Q^s`` / ``feasibility`` to maximize over (the
            action dimensions). The remaining axes are the state axes retained in the
            output.

    Returns:
        Tuple ``(V, D)`` where ``V`` maps each stakeholder name to its value array of
        shape ``(*state_axes,)`` (the action axes reduced away), and ``D`` is the
        boolean all-infeasible flag of shape ``(*state_axes,)``.
    """
    if not stakeholder_Q:
        msg = "collective_readout requires at least one stakeholder."
        raise ValueError(msg)
    if set(stakeholder_Q) != set(weights):
        msg = (
            "stakeholder_Q and weights must have identical keys; got "
            f"{sorted(stakeholder_Q)} vs {sorted(weights)}."
        )
        raise ValueError(msg)

    # Household scalarization O = Σ_s λ_s Q^s over the common action grid.
    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)

    # Joint argmax over the feasible action axes (first maximizer on ties). The max
    # value of O itself is discarded — E1 reads each stakeholder's OWN Q at a*.
    argmax_flat, _ = argmax_and_max(
        objective, axis=action_axes, initial=-jnp.inf, where=feasibility
    )

    # Gather each stakeholder's Q at the shared argmax, using the same flatten order
    # `argmax_and_max` used to produce the index.
    values = {
        name: _gather_along_actions(
            q=q, argmax_flat=argmax_flat, action_axes=action_axes
        )
        for name, q in stakeholder_Q.items()
    }

    # Divorce / empty-feasible-set flag: no feasible action anywhere over the action
    # axes for this state cell. Distinct from a numeric -inf value.
    divorce = ~jnp.any(feasibility, axis=action_axes)

    return values, divorce


def _weighted_sum(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    weights: Mapping[str, FloatND | float],
) -> FloatND:
    """Scalarize the per-stakeholder Q into the household objective Σ_s λ_s Q^s."""
    names = list(stakeholder_Q)
    objective = weights[names[0]] * stakeholder_Q[names[0]]
    for name in names[1:]:
        objective = objective + weights[name] * stakeholder_Q[name]
    return objective


def _gather_along_actions(
    *, q: FloatND, argmax_flat: IntND, action_axes: tuple[int, ...]
) -> FloatND:
    """Gather ``q`` at the flattened action argmax, mirroring ``argmax_and_max``.

    ``argmax_and_max`` moves ``action_axes`` to the back, flattens them, and argmaxes
    the last axis, so ``argmax_flat`` indexes into that flattened action space with
    the state axes as its shape. Reproduce the same layout on ``q`` and take along it.
    """
    q_moved = _move_axes_to_back(q, axes=action_axes)
    q_flat = _flatten_last_n_axes(q_moved, n=len(action_axes))
    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)
    return gathered[..., 0]
