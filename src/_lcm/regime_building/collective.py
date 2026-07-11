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


def collective_argmax_and_readout(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    feasibility: BoolND,
    weights: Mapping[str, FloatND | float],
    action_axes: tuple[int, ...],
) -> tuple[IntND, dict[str, FloatND], BoolND]:
    r"""Like `collective_readout`, but also returns the household argmax index.

    COLLECTIVE-REGIMES (E4). The solve-side readout (`collective_readout`)
    only needs the per-stakeholder VALUES at the shared argmax; the simulate-
    side value router additionally needs the argmax INDEX itself, so the
    engine can look up which action was actually taken (mirroring the
    singleton `argmax_and_max_Q_over_a`, whose flat index feeds
    `_lookup_values_from_indices`). Factored out so `collective_readout`
    (still the solve entry point) stays byte-identical.

    Returns:
        Tuple ``(argmax_flat, V, D)`` — the flat argmax index (in the same
        flattened-action layout `argmax_and_max` produces, directly
        compatible with the singleton simulate lookup), the per-stakeholder
        value mapping, and the divorce flag.
    """
    if not stakeholder_Q:
        msg = "collective_argmax_and_readout requires at least one stakeholder."
        raise ValueError(msg)
    if set(stakeholder_Q) != set(weights):
        msg = (
            "stakeholder_Q and weights must have identical keys; got "
            f"{sorted(stakeholder_Q)} vs {sorted(weights)}."
        )
        raise ValueError(msg)

    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)
    argmax_flat, _ = argmax_and_max(
        objective, axis=action_axes, initial=-jnp.inf, where=feasibility
    )
    divorce = ~jnp.any(feasibility, axis=action_axes)
    values = {
        name: jnp.where(
            divorce,
            -jnp.inf,
            _gather_along_actions(
                q=q, argmax_flat=argmax_flat, action_axes=action_axes
            ),
        )
        for name, q in stakeholder_Q.items()
    }
    return argmax_flat, values, divorce


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
    empty-``F`` marker); the returned ``V^s`` in such a cell is overwritten with the
    ``-inf`` sentinel — the masked argmax is arbitrary there, so the gathered
    ``Q^s`` would otherwise be an infeasible action's value — and must be routed
    by the caller through the divorce fallback, never read as a value.

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
    _argmax_flat, values, divorce = collective_argmax_and_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights=weights,
        action_axes=action_axes,
    )
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
