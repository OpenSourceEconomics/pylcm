"""Collective-regime readout: the stakeholder value gather at the household argmax.

A collective regime carries one per-stakeholder action-value array ``Q^s`` each,
chooses the action that maximizes a household *scalarization*
``O = Σ_s λ_s Q^s`` over the feasible set, and then reads off *each stakeholder's
own* ``Q^s`` at that common argmax — NOT the scalarized value ``O``. The household
maximizes the weighted objective, but the individual values are each stakeholder's
own utility stream under that joint choice.

This module is a pure, engine-topology-free building block: it takes already-computed
per-stakeholder ``Q`` arrays and the feasibility mask and returns the per-stakeholder
``V`` plus the all-infeasible flag ``D`` (the dissolution / empty-feasible-set marker,
kept distinct from a numeric ``-inf`` that can arise on-path). The terminal and
non-terminal solve kernels call it after building their ``Q^s``.
"""

from collections.abc import Mapping

import jax.numpy as jnp

from _lcm.regime_building.argmax import (
    _flatten_last_n_axes,
    _move_axes_to_back,
    argmax_and_max,
)
from _lcm.zero_safe import sum_in_value_order, zero_safe_weighted_term
from lcm.typing import BoolND, FloatND, IntND

# Up to this many stakeholders, the scalarization's reduction order is not a choice:
# one term is returned as-is and two admit a single association. From three terms on,
# a declaration-order left fold lets an economically inert relabeling select a
# different reduction tree, so the sum is canonicalized by contribution value instead.
_LARGEST_ORDER_FREE_HOUSEHOLD = 2


def collective_argmax_and_readout(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    feasibility: BoolND,
    weights: Mapping[str, FloatND | float],
    action_axes: tuple[int, ...],
) -> tuple[IntND, dict[str, FloatND], BoolND]:
    r"""Like `collective_readout`, but also returns the household argmax index.

    The solve-side readout (`collective_readout`) only needs the
    per-stakeholder VALUES at the shared argmax; the simulate-side value
    router additionally needs the argmax INDEX itself, so the engine can look
    up which action was actually taken (mirroring the singleton
    `argmax_and_max_Q_over_a`, whose flat index feeds
    `_lookup_values_from_indices`).

    Returns:
        Tuple ``(argmax_flat, V, D)`` — the flat argmax index (in the same
        flattened-action layout `argmax_and_max` produces, directly
        compatible with the singleton simulate lookup), the per-stakeholder
        value mapping, and the dissolution flag.
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
    dissolution = ~jnp.any(feasibility, axis=action_axes)
    values = {
        name: jnp.where(
            dissolution,
            -jnp.inf,
            _gather_along_actions(
                q=q, argmax_flat=argmax_flat, action_axes=action_axes
            ),
        )
        for name, q in stakeholder_Q.items()
    }
    return argmax_flat, values, dissolution


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
    ``Q^s``. A cell with no feasible action yields ``D = True`` (the dissolution /
    empty-``F`` marker); the returned ``V^s`` in such a cell is overwritten with the
    ``-inf`` sentinel — the masked argmax is arbitrary there, so the gathered
    ``Q^s`` would otherwise be an infeasible action's value — and must be routed
    by the caller through the dissolution fallback, never read as a value.

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
    _argmax_flat, values, dissolution = collective_argmax_and_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights=weights,
        action_axes=action_axes,
    )
    return values, dissolution


def _weighted_sum(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    weights: Mapping[str, FloatND | float],
) -> FloatND:
    """Scalarize the per-stakeholder Q into the household objective Σ_s λ_s Q^s.

    Zero-safe: a stakeholder excluded from the household scalarization via a
    zero Pareto weight may still hold an admissible on-path `-inf` `Q^s` (e.g.
    a feasible zero-consumption action for that partner); `weight * Q` would
    then be `0.0 * -inf = nan`, poisoning the objective — and, via the
    argmax, the WRONG stakeholder's `Q` would decide the household's choice.
    Each term goes through `zero_safe_weighted_term` so a zero-weight
    stakeholder contributes exactly `0.0` regardless of its own `Q^s`.
    """
    names = list(stakeholder_Q)

    # A Pareto weight may arrive as a plain Python float -- see this function's
    # own signature -- while the shared term takes arrays, so it is converted
    # here at the boundary rather than by a permissive union on the term.
    #
    # `subnormal_is_accounted_for=False`: the weight arrives at whatever size
    # the model chose and nothing upstream has put it on a scale, so the term
    # moves the exponent itself rather than let a below-normal weight flush and
    # price a stakeholder out of the household objective entirely.
    def _term(name: str) -> FloatND:
        return zero_safe_weighted_term(
            weight=jnp.asarray(weights[name]),
            value=stakeholder_Q[name],
            subnormal_is_accounted_for=False,
        )

    terms = [_term(name) for name in names]
    if len(terms) <= _LARGEST_ORDER_FREE_HOUSEHOLD:
        # Preserve the established one-/two-stakeholder arithmetic exactly.  With
        # two terms there is no association choice, so relabeling cannot select a
        # different reduction tree.
        objective = terms[0]
        for term in terms[1:]:
            objective = objective + term
        return objective

    # ``stakeholders`` is an ordered representation of identities, not an economic
    # ordering of summands.  For three or more terms a declaration-order left fold
    # can cross a strict action boundary under cancellation.  Canonicalise by the
    # contribution VALUES, the same invariant used for the regime mixture.
    return sum_in_value_order(jnp.stack(terms, axis=0), axis=0)


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
