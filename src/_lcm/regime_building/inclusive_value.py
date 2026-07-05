r"""Smoothed inclusive-value choice probabilities (opt-in smoothing diagnostic).

The solution/simulation path reduces the state-action value array `Q_arr` with a
hard `argmax` over the joint discrete-continuous action grid. Under common random
numbers that makes simulated choice-indicator moments step functions of the
parameters, so their finite-difference derivative is zero and the MSM
information matrix is singular.

This module provides the $\tau$-smoothed alternative used by the (opt-in, off by
default) smoothing diagnostic: the choice probability of a designated discrete
action, obtained by **marginalizing a masked joint softmax** over the other
actions. Equivalently it is the softmax of the inclusive value $I_\tau(j) =
\tau \, \operatorname{logsumexp}_{a_{-j}}(Q(j, a_{-j})/\tau)$, namely
$P_\tau(j) = \operatorname{softmax}_j(I_\tau(j)/\tau)$, with infeasible actions
masked to $-\infty$ before the reduction.

This is **not** a hard-max-then-softmax: the inner reduction is a `logsumexp`
over $a_{-j}$, so a second-best feasible action within a choice level raises that
level's value.

Nothing in the existing solve/simulate path calls these functions, so importing
this module leaves production output unchanged; the diagnostic opts in
explicitly.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from lcm.typing import BoolND, FloatND


def smoothed_inclusive_value(
    Q_arr: FloatND,
    *,
    feasible: BoolND,
    tau: float,
    choice_axis: int,
) -> FloatND:
    r"""Compute the `τ`-smoothed inclusive value of each choice level.

    Masks infeasible actions to $-\infty$, then reduces every action axis except
    `choice_axis` by $\tau\,\operatorname{logsumexp}(\cdot/\tau)$, leaving one
    value per level of the choice action.

    Args:
        Q_arr: State-action value array with one axis per action.
        feasible: Boolean array, same shape as `Q_arr`, marking feasible actions.
        tau: Smoothing temperature; must be strictly positive.
        choice_axis: The action axis to keep (the discrete choice of interest).

    Returns:
        The inclusive value along `choice_axis`, with every other axis reduced.

    Raises:
        ValueError: If `tau` is not strictly positive.

    """
    _fail_if_non_positive_tau(tau)
    masked = jnp.where(feasible, Q_arr / tau, -jnp.inf)
    other_axes = tuple(axis for axis in range(Q_arr.ndim) if axis != choice_axis)
    return tau * logsumexp(masked, axis=other_axes)


def smoothed_choice_probabilities(
    Q_arr: FloatND,
    *,
    feasible: BoolND,
    tau: float,
    choice_axis: int,
) -> FloatND:
    r"""Compute the smoothed choice probabilities of a discrete action.

    The probability of each level of the choice action is the marginal of the
    masked joint softmax over the other actions — equivalently the softmax of
    the inclusive value, $P_\tau(j) = \operatorname{softmax}_j(I_\tau(j)/\tau)$.
    A choice level with no feasible action receives probability zero. As $\tau
    \to 0$ (with a unique feasible maximum) the distribution concentrates on the
    `argmax` level, matching the hard policy.

    Args:
        Q_arr: State-action value array with one axis per action.
        feasible: Boolean array, same shape as `Q_arr`, marking feasible actions.
        tau: Smoothing temperature; must be strictly positive.
        choice_axis: The action axis whose levels the probabilities range over.

    Returns:
        A probability vector over the levels of the choice action.

    Raises:
        ValueError: If `tau` is not strictly positive.

    """
    inclusive_value = smoothed_inclusive_value(
        Q_arr, feasible=feasible, tau=tau, choice_axis=choice_axis
    )
    return jax.nn.softmax(inclusive_value / tau)


def _fail_if_non_positive_tau(tau: float) -> None:
    """Raise if the smoothing temperature is not strictly positive."""
    if tau <= 0.0:
        msg = f"Smoothing temperature tau must be strictly positive, got {tau}."
        raise ValueError(msg)
