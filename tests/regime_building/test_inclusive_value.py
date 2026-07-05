"""Tests for the smoothed inclusive-value choice probabilities.

These cover the masked joint-softmax construction used by the (opt-in, off by
default) smoothing diagnostic: the marginal of the masked joint softmax, the
distinction from the incorrect hard-max-then-softmax shortcut, the zero-
temperature limit, and infeasible-action masking.

The masked-joint-softmax equivalence folds in the Pro-audit counterexample
(RT3): marginalizing the masked joint softmax over the non-choice actions must
equal the softmax of the τ-scaled inclusive value.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.inclusive_value import (
    smoothed_choice_probabilities,
    smoothed_inclusive_value,
)


def test_choice_probs_equal_masked_joint_softmax_marginal():
    """P_τ(j) is the marginal over the non-choice actions of the joint softmax.

    Pro-audit RT3: the inclusive-value softmax must equal marginalizing a masked
    joint softmax, not any hard-max-then-softmax shortcut.
    """
    Q = jnp.array([[1.0, -jnp.inf, 0.5], [0.2, 0.4, -jnp.inf]])
    feasible = jnp.isfinite(Q)
    tau = 0.7

    probs = smoothed_choice_probabilities(Q, feasible=feasible, tau=tau, choice_axis=0)

    masked = np.where(np.asarray(feasible), np.asarray(Q) / tau, -np.inf)
    joint = np.exp(masked - np.nanmax(masked))
    joint = joint / joint.sum()
    expected_marginal = joint.sum(axis=1)
    np.testing.assert_allclose(np.asarray(probs), expected_marginal, atol=1e-6)


def test_choice_probs_differ_from_hard_max_then_softmax():
    """The inclusive value uses logsumexp over a_{-j}, not the within-level max.

    A second feasible action under choice level 0 raises its inclusive value
    above its own best action, so the inclusive-value probabilities differ from
    softmax over the per-level maxima (F2).
    """
    Q = jnp.array([[1.0, 0.99], [1.0, -jnp.inf]])
    feasible = jnp.isfinite(Q)
    tau = 0.5

    inclusive = smoothed_choice_probabilities(
        Q, feasible=feasible, tau=tau, choice_axis=0
    )

    level_maxima = jnp.array([1.0, 1.0])
    hard_max_then_softmax = np.asarray(
        jnp.exp(level_maxima / tau) / jnp.exp(level_maxima / tau).sum()
    )
    # Level 0 has extra feasible mass (0.99), so it must carry strictly more
    # probability than the hard-max-then-softmax shortcut assigns it.
    assert np.asarray(inclusive)[0] > hard_max_then_softmax[0]


def test_choice_probs_concentrate_on_argmax_as_tau_goes_to_zero():
    """With a unique feasible maximum, P_τ → one-hot at its choice level."""
    Q = jnp.array([[3.0, 1.0], [0.5, 2.0]])
    feasible = jnp.ones_like(Q, dtype=bool)

    probs = smoothed_choice_probabilities(Q, feasible=feasible, tau=1e-3, choice_axis=0)

    np.testing.assert_allclose(np.asarray(probs), [1.0, 0.0], atol=1e-6)


def test_choice_probs_sum_to_one():
    """The choice probabilities form a distribution over the choice levels."""
    Q = jnp.array([[1.0, -jnp.inf, 0.5], [0.2, 0.4, 0.9]])
    feasible = jnp.isfinite(Q)

    probs = smoothed_choice_probabilities(Q, feasible=feasible, tau=0.7, choice_axis=0)

    np.testing.assert_allclose(float(probs.sum()), 1.0, atol=1e-6)


def test_fully_infeasible_choice_level_gets_zero_probability():
    """A choice level with no feasible action carries zero probability."""
    Q = jnp.array([[1.0, 0.5], [-jnp.inf, -jnp.inf]])
    feasible = jnp.isfinite(Q)

    probs = smoothed_choice_probabilities(Q, feasible=feasible, tau=0.7, choice_axis=0)

    np.testing.assert_allclose(float(probs[1]), 0.0, atol=1e-12)


def test_inclusive_value_is_softmax_smoothing_of_the_hard_max():
    """The inclusive value lies at or above the feasible per-level maximum.

    `τ·logsumexp(Q/τ) ≥ max(Q)` always, approaching it as τ → 0, so the
    inclusive value is a smooth upper envelope of the hard within-level max.
    """
    Q = jnp.array([[1.0, 0.0, 0.5], [0.2, 0.4, -jnp.inf]])
    feasible = jnp.isfinite(Q)

    inclusive = smoothed_inclusive_value(Q, feasible=feasible, tau=0.3, choice_axis=0)
    level_maxima = jnp.array([1.0, 0.4])

    assert bool(jnp.all(inclusive >= level_maxima - 1e-9))


@pytest.mark.parametrize("tau", [-1.0, 0.0])
def test_non_positive_temperature_raises(tau: float):
    """The smoothing temperature must be strictly positive."""
    Q = jnp.array([[1.0, 0.5], [0.2, 0.4]])
    feasible = jnp.ones_like(Q, dtype=bool)

    with pytest.raises(ValueError, match="tau"):
        smoothed_choice_probabilities(Q, feasible=feasible, tau=tau, choice_axis=0)
