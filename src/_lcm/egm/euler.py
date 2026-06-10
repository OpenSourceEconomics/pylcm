"""Euler inversion on the exogenous savings grid.

The first-order condition at an interior optimum equates the marginal utility
of the continuous action with the discounted expected marginal continuation
value: $u'(c_j) = \\beta \\, \\mathbb{E}[\\partial V' / \\partial R' \\cdot
\\partial R' / \\partial A]$ at each savings node $A_j$. Inverting via the
regime's `inverse_marginal_utility` function yields the optimal action and
the endogenous resources point $R_j = A_j + c_j$ without any root finding.
"""

from collections.abc import Callable

import jax.numpy as jnp

from lcm.typing import ScalarFloat


def invert_euler(
    *,
    expected_marginal_continuation: ScalarFloat,
    discount_factor: ScalarFloat,
    inverse_marginal_utility: Callable[..., ScalarFloat],
) -> ScalarFloat:
    """Invert the Euler equation at one savings node.

    The expected marginal continuation is clamped at a small positive
    epsilon *before* inversion (the degenerate-inversion guard): an exactly
    zero marginal continuation — e.g. every reachable child is a stateless
    terminal regime — means saving has no marginal value, so the
    consume-everything corner is optimal. Without the clamp,
    $(u')^{-1}(0) = +\\infty$ would inject an infinite endogenous grid point
    that poisons the candidate sort and the envelope scan; with it, the
    inversion returns a very large but finite action and the closed-form
    credit-constrained segment represents the corner.

    Args:
        expected_marginal_continuation: Probability- and shock-weighted
            expected marginal continuation value
            $\\mathbb{E}[\\partial V'/\\partial R' \\cdot \\partial R'/\\partial A]$
            at the savings node.
        discount_factor: Discount factor $\\beta$ of the Bellman aggregator.
        inverse_marginal_utility: The regime's inverse-marginal-utility
            function with every parameter except `marginal_continuation`
            already bound.

    Returns:
        The optimal continuous action at the savings node.

    """
    eps = jnp.finfo(expected_marginal_continuation.dtype).eps
    clamped = jnp.maximum(expected_marginal_continuation, eps)
    return inverse_marginal_utility(marginal_continuation=discount_factor * clamped)
