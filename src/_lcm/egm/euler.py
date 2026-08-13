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

from lcm.typing import FloatND, ScalarFloat


def invert_euler(
    *,
    expected_marginal_continuation: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_marginal_utility: Callable[[FloatND], FloatND],
) -> FloatND:
    """Invert the Euler equation on the savings grid.

    The inversion is elementwise, so the same call serves a per-node scalar
    solve and a whole savings row.

    The discounted expected marginal continuation is clamped at a small
    positive epsilon *before* inversion (the degenerate-inversion guard): an
    exactly zero discounted marginal continuation — e.g. every reachable
    child is a stateless terminal regime, or the discount factor is zero —
    means saving has no marginal value, so the consume-everything corner is
    optimal. Without the clamp, $(u')^{-1}(0) = +\\infty$ would inject an
    infinite endogenous grid point that poisons the candidate sort and the
    envelope scan; with it, the inversion returns a very large but finite
    action and the closed-form credit-constrained segment represents the
    corner. The clamp acts on the discounted product, so a zero discount
    factor cannot reintroduce the degenerate inversion.

    The clamp is a numerical guard, not an economic identity: a discounted
    marginal continuation that is *positive but below the dtype's machine
    epsilon* is also clamped, so the returned action is $(u')^{-1}(eps)$ rather
    than the mathematical inverse at that tiny value. Such a candidate lies far
    to the right (a near-zero marginal implies near-total consumption) and is
    normally dominated or outside the queried resources range — a numerical
    artifact that the upper envelope drops where there is one, and that the
    closed-form constrained candidate covers where there is not.

    Args:
        expected_marginal_continuation: Probability- and shock-weighted
            expected marginal continuation value
            $\\mathbb{E}[\\partial V'/\\partial R' \\cdot \\partial R'/\\partial A]$
            at each savings node.
        discount_factor: Discount factor $\\beta$ of the Bellman aggregator.
        inverse_marginal_utility: The regime's inverse-marginal-utility
            function, reduced to a unary map of the marginal continuation with
            every other parameter already bound. It is called positionally, so
            its parameter name is its own business.

    Returns:
        The optimal continuous action at each savings node.

    """
    discounted = discount_factor * expected_marginal_continuation
    eps = jnp.finfo(discounted.dtype).eps
    return inverse_marginal_utility(jnp.maximum(discounted, eps))
