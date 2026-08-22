"""The hard-max discrete envelope carries a well-defined marginal at value ties.

Danskin's theorem gives the enveloped marginal as the winning branch's marginal.
Where two branches tie in value the envelope has a kink and the true derivative is
a subgradient set; the solver's argmax convention selects the lowest-index tied
branch, a well-defined single value (not a NaN, not an average). This locks that
convention so the parent period's Euler inversion reads a stable marginal at a
choice-indifference point.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.solution.nbegm import _discrete_envelope_over_branches


def test_hard_max_envelope_at_a_value_tie_takes_the_lower_index_branch_marginal():
    """At a value tie the enveloped marginal is the lower-index tied branch's."""
    # Column 0: both branches value 1.0 (a tie); column 1: branch 1 strictly wins.
    value_stack = jnp.array([[1.0, 2.0], [1.0, 5.0]])
    marginal_stack = jnp.array([[0.3, 0.4], [0.7, 0.9]])

    value, marginal = _discrete_envelope_over_branches(
        value_stack=value_stack,
        marginal_stack=marginal_stack,
        taste_shock_scale=0.0,
    )

    np.testing.assert_allclose(np.asarray(value), [1.0, 5.0])
    # Tie column → branch 0 (lowest index); strict-win column → branch 1.
    np.testing.assert_allclose(np.asarray(marginal), [0.3, 0.9])


def test_hard_max_envelope_marginal_is_the_winning_branch_marginal():
    """Away from ties the enveloped marginal is the strictly-winning branch's."""
    value_stack = jnp.array([[4.0, 1.0], [2.0, 6.0]])
    marginal_stack = jnp.array([[0.5, 0.1], [0.2, 0.8]])

    _value, marginal = _discrete_envelope_over_branches(
        value_stack=value_stack,
        marginal_stack=marginal_stack,
        taste_shock_scale=0.0,
    )

    # Branch 0 wins column 0, branch 1 wins column 1.
    np.testing.assert_allclose(np.asarray(marginal), [0.5, 0.8])
