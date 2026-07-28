"""The off-grid branch action must attain the value its branch is ranked by.

A discrete branch's returned (branch, action) pair must be an associated
optimizer pair: the continuous action returned for the winning branch has to
attain the objective the branch is ranked by, or a value-ranked branch can be
returned with an action worth less than a competitor's — a dominated pair.

Two recovery routes are contrasted on a stored solve node, using the branch
`V_A(R) = 2 log R` with optimal policy `c*_A(R) = R / 2` (log flow utility, so
the branch objective is `log(c) + log(R - c) + 2 log 2`), published on the
coarse nodes `R = (1, 10)`:

- inverting the resource-derivative of the cubic-Hermite *value* read does not
  recover the attaining action, because the Fritsch-Carlson limiter rewrites the
  node tangent before it is inverted;
- ranking the branch's piecewise-linear *policy* read by the objective it
  actually attains is associated by construction.
"""

import math

import jax
import jax.numpy as jnp

from _lcm.egm.interp import interp_on_padded_grid

_ENDOG = jnp.array([1.0, 10.0, jnp.nan])
_VALUE = jnp.array([0.0, 2.0 * math.log(10.0), jnp.nan])
_POLICY = jnp.array([0.5, 5.0, jnp.nan])
_MARGINAL = jnp.array([2.0, 0.2, jnp.nan])

# A feasible competitor branch B worth -0.05 at R = 1 (interior optimum).
_COMPETITOR_VALUE_AT_R1 = -0.05


def _branch_objective(resources: float, consumption: float) -> float:
    """Branch A's flow-plus-continuation objective `log c + W_A(R - c)`."""
    return (
        math.log(consumption) + math.log(resources - consumption) + 2.0 * math.log(2.0)
    )


def test_inverting_the_limited_hermite_value_derivative_returns_a_dominated_action():
    """Inverting the value read's own derivative yields a dominated action.

    At the stored node `R = 1` the branch optimum is `c* = 0.5` with value `0`,
    above the competitor's `-0.05`. The Fritsch-Carlson limiter clips the node
    tangent, so inverting the value-read derivative returns an action whose
    realized branch objective falls *below* the competitor — the value-ranked
    branch would be returned with a jointly dominated pair. This is why the
    re-decision does not read the action from the value derivative.
    """
    resources = jnp.asarray(1.0)
    derivative = jax.grad(
        lambda x: interp_on_padded_grid(
            x_query=x, xp=_ENDOG, fp=_VALUE, fp_slopes=_MARGINAL
        )
    )(resources)
    inverted_action = 1.0 / float(derivative)  # (u')^{-1}(m) = 1/m for log utility

    assert _branch_objective(1.0, inverted_action) < _COMPETITOR_VALUE_AT_R1


def test_objective_ranked_linear_policy_read_is_not_dominated():
    """Ranking the linear policy read by its attained objective is associated.

    The piecewise-linear policy read at `R = 1` returns the branch optimum
    `c* = 0.5`; the objective it attains is `0`, at or above the competitor's
    `-0.05`, so the returned pair is never dominated.
    """
    policy_action = float(
        interp_on_padded_grid(x_query=jnp.asarray(1.0), xp=_ENDOG, fp=_POLICY)
    )

    assert _branch_objective(1.0, policy_action) >= _COMPETITOR_VALUE_AT_R1
