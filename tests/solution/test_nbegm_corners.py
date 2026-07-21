"""Corner-completeness of the NBEGM per-case step.

A jumped continuation breaks concavity, so the per-case value is the upper
envelope over every feasible candidate: the Euler interior path, the
boundary-targeting branch that saves to the higher eligible continuation, and the
hard borrowing corner that saves nothing. These tests pin two corners a plain EGM
shortcut misses:

- the exact-boundary continuation read returns the equality-owning side, not a
  bridged average across the jump;
- the per-case value dominates a dense brute that searches every consumption
  level, so no corner is dropped from the envelope.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.nbegm_step import _case_step, _kink_aware_interp


def test_kink_aware_interp_returns_the_otherwise_side_at_the_exact_limit():
    """An `otherwise`-owned boundary reads the otherwise value at exactly the limit.

    The grid node below the limit holds the eligible (high) continuation and the
    node above holds the otherwise (low) one. A query landing exactly on the limit
    belongs to the otherwise side, so it must read the low value, not the average
    of the two.
    """
    grid = jnp.asarray([7.0, 7.5, 8.5, 9.0])
    values = jnp.asarray([10.0, 10.0, 0.0, 0.0])
    limit = 8.0
    at_limit = float(
        _kink_aware_interp(
            jnp.asarray([limit]), grid, values, limit, equality_owner="otherwise"
        )[0]
    )
    np.testing.assert_allclose(at_limit, 0.0, atol=1e-9)


def test_kink_aware_interp_returns_the_when_side_at_the_exact_limit():
    """A `when`-owned boundary reads the eligible value at exactly the limit."""
    grid = jnp.asarray([7.0, 7.5, 8.5, 9.0])
    values = jnp.asarray([10.0, 10.0, 0.0, 0.0])
    limit = 8.0
    at_limit = float(
        _kink_aware_interp(
            jnp.asarray([limit]), grid, values, limit, equality_owner="when"
        )[0]
    )
    np.testing.assert_allclose(at_limit, 10.0, atol=1e-9)


def test_case_step_matches_a_dense_brute_through_a_value_jump():
    """The per-case value equals a candidate-complete dense brute, jump and all.

    The continuation jumps up by a constant below the limit with the same slope on
    each side. A dense consumption search over the whole budget recovers the true
    single-case Bellman value — including the zero-savings corner and the
    save-to-the-limit branch — so the case-piece step must match it across the
    asset interior.
    """
    crra, beta, return_liquid, income, subsidy, limit = 2.0, 0.95, 0.03, 1.0, 3.0, 2.0
    gross_return = 1.0 + return_liquid
    liquid_grid = jnp.linspace(0.1, 10.0, 200)
    savings_grid = jnp.linspace(0.0, 10.0, 200)

    def value_function(liquid):
        return jnp.where(liquid < limit, 5.0 + jnp.log1p(liquid), jnp.log1p(liquid))

    next_value = value_function(liquid_grid)
    next_marginal = 1.0 / (1.0 + liquid_grid)

    value, _marginal, _policy = _case_step(
        next_value=next_value,
        next_marginal=next_marginal,
        liquid_grid=liquid_grid,
        savings_grid=savings_grid,
        discount_factor=beta,
        crra=crra,
        return_liquid=return_liquid,
        income=income,
        subsidy=subsidy,
        asset_limit=limit,
        equality_owner="otherwise",
    )

    coh = liquid_grid + subsidy
    fractions = jnp.linspace(1e-4, 1.0 - 1e-9, 6000)
    consumption = fractions[:, None] * coh[None, :]
    next_liquid = gross_return * (coh[None, :] - consumption) + income
    crra_utility = consumption ** (1.0 - crra) / (1.0 - crra)
    candidate = crra_utility + beta * value_function(next_liquid)
    brute = jnp.max(candidate, axis=0)

    interior = (np.asarray(liquid_grid) > 0.5) & (np.asarray(liquid_grid) < 9.5)
    np.testing.assert_allclose(
        np.asarray(value)[interior],
        np.asarray(brute)[interior],
        atol=2e-2,
        rtol=5e-3,
    )
