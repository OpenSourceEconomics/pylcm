"""Tests for the built-in Koopmans aggregators."""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.ez_kernel import ez_period_value
from lcm import H_epstein_zin, H_linear
from tests.test_models.taste_shocks_toy import get_model as get_toy_model


def test_H_linear_is_discounted_sum():
    """`H_linear(u, ce, beta) == u + beta * ce`."""
    result = H_linear(
        utility=jnp.asarray(2.0),
        E_next_V=jnp.asarray(3.0),
        discount_factor=jnp.asarray(0.9),
    )
    np.testing.assert_allclose(result, 2.0 + 0.9 * 3.0, rtol=1e-6)


def test_H_epstein_zin_is_ces_in_utility_and_continuation():
    """`H_epstein_zin` is the CES form with curvature `rho = 1 - 1/psi`."""
    utility, ce, beta, ies = 2.0, 3.0, 0.9, 2.0
    rho = 1.0 - 1.0 / ies
    expected = ((1.0 - beta) * utility**rho + beta * ce**rho) ** (1.0 / rho)
    result = H_epstein_zin(
        utility=jnp.asarray(utility),
        E_next_V=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(ies),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_H_epstein_zin_unit_ies_is_cobb_douglas():
    """At `psi = 1` the aggregator is the Cobb-Douglas limit `u^(1-beta) * ce^beta`."""
    utility, ce, beta = 2.0, 3.0, 0.9
    result = H_epstein_zin(
        utility=jnp.asarray(utility),
        E_next_V=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(result, utility ** (1.0 - beta) * ce**beta, rtol=1e-6)


def test_default_H_is_H_linear():
    """A non-terminal regime without an explicit `H` gets `H_linear` at model build."""
    toy = get_toy_model()
    assert toy.user_regimes["alive"].functions["H"] is H_linear


def test_H_epstein_zin_is_stable_where_raw_powers_underflow() -> None:
    """The aggregator stays exact where raw CES powers leave float64's range.

    With `psi = 1/25` and inputs near `1e-14`, the raw terms `U^(1-rho)` are
    ~1e336 — past float64 — while the aggregated value (~1e-14) is comfortably
    representable. GridSearch reads this aggregator directly, so it must
    publish the same value the NBEGM period kernel computes; the reference is
    the log-domain CES.
    """
    inverse_eis = 25.0
    beta = 0.5
    utility = 1e-14
    certainty_equivalent = 2e-14

    got = H_epstein_zin(
        utility=jnp.asarray(utility),
        E_next_V=jnp.asarray(certainty_equivalent),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(1.0 / inverse_eis),
    )

    logs = np.array(
        [
            np.log(1.0 - beta) + (1.0 - inverse_eis) * np.log(utility),
            np.log(beta) + (1.0 - inverse_eis) * np.log(certainty_equivalent),
        ]
    )
    shift = logs.max()
    expected = np.exp(
        (shift + np.log(np.sum(np.exp(logs - shift)))) / (1.0 - inverse_eis)
    )
    assert float(got) > 0.0
    np.testing.assert_allclose(float(got), expected, rtol=1e-10)


def test_H_epstein_zin_matches_the_nbegm_period_kernel() -> None:
    """GridSearch and NBEGM evaluate one CES aggregator, bit for bit.

    The brute solver reads `H_epstein_zin` while NBEGM reads
    `ez_period_value`; publishing different cardinal values for the same
    model would break cross-solver validation, so the public aggregator must
    be the same computation.
    """
    got_public = H_epstein_zin(
        utility=jnp.asarray(2.0),
        E_next_V=jnp.asarray(3.0),
        discount_factor=jnp.asarray(0.9),
        intertemporal_elasticity_of_substitution=jnp.asarray(0.5),
    )
    got_kernel = ez_period_value(
        flow=jnp.asarray(2.0),
        nu=jnp.asarray(3.0),
        discount_factor=jnp.asarray(0.9),
        inverse_eis=jnp.asarray(2.0),
    )
    np.testing.assert_array_equal(np.asarray(got_public), np.asarray(got_kernel))
