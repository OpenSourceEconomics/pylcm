"""Tests for the built-in Koopmans aggregators."""

import jax.numpy as jnp
import numpy as np

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
