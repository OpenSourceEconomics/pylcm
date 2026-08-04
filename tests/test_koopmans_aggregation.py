"""Tests for the built-in Koopmans aggregators."""

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import PowerMean, W_epstein_zin, W_linear
from tests.test_models.taste_shocks_toy import get_model as get_toy_model


def test_W_linear_is_discounted_sum():
    """`W_linear(u, ce, beta) == u + beta * ce`."""
    result = W_linear(
        utility=jnp.asarray(2.0),
        CE=jnp.asarray(3.0),
        discount_factor=jnp.asarray(0.9),
    )
    np.testing.assert_allclose(result, 2.0 + 0.9 * 3.0, rtol=1e-6)


def test_W_epstein_zin_is_ces_in_utility_and_continuation():
    """`W_epstein_zin` is the CES form with curvature `rho = 1 - 1/psi`."""
    utility, ce, beta, ies = 2.0, 3.0, 0.9, 2.0
    rho = 1.0 - 1.0 / ies
    expected = ((1.0 - beta) * utility**rho + beta * ce**rho) ** (1.0 / rho)
    result = W_epstein_zin(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(ies),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_W_epstein_zin_unit_ies_is_cobb_douglas():
    """At `psi = 1` the aggregator is the Cobb-Douglas limit `u^(1-beta) * ce^beta`."""
    utility, ce, beta = 2.0, 3.0, 0.9
    result = W_epstein_zin(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(result, utility ** (1.0 - beta) * ce**beta, rtol=1e-6)


def test_default_aggregator_is_W_linear():
    """A non-terminal regime without an explicit `H` gets `W_linear` at model build."""
    toy = get_toy_model()
    assert toy.user_regimes["alive"].koopmans_aggregator is W_linear


@pytest.mark.parametrize("value", [1e-50, 2e-50])
def test_W_epstein_zin_is_idempotent_at_tiny_values(x64_enabled: None, value: float):
    """`W(x, x) == x`: aggregating a value with itself returns it, at any scale."""
    result = W_epstein_zin(
        utility=jnp.asarray(value),
        CE=jnp.asarray(value),
        discount_factor=jnp.asarray(0.5),
        intertemporal_elasticity_of_substitution=jnp.asarray(0.125),
    )
    np.testing.assert_allclose(float(result), value, rtol=1e-12)


@pytest.mark.parametrize("value", [1e-8, 2e-8])
def test_W_epstein_zin_is_idempotent_at_tiny_values_float32(value: float):
    """`W(x, x) == x` holds in single precision too, where `x^rho` underflows."""
    result = W_epstein_zin(
        utility=jnp.asarray(value, dtype=jnp.float32),
        CE=jnp.asarray(value, dtype=jnp.float32),
        discount_factor=jnp.asarray(0.5, dtype=jnp.float32),
        intertemporal_elasticity_of_substitution=jnp.asarray(0.125, dtype=jnp.float32),
    )
    np.testing.assert_allclose(float(result), value, rtol=1e-5)


def test_W_epstein_zin_ranks_actions_correctly_near_unit_ies_float32():
    """Just off `psi = 1`, the aggregator still ranks two actions correctly.

    The near-zero CES exponent makes the naive `((1-b) u^r + b ce^r)^(1/r)`
    cancel; here it is enough to reverse the preferred action.
    """
    utility = jnp.asarray([1.7782794, 0.04216965], dtype=jnp.float32)
    ce = jnp.asarray([1.9952623, 94.40609], dtype=jnp.float32)
    result = W_epstein_zin(
        utility=utility,
        CE=ce,
        discount_factor=jnp.asarray(0.5, dtype=jnp.float32),
        intertemporal_elasticity_of_substitution=jnp.asarray(
            1.000001, dtype=jnp.float32
        ),
    )
    # 60-digit-exact values of the CES form at these inputs.
    np.testing.assert_allclose(
        np.asarray(result), [1.8836490802, 1.9952771719], rtol=1e-6
    )
    assert int(jnp.argmax(result)) == 1


@pytest.mark.parametrize("ies", [0.125, 0.5, 0.999999, 1.0, 1.000001, 2.0, 8.0])
def test_W_epstein_zin_matches_the_exact_ces_value(x64_enabled: None, ies: float):
    """The aggregator reproduces the CES value to within a few ulps of exact.

    The reference is evaluated in 60-digit arithmetic rather than as the
    literal float expression, which itself loses eleven digits as `psi`
    approaches one.
    """
    utility, ce, beta = 2.0, 3.0, 0.9
    with localcontext() as context:
        context.prec = 60
        rho = 1 - 1 / Decimal(str(ies))
        expected = (
            Decimal(str(utility)) ** (1 - Decimal(str(beta)))
            * Decimal(str(ce)) ** Decimal(str(beta))
            if rho == 0
            else (
                (1 - Decimal(str(beta))) * Decimal(str(utility)) ** rho
                + Decimal(str(beta)) * Decimal(str(ce)) ** rho
            )
            ** (1 / rho)
        )
    result = W_epstein_zin(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(ies),
    )
    np.testing.assert_allclose(float(result), float(expected), rtol=1e-14)


def test_W_epstein_zin_is_continuous_across_unit_ies(x64_enabled: None):
    """The Cobb-Douglas limit is approached smoothly from both sides."""
    utility, ce, beta = 2.0, 3.0, 0.9

    def aggregate(ies: float) -> float:
        return float(
            W_epstein_zin(
                utility=jnp.asarray(utility),
                CE=jnp.asarray(ce),
                discount_factor=jnp.asarray(beta),
                intertemporal_elasticity_of_substitution=jnp.asarray(ies),
            )
        )

    limit = utility ** (1.0 - beta) * ce**beta
    for ies in (1.0 - 1e-9, 1.0, 1.0 + 1e-9):
        np.testing.assert_allclose(aggregate(ies), limit, rtol=1e-8)


@pytest.mark.parametrize(("discount_factor", "expected"), [(0.0, 2.0), (1.0, 3.0)])
def test_W_epstein_zin_at_extreme_discount_factors(
    x64_enabled: None, discount_factor: float, expected: float
):
    """`beta = 0` returns the utility and `beta = 1` the certainty equivalent."""
    result = W_epstein_zin(
        utility=jnp.asarray(2.0),
        CE=jnp.asarray(3.0),
        discount_factor=jnp.asarray(discount_factor),
        intertemporal_elasticity_of_substitution=jnp.asarray(0.5),
    )
    np.testing.assert_allclose(float(result), expected, rtol=1e-12)


def test_W_epstein_zin_is_symmetric_in_its_two_arguments(x64_enabled: None):
    """Swapping the two values and complementing the weight leaves it unchanged."""
    kwargs = {
        "discount_factor": jnp.asarray(0.3),
        "intertemporal_elasticity_of_substitution": jnp.asarray(0.25),
    }
    direct = W_epstein_zin(utility=jnp.asarray(2.0), CE=jnp.asarray(7.0), **kwargs)
    swapped = W_epstein_zin(
        utility=jnp.asarray(7.0),
        CE=jnp.asarray(2.0),
        **(kwargs | {"discount_factor": jnp.asarray(0.7)}),
    )
    np.testing.assert_allclose(float(direct), float(swapped), rtol=1e-12)


def test_W_epstein_zin_agrees_under_jit(x64_enabled: None):
    """Tracing the aggregator does not change what it computes."""
    kwargs = {
        "utility": jnp.asarray(1e-30),
        "CE": jnp.asarray(4e-30),
        "discount_factor": jnp.asarray(0.5),
        "intertemporal_elasticity_of_substitution": jnp.asarray(0.125),
    }
    np.testing.assert_allclose(
        float(jax.jit(W_epstein_zin)(**kwargs)), float(W_epstein_zin(**kwargs)), rtol=0
    )


def test_W_epstein_zin_is_the_power_mean_of_utility_and_continuation(
    x64_enabled: None,
):
    """The aggregator is `PowerMean` over `(U, CE)` with weights `(1-beta, beta)`.

    Its exponent is `1 - 1/psi`, so the equivalent `PowerMean` risk aversion
    is `1/psi`.
    """
    utility, ce, beta, ies = 2.0, 7.0, 0.3, 0.25
    aggregated = W_epstein_zin(
        utility=jnp.asarray(utility),
        CE=jnp.asarray(ce),
        discount_factor=jnp.asarray(beta),
        intertemporal_elasticity_of_substitution=jnp.asarray(ies),
    )
    as_power_mean = PowerMean().aggregate(
        values=jnp.array([utility, ce]),
        weights=jnp.array([1.0 - beta, beta]),
        params={"risk_aversion": jnp.asarray(1.0 / ies)},
    )
    np.testing.assert_allclose(float(aggregated), float(as_power_mean), rtol=1e-12)


def test_W_epstein_zin_applies_a_batched_ies_pointwise(x64_enabled: None):
    """A per-state IES applies to its own state, not to a lottery node.

    `psi` is a state/action-batched quantity: it broadcasts over the aggregated
    points, and each point's own `psi` governs it.
    """
    utility = jnp.array([0.1, 0.1])
    ce = jnp.array([0.1, 0.05])
    result = W_epstein_zin(
        utility=utility,
        CE=ce,
        discount_factor=jnp.asarray(0.1),
        intertemporal_elasticity_of_substitution=jnp.array([0.125, 0.25]),
    )
    pointwise = [
        float(
            W_epstein_zin(
                utility=utility[i],
                CE=ce[i],
                discount_factor=jnp.asarray(0.1),
                intertemporal_elasticity_of_substitution=jnp.asarray(psi),
            )
        )
        for i, psi in enumerate((0.125, 0.25))
    ]
    np.testing.assert_allclose(np.asarray(result), pointwise, rtol=1e-12)
    assert int(jnp.argmax(result)) == 0


def test_W_epstein_zin_accepts_a_batched_ies_of_any_length(x64_enabled: None):
    """A batched IES whose length differs from the two aggregated values works."""
    result = W_epstein_zin(
        utility=jnp.array([0.1, 0.1, 0.1]),
        CE=jnp.array([0.1, 0.05, 0.2]),
        discount_factor=jnp.asarray(0.1),
        intertemporal_elasticity_of_substitution=jnp.array([0.5, 2.0, 4.0]),
    )
    np.testing.assert_allclose(
        np.asarray(result), [0.1, 0.09422792, 0.10919235], rtol=1e-6
    )
