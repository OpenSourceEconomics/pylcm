"""Pure Epstein-Zin EGM kernel: consumption inversion and value update.

Conditional on the continuation certainty equivalent `nu` and its savings
derivative `dnu/ds`, the interior Euler equation inverts for consumption in
closed form. For a power period flow `q(c, s) = c^phi * s^(1-phi)` with `s`
fixed and positive, the marginal `q^(-rho) q_c` is a single power of `c`, so
the inverse has exponent `1/[phi(1-rho)-1]`. The basic single-good flow is
`phi = 1`, where the exponent reduces to `-1/rho`.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.ez_kernel import (
    ez_consumption_from_euler,
    ez_continuation,
    ez_invert_partials,
    ez_marginal_of_resource,
    ez_period_value,
    ez_transform_partials,
    ez_transform_scalar,
)


def test_basic_flow_consumption_matches_the_crra_power_inversion() -> None:
    """With `phi = 1` the inversion is `c = (beta/(1-beta) nu^-rho dnu_ds)^(-1/rho)`."""
    rho = 2.0
    beta = 0.95
    nu = 3.0
    dnu_ds = 0.4
    consumption = ez_consumption_from_euler(
        nu=jnp.asarray(nu),
        dnu_ds=jnp.asarray(dnu_ds),
        discount_factor=beta,
        inverse_eis=rho,
        flow_coefficient=1.0,
        flow_exponent=-rho,
    )
    reference = (beta / (1.0 - beta) * nu ** (-rho) * dnu_ds) ** (-1.0 / rho)
    np.testing.assert_allclose(np.asarray(consumption), reference, rtol=1e-10)


def test_composite_flow_uses_the_cobb_douglas_exponent_not_minus_one_over_rho() -> None:
    """Fixed-service composite flow inverts with exponent `1/[phi(1-rho)-1]`.

    The recovered consumption satisfies the composite-flow FOC
    `(1-beta) q^(-rho) q_c = beta nu^(-rho) dnu_ds` exactly, which the plain
    `-1/rho` exponent would not.
    """
    phi = 0.6
    rho = 2.0
    beta = 0.95
    service = 3.0
    nu = 4.0
    dnu_ds = 0.05
    flow_exponent = phi * (1.0 - rho) - 1.0
    flow_coefficient = phi * service ** ((1.0 - phi) * (1.0 - rho))
    consumption = ez_consumption_from_euler(
        nu=jnp.asarray(nu),
        dnu_ds=jnp.asarray(dnu_ds),
        discount_factor=beta,
        inverse_eis=rho,
        flow_coefficient=flow_coefficient,
        flow_exponent=flow_exponent,
    )
    c = float(consumption)
    q = c**phi * service ** (1.0 - phi)
    q_c = phi * c ** (phi - 1.0) * service ** (1.0 - phi)
    lhs = (1.0 - beta) * q ** (-rho) * q_c
    rhs = beta * nu ** (-rho) * dnu_ds
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9)


def test_period_value_is_the_epstein_zin_aggregator() -> None:
    """`V = [(1-beta) q^(1-rho) + beta nu^(1-rho)]^(1/(1-rho))`."""
    rho = 2.0
    beta = 0.95
    flow = 2.0
    nu = 3.0
    value = ez_period_value(
        flow=jnp.asarray(flow),
        nu=jnp.asarray(nu),
        discount_factor=beta,
        inverse_eis=rho,
    )
    reference = ((1.0 - beta) * flow ** (1.0 - rho) + beta * nu ** (1.0 - rho)) ** (
        1.0 / (1.0 - rho)
    )
    np.testing.assert_allclose(np.asarray(value), reference, rtol=1e-10)


def test_marginal_of_resource_matches_the_foc_substituted_continuation_form() -> None:
    """`dV/dm = (1-beta) V^rho c^-rho` equals `V^rho beta nu^-rho dnu_ds` at optimum.

    The envelope marginal of the resource and the continuation form of the same
    derivative agree once the interior Euler equation
    `(1-beta) c^-rho = beta nu^-rho dnu_ds` holds, so the consumption recovered
    from the Euler inversion and the value it induces produce a consistent
    marginal value of liquid.
    """
    rho = 2.0
    beta = 0.95
    nu = 3.0
    dnu_ds = 0.4
    consumption = ez_consumption_from_euler(
        nu=jnp.asarray(nu),
        dnu_ds=jnp.asarray(dnu_ds),
        discount_factor=beta,
        inverse_eis=rho,
        flow_coefficient=1.0,
        flow_exponent=-rho,
    )
    value = ez_period_value(
        flow=consumption,
        nu=jnp.asarray(nu),
        discount_factor=beta,
        inverse_eis=rho,
    )
    marginal = ez_marginal_of_resource(
        flow_marginal=consumption ** (-rho),
        value=value,
        discount_factor=beta,
        inverse_eis=rho,
    )
    foc_form = float(value) ** rho * beta * nu ** (-rho) * dnu_ds
    np.testing.assert_allclose(np.asarray(marginal), foc_form, rtol=1e-9)


def test_period_value_at_unit_eis_is_the_cobb_douglas_limit() -> None:
    """At `rho = 1` the aggregator is the Cobb-Douglas limit `flow^(1-beta) nu^beta`.

    The CES exponent `1/(1-rho)` is singular at unit elasticity; the recursion's
    well-defined limit is the geometric aggregator, matching `H_epstein_zin`.
    """
    beta = 0.3
    flow = 2.0
    nu = 8.0
    value = ez_period_value(
        flow=jnp.asarray(flow),
        nu=jnp.asarray(nu),
        discount_factor=beta,
        inverse_eis=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(value), flow ** (1.0 - beta) * nu**beta, rtol=1e-10
    )


def test_period_value_is_strictly_positive_for_positive_inputs() -> None:
    """The recursive value index stays strictly positive, as the recursion needs."""
    value = ez_period_value(
        flow=jnp.asarray(1e-6),
        nu=jnp.asarray(1e-6),
        discount_factor=0.95,
        inverse_eis=3.0,
    )
    assert float(value) > 0.0


def test_continuation_nu_is_the_power_mean_of_child_values() -> None:
    """`nu = (E[V'^(1-gamma)])^(1/(1-gamma))` over the continuation lottery."""
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    nu, _ = ez_continuation(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    reference = (0.5 * 1.0 ** (1.0 - gamma) + 0.5 * 2.0 ** (1.0 - gamma)) ** (
        1.0 / (1.0 - gamma)
    )
    np.testing.assert_allclose(np.asarray(nu)[0], reference, rtol=1e-10)


def test_continuation_marginal_is_the_risk_reweighted_covariation() -> None:
    """`dnu/ds = nu^gamma * E[V'^(-gamma) * dV'/ds]` for exogenous probabilities.

    The certainty equivalent's savings derivative reweights each child's marginal
    by its risk-transformed value share, so a high-value state contributes less to
    the marginal incentive than a linear expectation would credit it.
    """
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    nu, dnu_ds = ez_continuation(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    nu_val = float(np.asarray(nu)[0])
    reference = nu_val**gamma * (
        0.5 * 1.0 ** (-gamma) * 0.5 + 0.5 * 2.0 ** (-gamma) * 0.25
    )
    np.testing.assert_allclose(np.asarray(dnu_ds)[0], reference, rtol=1e-9)


def test_continuation_reduces_to_linear_expectation_at_zero_risk_aversion() -> None:
    """At `risk_aversion = 0` the pair is the plain `(E[V'], E[dV'/ds])`."""
    child_values = jnp.array([[1.0, 3.0]])
    child_marginals = jnp.array([[0.4, 0.2]])
    weights = jnp.array([0.25, 0.75])
    nu, dnu_ds = ez_continuation(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(0.0),
    )
    np.testing.assert_allclose(np.asarray(nu)[0], 0.25 * 1.0 + 0.75 * 3.0, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(dnu_ds)[0], 0.25 * 0.4 + 0.75 * 0.2, rtol=1e-10
    )


def test_transform_partials_are_the_generator_weighted_sums() -> None:
    """`S = sum_j w_j V_j^(1-gamma)` and `T = sum_j w_j V_j^(-gamma) dV_j/ds`."""
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    transformed_value, transformed_marginal = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    exp_value = 0.5 * 1.0 ** (1.0 - gamma) + 0.5 * 2.0 ** (1.0 - gamma)
    exp_marginal = 0.5 * 1.0 ** (-gamma) * 0.5 + 0.5 * 2.0 ** (-gamma) * 0.25
    np.testing.assert_allclose(np.asarray(transformed_value)[0], exp_value, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(transformed_marginal)[0], exp_marginal, rtol=1e-10
    )


def test_transform_scalar_applies_the_generator() -> None:
    """`g(V) = V^(1-gamma)` for a certain continuation value."""
    gamma = 4.0
    value = jnp.asarray(2.0)
    np.testing.assert_allclose(
        np.asarray(ez_transform_scalar(value=value, risk_aversion=jnp.asarray(gamma))),
        2.0 ** (1.0 - gamma),
        rtol=1e-10,
    )


def test_transform_scalar_is_the_log_generator_at_unit_risk_aversion() -> None:
    """At `gamma = 1` the generator degenerates to `log V`."""
    value = jnp.asarray(3.0)
    np.testing.assert_allclose(
        np.asarray(ez_transform_scalar(value=value, risk_aversion=jnp.asarray(1.0))),
        np.log(3.0),
        rtol=1e-10,
    )


def test_invert_partials_recovers_the_continuation_certainty_equivalent() -> None:
    """`ez_invert_partials(ez_transform_partials(.)) == ez_continuation(.)`.

    The single-regime certainty equivalent is the transform partials inverted
    with no intervening regime blend, so the two must agree exactly.
    """
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    transformed_value, transformed_marginal = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    nu, dnu_ds = ez_invert_partials(
        transformed_value=transformed_value,
        transformed_marginal=transformed_marginal,
        risk_aversion=jnp.asarray(gamma),
    )
    ref_nu, ref_dnu_ds = ez_continuation(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    np.testing.assert_allclose(np.asarray(nu), np.asarray(ref_nu), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(dnu_ds), np.asarray(ref_dnu_ds), rtol=1e-10)


def test_single_node_transform_partial_matches_the_deterministic_inline_form() -> None:
    """A one-node lottery transforms to the deterministic-target inline partial.

    The continuation reader transforms a child with no own shock lottery inline as
    `(g(V), V^(-gamma) dV/ds)`; `ez_transform_partials` on a single unit-weight node
    must produce exactly that, so the two continuation paths carry identical
    transform-space partials into the regime blend.
    """
    gamma = 4.0
    value = jnp.asarray(2.0)
    marginal = jnp.asarray(0.25)
    transformed_value, transformed_marginal = ez_transform_partials(
        child_values=value[None],
        child_marginals=marginal[None],
        weights=jnp.array([1.0]),
        risk_aversion=jnp.asarray(gamma),
    )
    np.testing.assert_allclose(
        np.asarray(transformed_value),
        np.asarray(ez_transform_scalar(value=value, risk_aversion=jnp.asarray(gamma))),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(transformed_marginal),
        np.asarray(value ** (-gamma) * marginal),
        rtol=1e-10,
    )
