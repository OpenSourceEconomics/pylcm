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
    ez_blend_partials,
    ez_consumption_from_euler,
    ez_continuation,
    ez_invert_partials,
    ez_marginal_of_resource,
    ez_period_value,
    ez_transform_partials,
    ez_transform_scalar,
)
from _lcm.egm.nbegm_step import _ez_flow_power_structure


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
        log_flow_coefficient=0.0,
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
    log_flow_coefficient = np.log(phi) + (1.0 - phi) * (1.0 - rho) * np.log(service)
    consumption = ez_consumption_from_euler(
        nu=jnp.asarray(nu),
        dnu_ds=jnp.asarray(dnu_ds),
        discount_factor=beta,
        inverse_eis=rho,
        log_flow_coefficient=log_flow_coefficient,
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
        log_flow_coefficient=0.0,
        flow_exponent=-rho,
    )
    value = ez_period_value(
        flow=consumption,
        nu=jnp.asarray(nu),
        discount_factor=beta,
        inverse_eis=rho,
    )
    marginal = ez_marginal_of_resource(
        log_flow_marginal=-rho * jnp.log(consumption),
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


def test_period_value_stays_finite_at_high_inverse_eis_near_the_constraint() -> None:
    """`rho = 25` with a flow near the borrowing constraint keeps a finite value.

    The CES terms `flow^(1-rho)` stay inside float64's power range up to
    `|1-rho| * |log10 flow| < 308`, which covers every economically meaningful
    inverse elasticity; the value matches the log-domain reference
    `log V = LSE(log(1-beta) + (1-rho) log q, log beta + (1-rho) log nu)
    / (1-rho)`.
    """
    rho = 25.0
    beta = 0.95
    flow = 1e-10
    nu = 2e-10
    logs = np.array(
        [
            np.log(1.0 - beta) + (1.0 - rho) * np.log(flow),
            np.log(beta) + (1.0 - rho) * np.log(nu),
        ]
    )
    shift = logs.max()
    reference = np.exp((shift + np.log(np.sum(np.exp(logs - shift)))) / (1.0 - rho))
    value = ez_period_value(
        flow=jnp.asarray(flow),
        nu=jnp.asarray(nu),
        discount_factor=beta,
        inverse_eis=rho,
    )
    np.testing.assert_allclose(np.asarray(value), reference, rtol=1e-9)


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


def test_continuation_stays_finite_at_high_risk_aversion_near_the_constraint() -> None:
    """`gamma = 50` with values near the borrowing constraint keeps a finite pair.

    The transformed terms `V^(1-gamma)` overflow float64 for `V ~ 1e-10`, so the
    certainty equivalent must be evaluated in the log domain: the correct `nu` is
    the power mean (finite, close to the smallest value at high risk aversion) and
    `dnu/ds = sum_j w_j (nu/V_j)^gamma dV_j/ds` weights the worst state's marginal
    up without overflowing.
    """
    gamma = 50.0
    values = np.array([1e-10, 2e-10])
    weights = np.array([0.5, 0.5])
    marginals = np.array([1.0, 1.0])
    logs = np.log(weights) + (1.0 - gamma) * np.log(values)
    shift = logs.max()
    log_nu = (shift + np.log(np.sum(np.exp(logs - shift)))) / (1.0 - gamma)
    nu_reference = np.exp(log_nu)
    dnu_reference = np.sum(
        weights * np.exp(gamma * (log_nu - np.log(values))) * marginals
    )
    nu, dnu_ds = ez_continuation(
        child_values=jnp.asarray(values)[None, :],
        child_marginals=jnp.asarray(marginals)[None, :],
        weights=jnp.asarray(weights),
        risk_aversion=jnp.asarray(gamma),
    )
    np.testing.assert_allclose(np.asarray(nu)[0], nu_reference, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(dnu_ds)[0], dnu_reference, rtol=1e-9)


def test_continuation_marginal_matches_finite_differences() -> None:
    """`dnu/ds` equals the finite-difference derivative of `nu(s)`.

    With affine child values `V_j(s) = a_j + b_j s` (marginals `b_j`), the
    certainty equivalent's savings derivative from the anchored `T` channel
    must equal the numerical derivative of the certainty equivalent itself.
    """
    gamma = 6.0
    weights = jnp.array([0.3, 0.7])
    intercepts = np.array([1.0, 2.5])
    slopes = np.array([0.4, 0.15])
    savings = 2.0
    step = 1e-6

    def nu_at(s: float) -> float:
        nu, _ = ez_continuation(
            child_values=jnp.asarray(intercepts + slopes * s)[None, :],
            child_marginals=jnp.asarray(slopes)[None, :],
            weights=weights,
            risk_aversion=jnp.asarray(gamma),
        )
        return float(np.asarray(nu)[0])

    _, dnu_ds = ez_continuation(
        child_values=jnp.asarray(intercepts + slopes * savings)[None, :],
        child_marginals=jnp.asarray(slopes)[None, :],
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    finite_difference = (nu_at(savings + step) - nu_at(savings - step)) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(dnu_ds)[0], finite_difference, rtol=1e-6)


def test_continuation_is_continuous_at_unit_risk_aversion() -> None:
    """The pair `(nu, dnu/ds)` is continuous through `gamma = 1`.

    The generator switches to its logarithmic branch at unit risk aversion;
    values just below and just above must bracket the exact-limit value, not
    jump across the branch switch.
    """
    child_values = jnp.array([[1.0, 3.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])

    def pair(gamma: float) -> tuple[float, float]:
        nu, dnu = ez_continuation(
            child_values=child_values,
            child_marginals=child_marginals,
            weights=weights,
            risk_aversion=jnp.asarray(gamma),
        )
        return float(np.asarray(nu)[0]), float(np.asarray(dnu)[0])

    at_limit = pair(1.0)
    below = pair(1.0 - 1e-7)
    above = pair(1.0 + 1e-7)
    np.testing.assert_allclose(below, at_limit, rtol=1e-5)
    np.testing.assert_allclose(above, at_limit, rtol=1e-5)


def test_period_value_is_continuous_at_unit_eis() -> None:
    """`ez_period_value` is continuous through `rho = 1`.

    The CES aggregator switches to its Cobb-Douglas branch at unit elasticity;
    values just off the singular exponent must approach the limit value.
    """

    def value(rho: float) -> float:
        return float(
            ez_period_value(
                flow=jnp.asarray(2.0),
                nu=jnp.asarray(8.0),
                discount_factor=0.3,
                inverse_eis=rho,
            )
        )

    at_limit = value(1.0)
    np.testing.assert_allclose(value(1.0 - 1e-7), at_limit, rtol=1e-5)
    np.testing.assert_allclose(value(1.0 + 1e-7), at_limit, rtol=1e-5)


def test_gamma_equal_rho_is_expected_utility_in_the_transformed_index() -> None:
    """At `gamma = rho` the recursion is linear in `W = V^(1-rho)`, not in `V`.

    The transformed index satisfies
    `W = (1-beta) q^(1-rho) + beta E[W']`; composing `ez_continuation` and
    `ez_period_value` at `gamma = rho` must reproduce `V = W^(1/(1-rho))`
    exactly — and differ from the recursion with an arithmetic (linear-in-`V`)
    continuation.
    """
    gamma = rho = 2.0
    beta = 0.9
    flow = 1.5
    child_values = np.array([1.0, 3.0])
    weights = np.array([0.5, 0.5])
    nu, _ = ez_continuation(
        child_values=jnp.asarray(child_values)[None, :],
        child_marginals=jnp.zeros((1, 2)),
        weights=jnp.asarray(weights),
        risk_aversion=jnp.asarray(gamma),
    )
    value = ez_period_value(
        flow=jnp.asarray(flow),
        nu=nu,
        discount_factor=beta,
        inverse_eis=rho,
    )
    transformed = (1.0 - beta) * flow ** (1.0 - rho) + beta * np.sum(
        weights * child_values ** (1.0 - rho)
    )
    reference = transformed ** (1.0 / (1.0 - rho))
    np.testing.assert_allclose(np.asarray(value)[0], reference, rtol=1e-10)
    arithmetic_nu = np.sum(weights * child_values)
    assert not np.isclose(float(np.asarray(nu)[0]), arithmetic_nu)


def test_continuation_stays_finite_at_high_risk_aversion_in_float32() -> None:
    """The anchored kernel survives the high-`gamma` stress in float32.

    Raw transformed terms `V^(1-gamma)` at `gamma = 50` overflow float32 many
    orders of magnitude before float64; the anchored form keeps every exponent
    nonpositive on the value channel, so single precision returns the same
    certainty equivalent as the float64 reference.
    """
    gamma = 50.0
    values64 = np.array([1e-6, 2e-6])
    weights64 = np.array([0.5, 0.5])
    marginals64 = np.array([1.0, 1.0])
    nu64, dnu64 = ez_continuation(
        child_values=jnp.asarray(values64)[None, :],
        child_marginals=jnp.asarray(marginals64)[None, :],
        weights=jnp.asarray(weights64),
        risk_aversion=jnp.asarray(gamma),
    )
    nu32, dnu32 = ez_continuation(
        child_values=jnp.asarray(values64, dtype=jnp.float32)[None, :],
        child_marginals=jnp.asarray(marginals64, dtype=jnp.float32)[None, :],
        weights=jnp.asarray(weights64, dtype=jnp.float32),
        risk_aversion=jnp.asarray(gamma, dtype=jnp.float32),
    )
    assert np.asarray(nu32).dtype == np.float32
    assert np.isfinite(np.asarray(nu32)).all()
    assert np.isfinite(np.asarray(dnu32)).all()
    np.testing.assert_allclose(
        np.asarray(nu32), np.asarray(nu64, dtype=np.float32), rtol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(dnu32), np.asarray(dnu64, dtype=np.float32), rtol=1e-3
    )


def test_transform_partials_carry_the_generator_weighted_sums() -> None:
    """The anchored partials represent `S = sum_j w_j V_j^(1-gamma)` and `T`.

    De-scaling with `S = e^((1-gamma) a) S~` and `T = e^b T~` recovers the
    plain generator-weighted sums the joint-lottery theorem is stated in.
    """
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    anchor, scaled_value, marginal_log_scale, marginal_mantissa = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    exp_value = 0.5 * 1.0 ** (1.0 - gamma) + 0.5 * 2.0 ** (1.0 - gamma)
    exp_marginal = 0.5 * 1.0 ** (-gamma) * 0.5 + 0.5 * 2.0 ** (-gamma) * 0.25
    a = np.asarray(anchor)[0]
    np.testing.assert_allclose(
        np.exp((1.0 - gamma) * a) * np.asarray(scaled_value)[0], exp_value, rtol=1e-10
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(marginal_log_scale)[0]) * np.asarray(marginal_mantissa)[0],
        exp_marginal,
        rtol=1e-10,
    )


def test_transform_scalar_anchors_the_generator() -> None:
    """A certain value enters transform space as its own anchor: `(log V, 1)`."""
    gamma = 4.0
    value = jnp.asarray(2.0)
    anchor, scaled = ez_transform_scalar(value=value, risk_aversion=jnp.asarray(gamma))
    np.testing.assert_allclose(
        np.exp((1.0 - gamma) * np.asarray(anchor)) * np.asarray(scaled),
        2.0 ** (1.0 - gamma),
        rtol=1e-10,
    )


def test_transform_scalar_is_the_log_generator_at_unit_risk_aversion() -> None:
    """At `gamma = 1` the generator degenerates to `log V` with a zero anchor."""
    value = jnp.asarray(3.0)
    anchor, scaled = ez_transform_scalar(value=value, risk_aversion=jnp.asarray(1.0))
    np.testing.assert_allclose(np.asarray(anchor), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(scaled), np.log(3.0), rtol=1e-10)


def test_invert_partials_recovers_the_continuation_certainty_equivalent() -> None:
    """`ez_invert_partials(ez_transform_partials(.)) == ez_continuation(.)`.

    The single-regime certainty equivalent is the transform partials inverted
    with no intervening regime blend, so the two must agree exactly.
    """
    gamma = 4.0
    child_values = jnp.array([[1.0, 2.0]])
    child_marginals = jnp.array([[0.5, 0.25]])
    weights = jnp.array([0.5, 0.5])
    anchor, scaled_value, marginal_log_scale, marginal_mantissa = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=jnp.asarray(gamma),
    )
    nu, dnu_ds = ez_invert_partials(
        log_anchor=anchor,
        scaled_value=scaled_value,
        marginal_log_scale=marginal_log_scale,
        marginal_mantissa=marginal_mantissa,
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


def test_blend_partials_matches_the_single_joint_lottery() -> None:
    """Blending per-regime anchored partials equals one joint-lottery reduction.

    Two target regimes with probabilities `(p, 1-p)` and value scales twelve
    orders of magnitude apart blend into exactly the certainty equivalent of the
    concatenated `(regime x shock)` lottery — the re-anchoring in
    `ez_blend_partials` preserves additivity of the underlying transform sums
    without leaving the representable range.
    """
    gamma = 30.0
    prob = jnp.array([0.3, 0.7])
    values_a = jnp.array([1e-9, 3e-9])
    values_b = jnp.array([2e3, 5e3])
    marginals_a = jnp.array([0.5, 0.25])
    marginals_b = jnp.array([0.1, 0.05])
    weights = jnp.array([0.5, 0.5])
    partials = [
        ez_transform_partials(
            child_values=values[None, :],
            child_marginals=marginals[None, :],
            weights=weights,
            risk_aversion=jnp.asarray(gamma),
        )
        for values, marginals in ((values_a, marginals_a), (values_b, marginals_b))
    ]
    joint_anchor, blended_value, joint_marginal_scale, blended_mantissa = (
        ez_blend_partials(
            log_anchors=jnp.stack([p[0] for p in partials]),
            scaled_values=jnp.stack([p[1] for p in partials]),
            marginal_log_scales=jnp.stack([p[2] for p in partials]),
            marginal_mantissas=jnp.stack([p[3] for p in partials]),
            probs=prob[:, None],
            risk_aversion=jnp.asarray(gamma),
        )
    )
    nu, dnu_ds = ez_invert_partials(
        log_anchor=joint_anchor,
        scaled_value=blended_value,
        marginal_log_scale=joint_marginal_scale,
        marginal_mantissa=blended_mantissa,
        risk_aversion=jnp.asarray(gamma),
    )
    ref_nu, ref_dnu_ds = ez_continuation(
        child_values=jnp.concatenate([values_a, values_b])[None, :],
        child_marginals=jnp.concatenate([marginals_a, marginals_b])[None, :],
        weights=jnp.concatenate([0.3 * weights, 0.7 * weights]),
        risk_aversion=jnp.asarray(gamma),
    )
    assert np.isfinite(np.asarray(nu)).all()
    np.testing.assert_allclose(np.asarray(nu), np.asarray(ref_nu), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(dnu_ds), np.asarray(ref_dnu_ds), rtol=1e-9)


def test_single_node_transform_partial_matches_the_scalar_anchor() -> None:
    """A one-node lottery transforms to the stateless target's anchored pair.

    The continuation reader transforms a child with no own shock lottery via
    `ez_transform_partials` on a single unit-weight node; a stateless target
    enters through `ez_transform_scalar`. Both paths must carry identical
    transform-space contributions into the regime blend.
    """
    gamma = 4.0
    value = jnp.asarray(2.0)
    marginal = jnp.asarray(0.25)
    anchor, scaled_value, marginal_log_scale, marginal_mantissa = ez_transform_partials(
        child_values=value[None],
        child_marginals=marginal[None],
        weights=jnp.array([1.0]),
        risk_aversion=jnp.asarray(gamma),
    )
    scalar_anchor, scalar_scaled = ez_transform_scalar(
        value=value, risk_aversion=jnp.asarray(gamma)
    )
    np.testing.assert_allclose(
        np.asarray(anchor), np.asarray(scalar_anchor), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(scaled_value), np.asarray(scalar_scaled), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.exp(np.asarray(marginal_log_scale)) * np.asarray(marginal_mantissa),
        np.asarray(value ** (-gamma) * marginal),
        rtol=1e-10,
    )


def test_flow_power_structure_poisons_a_degenerate_euler_exponent() -> None:
    """`xi = phi (1-rho) - 1 = 0` yields a NaN exponent, not a spurious policy.

    At `xi = 0` the Euler equation is constant in consumption — the closed-form
    inversion `c = x^(1/xi)` is undefined. The exponent `phi` and the inverse
    EIS `rho` are runtime parameters, so the degenerate combination cannot be
    rejected at model build; the structure reader poisons the exponent with NaN
    so the solve's NaN fail-fast surfaces the (regime, period) instead of the
    inversion computing a finite but meaningless consumption.
    """
    log_flow_coefficient, flow_exponent = _ez_flow_power_structure(
        utility_of_action=lambda consumption: consumption**2,
        inverse_eis=jnp.asarray(0.5),
    )
    assert bool(jnp.isnan(flow_exponent))
    assert bool(jnp.isfinite(log_flow_coefficient))


def test_flow_power_structure_is_exact_away_from_the_degenerate_exponent() -> None:
    """For `q = c` and `rho = 2` the structure is `(log 1, -rho) = (0, -2)` exactly."""
    log_flow_coefficient, flow_exponent = _ez_flow_power_structure(
        utility_of_action=lambda consumption: consumption,
        inverse_eis=jnp.asarray(2.0),
    )
    np.testing.assert_allclose(np.asarray(log_flow_coefficient), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(flow_exponent), -2.0, rtol=1e-12)


def test_continuation_marginal_is_finite_for_gamma_below_one_extreme_ratio() -> None:
    """`gamma < 1` with extreme child-value ratios keeps `dnu/ds` finite.

    For `gamma < 1` the value channel is anchored to the largest child value,
    but the transform marginal `T = sum w V^(-gamma) dV/ds` is dominated by
    the *smallest* — the two channels need independent scaling. With child
    values spanning the normal float64 range, an anchor shared with the value
    channel puts `e^(gamma * spread)` (here `e^1368`) in the marginal's
    intermediate even though the exact `(nu, dnu/ds)` pair is comfortably
    representable. The reference is the plain log-domain arithmetic of the
    two-node lottery.
    """
    gamma = 0.99
    values = np.array([1e-300, 1e300])
    marginals = np.array([1e-290, 1.0])
    nu, dnu_ds = ez_continuation(
        child_values=jnp.asarray(values),
        child_marginals=jnp.asarray(marginals),
        weights=jnp.array([0.5, 0.5]),
        risk_aversion=gamma,
    )

    log_v = np.log(values)
    log_transformed = np.logaddexp(
        np.log(0.5) + (1.0 - gamma) * log_v[0],
        np.log(0.5) + (1.0 - gamma) * log_v[1],
    )
    log_nu = log_transformed / (1.0 - gamma)
    log_marginal_sum = np.logaddexp(
        np.log(0.5) - gamma * log_v[0] + np.log(marginals[0]),
        np.log(0.5) - gamma * log_v[1] + np.log(marginals[1]),
    )
    expected_nu = np.exp(log_nu)
    expected_dnu_ds = np.exp(gamma * log_nu + log_marginal_sum)
    assert bool(jnp.isfinite(dnu_ds))
    np.testing.assert_allclose(float(nu), expected_nu, rtol=1e-10)
    np.testing.assert_allclose(float(dnu_ds), expected_dnu_ds, rtol=1e-10)


def test_continuation_is_graceful_when_a_child_value_flushes_to_zero() -> None:
    """A zero (flushed-subnormal) child value never turns `dnu/ds` into NaN.

    Accelerator float32 flushes subnormal inputs to zero; the transform
    marginal must then drop that node exactly (its marginal is zero) rather
    than form `inf * 0`. The float32 result equals the float64 computation on
    the same effective inputs.
    """
    values_32 = jnp.array([1e-38, 1.0], dtype=jnp.float32)
    marginals_32 = jnp.array([1e-38, 1.0], dtype=jnp.float32)
    weights_32 = jnp.array([0.5, 0.5], dtype=jnp.float32)
    nu_32, dnu_ds_32 = ez_continuation(
        child_values=values_32,
        child_marginals=marginals_32,
        weights=weights_32,
        risk_aversion=0.99,
    )
    nu_64, dnu_ds_64 = ez_continuation(
        child_values=values_32.astype(jnp.float64),
        child_marginals=marginals_32.astype(jnp.float64),
        weights=weights_32.astype(jnp.float64),
        risk_aversion=0.99,
    )

    assert bool(jnp.isfinite(dnu_ds_32))
    np.testing.assert_allclose(float(nu_32), float(nu_64), rtol=1e-3)
    np.testing.assert_allclose(float(dnu_ds_32), float(dnu_ds_64), rtol=1e-3)


def test_period_value_is_computed_in_the_log_domain() -> None:
    """The CES aggregator stays exact where raw powers overflow float32.

    With `rho = 5` and inputs near `1e-10`, `flow^(1-rho)` is ~1e40 — past
    float32 — while the aggregated value (~1e-10) is comfortably representable.
    The log-domain reference is
    `exp(LSE(log(1-beta) + (1-rho) log q, log beta + (1-rho) log nu) / (1-rho))`.
    """
    rho = 5.0
    beta = 0.5
    flow = 1e-10
    nu = 2e-10

    got = ez_period_value(
        flow=jnp.asarray(flow, dtype=jnp.float32),
        nu=jnp.asarray(nu, dtype=jnp.float32),
        discount_factor=beta,
        inverse_eis=rho,
    )

    logs = np.array(
        [
            np.log(1.0 - beta) + (1.0 - rho) * np.log(flow),
            np.log(beta) + (1.0 - rho) * np.log(nu),
        ]
    )
    shift = logs.max()
    expected = np.exp((shift + np.log(np.exp(logs - shift).sum())) / (1.0 - rho))
    assert float(got) > 0.0
    np.testing.assert_allclose(float(got), expected, rtol=1e-4)


def test_consumption_from_euler_is_computed_in_the_log_domain() -> None:
    """The Euler inversion stays exact where `nu^(-rho)` overflows float32.

    With `nu = dnu/ds = 1e-10` and `rho = 5`, the Euler target `nu^(-rho) dnu/ds`
    is ~1e40 — past float32 — while the inverted consumption `1e-8` is
    comfortably representable.
    """
    got = ez_consumption_from_euler(
        nu=jnp.asarray(1e-10, dtype=jnp.float32),
        dnu_ds=jnp.asarray(1e-10, dtype=jnp.float32),
        discount_factor=0.5,
        inverse_eis=5.0,
        log_flow_coefficient=0.0,
        flow_exponent=-5.0,
    )

    log_target = -5.0 * np.log(1e-10) + np.log(1e-10)
    expected = np.exp(-log_target / 5.0)
    assert float(got) > 0.0
    np.testing.assert_allclose(float(got), expected, rtol=1e-4)


def test_marginal_of_resource_is_computed_in_the_log_domain() -> None:
    """The envelope marginal stays exact where `V^rho` underflows float32.

    With `V = 1e-10` and `rho = 5`, `V^rho = 1e-50` underflows float32 to zero,
    while the marginal `(1-beta) V^rho q_m` with a large Euler-form flow
    marginal (`1e30`, itself float32-representable) is ~5e-21 — comfortably
    representable.
    """
    got = ez_marginal_of_resource(
        log_flow_marginal=jnp.log(jnp.asarray(1e30, dtype=jnp.float32)),
        value=jnp.asarray(1e-10, dtype=jnp.float32),
        discount_factor=0.5,
        inverse_eis=5.0,
    )

    expected = 0.5 * np.exp(5.0 * np.log(1e-10) + np.log(1e30))
    assert float(got) > 0.0
    np.testing.assert_allclose(float(got), expected, rtol=1e-4)


def test_flow_power_structure_returns_a_finite_log_coefficient() -> None:
    """A tiny flow scale yields a finite log coefficient where the raw overflows.

    For `q = A c^phi` with `A = 1e-14` and `rho = 25`, the raw coefficient
    `A^(1-rho) phi` is ~1e336 — past float64 — while its logarithm (~773) is
    ordinary. The Euler inversion consumes the log form, so the exact solution
    `c = 1` (at `nu = A`, `dnu/ds = phi A`, `beta = 1/2`) comes out exactly.
    """
    scale = 1e-14
    power = 0.5
    rho = 25.0

    log_coefficient, flow_exponent = _ez_flow_power_structure(
        utility_of_action=lambda consumption: scale * consumption**power,
        inverse_eis=jnp.asarray(rho),
    )
    consumption = ez_consumption_from_euler(
        nu=jnp.asarray(scale),
        dnu_ds=jnp.asarray(power * scale),
        discount_factor=0.5,
        inverse_eis=rho,
        log_flow_coefficient=log_coefficient,
        flow_exponent=flow_exponent,
    )

    expected_log_coefficient = (1.0 - rho) * np.log(scale) + np.log(power)
    assert bool(jnp.isfinite(log_coefficient))
    np.testing.assert_allclose(
        float(log_coefficient), expected_log_coefficient, rtol=1e-12
    )
    np.testing.assert_allclose(float(consumption), 1.0, rtol=1e-12)


def test_marginal_of_resource_consumes_the_log_flow_marginal() -> None:
    """A large Euler-form flow marginal enters as its log, exact in float32.

    With `c = 1e-10` and `rho = 5` the raw flow marginal `c^(-rho)` is 1e50 —
    past float32 — while the resource marginal `(1-beta) V^rho c^(-rho)` is
    exactly 0.5 for `V = c`. Passing the marginal in log form keeps the whole
    chain inside the dtype.
    """
    rho = 5.0
    log_consumption = jnp.log(jnp.asarray(1e-10, dtype=jnp.float32))

    got = ez_marginal_of_resource(
        log_flow_marginal=-rho * log_consumption,
        value=jnp.asarray(1e-10, dtype=jnp.float32),
        discount_factor=0.5,
        inverse_eis=rho,
    )

    np.testing.assert_allclose(float(got), 0.5, rtol=1e-5)
