"""Epstein-Zin EGM kernel — closed-form consumption inversion and value update.

The recursive value satisfies
`V = [(1-beta) q^(1-rho) + beta nu^(1-rho)]^(1/(1-rho))`, where `q` is the period
flow, `rho` the inverse elasticity of intertemporal substitution, `beta` the
discount factor, and `nu` the certainty equivalent of the next-period value over
the joint continuation lottery (`PowerMean.aggregate`).

Conditional on `nu` and its end-of-period-savings derivative `dnu/ds`, the
first-order condition `(1-beta) q^(-rho) q_c = beta nu^(-rho) dnu/ds` inverts for
consumption in closed form whenever `q^(-rho) q_c` is a single power of `c` — the
basic single-good flow and the fixed-service Cobb-Douglas flow both qualify. The
inversion carries the certainty equivalent's savings derivative directly, so it
retains the nonlinear next-resource terms (the composed `dR'/ds` gradients) that
a policy-only formula would drop. Transition probabilities and quadrature
weights must be savings-independent (`dP/ds = 0`): the transform marginal `T`
carries no probability-derivative term.

Reference: Alan Lujan, "The Endogenous Grid Method for Epstein-Zin Preferences,"
arXiv:2601.04438 (2026), direct route (his Section 2.2).

The `exponent == 0.0` and `one_minus_rho == 0.0` tests are exact on purpose,
against the project's usual rule for comparing floats. Each selects the limiting
closed form a *user-supplied parameter* asks for — the log / Cobb-Douglas limit —
and the parameter arrives as the literal the user wrote, so the question is which
of two `jnp.where` branches applies rather than whether two computed quantities
are close. `jnp.where` evaluates both, so the unselected branch is what would
otherwise produce the `log(0)` or `0 ** 0`; a tolerance would route a genuinely
small-but-nonzero elasticity into the wrong formula.
"""

import jax.numpy as jnp

from _lcm.power_mean import weighted_power_mean, weighted_power_mean_of_pair
from lcm.typing import FloatND, ScalarFloat


def ez_continuation(
    *,
    child_values: FloatND,
    child_marginals: FloatND,
    weights: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Aggregate the continuation certainty equivalent and its savings derivative.

    Reduces over the last axis (the continuation lottery — the joint stochastic
    node and target regime). The certainty equivalent is the power mean
    `nu = (E[V'^(1-gamma)])^(1/(1-gamma))`, evaluated in the log domain so it stays
    finite near the borrowing constraint. Its savings derivative reweights each
    child's marginal by the child's risk-transformed value share,
    `dnu/ds = sum_j w_j (nu/V_j')^gamma * dV_j'/ds = nu^gamma * E[V'^(-gamma) dV'/ds]`,
    computed through the value ratio to keep the powers near one.

    This is the exogenous-probability form: the transition weights do not depend
    on end-of-period savings (`dP/ds = 0`), which holds whenever next-period
    uncertainty is an exogenous shock or regime lottery. `risk_aversion = 0`
    recovers the linear pair `(E[V'], E[dV'/ds])`.

    Args:
        child_values: Strictly positive next-period values on the continuation
            lottery, reduced over the last axis.
        child_marginals: The next-period value derivatives `dV'/ds` on the same
            lottery axis.
        weights: Nonnegative lottery probabilities over the last axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the certainty equivalent `nu` and its savings derivative
        `dnu/ds`, each reduced over the last axis.

    """
    certainty_equivalent, weight_sum, marginal_log_scale, marginal_mantissa = (
        ez_transform_partials(
            child_values=child_values,
            child_marginals=child_marginals,
            weights=weights,
            risk_aversion=risk_aversion,
        )
    )
    return ez_invert_partials(
        certainty_equivalent=certainty_equivalent,
        weight_sum=weight_sum,
        marginal_log_scale=marginal_log_scale,
        marginal_mantissa=marginal_mantissa,
        risk_aversion=risk_aversion,
    )


def ez_transform_partials(
    *,
    child_values: FloatND,
    child_marginals: FloatND,
    weights: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Reduce one continuation lottery to stable Epstein-Zin partials.

    The value channel is the public power mean itself, not a second numerical
    implementation of the same operator. The remaining channels carry the
    lottery mass and a signed log-scaled derivative statistic. These partials
    compose across regime targets and streamed node blocks without retaining
    the underlying lottery.
    """
    exponent = jnp.asarray(1.0 - risk_aversion)
    broadcast_weights = jnp.broadcast_to(weights, child_values.shape)
    positive = broadcast_weights > 0.0
    masked_weights = jnp.where(positive, broadcast_weights, broadcast_weights * 0.0)
    weight_sum = jnp.sum(masked_weights, axis=-1)
    certainty_equivalent = weighted_power_mean(
        values=child_values,
        weights=broadcast_weights,
        exponent=exponent,
        shifts=jnp.zeros_like(broadcast_weights, dtype=jnp.int32),
    )

    # A NaN marginal contributes because `NaN != 0`; zero-probability and
    # zero-marginal nodes drop out exactly. The signed mantissa keeps
    # cancellation visible without forming powers that overflow.
    contributing = positive & (child_marginals != 0.0)
    log_v = jnp.log(child_values)
    log_magnitude = jnp.where(
        contributing,
        jnp.log(jnp.where(positive, broadcast_weights, 1.0))
        - risk_aversion * log_v
        + jnp.log(jnp.abs(child_marginals)),
        -jnp.inf,
    )
    peak = jnp.max(log_magnitude, axis=-1)
    marginal_log_scale = jnp.where(jnp.isfinite(peak), peak, 0.0)
    marginal_mantissa = jnp.sum(
        jnp.where(
            contributing,
            jnp.sign(child_marginals)
            * jnp.exp(log_magnitude - marginal_log_scale[..., None]),
            broadcast_weights * 0.0,
        ),
        axis=-1,
    )
    return (
        certainty_equivalent,
        weight_sum,
        marginal_log_scale,
        marginal_mantissa,
    )


def ez_blend_partials(
    *,
    certainty_equivalents: FloatND,
    weight_sums: FloatND,
    marginal_log_scales: FloatND,
    marginal_mantissas: FloatND,
    probs: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Blend per-target partials into the joint continuation lottery.

    Reduces over the leading target-regime axis. Associativity of a power mean
    means each target lottery can be represented by its own certainty equivalent
    and mass: target `r` receives joint weight `p_r W_r`. The shared public
    reduction preserves the same zero, rare-node, and geometric limiting
    semantics as `PowerMean.aggregate`.
    """
    reachable = probs > 0.0
    joint_weights = jnp.where(reachable, probs * weight_sums, probs * 0.0)
    blended_weight = jnp.sum(joint_weights, axis=0)
    moved_weights = jnp.moveaxis(joint_weights, 0, -1)
    joint_certainty_equivalent = weighted_power_mean(
        values=jnp.moveaxis(certainty_equivalents, 0, -1),
        weights=moved_weights,
        exponent=jnp.asarray(1.0 - risk_aversion),
        shifts=jnp.zeros_like(moved_weights, dtype=jnp.int32),
    )

    contributing = reachable & (marginal_mantissas != 0.0)
    candidate = jnp.where(
        contributing,
        jnp.log(jnp.where(reachable, probs, 1.0)) + marginal_log_scales,
        -jnp.inf,
    )
    peak = jnp.max(candidate, axis=0)
    joint_marginal_scale = jnp.where(jnp.isfinite(peak), peak, 0.0)
    blended_mantissa = jnp.sum(
        jnp.where(
            contributing,
            marginal_mantissas * jnp.exp(candidate - joint_marginal_scale),
            probs * 0.0,
        ),
        axis=0,
    )
    return (
        joint_certainty_equivalent,
        blended_weight,
        joint_marginal_scale,
        blended_mantissa,
    )


def ez_invert_partials(
    *,
    certainty_equivalent: FloatND,
    weight_sum: FloatND,
    marginal_log_scale: FloatND,
    marginal_mantissa: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Return a partial lottery certainty equivalent and savings derivative."""
    safe_weight = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    log_nu = jnp.log(certainty_equivalent)
    dnu_ds = (
        jnp.exp(risk_aversion * log_nu + marginal_log_scale)
        * marginal_mantissa
        / safe_weight
    )
    return certainty_equivalent, dnu_ds


def ez_transform_scalar(
    *, value: FloatND, risk_aversion: ScalarFloat | float
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Represent a certain stateless continuation as one unit-mass partial."""
    del risk_aversion
    zero = jnp.zeros_like(value)
    return value, jnp.ones_like(value), zero, zero


def ez_consumption_from_euler(
    *,
    nu: FloatND,
    dnu_ds: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
    log_flow_coefficient: FloatND | float,
    flow_exponent: ScalarFloat | float,
) -> FloatND:
    """Invert the Epstein-Zin Euler equation for consumption at a savings node.

    Solves `(1-beta) q_c(c) = beta nu^(-rho) dnu/ds` where the period-flow
    marginal is the single power `q^(-rho) q_c = kappa · c^flow_exponent`,
    with the coefficient supplied as `log_flow_coefficient = log(kappa)` —
    the raw `kappa = A^(1-rho) phi` overflows the dtype long before the
    inverted consumption does, so it is never materialized. For the basic
    single-good flow `q = c`, `log_flow_coefficient = 0` and
    `flow_exponent = -rho`. For the fixed-service Cobb-Douglas flow
    `q = c^phi s^(1-phi)`, `log_flow_coefficient =
    (1-phi)(1-rho) log(s) + log(phi)` and `flow_exponent = phi(1-rho) - 1`.

    Args:
        nu: Certainty equivalent of the next-period value at the savings node.
        dnu_ds: Derivative of `nu` with respect to end-of-period savings.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.
        log_flow_coefficient: The log of the constant multiplying
            `c^flow_exponent` in the period-flow marginal.
        flow_exponent: The power of `c` in the period-flow marginal
            (`phi(1-rho) - 1`, or `-rho` for the basic flow).

    Returns:
        The optimal consumption at the savings node.

    """
    # Log-domain inversion: the Euler target `nu^(-rho) dnu/ds` overflows the
    # dtype long before the inverted consumption does (small `nu` and large
    # `rho` push the target past the exponent range while `c` stays ordinary).
    # `dnu_ds = 0` reads `log(0) = -inf` and inverts to the same limit as the
    # raw power; a negative `dnu_ds` reads NaN and poisons the candidate.
    log_target = (
        jnp.log(discount_factor)
        - jnp.log1p(-discount_factor)
        - inverse_eis * jnp.log(nu)
        + jnp.log(dnu_ds)
    )
    return jnp.exp((log_target - log_flow_coefficient) / flow_exponent)


def ez_marginal_of_resource(
    *,
    log_flow_marginal: FloatND,
    value: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
) -> FloatND:
    """Return the envelope marginal value of the resource at an interior optimum.

    By the envelope theorem the derivative of the recursive value with respect to
    the Euler state (cash-on-hand `m`) is `dV/dm = (1-beta) V^rho (q^(-rho) q_c)`,
    where `q^(-rho) q_c` is the period flow's Euler-form marginal and `rho` the
    inverse elasticity of intertemporal substitution. The marginal enters as its
    logarithm — for a single-power flow,
    `log_flow_marginal = log_flow_coefficient + flow_exponent * log(c)`
    (`-rho log(c)` for the basic single-good flow `q = c`) — because the raw
    power leaves the dtype's range long before `dV/dm` does. Substituting the
    interior Euler equation `(1-beta) q^(-rho) q_c = beta nu^(-rho) dnu/ds`
    recovers the equivalent continuation form `V^rho beta nu^(-rho) dnu/ds`, so
    the marginal is consistent with the consumption the Euler inversion returns.

    Args:
        log_flow_marginal: The log of the period flow's Euler-form marginal
            `q^(-rho) q_c` at the optimum.
        value: The recursive value index `V` at the state.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The marginal value of the resource `dV/dm`.

    """
    # Log-domain product: `V^rho` underflows the dtype long before the
    # marginal `(1-beta) V^rho q_m` does (its factors' exponents cancel).
    return jnp.exp(
        jnp.log1p(-discount_factor) + inverse_eis * jnp.log(value) + log_flow_marginal
    )


def ez_period_value(
    *,
    flow: FloatND,
    nu: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
) -> FloatND:
    """Return the Epstein-Zin recursive value index at a state.

    `V = [(1-beta) flow^(1-rho) + beta nu^(1-rho)]^(1/(1-rho))`, with the
    Cobb-Douglas limit `flow^(1-beta) nu^beta` at unit elasticity (`rho = 1`).
    The aggregator is a CES combination of the current-period flow and the
    continuation certainty equivalent; it stays strictly positive for strictly
    positive inputs, which the recursion (and the power-mean certainty
    equivalent) require.

    This is the weighted power mean of `(flow, nu)` at weights
    `(1 - beta, beta)` and exponent `1 - rho`, evaluated by the same routine
    the public `CESAggregator` calls. Both therefore publish one cardinal
    value bit for bit, and the kernel inherits the anchored log form, the
    exact geometric-mean limit at unit elasticity, and the weight rescaling
    that routine documents.

    Args:
        flow: The current-period flow `q` (consumption in the single-good case).
        nu: The certainty equivalent of the next-period value.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The recursive value index.

    """
    beta = jnp.asarray(discount_factor)
    return weighted_power_mean_of_pair(
        first=flow,
        second=nu,
        first_weight=1.0 - beta,
        second_weight=beta,
        exponent=jnp.asarray(1.0 - inverse_eis),
    )
