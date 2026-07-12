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
retains the savings-dependent transition-probability and nonlinear next-resource
terms that a policy-only formula would drop.

Reference: Alan Lujan, "The Endogenous Grid Method for Epstein-Zin Preferences,"
arXiv:2601.04438 (2026), direct route (his Section 2.2).
"""

import jax.numpy as jnp

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
    transformed_value, transformed_marginal = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=risk_aversion,
    )
    return ez_invert_partials(
        transformed_value=transformed_value,
        transformed_marginal=transformed_marginal,
        risk_aversion=risk_aversion,
    )


def ez_transform_partials(
    *,
    child_values: FloatND,
    child_marginals: FloatND,
    weights: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Reduce one continuation lottery to Epstein-Zin transformed partial sums.

    Reduces over the last axis (a continuation lottery — the stochastic node combo
    of one target regime) into the certainty-equivalent generator's transform space:

    - transformed value `S = sum_j w_j g(V_j)`, generator `g(V) = V^(1-gamma)` (or
      `g(V) = log V` at `gamma = 1`);
    - transformed marginal `T = sum_j w_j V_j^(-gamma) dV_j/ds`.

    These partials sum linearly across a regime lottery, so the joint certainty
    equivalent over the `(regime x shock)` lottery is `ez_invert_partials` applied
    to the regime-probability-weighted sum of the per-regime `(S, T)`. The generator
    and its weighting match the brute-force joint-lottery operator.

    Args:
        child_values: Strictly positive next-period values on the lottery axis.
        child_marginals: The next-period value derivatives `dV'/ds` on that axis.
        weights: Nonnegative lottery probabilities over the last axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the transformed value partial `S` and marginal partial `T`, each
        reduced over the last axis.

    """
    exponent = 1.0 - risk_aversion
    log_v = jnp.log(child_values)
    geometric = jnp.where(weights > 0.0, weights * log_v, 0.0)
    power = weights * child_values**exponent
    transformed_value = jnp.sum(jnp.where(exponent == 0.0, geometric, power), axis=-1)
    transformed_marginal = jnp.sum(
        weights * child_values ** (-risk_aversion) * child_marginals, axis=-1
    )
    return transformed_value, transformed_marginal


def ez_invert_partials(
    *,
    transformed_value: FloatND,
    transformed_marginal: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Invert Epstein-Zin transformed partial sums to `(nu, dnu/ds)`.

    The inverse of `ez_transform_partials`, applied after the regime-probability
    weighted sum of the per-regime partials: `nu = g^-1(S)` with
    `g^-1(x) = x^(1/(1-gamma))` (or `exp(x)` at `gamma = 1`), and the certainty
    equivalent's savings derivative `dnu/ds = nu^gamma * T`.

    Args:
        transformed_value: The transform-space value `S`, summed over the joint
            (regime x shock) lottery.
        transformed_marginal: The transform-space marginal `T`, summed over the
            same joint lottery.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the certainty equivalent `nu` and its savings derivative `dnu/ds`.

    """
    exponent = 1.0 - risk_aversion
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
    nu = jnp.where(
        exponent == 0.0,
        jnp.exp(transformed_value),
        transformed_value ** (1.0 / safe_exponent),
    )
    dnu_ds = nu**risk_aversion * transformed_marginal
    return nu, dnu_ds


def ez_transform_scalar(
    *, value: FloatND, risk_aversion: ScalarFloat | float
) -> FloatND:
    """Apply the Epstein-Zin generator `g` to a single certain continuation value.

    `g(V) = V^(1-gamma)` (or `log V` at `gamma = 1`). A stateless target regime — a
    terminal bequest constant with no savings derivative — contributes
    `p_r * g(const_r)` to the joint transformed value and nothing to the marginal.
    """
    exponent = 1.0 - risk_aversion
    return jnp.where(exponent == 0.0, jnp.log(value), value**exponent)


def ez_consumption_from_euler(
    *,
    nu: FloatND,
    dnu_ds: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
    flow_coefficient: FloatND | float,
    flow_exponent: ScalarFloat | float,
) -> FloatND:
    """Invert the Epstein-Zin Euler equation for consumption at a savings node.

    Solves `(1-beta) q_c(c) = beta nu^(-rho) dnu/ds` where the period-flow
    marginal is the single power `q^(-rho) q_c = flow_coefficient · c^flow_exponent`.
    For the basic single-good flow `q = c`, `flow_coefficient = 1` and
    `flow_exponent = -rho`. For the fixed-service Cobb-Douglas flow
    `q = c^phi s^(1-phi)`, `flow_coefficient = phi · s^((1-phi)(1-rho))` and
    `flow_exponent = phi(1-rho) - 1`.

    Args:
        nu: Certainty equivalent of the next-period value at the savings node.
        dnu_ds: Derivative of `nu` with respect to end-of-period savings.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.
        flow_coefficient: The constant multiplying `c^flow_exponent` in the
            period-flow marginal (depends on the fixed service level).
        flow_exponent: The power of `c` in the period-flow marginal
            (`phi(1-rho) - 1`, or `-rho` for the basic flow).

    Returns:
        The optimal consumption at the savings node.

    """
    target_marginal = (
        discount_factor * nu ** (-inverse_eis) * dnu_ds / (1.0 - discount_factor)
    )
    return (target_marginal / flow_coefficient) ** (1.0 / flow_exponent)


def ez_marginal_of_resource(
    *,
    flow_marginal: FloatND,
    value: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
) -> FloatND:
    """Return the envelope marginal value of the resource at an interior optimum.

    By the envelope theorem the derivative of the recursive value with respect to
    the Euler state (cash-on-hand `m`) is `dV/dm = (1-beta) V^rho (q^(-rho) q_c)`,
    where `flow_marginal = q^(-rho) q_c` is the period flow's Euler-form marginal
    and `rho` the inverse elasticity of intertemporal substitution. For a
    single-power flow `q = flow_coefficient**... c^phi`, `flow_marginal =
    flow_coefficient * c^flow_exponent` (`c^(-rho)` for the basic single-good flow
    `q = c`). Substituting the interior Euler equation `(1-beta) q^(-rho) q_c =
    beta nu^(-rho) dnu/ds` recovers the equivalent continuation form `V^rho beta
    nu^(-rho) dnu/ds`, so the marginal is consistent with the consumption the Euler
    inversion returns.

    Args:
        flow_marginal: The period flow's Euler-form marginal `q^(-rho) q_c` at the
            optimum (`flow_coefficient * c^flow_exponent`; `c^(-rho)` for `q = c`).
        value: The recursive value index `V` at the state.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The marginal value of the resource `dV/dm`.

    """
    return (1.0 - discount_factor) * value**inverse_eis * flow_marginal


def ez_period_value(
    *,
    flow: FloatND,
    nu: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
) -> FloatND:
    """Return the Epstein-Zin recursive value index at a state.

    `V = [(1-beta) flow^(1-rho) + beta nu^(1-rho)]^(1/(1-rho))`. The aggregator is
    a CES combination of the current-period flow and the continuation certainty
    equivalent; it stays strictly positive for strictly positive inputs, which
    the recursion (and the power-mean certainty equivalent) require.

    Args:
        flow: The current-period flow `q` (consumption in the single-good case).
        nu: The certainty equivalent of the next-period value.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The recursive value index.

    """
    one_minus_rho = 1.0 - inverse_eis
    aggregate = (
        1.0 - discount_factor
    ) * flow**one_minus_rho + discount_factor * nu**one_minus_rho
    return aggregate ** (1.0 / one_minus_rho)
