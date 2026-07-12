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
from jax.scipy.special import logsumexp

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
    log_v = jnp.log(child_values)
    exponent = 1.0 - risk_aversion
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
    log_nu_power = logsumexp(jnp.log(weights) + exponent * log_v, axis=-1) / (
        safe_exponent
    )
    log_nu_geometric = jnp.sum(jnp.where(weights > 0.0, weights * log_v, 0.0), axis=-1)
    nu = jnp.exp(jnp.where(exponent == 0.0, log_nu_geometric, log_nu_power))
    value_share = (nu[..., None] / child_values) ** risk_aversion
    dnu_ds = jnp.sum(weights * value_share * child_marginals, axis=-1)
    return nu, dnu_ds


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
    flow: FloatND,
    value: FloatND,
    discount_factor: ScalarFloat | float,
    inverse_eis: ScalarFloat | float,
) -> FloatND:
    """Return the envelope marginal value of the resource at an interior optimum.

    By the envelope theorem the derivative of the recursive value with respect to
    the Euler state (cash-on-hand `m`) is `dV/dm = (1-beta) V^rho flow^(-rho)`,
    where `flow` is the period consumption good (`q = c` in the basic single-good
    case) and `rho` the inverse elasticity of intertemporal substitution.
    Substituting the interior Euler equation `(1-beta) flow^(-rho) = beta
    nu^(-rho) dnu/ds` recovers the equivalent continuation form
    `V^rho beta nu^(-rho) dnu/ds`, so the marginal is consistent with the
    consumption the Euler inversion returns.

    Args:
        flow: The period flow at the optimum (consumption in the basic case).
        value: The recursive value index `V` at the state.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The marginal value of the resource `dV/dm`.

    """
    return (1.0 - discount_factor) * value**inverse_eis * flow ** (-inverse_eis)


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
