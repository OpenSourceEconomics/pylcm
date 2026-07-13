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
    anchor, scaled_value, scaled_marginal = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=risk_aversion,
    )
    return ez_invert_partials(
        log_anchor=anchor,
        scaled_value=scaled_value,
        scaled_marginal=scaled_marginal,
        risk_aversion=risk_aversion,
    )


def ez_transform_partials(
    *,
    child_values: FloatND,
    child_marginals: FloatND,
    weights: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND]:
    """Reduce one continuation lottery to anchored Epstein-Zin partial sums.

    Reduces over the last axis (a continuation lottery — the stochastic node combo
    of one target regime) into the certainty-equivalent generator's transform
    space, represented in the log domain against a per-lottery anchor `a` so the
    powers never leave the exponent range:

    - transformed value `S = sum_j w_j V_j^(1-gamma) = e^((1-gamma) a) * S~` with
      `S~ = sum_j w_j e^((1-gamma)(log V_j - a))`;
    - transformed marginal `T = sum_j w_j V_j^(-gamma) dV_j/ds = e^(-gamma a) * T~`
      with `T~ = sum_j w_j e^(-gamma (log V_j - a)) dV_j/ds`.

    The anchor is the positive-weight nodes' extremal log value on the side that
    keeps every `(1-gamma)(log V_j - a)` nonpositive (the smallest value dominates
    `S` for `gamma > 1`, the largest for `gamma < 1`), so `S~` sums terms in
    `(0, 1]` and `T~` stays scaled to the dominating node. At `gamma = 1` the
    generator is `log V`: the anchor is zero and the scaled value is the plain
    weighted sum `sum_j w_j log V_j`.

    Anchored partials from different lotteries combine additively through
    `ez_blend_partials`, so the joint certainty equivalent over the
    `(regime x shock)` lottery is `ez_invert_partials` applied to the
    regime-probability-weighted blend. The generator and its weighting match the
    brute-force joint-lottery operator.

    Args:
        child_values: Strictly positive next-period values on the lottery axis.
        child_marginals: The next-period value derivatives `dV'/ds` on that axis.
        weights: Nonnegative lottery probabilities over the last axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the log anchor `a`, the anchored value partial `S~`, and the
        anchored marginal partial `T~`, each reduced over the last axis.

    """
    exponent = 1.0 - risk_aversion
    log_v = jnp.log(child_values)
    positive = weights > 0.0
    anchor_high = jnp.max(jnp.where(positive, log_v, -jnp.inf), axis=-1)
    anchor_low = jnp.min(jnp.where(positive, log_v, jnp.inf), axis=-1)
    anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    anchor = jnp.where(exponent == 0.0, 0.0, anchor)
    centered = log_v - anchor[..., None]
    power_terms = weights * jnp.exp(exponent * centered)
    log_terms = weights * log_v
    scaled_value = jnp.sum(
        jnp.where(
            positive,
            jnp.where(exponent == 0.0, log_terms, power_terms),
            weights * 0.0,
        ),
        axis=-1,
    )
    scaled_marginal = jnp.sum(
        jnp.where(
            positive,
            weights * jnp.exp(-risk_aversion * centered) * child_marginals,
            weights * 0.0,
        ),
        axis=-1,
    )
    return anchor, scaled_value, scaled_marginal


def ez_blend_partials(
    *,
    log_anchors: FloatND,
    scaled_values: FloatND,
    scaled_marginals: FloatND,
    probs: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND]:
    """Blend per-target anchored partials with the regime probabilities.

    Reduces over the leading axis (the target regimes of one regime lottery).
    Each target `r` contributes `p_r * S_r` and `p_r * T_r` to the joint
    transform-space sums; in the anchored representation that is a re-anchoring
    to the targets' joint extremal anchor followed by the probability-weighted
    sum, with every re-anchoring factor `e^((1-gamma)(a_r - a))` and
    `e^(-gamma (a_r - a))` at most one by the anchor choice. Zero-probability
    targets contribute exactly zero and are excluded from the joint anchor
    (`p * 0` keeps a NaN probability poisoning the sum). At `gamma = 1` the
    anchors are all zero and the blend is the plain probability-weighted sum.

    Args:
        log_anchors: Per-target log anchors, stacked on the leading axis.
        scaled_values: Per-target anchored value partials `S~_r`, same stacking.
        scaled_marginals: Per-target anchored marginal partials `T~_r` (zero for
            a stateless target), same stacking.
        probs: Regime transition probabilities over the leading axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the joint log anchor, the blended value partial, and the
        blended marginal partial.

    """
    exponent = 1.0 - risk_aversion
    reachable = probs > 0.0
    anchor_high = jnp.max(jnp.where(reachable, log_anchors, -jnp.inf), axis=0)
    anchor_low = jnp.min(jnp.where(reachable, log_anchors, jnp.inf), axis=0)
    joint_anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    joint_anchor = jnp.where(exponent == 0.0, 0.0, joint_anchor)
    shift = log_anchors - joint_anchor
    blended_value = jnp.sum(
        jnp.where(
            reachable,
            probs * jnp.exp(exponent * shift) * scaled_values,
            probs * 0.0,
        ),
        axis=0,
    )
    blended_marginal = jnp.sum(
        jnp.where(
            reachable,
            probs * jnp.exp(-risk_aversion * shift) * scaled_marginals,
            probs * 0.0,
        ),
        axis=0,
    )
    return joint_anchor, blended_value, blended_marginal


def ez_invert_partials(
    *,
    log_anchor: FloatND,
    scaled_value: FloatND,
    scaled_marginal: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Invert anchored Epstein-Zin partial sums to `(nu, dnu/ds)`.

    The inverse of the anchored transform, applied after the regime-probability
    blend: with `S = e^((1-gamma) a) S~` and `T = e^(-gamma a) T~`,

    - `nu = S^(1/(1-gamma)) = e^a * S~^(1/(1-gamma))` (or `exp(S~)` at
      `gamma = 1`, where the anchor is zero);
    - `dnu/ds = nu^gamma T = S~^(gamma/(1-gamma)) T~` (or `nu * T~` at
      `gamma = 1`).

    Every power acts on the anchored `S~` — a sum of terms at most one each —
    so the inversion stays finite wherever the exact certainty equivalent is
    representable.

    Args:
        log_anchor: The joint log anchor `a`.
        scaled_value: The anchored transform-space value `S~`, blended over the
            joint (regime x shock) lottery.
        scaled_marginal: The anchored transform-space marginal `T~`, blended over
            the same joint lottery.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the certainty equivalent `nu` and its savings derivative `dnu/ds`.

    """
    exponent = 1.0 - risk_aversion
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
    nu = jnp.where(
        exponent == 0.0,
        jnp.exp(scaled_value),
        jnp.exp(log_anchor) * scaled_value ** (1.0 / safe_exponent),
    )
    dnu_ds = (
        jnp.where(
            exponent == 0.0,
            nu,
            scaled_value ** (risk_aversion / safe_exponent),
        )
        * scaled_marginal
    )
    return nu, dnu_ds


def ez_transform_scalar(
    *, value: FloatND, risk_aversion: ScalarFloat | float
) -> tuple[FloatND, FloatND]:
    """Anchor a single certain continuation value in the generator's transform space.

    A stateless target regime — a terminal bequest constant with no savings
    derivative — contributes `p_r * g(const_r)` to the joint transformed value and
    nothing to the marginal. In the anchored representation that is the pair
    `(log V, 1)` (the value is its own anchor, so `S = e^((1-gamma) log V) * 1`),
    or `(0, log V)` at `gamma = 1` where the generator is the plain logarithm.
    """
    exponent = 1.0 - risk_aversion
    anchor = jnp.where(exponent == 0.0, 0.0, jnp.log(value))
    scaled = jnp.where(exponent == 0.0, jnp.log(value), jnp.ones_like(value))
    return anchor, scaled


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

    `V = [(1-beta) flow^(1-rho) + beta nu^(1-rho)]^(1/(1-rho))`, with the
    Cobb-Douglas limit `flow^(1-beta) nu^beta` at unit elasticity (`rho = 1`),
    matching `H_epstein_zin`. The aggregator is a CES combination of the
    current-period flow and the continuation certainty equivalent; it stays
    strictly positive for strictly positive inputs, which the recursion (and
    the power-mean certainty equivalent) require.

    Args:
        flow: The current-period flow `q` (consumption in the single-good case).
        nu: The certainty equivalent of the next-period value.
        discount_factor: The discount factor `beta`.
        inverse_eis: The inverse elasticity of intertemporal substitution `rho`.

    Returns:
        The recursive value index.

    """
    one_minus_rho = 1.0 - inverse_eis
    # The unselected CES branch must not divide by zero at `rho = 1`.
    safe_one_minus_rho = jnp.where(one_minus_rho == 0.0, 1.0, one_minus_rho)
    cobb_douglas = flow ** (1.0 - discount_factor) * nu**discount_factor
    aggregate = (1.0 - discount_factor) * flow**safe_one_minus_rho + (
        discount_factor
    ) * nu**safe_one_minus_rho
    ces = aggregate ** (1.0 / safe_one_minus_rho)
    return jnp.where(one_minus_rho == 0.0, cobb_douglas, ces)
