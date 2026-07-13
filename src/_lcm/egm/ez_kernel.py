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
    anchor, scaled_value, marginal_log_scale, marginal_mantissa = ez_transform_partials(
        child_values=child_values,
        child_marginals=child_marginals,
        weights=weights,
        risk_aversion=risk_aversion,
    )
    return ez_invert_partials(
        log_anchor=anchor,
        scaled_value=scaled_value,
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
    """Reduce one continuation lottery to anchored Epstein-Zin partial sums.

    Reduces over the last axis (a continuation lottery — the stochastic node combo
    of one target regime) into the certainty-equivalent generator's transform
    space. The two channels need independent scaling — for `gamma < 1` the value
    sum is dominated by the *largest* child value while the marginal sum is
    dominated by the *smallest* — so each carries its own log scale:

    - transformed value `S = sum_j w_j V_j^(1-gamma) = e^((1-gamma) a) * S~` with
      `S~ = sum_j w_j e^((1-gamma)(log V_j - a))`; the anchor `a` is the
      positive-weight nodes' extremal log value on the side that keeps every
      exponent nonpositive (the smallest value dominates for `gamma > 1`, the
      largest for `gamma < 1`), so `S~` sums terms in `(0, 1]`. At `gamma = 1`
      the generator is `log V`: the anchor is zero and the scaled value is the
      plain weighted sum `sum_j w_j log V_j`.
    - transformed marginal `T = sum_j w_j V_j^(-gamma) dV_j/ds = e^b * T~`, held
      as a signed mantissa against its own log scale
      `b = max_j [log w_j - gamma log V_j + log |dV_j/ds|]`, so every mantissa
      term has magnitude at most one whatever the value spread. A lottery whose
      marginals are all zero reads `(b, T~) = (0, 0)`.

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
        Tuple of the value log anchor `a`, the anchored value partial `S~`, the
        marginal log scale `b`, and the marginal mantissa `T~`, each reduced
        over the last axis.

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
    # `contributing` keeps NaN marginals in (`NaN != 0` is true), so a poisoned
    # carry still propagates; zero-probability and zero-marginal nodes drop out
    # exactly.
    contributing = positive & (child_marginals != 0.0)
    log_magnitude = jnp.where(
        contributing,
        jnp.log(jnp.where(positive, weights, 1.0))
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
            weights * 0.0,
        ),
        axis=-1,
    )
    return anchor, scaled_value, marginal_log_scale, marginal_mantissa


def ez_blend_partials(
    *,
    log_anchors: FloatND,
    scaled_values: FloatND,
    marginal_log_scales: FloatND,
    marginal_mantissas: FloatND,
    probs: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND, FloatND]:
    """Blend per-target anchored partials with the regime probabilities.

    Reduces over the leading axis (the target regimes of one regime lottery).
    Each target `r` contributes `p_r * S_r` and `p_r * T_r` to the joint
    transform-space sums. On the value channel that is a re-anchoring to the
    targets' joint extremal anchor followed by the probability-weighted sum,
    with every re-anchoring factor `e^((1-gamma)(a_r - a))` at most one by the
    anchor choice; at `gamma = 1` the anchors are all zero and the blend is the
    plain probability-weighted sum. On the marginal channel the joint log scale
    is `max_r [log p_r + b_r]` over contributing targets, so every re-scaled
    mantissa keeps magnitude at most one. Zero-probability targets contribute
    exactly zero and are excluded from both joint scales (`p * 0` keeps a NaN
    probability poisoning the sums).

    Args:
        log_anchors: Per-target value log anchors, stacked on the leading axis.
        scaled_values: Per-target anchored value partials `S~_r`, same stacking.
        marginal_log_scales: Per-target marginal log scales `b_r`, same
            stacking (zero for a stateless target).
        marginal_mantissas: Per-target marginal mantissas `T~_r` (zero for a
            stateless target), same stacking.
        probs: Regime transition probabilities over the leading axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the joint value log anchor, the blended value partial, the
        joint marginal log scale, and the blended marginal mantissa.

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
    return joint_anchor, blended_value, joint_marginal_scale, blended_mantissa


def ez_invert_partials(
    *,
    log_anchor: FloatND,
    scaled_value: FloatND,
    marginal_log_scale: FloatND,
    marginal_mantissa: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Invert anchored Epstein-Zin partial sums to `(nu, dnu/ds)`.

    The inverse of the anchored transform, applied after the regime-probability
    blend: with `S = e^((1-gamma) a) S~` and `T = e^b T~`,

    - `nu = S^(1/(1-gamma))`, computed as
      `log nu = a + log(S~) / (1-gamma)` (or `log nu = S~` at `gamma = 1`,
      where the generator is the plain logarithm);
    - `dnu/ds = nu^gamma T = e^(gamma log nu + b) * T~`.

    All scale arithmetic happens on log quantities and the mantissa `T~` keeps
    magnitude of order one, so the inversion stays finite wherever the exact
    `(nu, dnu/ds)` pair is representable.

    Args:
        log_anchor: The joint value log anchor `a`.
        scaled_value: The anchored transform-space value `S~`, blended over the
            joint (regime x shock) lottery.
        marginal_log_scale: The joint marginal log scale `b`, blended over the
            same joint lottery.
        marginal_mantissa: The transform-space marginal mantissa `T~`, blended
            over the same joint lottery.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the certainty equivalent `nu` and its savings derivative `dnu/ds`.

    """
    exponent = 1.0 - risk_aversion
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
    log_nu = jnp.where(
        exponent == 0.0,
        scaled_value,
        log_anchor + jnp.log(scaled_value) / safe_exponent,
    )
    nu = jnp.exp(log_nu)
    dnu_ds = jnp.exp(risk_aversion * log_nu + marginal_log_scale) * marginal_mantissa
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
    return jnp.exp((log_target - jnp.log(flow_coefficient)) / flow_exponent)


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
    # Log-domain product: `V^rho` underflows the dtype long before the
    # marginal `(1-beta) V^rho q_m` does (its factors' exponents cancel).
    # A zero flow marginal reads `log(0) = -inf` and returns exactly zero.
    return jnp.exp(
        jnp.log1p(-discount_factor)
        + inverse_eis * jnp.log(value)
        + jnp.log(flow_marginal)
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
    log_flow = jnp.log(flow)
    log_nu = jnp.log(nu)
    cobb_douglas = jnp.exp(
        (1.0 - discount_factor) * log_flow + discount_factor * log_nu
    )
    # Log-domain CES: `flow^(1-rho)` overflows the dtype long before the
    # aggregated value does (curved aggregators push the raw powers past the
    # exponent range while `V` stays between its two inputs).
    ces = jnp.exp(
        jnp.logaddexp(
            jnp.log1p(-discount_factor) + safe_one_minus_rho * log_flow,
            jnp.log(discount_factor) + safe_one_minus_rho * log_nu,
        )
        / safe_one_minus_rho
    )
    return jnp.where(one_minus_rho == 0.0, cobb_douglas, ces)
