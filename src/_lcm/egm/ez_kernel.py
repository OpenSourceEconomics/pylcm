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
    anchor, weight_sum, scaled_value, marginal_log_scale, marginal_mantissa = (
        ez_transform_partials(
            child_values=child_values,
            child_marginals=child_marginals,
            weights=weights,
            risk_aversion=risk_aversion,
        )
    )
    return ez_invert_partials(
        log_anchor=anchor,
        weight_sum=weight_sum,
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
) -> tuple[FloatND, FloatND, FloatND, FloatND, FloatND]:
    """Reduce one continuation lottery to anchored Epstein-Zin partial sums.

    Reduces over the last axis (a continuation lottery — the stochastic node combo
    of one target regime) into the certainty-equivalent generator's transform
    space. The two channels need independent scaling — for `gamma < 1` the value
    sum is dominated by the *largest* child value while the marginal sum is
    dominated by the *smallest* — so each carries its own log scale:

    - transformed value `S = sum_j w_j V_j^(1-gamma) = e^((1-gamma) a) (W + E)`,
      carried as the weight sum `W = sum_j w_j` and the *deviation* sum
      `E = sum_j w_j expm1((1-gamma)(log V_j - a))`. The anchor `a` is the
      positive-weight nodes' extremal log value on the side that keeps every
      exponent nonpositive (the smallest value dominates for `gamma > 1`, the
      largest for `gamma < 1`), so `E` sums terms in `[-w_j, 0]`. Splitting off
      `W` keeps the information that survives division by `1-gamma` at the
      inversion — near `gamma = 1` the sum `W + E` rounds to `W` while `E`
      still carries the first-order generator mean exactly. At `gamma = 1` the
      generator is `log V`: the anchor is zero and the deviation slot holds the
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
    brute-force joint-lottery operator; the inversion de-scales the true mass
    `W + E` as `log(W) + log1p(E/W)`, treating a weight sum within summation
    roundoff of one as exactly normalized, while the near-unit generator
    information rides in the deviation ratio.

    Args:
        child_values: Strictly positive next-period values on the lottery axis.
        child_marginals: The next-period value derivatives `dV'/ds` on that axis.
        weights: Nonnegative lottery probabilities over the last axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the value log anchor `a`, the weight sum `W`, the anchored
        deviation sum `E` (the weighted log generator at `gamma = 1`), the
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
    deviation_terms = weights * jnp.expm1(exponent * centered)
    log_terms = weights * log_v
    # Broadcast against the value batch so the weight channel carries the same
    # leading axes as the deviation channel (weights are often a plain 1-D
    # probability vector shared across the batch).
    masked_weights = jnp.where(positive, weights, weights * 0.0)
    weight_sum = jnp.sum(jnp.broadcast_to(masked_weights, centered.shape), axis=-1)
    scaled_value = jnp.sum(
        jnp.where(
            positive,
            jnp.where(exponent == 0.0, log_terms, deviation_terms),
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
    return anchor, weight_sum, scaled_value, marginal_log_scale, marginal_mantissa


def ez_blend_partials(
    *,
    log_anchors: FloatND,
    weight_sums: FloatND,
    scaled_values: FloatND,
    marginal_log_scales: FloatND,
    marginal_mantissas: FloatND,
    probs: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND, FloatND, FloatND, FloatND]:
    """Blend per-target anchored partials with the regime probabilities.

    Reduces over the leading axis (the target regimes of one regime lottery).
    Each target `r` contributes `p_r * S_r` and `p_r * T_r` to the joint
    transform-space sums. On the value channel — carried as the weight/deviation
    split `S_r = e^((1-gamma) a_r) (W_r + E_r)` — the re-anchoring to the joint
    extremal anchor distributes as

    `e^((1-gamma) s_r) (W_r + E_r) = W_r + [expm1((1-gamma) s_r) W_r
    + e^((1-gamma) s_r) E_r]`

    with `s_r = a_r - a` on the side that keeps every factor at most one, so
    the joint weight sum stays the plain `sum_r p_r W_r` and the deviation slot
    keeps the first-order generator information that survives the near-unit
    inversion. At `gamma = 1` the anchors are all zero, the shifts vanish, and
    the same formula reduces to the plain probability-weighted sums. On the
    marginal channel the joint log scale is `max_r [log p_r + b_r]` over
    contributing targets, so every re-scaled mantissa keeps magnitude at most
    one. Zero-probability targets contribute exactly zero and are excluded
    from both joint scales (`p * 0` keeps a NaN probability poisoning the
    sums).

    Args:
        log_anchors: Per-target value log anchors, stacked on the leading axis.
        weight_sums: Per-target weight sums `W_r`, same stacking.
        scaled_values: Per-target anchored deviation sums `E_r` (weighted log
            generators at `gamma = 1`), same stacking.
        marginal_log_scales: Per-target marginal log scales `b_r`, same
            stacking (zero for a stateless target).
        marginal_mantissas: Per-target marginal mantissas `T~_r` (zero for a
            stateless target), same stacking.
        probs: Regime transition probabilities over the leading axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the joint value log anchor, the blended weight sum, the
        blended deviation sum, the joint marginal log scale, and the blended
        marginal mantissa.

    """
    exponent = 1.0 - risk_aversion
    reachable = probs > 0.0
    anchor_high = jnp.max(jnp.where(reachable, log_anchors, -jnp.inf), axis=0)
    anchor_low = jnp.min(jnp.where(reachable, log_anchors, jnp.inf), axis=0)
    joint_anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    joint_anchor = jnp.where(exponent == 0.0, 0.0, joint_anchor)
    shift = log_anchors - joint_anchor
    growth = jnp.expm1(exponent * shift)
    blended_weight = jnp.sum(
        jnp.where(reachable, probs * weight_sums, probs * 0.0), axis=0
    )
    blended_value = jnp.sum(
        jnp.where(
            reachable,
            probs * (growth * weight_sums + (growth + 1.0) * scaled_values),
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
    return (
        joint_anchor,
        blended_weight,
        blended_value,
        joint_marginal_scale,
        blended_mantissa,
    )


def ez_invert_partials(
    *,
    log_anchor: FloatND,
    weight_sum: FloatND,
    scaled_value: FloatND,
    marginal_log_scale: FloatND,
    marginal_mantissa: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Invert anchored Epstein-Zin partial sums to `(nu, dnu/ds)`.

    The inverse of the anchored transform, applied after the regime-probability
    blend: with `S = e^((1-gamma) a) (W + E)` and `T = e^b T~`,

    - `nu = S^(1/(1-gamma))`, computed as
      `log nu = a + [log(W) + log1p(E / W)] / (1-gamma)` (or `log nu = E / W`
      at `gamma = 1`, where the deviation slot holds the weighted log
      generator). Splitting `log(W + E)` into the mass term and the `log1p`
      deviation ratio keeps the quotient exact through the `gamma -> 1`
      limit: one ULP from unit risk aversion, `E / W` carries the first-order
      generator mean that a rounded `log(W + E)` would lose to cancellation
      before the division.
    - The mass term is gated on how far `W` sits from one:
      - within sqrt(eps) — floating summation roundoff on a mathematically
        unit-mass lottery — the lottery inverts as exactly normalized
        (`log(W)` dropped, marginal divided by `W`), because the power mean
        has a finite `gamma -> 1` limit only at unit mass and a roundoff gap
        would otherwise blow up as `log(W)/(1-gamma)`;
      - materially away from one, the exact `log(W)` contribution is kept
        (a fixed-`gamma` sub-probability operator), and the `gamma = 1`
        branch publishes the normalized geometric pair, the only finite
        choice there.
    - `dnu/ds = nu^gamma T = e^(gamma log nu + b) * T~`, divided by `W`
      exactly where the value channel is normalized so the pair stays
      consistent.

    All scale arithmetic happens on log quantities and the mantissa `T~` keeps
    magnitude of order one, so the inversion stays finite wherever the exact
    `(nu, dnu/ds)` pair is representable.

    Args:
        log_anchor: The joint value log anchor `a`.
        weight_sum: The blended weight sum `W`.
        scaled_value: The blended deviation sum `E` (the weighted log
            generator at `gamma = 1`), blended over the joint (regime x shock)
            lottery.
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
    safe_weight = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    # A mass gap below sqrt(eps) is floating summation roundoff on a
    # mathematically unit-mass lottery (quadrature weights rarely sum to one
    # bit-exactly), not economic mass: the raw `log(W)/(1-gamma)` term would
    # amplify it to an order-one error near `gamma = 1`, while its true effect
    # on the certainty equivalent is below sqrt(eps) relative at any gamma.
    # Such lotteries invert as exactly normalized (`W = 1`, deviations `E/W`);
    # a materially non-unit mass keeps its exact `log(W)` contribution.
    roundoff_mass = (
        jnp.abs(weight_sum - 1.0) <= jnp.sqrt(jnp.finfo(safe_weight.dtype).eps)
    )
    log_mass = jnp.where(roundoff_mass, 0.0, jnp.log(safe_weight))
    log_nu = jnp.where(
        exponent == 0.0,
        scaled_value / safe_weight,
        log_anchor + (log_mass + jnp.log1p(scaled_value / safe_weight)) / safe_exponent,
    )
    nu = jnp.exp(log_nu)
    # Where the value channel is normalized — the roundoff snap, and the
    # `gamma = 1` branch (whose unnormalized form has no finite value) — the
    # marginal divides by the same mass so `(nu, dnu/ds)` stay an exact pair.
    mass_normalizer = jnp.where(
        roundoff_mass | (exponent == 0.0), safe_weight, 1.0
    )
    dnu_ds = (
        jnp.exp(risk_aversion * log_nu + marginal_log_scale)
        * marginal_mantissa
        / mass_normalizer
    )
    return nu, dnu_ds


def ez_transform_scalar(
    *, value: FloatND, risk_aversion: ScalarFloat | float
) -> tuple[FloatND, FloatND, FloatND]:
    """Anchor a single certain continuation value in the generator's transform space.

    A stateless target regime — a terminal bequest constant with no savings
    derivative — contributes `p_r * g(const_r)` to the joint transformed value and
    nothing to the marginal. In the weight/deviation representation that is the
    triple `(log V, 1, 0)` (the value is its own anchor with unit weight and
    zero deviation, so `S = e^((1-gamma) log V) * (1 + 0)`), or `(0, 1, log V)`
    at `gamma = 1` where the deviation slot holds the log generator.
    """
    exponent = 1.0 - risk_aversion
    anchor = jnp.where(exponent == 0.0, 0.0, jnp.log(value))
    scaled = jnp.where(exponent == 0.0, jnp.log(value), jnp.zeros_like(value))
    return anchor, jnp.ones_like(value), scaled


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
    # Log-domain CES with two complementary stable forms:
    # - the deviation form `log V = log q + log1p(beta expm1(e d)) / e` with
    #   `d = log nu - log q` is algebraically exact and keeps the quotient
    #   accurate arbitrarily close to `rho = 1` (a rounded log-sum divided by a
    #   near-zero `e` loses the Cobb-Douglas limit to cancellation);
    # - the log-sum-exp form covers the deviation form's one blind spot,
    #   `e d` past the dtype's exp range, where `expm1` overflows while the
    #   aggregate is still representable.
    deviation = safe_one_minus_rho * (log_nu - log_flow)
    deviation_form = (
        log_flow
        + jnp.log1p(discount_factor * jnp.expm1(deviation)) / safe_one_minus_rho
    )
    lse_form = (
        jnp.logaddexp(
            jnp.log1p(-discount_factor) + safe_one_minus_rho * log_flow,
            jnp.log(discount_factor) + safe_one_minus_rho * log_nu,
        )
        / safe_one_minus_rho
    )
    exp_range = 0.9 * jnp.log(jnp.finfo(jnp.result_type(flow)).max)
    ces = jnp.exp(jnp.where(deviation > exp_range, lse_form, deviation_form))
    return jnp.where(one_minus_rho == 0.0, cobb_douglas, ces)
