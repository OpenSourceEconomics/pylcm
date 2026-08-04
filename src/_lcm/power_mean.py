"""Stable evaluation of the weighted power mean.

Both nonlinear pieces of an Epstein-Zin recursion are weighted power means
over the same kernel, differing only in what they average and at which
exponent:

- the certainty equivalent `PowerMean` averages the continuation lottery at
  exponent `1 - risk_aversion`, weighted by the lottery's probabilities;
- the Koopmans aggregator `W_epstein_zin` averages `(utility, CE)` at
  exponent `1 - 1/psi`, weighted by `(1 - discount_factor, discount_factor)`.

`weighted_power_mean` is the single evaluation both route through, so a
range that one survives the other survives too.
"""

import jax.numpy as jnp

from lcm.typing import FloatND

# Deviation ratio at which `weighted_power_mean` switches from the `log1p`
# moment representation to the `log` one. Either side of it both are accurate
# to a couple of ulps: `log1p` degrades only as the ratio approaches `-1` and
# `log` only as it approaches `0`, so any interior crossover works.
_DEVIATION_RATIO_CROSSOVER = -0.5


def weighted_power_mean(
    *,
    values: FloatND,
    weights: FloatND,
    exponent: FloatND,
) -> FloatND:
    """Return `(Σ w̃ · v^p)^(1/p)` over the last axis, stably.

    `p` is `exponent` and `w̃` the mass-normalized weights. The naive form
    overflows when `p` is negative and `v` is small: the intermediate `v^p`
    exceeds the dtype's range and the mean collapses to zero or infinity.
    The evaluation is anchored in the log domain instead,
    `log mean = a + log M / p`, with `a` the extremal log value and `M` the
    moment of the anchored, mass-normalized lottery.

    `M` has two representations — an `expm1`-deviation sum that survives
    `p -> 0` and an `exp` sum that survives a lottery carrying near-zero mass
    on the anchor — and the evaluation takes whichever one the lottery does
    not cancel. The result therefore stays finite wherever the mathematical
    value is, and the geometric-mean limit stays exact arbitrarily close to
    `p = 0`.

    The geometric mean is selected by an exact `exponent == 0` test, so at
    that one point the result carries no dependence on `exponent` and its
    derivative there reads as zero rather than the true finite value.
    Gradient-based work that starts exactly there should offset the starting
    value.

    Args:
        values: Strictly positive values along the last axis. A value of
            exactly zero is admitted as the limiting case: it sends the mean
            to zero at negative exponents and contributes nothing at positive
            ones.
        weights: Nonnegative weights over `values`, broadcast against them.
            They are normalized by their sum, so scaling them all by a
            constant leaves the result unchanged and a lottery carrying no
            mass averages to NaN. Zero-weight entries drop out exactly, while
            a NaN weight propagates.
        exponent: The power, broadcast against the reduced shape. `0` is the
            weighted geometric mean `exp(E[log v])`, `1` the arithmetic mean.

    Returns:
        The weighted power mean, reduced over the last axis.

    """
    live = weights > 0.0
    # A node carrying no weight must not be able to inject a non-finite
    # quantity into the reductions. `log(0)` is `-inf` and its derivative
    # `1/0` is infinite, and a weight of exactly zero cancels neither —
    # forward it would take a `jnp.where` in every reduction, and in
    # reverse mode the masked cotangent would still be `0 * inf`. Replacing
    # the value itself keeps both directions finite, and leaves the result
    # unchanged because the node's normalized weight is exactly zero.
    log_v = jnp.log(jnp.where(live, values, 1.0))
    # The `exponent == 0` power branch must not divide by zero.
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)
    # Anchored form: with `a` the extremal log value on the side that
    # keeps every scaled exponent nonpositive,
    # `log mean = a + log M / p` where `M = sum (w/W) exp(p (log v - a))`
    # over the mass-normalized lottery.
    #
    # Only a node with a finite log value may anchor. A live node at value
    # zero has `log v = -inf`, and anchoring there would make its own
    # `log v - a` the indeterminate `-inf - (-inf)`. Anchored to a finite
    # node instead, that value's scaled exponent is `+inf` at a negative
    # exponent, the moment diverges, and the mean goes to zero — which is
    # what `v^p -> inf` gives.
    anchorable = live & jnp.isfinite(log_v)
    anchor_high = jnp.max(jnp.where(anchorable, log_v, -jnp.inf), axis=-1)
    anchor_low = jnp.min(jnp.where(anchorable, log_v, jnp.inf), axis=-1)
    anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    # With no anchorable node the reductions sit at their sentinels; any
    # finite anchor will do, and zero keeps `centered` well defined.
    anchor = jnp.where(jnp.isfinite(anchor), anchor, 0.0)
    anchor = jnp.where(exponent == 0.0, 0.0, anchor)
    centered = log_v - anchor[..., None]
    broadcast_live = jnp.broadcast_to(live, centered.shape)
    broadcast_weights = jnp.broadcast_to(weights, centered.shape)
    # A NaN weight is deliberately kept rather than masked away: a
    # malformed lottery must surface as NaN, not silently lose a branch.
    masked_weights = jnp.where(
        broadcast_live | jnp.isnan(broadcast_weights), broadcast_weights, 0.0
    )
    weight_sum = jnp.sum(masked_weights, axis=-1)
    # A lottery carrying no mass has no mean. The linear path's `jnp.average`
    # reports the same NaN, so either aggregation reaches the NaN diagnostics
    # on the same model.
    safe_weight = jnp.where(weight_sum > 0.0, weight_sum, jnp.nan)
    normalized_weights = masked_weights / safe_weight[..., None]
    scaled = exponent * centered
    # The moment `M = Σ w̃ exp(p (log v - a))` has two representations, each
    # exact where the other cancels. Both reduce the same anchored lottery,
    # whose terms all lie in `[0, 1]` with the anchor's own term at exactly
    # its weight — so neither can overflow, and only a value of exactly zero
    # can send a term to infinity:
    # - `log1p(Σ w̃ expm1(...))` sums small negatives, so it survives
    #   `p -> 0`, where `log(M)` would be a rounded `0/0` against the
    #   exponent and the geometric-mean limit would be lost.
    # - `log(Σ w̃ exp(...))` sums strict positives, so it survives a wide
    #   lottery carrying near-zero mass on the anchor, where every other
    #   `expm1` rounds to `-1`, the deviation ratio rounds to `-1` too,
    #   and `log1p` would report a mathematically positive moment as zero.
    deviation_ratio = jnp.sum(normalized_weights * jnp.expm1(scaled), axis=-1)
    moment = jnp.sum(normalized_weights * jnp.exp(scaled), axis=-1)
    # `jnp.where` evaluates both branches, so each has to stay finite even
    # where it is discarded — an infinity in the dead branch would still
    # reach a gradient.
    near_geometric = deviation_ratio > _DEVIATION_RATIO_CROSSOVER
    safe_deviation_ratio = jnp.where(near_geometric, deviation_ratio, 0.0)
    safe_moment = jnp.where(near_geometric, 1.0, moment)
    log_moment = jnp.where(
        near_geometric,
        jnp.log1p(safe_deviation_ratio),
        jnp.log(safe_moment),
    )
    log_mean_power = anchor + log_moment / safe_exponent
    log_mean_geometric = jnp.sum(normalized_weights * log_v, axis=-1)
    return jnp.exp(jnp.where(exponent == 0.0, log_mean_geometric, log_mean_power))
