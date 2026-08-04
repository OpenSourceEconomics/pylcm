"""Stable evaluation of the weighted power mean.

Both nonlinear pieces of an Epstein-Zin recursion are weighted power means
over the same kernel, differing only in what they average and at which
exponent:

- the certainty equivalent `PowerMean` averages the continuation lottery at
  exponent `1 - risk_aversion`, weighted by the lottery's probabilities;
- the Koopmans aggregator `W_epstein_zin` averages `(utility, CE)` at
  exponent `1 - 1/psi`, weighted by `(1 - discount_factor, discount_factor)`.

`weighted_power_mean` reduces a lottery of any length over its last axis.
`weighted_power_mean_of_pair` is the same derivation written out for the
two-node case, which is what the Koopmans aggregator needs: a trailing axis
of length two reduces poorly, and carrying the pair as two arrays avoids
materializing that axis at all. The two agree to a few ulps everywhere,
which `test_W_epstein_zin_equals_the_general_power_mean` pins.
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
    # `exponent` broadcasts against the reduced shape, so it needs the lottery
    # axis inserted before it multiplies `centered`. Without it, ordinary
    # broadcasting aligns the exponent's trailing dimension with the lottery
    # nodes: same length and every node silently gets the wrong exponent,
    # different length and the multiply raises.
    scaled = jnp.expand_dims(exponent, axis=-1) * centered
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


def weighted_power_mean_of_pair(
    *,
    first: FloatND,
    second: FloatND,
    first_weight: FloatND,
    second_weight: FloatND,
    exponent: FloatND,
) -> FloatND:
    """Return the weighted power mean of exactly two values, stably.

    The arithmetic is `weighted_power_mean`'s, written out for a two-node
    lottery so the pair never has to be stacked into a trailing axis of
    length two. Every guarantee stated there holds here — the anchored log
    form, the two moment representations, the exact geometric-mean limit at
    `exponent == 0`, zero weights dropping out exactly, a NaN weight
    propagating, and a massless pair averaging to NaN.

    Args:
        first: Nonnegative first value.
        second: Nonnegative second value, broadcast against `first`.
        first_weight: Nonnegative weight on `first`.
        second_weight: Nonnegative weight on `second`.
        exponent: The power, broadcast against the two values.

    Returns:
        The weighted power mean of the pair.

    """
    first_live = first_weight > 0.0
    second_live = second_weight > 0.0
    log_first = jnp.log(jnp.where(first_live, first, 1.0))
    log_second = jnp.log(jnp.where(second_live, second, 1.0))
    safe_exponent = jnp.where(exponent == 0.0, 1.0, exponent)

    first_anchorable = first_live & jnp.isfinite(log_first)
    second_anchorable = second_live & jnp.isfinite(log_second)
    anchor_high = jnp.maximum(
        jnp.where(first_anchorable, log_first, -jnp.inf),
        jnp.where(second_anchorable, log_second, -jnp.inf),
    )
    anchor_low = jnp.minimum(
        jnp.where(first_anchorable, log_first, jnp.inf),
        jnp.where(second_anchorable, log_second, jnp.inf),
    )
    anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    anchor = jnp.where(jnp.isfinite(anchor), anchor, 0.0)
    anchor = jnp.where(exponent == 0.0, 0.0, anchor)

    masked_first_weight = jnp.where(
        first_live | jnp.isnan(first_weight), first_weight, 0.0
    )
    masked_second_weight = jnp.where(
        second_live | jnp.isnan(second_weight), second_weight, 0.0
    )
    weight_sum = masked_first_weight + masked_second_weight
    safe_weight = jnp.where(weight_sum > 0.0, weight_sum, jnp.nan)
    normalized_first = masked_first_weight / safe_weight
    normalized_second = masked_second_weight / safe_weight

    scaled_first = exponent * (log_first - anchor)
    scaled_second = exponent * (log_second - anchor)
    deviation_ratio = normalized_first * jnp.expm1(
        scaled_first
    ) + normalized_second * jnp.expm1(scaled_second)
    moment = normalized_first * jnp.exp(scaled_first) + normalized_second * jnp.exp(
        scaled_second
    )
    near_geometric = deviation_ratio > _DEVIATION_RATIO_CROSSOVER
    log_moment = jnp.where(
        near_geometric,
        jnp.log1p(jnp.where(near_geometric, deviation_ratio, 0.0)),
        jnp.log(jnp.where(near_geometric, 1.0, moment)),
    )
    log_mean_power = anchor + log_moment / safe_exponent
    log_mean_geometric = normalized_first * log_first + normalized_second * log_second
    return jnp.exp(jnp.where(exponent == 0.0, log_mean_geometric, log_mean_power))
