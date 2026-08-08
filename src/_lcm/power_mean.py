"""Stable evaluation of the weighted power mean.

Both nonlinear pieces of an Epstein-Zin recursion are weighted power means
over the same kernel, differing only in what they average and at which
exponent:

- the certainty equivalent `PowerMean` averages the continuation lottery at
  exponent `1 - risk_aversion`, weighted by the lottery's probabilities;
- the Koopmans aggregator `CESAggregator` averages `(utility, CE)` at
  exponent `1 - 1/psi`, weighted by `(1 - discount_factor, discount_factor)`.

`weighted_power_mean` reduces a lottery of any length over its last axis.
`weighted_power_mean_of_pair` is the same derivation written out for the
two-node case, which is what the Koopmans aggregator needs: a trailing axis
of length two reduces poorly, and carrying the pair as two arrays avoids
materializing that axis at all. The two agree to a few ulps everywhere,
which `test_CESAggregator_equals_the_general_power_mean` pins.
"""

import jax.numpy as jnp

from _lcm.probability import (
    is_live,
    is_negative,
    rescaled_lottery_weights,
    rescaled_weight_pair,
)
from lcm.typing import BoolND, FloatND

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

    Three exact points carry a derivative the primal does not justify, all
    from a branch selected by an exact test. Gradient-based work that would
    start at one of them should offset the starting value:

    - at `exponent == 0` the result carries no dependence on `exponent`, so
      that derivative reads as zero rather than its true finite value;
    - a weight of exactly zero is replaced by a constant, so the derivative
      with respect to it reads as zero — which makes `dW/d(discount_factor)`
      zero at `discount_factor` of exactly `0` or `1`;
    - a value of exactly zero has `log v = -inf`, and while the primal is the
      documented limit, the reverse-mode gradient is NaN for *every* input of
      the mean, not only for that value.

    Args:
        values: Strictly positive values along the last axis. A value of
            exactly zero is admitted as the limiting case: it sends the mean
            to zero at negative exponents and contributes nothing at positive
            ones.
        weights: Nonnegative weights over `values`, broadcast against them.
            They are normalized by their sum, so scaling them all by a
            constant leaves the result unchanged and a lottery carrying no
            mass averages to NaN. Zero-weight entries drop out exactly, while
            a NaN weight propagates. So does a negative one, which is not a
            lottery rather than a dead node.

            That invariance is a property of the mean, not a licence for the
            caller. Where the weights are probabilities that are supposed to
            sum to one, normalization divides any lost mass straight back out
            and leaves no trace in the result, so the total has to be checked
            before the call — `_lcm.regime_building.Q_and_F` does so for the
            continuation lottery.
        exponent: The power, broadcast against the reduced shape. `0` is the
            weighted geometric mean `exp(E[log v])`, `1` the arithmetic mean.

    Returns:
        The weighted power mean, reduced over the last axis.

    """
    # The mean depends on the weights only through their ratios, so lifting the
    # whole lottery onto a scale the dtype can multiply changes nothing about
    # the answer and everything about whether it can be computed: a weight
    # below the normal range is flushed as an operand, and its term — which
    # after the `v^p` transform can be of order one however small the weight —
    # would drop out of the moment entirely.
    weights = rescaled_lottery_weights(weights)
    # A negative weight is not a lottery. Mapping it to NaN makes it propagate
    # like any other malformed weight instead of being read as "dead" by the
    # liveness test below and silently dropped — which would return the
    # surviving nodes' mean as though the caller had asked for it.
    weights = jnp.where(is_negative(weights), jnp.nan, weights)
    live = is_live(weights)
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
    #
    # The claim above — that the anchor's own term sits at exactly its weight,
    # so the moment cannot underflow — needs that weight to be a normal number.
    # Anchored on a node the dtype cannot multiply, every term of the moment
    # would be below the smallest normal, the sum would flush to exactly zero,
    # `log M` would be `-inf`, and the mean would come back as an infinity for
    # a lottery whose answer is ordinary. The rescaling above is what makes it
    # hold: no live weight is subnormal by the time any node anchors.
    anchorable = live & jnp.isfinite(log_v)
    anchor_high = jnp.max(jnp.where(anchorable, log_v, -jnp.inf), axis=-1)
    anchor_low = jnp.min(jnp.where(anchorable, log_v, jnp.inf), axis=-1)
    anchor = jnp.where(exponent >= 0.0, anchor_high, anchor_low)
    # With no anchorable node the reductions sit at their sentinels; any
    # finite anchor will do, and zero keeps `centered` well defined.
    anchor = jnp.where(jnp.isfinite(anchor), anchor, 0.0)
    anchor = jnp.where(exponent == 0.0, 0.0, anchor)
    # A dead node's value was replaced by `1.0` above, so its `log v` is `0` —
    # a point the anchor knows nothing about, since only live nodes may anchor.
    # Left alone it can sit arbitrarily far above the anchored range, overflow
    # `exp` to infinity, and turn its own `0 * inf` contribution into a NaN that
    # takes the whole reduction with it. Pinning it to the anchor puts its term
    # at exactly `1`, which its zero weight then cancels exactly.
    centered = jnp.where(live, log_v - anchor[..., None], 0.0)
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
    #
    # The mass division comes after the sum in both, not before it. Normalizing
    # each weight first divides the rarest node's weight by the total and can
    # put it back below the normal range, undoing the rescaling above.
    deviation_ratio = jnp.sum(masked_weights * jnp.expm1(scaled), axis=-1) / safe_weight
    # `jnp.where` evaluates both branches, so each has to stay finite even
    # where it is discarded — an infinity in the dead branch would still
    # reach a gradient.
    near_geometric = deviation_ratio > _DEVIATION_RATIO_CROSSOVER
    safe_deviation_ratio = jnp.where(near_geometric, deviation_ratio, 0.0)
    log_moment = jnp.where(
        near_geometric,
        jnp.log1p(safe_deviation_ratio),
        _log_moment(
            log_weights=jnp.log(jnp.where(broadcast_live, masked_weights, 1.0)),
            scaled=scaled,
            live=broadcast_live,
            weights=broadcast_weights,
            weight_sum=safe_weight,
            discarded=near_geometric,
        ),
    )
    log_mean_power = anchor + log_moment / safe_exponent
    log_mean_geometric = jnp.sum(masked_weights * log_v, axis=-1) / safe_weight
    return jnp.exp(jnp.where(exponent == 0.0, log_mean_geometric, log_mean_power))


def _log_moment(
    *,
    log_weights: FloatND,
    scaled: FloatND,
    live: BoolND,
    weights: FloatND,
    weight_sum: FloatND,
    discarded: BoolND,
) -> FloatND:
    """Return `log(Σ w exp(s) / Σ w)` without forming `exp(s)` on its own.

    The anchoring keeps every `exp(s)` at or below one, which stops the moment
    overflowing, but not underflowing: a lottery whose values span more than
    the dtype's exponent range — a harmonic mean of a value at `1` beside one
    at `tiny` is exactly that — sends the far node's `exp(s)` below the
    smallest subnormal, where it becomes zero. The node lost that way is not
    negligible. It is the one the anchor measures everything else against, and
    dropping it moves the mean by its whole share, which for the two-node
    lottery above is a factor of two.

    Adding `log w` before exponentiating puts each term at the scale it
    actually contributes at, so a term is only lost when the *product* is
    genuinely below the format, and the peak subtraction leaves at least one
    term at exactly one.

    Args:
        log_weights: `log w` at every live node, and anything finite elsewhere.
        scaled: The anchored exponents `p (log v - a)`.
        live: Which nodes carry weight.
        weights: The weights themselves, read only for their NaNs.
        weight_sum: The mass to divide by, already NaN where there is none.
        discarded: Where the caller takes its other branch, and this one only
            has to stay finite for the gradient.

    Returns:
        The log moment of the mass-normalized, anchored lottery.

    """
    # A dead node contributes nothing; a NaN weight is not a probability and
    # has to reach the result rather than be dropped as one.
    log_terms = jnp.where(
        live,
        log_weights + scaled,
        jnp.where(jnp.isnan(weights), jnp.nan, -jnp.inf),
    )
    log_terms = jnp.where(discarded[..., None], 0.0, log_terms)
    peak = jnp.max(log_terms, axis=-1, keepdims=True)
    safe_peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    log_sum = jnp.squeeze(safe_peak, axis=-1) + jnp.log(
        jnp.sum(jnp.exp(log_terms - safe_peak), axis=-1)
    )
    return log_sum - jnp.log(weight_sum)


def _log_term(
    *,
    weight: FloatND,
    masked_weight: FloatND,
    scaled: FloatND,
    live: BoolND,
    discarded: BoolND,
) -> FloatND:
    """Return `log(w) + s` at a live node, `-inf` at a dead one, NaN at a NaN."""
    log_weight = jnp.log(jnp.where(live, masked_weight, 1.0))
    term = jnp.where(
        live, log_weight + scaled, jnp.where(jnp.isnan(weight), jnp.nan, -jnp.inf)
    )
    return jnp.where(discarded, 0.0, term)


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
    propagating, and a massless pair averaging to NaN. Its gradient carries
    the same three exact-point caveats, so `dW/d(discount_factor)` reads as
    zero at a discount factor of exactly `0` or `1`.

    Args:
        first: Nonnegative first value.
        second: Nonnegative second value, broadcast against `first`.
        first_weight: Nonnegative weight on `first`. A negative weight is
            malformed and propagates as NaN.
        second_weight: Nonnegative weight on `second`, same contract.
        exponent: The power, broadcast against the two values.

    Returns:
        The weighted power mean of the pair.

    """
    # One common power of two puts both weights on a scale the dtype can
    # multiply, leaving their ratio — all the mean depends on — untouched.
    first_weight, second_weight = rescaled_weight_pair(first_weight, second_weight)
    # A negative weight is malformed and propagates as NaN rather than reading
    # as a dead node — see `weighted_power_mean`. For `CESAggregator` this is
    # what a `discount_factor` outside `[0, 1]` produces, and dropping the node
    # would silently return the other argument unchanged.
    first_weight = jnp.where(is_negative(first_weight), jnp.nan, first_weight)
    second_weight = jnp.where(is_negative(second_weight), jnp.nan, second_weight)
    first_live = is_live(first_weight)
    second_live = is_live(second_weight)
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

    # Dead nodes are pinned to the anchor for the reason `weighted_power_mean`
    # states: their substituted `log v = 0` is outside the anchored range and
    # would overflow `exp`, and `0 * inf` is NaN, not the zero their weight says.
    scaled_first = exponent * jnp.where(first_live, log_first - anchor, 0.0)
    scaled_second = exponent * jnp.where(second_live, log_second - anchor, 0.0)
    # Dividing by the mass after the sum rather than before it, for the reason
    # `weighted_power_mean` states: normalizing first can push the rarer of the
    # two weights back below the normal range.
    deviation_ratio = (
        masked_first_weight * jnp.expm1(scaled_first)
        + masked_second_weight * jnp.expm1(scaled_second)
    ) / safe_weight
    near_geometric = deviation_ratio > _DEVIATION_RATIO_CROSSOVER
    # Each term is formed as `exp(log w + s)` rather than as `w * exp(s)`, for
    # the reason `_log_moment` states: `exp(s)` alone underflows once the pair
    # spans more than the dtype's exponent range, and the term it loses is the
    # anchor's own.
    log_first_term = _log_term(
        weight=first_weight,
        masked_weight=masked_first_weight,
        scaled=scaled_first,
        live=first_live,
        discarded=near_geometric,
    )
    log_second_term = _log_term(
        weight=second_weight,
        masked_weight=masked_second_weight,
        scaled=scaled_second,
        live=second_live,
        discarded=near_geometric,
    )
    peak = jnp.maximum(log_first_term, log_second_term)
    safe_peak = jnp.where(jnp.isfinite(peak), peak, 0.0)
    log_moment_power = (
        safe_peak
        + jnp.log(
            jnp.exp(log_first_term - safe_peak) + jnp.exp(log_second_term - safe_peak)
        )
        - jnp.log(safe_weight)
    )
    log_moment = jnp.where(
        near_geometric,
        jnp.log1p(jnp.where(near_geometric, deviation_ratio, 0.0)),
        log_moment_power,
    )
    log_mean_power = anchor + log_moment / safe_exponent
    log_mean_geometric = (
        masked_first_weight * log_first + masked_second_weight * log_second
    ) / safe_weight
    return jnp.exp(jnp.where(exponent == 0.0, log_mean_geometric, log_mean_power))
