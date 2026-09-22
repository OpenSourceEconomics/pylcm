"""Stable blockwise log-sum-exp reduction over already-formed branch values.

GridSearch uses this reduction after maximizing the continuous cells within each
discrete EV1 branch. Choice probabilities remain a separate simulation concern.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax.numpy as jnp

from lcm.typing import BoolND, FloatND


class LogSumExpAccumulator(NamedTuple):
    """Mergeable stable exponential mass for EV1 branch values."""

    running_max: FloatND
    rescaled_sum: FloatND
    any_finite: BoolND
    any_positive_infinity: BoolND
    any_nan: BoolND


class LogSumExpResult(NamedTuple):
    """Smoothed maximum obtained from a complete branch reduction."""

    smoothed_value: FloatND


@dataclass(frozen=True)
class BoundLogSumExpReduction:
    """Value reduction whose one dynamic scale is fixed for its full lifetime."""

    scale: FloatND

    def initialize(self, *, value_template: FloatND) -> LogSumExpAccumulator:
        """Create an empty exponential-mass accumulator."""
        return LogSumExpAccumulator(
            running_max=jnp.full_like(value_template, -jnp.inf),
            rescaled_sum=jnp.zeros_like(value_template),
            any_finite=jnp.zeros(value_template.shape, dtype=bool),
            any_positive_infinity=jnp.zeros(value_template.shape, dtype=bool),
            any_nan=jnp.zeros(value_template.shape, dtype=bool),
        )

    def add(
        self,
        *,
        accumulator: LogSumExpAccumulator,
        values: FloatND,
    ) -> LogSumExpAccumulator:
        """Reduce one branch block and merge its stable exponential mass."""
        finite = jnp.isfinite(values)
        block_max = jnp.max(
            jnp.where(finite, values, -jnp.inf), axis=-1, initial=-jnp.inf
        )
        block_sum = jnp.sum(
            jnp.where(
                finite,
                jnp.exp((values - block_max[..., jnp.newaxis]) / self.scale),
                0.0,
            ),
            axis=-1,
        )
        block = LogSumExpAccumulator(
            running_max=block_max,
            rescaled_sum=block_sum,
            any_finite=jnp.any(finite, axis=-1),
            any_positive_infinity=jnp.any(jnp.isposinf(values), axis=-1),
            any_nan=jnp.any(jnp.isnan(values), axis=-1),
        )
        return self.merge(left=accumulator, right=block)

    def merge(
        self,
        *,
        left: LogSumExpAccumulator,
        right: LogSumExpAccumulator,
    ) -> LogSumExpAccumulator:
        """Merge two stable masses after rescaling to one running maximum."""
        running_max = jnp.maximum(left.running_max, right.running_max)
        left_factor = jnp.where(
            left.any_finite,
            jnp.exp((left.running_max - running_max) / self.scale),
            0.0,
        )
        right_factor = jnp.where(
            right.any_finite,
            jnp.exp((right.running_max - running_max) / self.scale),
            0.0,
        )
        return LogSumExpAccumulator(
            running_max=running_max,
            rescaled_sum=(
                left.rescaled_sum * left_factor + right.rescaled_sum * right_factor
            ),
            any_finite=left.any_finite | right.any_finite,
            any_positive_infinity=(
                left.any_positive_infinity | right.any_positive_infinity
            ),
            any_nan=left.any_nan | right.any_nan,
        )

    def finalize(self, *, accumulator: LogSumExpAccumulator) -> LogSumExpResult:
        """Publish the value with the dense helper's nonfinite semantics."""
        finite_result = accumulator.running_max + self.scale * jnp.log(
            accumulator.rescaled_sum
        )
        smoothed = jnp.where(
            accumulator.any_finite,
            finite_result,
            jnp.full_like(finite_result, -jnp.inf),
        )
        smoothed = jnp.where(
            accumulator.any_positive_infinity,
            jnp.full_like(smoothed, jnp.nan),
            smoothed,
        )
        smoothed = jnp.where(
            accumulator.any_nan,
            jnp.full_like(smoothed, jnp.nan),
            smoothed,
        )
        return LogSumExpResult(smoothed_value=smoothed)


class LogSumExpReduction:
    """Stable log-sum-exp over branch blocks, folded at a caller-supplied scale.

    Every fold step names the ``scale`` its exponential mass is rescaled by, since
    the scale is a dynamic value rather than part of the contract. ``bind`` fixes
    one scale for a whole session so partial steps cannot disagree about it.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Stable identity of the value-only log-sum-exp contract."""
        return ("logsumexp", 1)

    @property
    def exactness(self) -> Literal["tolerance_equivalent"]:
        """Return `"tolerance_equivalent"`: the mass is a floating-point sum."""
        return "tolerance_equivalent"

    def bind(self, *, scale: FloatND) -> BoundLogSumExpReduction:
        """Bind ``scale`` once so partial operations cannot disagree about it."""
        return BoundLogSumExpReduction(scale=scale)

    def initialize(self, *, value_template: FloatND) -> LogSumExpAccumulator:
        """Create an empty exponential-mass accumulator; the scale does not enter."""
        return BoundLogSumExpReduction(scale=jnp.ones_like(value_template)).initialize(
            value_template=value_template
        )

    def add(
        self,
        *,
        accumulator: LogSumExpAccumulator,
        values: FloatND,
        scale: FloatND,
    ) -> LogSumExpAccumulator:
        """Reduce one branch block at ``scale`` and merge its exponential mass."""
        return self.bind(scale=scale).add(accumulator=accumulator, values=values)

    def merge(
        self,
        *,
        left: LogSumExpAccumulator,
        right: LogSumExpAccumulator,
        scale: FloatND,
    ) -> LogSumExpAccumulator:
        """Merge two partial masses after rescaling to one running maximum."""
        return self.bind(scale=scale).merge(left=left, right=right)

    def finalize(
        self, *, accumulator: LogSumExpAccumulator, scale: FloatND
    ) -> LogSumExpResult:
        """Publish the smoothed maximum of a complete mass at ``scale``."""
        return self.bind(scale=scale).finalize(accumulator=accumulator)


LOGSUMEXP_REDUCTION = LogSumExpReduction()
# Shared stable log-sum-exp reduction specification.
