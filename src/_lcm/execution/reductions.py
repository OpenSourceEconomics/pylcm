"""Reduction specifications shared by every solver family.

A reduction states how the planner folds the blocks of one reduced axis. The three
action reductions live next to their solvers; this module holds the specifications
whose accumulator is not an action winner, and re-exports the protocol they satisfy.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from _lcm.execution.core_program import ReductionSemantics
from lcm.typing import BoolND, FloatND, IntND

__all__ = [
    "EXACTNESS_VALUES",
    "HardMaxWithCarryReduction",
    "IntervalEnvelopeReduction",
    "ReductionSemantics",
    "WeightedExpectationAccumulator",
    "WeightedExpectationReduction",
]

EXACTNESS_VALUES: tuple[Literal["exact"], Literal["tolerance_equivalent"]] = (
    "exact",
    "tolerance_equivalent",
)


@dataclass(frozen=True)
class WeightedExpectationAccumulator:
    """Running weighted sum and the weight mass it covers."""

    weighted_sum: FloatND
    """Sum of `weight * value` over every block folded so far."""

    weight_mass: FloatND
    """Weight mass those blocks covered."""


@dataclass(frozen=True)
class WeightedExpectationReduction:
    """Weighted sum over stochastic nodes; blocks contribute partial sums.

    Summation order differs between block schedules, so results agree to rounding,
    not bit for bit.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the weighted-expectation contract."""
        return ("weighted-expectation", 1)

    @property
    def exactness(self) -> Literal["tolerance_equivalent"]:
        """Return `"tolerance_equivalent"`: floating-point sums are order dependent."""
        return "tolerance_equivalent"

    def initialize(self, *, value_template: FloatND) -> WeightedExpectationAccumulator:
        """Start from a zero sum with zero covered mass."""
        return WeightedExpectationAccumulator(
            weighted_sum=jnp.zeros_like(value_template),
            weight_mass=jnp.zeros_like(value_template),
        )

    def add(
        self,
        *,
        accumulator: WeightedExpectationAccumulator,
        values: FloatND,
        feasible: BoolND,
        action_ids: IntND,
    ) -> WeightedExpectationAccumulator:
        """Add one block of weighted node values.

        `values` carries `weight * value` and `feasible` marks the real nodes.
        `action_ids` is the node index and is unused; padded nodes are infeasible
        and contribute neither value nor mass.
        """
        del action_ids
        contribution = jnp.where(feasible, values, jnp.zeros_like(values))
        return WeightedExpectationAccumulator(
            weighted_sum=accumulator.weighted_sum + jnp.sum(contribution, axis=-1),
            weight_mass=accumulator.weight_mass
            + jnp.sum(feasible.astype(values.dtype), axis=-1),
        )

    def merge(
        self,
        *,
        left: WeightedExpectationAccumulator,
        right: WeightedExpectationAccumulator,
    ) -> WeightedExpectationAccumulator:
        """Combine two partial sums."""
        return WeightedExpectationAccumulator(
            weighted_sum=left.weighted_sum + right.weighted_sum,
            weight_mass=left.weight_mass + right.weight_mass,
        )

    def finalize(self, *, accumulator: WeightedExpectationAccumulator) -> FloatND:
        """Publish the weighted sum; weights are normalized by the caller."""
        return accumulator.weighted_sum


@dataclass(frozen=True)
class HardMaxWithCarryReduction:
    """Hard max over outer candidates that also carries the winner's payload.

    Ties resolve to the lowest global candidate id, so the fold is exact and order
    independent. The accumulator and payload types are supplied by the NEGM outer
    sweep, which owns `initialize`, `add`, `merge`, and `finalize`.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the hard-max-with-carry contract."""
        return ("hard-max-with-carry", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: a max with deterministic tie-break is order independent."""
        return "exact"

    def initialize(self, *, value_template: FloatND) -> object:
        """Start from an all-infeasible accumulator shaped like `value_template`."""
        raise NotImplementedError(_CARRY_FOLD_OWNER)

    def add(
        self,
        *,
        accumulator: object,
        values: FloatND,
        feasible: BoolND,
        action_ids: IntND,
    ) -> object:
        """Fold one block of candidates into the accumulator."""
        raise NotImplementedError(_CARRY_FOLD_OWNER)

    def merge(self, *, left: object, right: object) -> object:
        """Combine two partial accumulators."""
        raise NotImplementedError(_CARRY_FOLD_OWNER)

    def finalize(self, *, accumulator: object) -> object:
        """Publish the winning candidate and its carried payload."""
        raise NotImplementedError(_CARRY_FOLD_OWNER)


@dataclass(frozen=True)
class IntervalEnvelopeReduction:
    """Upper envelope over intervals; the fold is the NBEGM interval merge.

    The merge is exact because interval ownership is decided by comparison
    arithmetic, not by a rounded sum.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the interval-envelope contract."""
        return ("interval-envelope", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: ownership decisions are order independent."""
        return "exact"

    def initialize(self, *, value_template: FloatND) -> object:
        """Start from an empty envelope shaped like `value_template`."""
        raise NotImplementedError(_INTERVAL_FOLD_OWNER)

    def add(
        self,
        *,
        accumulator: object,
        values: FloatND,
        feasible: BoolND,
        action_ids: IntND,
    ) -> object:
        """Fold one block of intervals into the envelope."""
        raise NotImplementedError(_INTERVAL_FOLD_OWNER)

    def merge(self, *, left: object, right: object) -> object:
        """Combine two partial envelopes."""
        raise NotImplementedError(_INTERVAL_FOLD_OWNER)

    def finalize(self, *, accumulator: object) -> object:
        """Publish the envelope."""
        raise NotImplementedError(_INTERVAL_FOLD_OWNER)


_CARRY_FOLD_OWNER = "The hard-max-with-carry fold is owned by the NEGM outer sweep."
_INTERVAL_FOLD_OWNER = "The interval-envelope fold is owned by the NBEGM interval step."
