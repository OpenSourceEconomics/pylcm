"""Reduction contracts and the specifications that are pure declarations.

A reduced axis names a reduction at two levels:

- `ReductionDeclaration` is the contract an axis references — a stable
  `semantic_key` that enters static program identity, and an `exactness` that says
  whether block order can move the published value. That is all the planner needs
  to decide a width and all a program's identity records.
- `ReductionSemantics` is a declaration that also carries the fold the planner may
  drive block by block. A reduction whose kernel lives with its solver is a
  declaration only: the axis's `width_keyword` reaches that kernel the same way it
  reaches every other streamed core.

The three action reductions live next to their solvers; this module holds the
protocols and the specifications whose accumulator is not an action winner.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp

from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import FloatND

__all__ = [
    "EXACTNESS_VALUES",
    "HARD_MAX_WITH_CARRY_REDUCTION",
    "INTERVAL_ENVELOPE_REDUCTION",
    "WEIGHTED_EXPECTATION_REDUCTION",
    "BoundWeightedExpectationReduction",
    "HardMaxWithCarryReduction",
    "IntervalEnvelopeReduction",
    "ReductionDeclaration",
    "ReductionSemantics",
    "WeightedExpectationAccumulator",
    "WeightedExpectationReduction",
    "WeightedExpectationResult",
]

EXACTNESS_VALUES: tuple[Literal["exact"], Literal["tolerance_equivalent"]] = (
    "exact",
    "tolerance_equivalent",
)


@runtime_checkable
class ReductionDeclaration(Protocol):
    """The contract a reduced axis names, without the arithmetic behind it.

    - `semantic_key` names the numerical contract and enters static program
      identity, so two programs reducing the same axis differently never share a
      compiled executable.
    - `exactness` says whether block order can change the published value:
      `"exact"` results are bit-identical across widths, `"tolerance_equivalent"`
      results agree to the working format's rounding.
    """

    @property
    def semantic_key(self) -> Hashable:
        """Return a stable key for the reduction's numerical contract."""
        ...

    @property
    def exactness(self) -> Literal["exact", "tolerance_equivalent"]:
        """Return whether block order can move the published value."""
        ...


@runtime_checkable
class ReductionSemantics(ReductionDeclaration, Protocol):
    """A declaration that also carries the fold the planner drives per block.

    The fold is four steps: one state created from a template, one contribution
    per block, an associative and commutative merge of two partial states, and a
    finalization that publishes the reduced result. Every block sees only its own
    cells, so any partition of the axis — including a shorter last block — reaches
    the same state that one pass over the whole axis would.

    Only the four step names and the state each step threads are shared; the block
    inputs are the family's own, so every step takes its family's keywords and
    nothing is positional:

    - the hard maxes take the block's `values`, its `feasible` mask and its
      canonical global `action_ids`, and break a tie toward the smallest global
      identity, which is what makes them exact under any partition;
    - the collective one takes the block's `objectives` and `stakeholder_values`
      alongside those, and reads every stakeholder at the one identity the
      household objective selects;
    - the smoothed one takes `values` and the `scale` its exponential mass is
      rescaled by;
    - the weighted expectation takes `values` and their `weights`.

    A family whose statement must not vary between two steps of one fold fixes it
    once with a `bind` of its own — `bind(*, scale)` for the smoothed maximum,
    `bind(*, subnormal_is_accounted_for)` for the weighted expectation — and the
    bound object then takes only the block. A family whose state carries an extra
    trailing axis names its own template: the collective reduction initializes
    from a `stakeholder_template` where the others take a `value_template`.
    """

    def initialize(self, **template: FloatND) -> object:
        """Return the empty state, shaped and typed like the family's template."""
        ...

    def add(self, *, accumulator: object, **block: object) -> object:
        """Fold one block into `accumulator` and return the new state."""
        ...

    def merge(self, *, left: object, right: object, **binding: object) -> object:
        """Combine two partial states into the state covering both their blocks."""
        ...

    def finalize(self, *, accumulator: object, **binding: object) -> object:
        """Publish the reduced result of a complete state."""
        ...


class WeightedExpectationAccumulator(NamedTuple):
    """Mergeable state of a weighted expectation over streamed node blocks."""

    weighted_sum: FloatND
    """Sum of `weight * value` over every node folded so far."""

    weight_mass: FloatND
    """Sum of the weights of those same nodes."""


class WeightedExpectationResult(NamedTuple):
    """The published expectation and the two sums it is formed from."""

    expectation: FloatND
    """Mass-normalized weighted mean; NaN wherever the mass is exactly zero."""

    weighted_sum: FloatND
    """Sum of `weight * value` over the complete axis."""

    weight_mass: FloatND
    """Sum of the weights over the complete axis."""


@dataclass(frozen=True)
class BoundWeightedExpectationReduction:
    """A weighted expectation whose subnormal-weight statement is fixed.

    A block's nodes occupy the **last** axis of `values`, and `weights`
    broadcasts to that shape; `add` rejects a `values` whose rank the weights
    cannot broadcast against, so a caller whose block axis leads has to move it
    rather than silently reduce the wrong one.

    A block contributes its own weighted sum and its own weight mass, and the
    merge adds both, so any partition of the axis — including a shorter last
    block, or a block padded with exactly-zero weights — reaches a state that
    names the same real number as one pass over the whole axis. Floating-point
    addition is not associative, so the states agree to the format's rounding
    rather than bit for bit.

    A node of represented-zero weight contributes exactly `0.0` even where its
    value is an infinity: a lottery's impossible outcome is not priced. A NaN
    weight stays poison, and a negative one stays visible.
    """

    subnormal_is_accounted_for: bool
    """Whether the caller has established a subnormal weight cannot matter here."""

    def initialize(self, *, value_template: FloatND) -> WeightedExpectationAccumulator:
        """Create an empty accumulator with `value_template`'s shape and dtype."""
        return WeightedExpectationAccumulator(
            weighted_sum=jnp.zeros_like(value_template),
            weight_mass=jnp.zeros_like(value_template),
        )

    def add(
        self,
        *,
        accumulator: WeightedExpectationAccumulator,
        values: FloatND,
        weights: FloatND,
    ) -> WeightedExpectationAccumulator:
        """Reduce one block of weighted nodes and merge it into `accumulator`.

        Raises:
            ValueError: `weights` does not broadcast against `values`, which is
                what a caller whose block axis is not last runs into.

        """
        _fail_if_block_axis_is_not_last(values=values, weights=weights)
        block = WeightedExpectationAccumulator(
            weighted_sum=jnp.sum(
                zero_safe_weighted_term(
                    weight=weights,
                    value=values,
                    subnormal_is_accounted_for=self.subnormal_is_accounted_for,
                ),
                axis=-1,
            ),
            weight_mass=jnp.sum(
                jnp.broadcast_to(weights, jnp.asarray(values).shape), axis=-1
            ),
        )
        return self.merge(left=accumulator, right=block)

    def merge(
        self,
        *,
        left: WeightedExpectationAccumulator,
        right: WeightedExpectationAccumulator,
    ) -> WeightedExpectationAccumulator:
        """Add two partial states; addition is associative up to rounding alone."""
        return WeightedExpectationAccumulator(
            weighted_sum=left.weighted_sum + right.weighted_sum,
            weight_mass=left.weight_mass + right.weight_mass,
        )

    def finalize(
        self, *, accumulator: WeightedExpectationAccumulator
    ) -> WeightedExpectationResult:
        """Publish the expectation alongside the two sums it is formed from.

        A lottery of exactly zero mass has no expectation, and the published
        value says so with a NaN rather than a laundered zero. The two sums ride
        along because a caller whose weights carry a common base-two scale
        normalizes by that exact power of two instead of by the accumulated
        mass, and takes `weighted_sum`.
        """
        return WeightedExpectationResult(
            expectation=jnp.where(
                accumulator.weight_mass == 0,
                jnp.full_like(accumulator.weighted_sum, jnp.nan),
                accumulator.weighted_sum / accumulator.weight_mass,
            ),
            weighted_sum=accumulator.weighted_sum,
            weight_mass=accumulator.weight_mass,
        )


def _fail_if_block_axis_is_not_last(*, values: FloatND, weights: FloatND) -> None:
    """Require the weights to broadcast against the block's trailing node axis."""
    value_shape = jnp.asarray(values).shape
    weight_shape = jnp.asarray(weights).shape
    if not value_shape or (
        weight_shape and weight_shape[-1] not in (1, value_shape[-1])
    ):
        msg = (
            "A weighted-expectation block reduces its last axis, so the node "
            f"weights of shape {weight_shape} must broadcast against values of "
            f"shape {value_shape}."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class WeightedExpectationReduction:
    """Weighted expectation over stochastic nodes; blocks contribute partial sums.

    Whether a weight below the format's normal range needs its exponent moved
    onto the value cannot be read off the operands, so it is stated rather than
    guessed: `bind(*, subnormal_is_accounted_for)` fixes the statement once for a
    whole fold, and the unbound steps take it per call for a single-shot use.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the weighted-expectation contract."""
        return ("weighted-expectation", 1)

    @property
    def exactness(self) -> Literal["tolerance_equivalent"]:
        """Return `"tolerance_equivalent"`: floating-point sums are order dependent."""
        return "tolerance_equivalent"

    def bind(
        self, *, subnormal_is_accounted_for: bool
    ) -> BoundWeightedExpectationReduction:
        """Fix the subnormal-weight statement so partial steps cannot disagree."""
        return BoundWeightedExpectationReduction(
            subnormal_is_accounted_for=subnormal_is_accounted_for
        )

    def initialize(self, *, value_template: FloatND) -> WeightedExpectationAccumulator:
        """Create an empty accumulator; the subnormal statement does not enter."""
        return _STATEMENT_FREE_STEPS.initialize(value_template=value_template)

    def add(
        self,
        *,
        accumulator: WeightedExpectationAccumulator,
        values: FloatND,
        weights: FloatND,
        subnormal_is_accounted_for: bool,
    ) -> WeightedExpectationAccumulator:
        """Reduce one block of weighted nodes under the stated subnormal rule."""
        return self.bind(subnormal_is_accounted_for=subnormal_is_accounted_for).add(
            accumulator=accumulator, values=values, weights=weights
        )

    def merge(
        self,
        *,
        left: WeightedExpectationAccumulator,
        right: WeightedExpectationAccumulator,
    ) -> WeightedExpectationAccumulator:
        """Add two partial states; the subnormal statement does not enter."""
        return _STATEMENT_FREE_STEPS.merge(left=left, right=right)

    def finalize(
        self, *, accumulator: WeightedExpectationAccumulator
    ) -> WeightedExpectationResult:
        """Publish the expectation; the subnormal statement does not enter."""
        return _STATEMENT_FREE_STEPS.finalize(accumulator=accumulator)


@dataclass(frozen=True)
class HardMaxWithCarryReduction:
    """Hard max over outer candidates that also carries the winner's payload.

    Ties resolve to the lowest global candidate id, so the fold is exact and order
    independent.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the hard-max-with-carry contract."""
        return ("hard-max-with-carry", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: a max with deterministic tie-break is order independent."""
        return "exact"


@dataclass(frozen=True)
class IntervalEnvelopeReduction:
    """Upper envelope over NB-EGM interval blocks, as a declaration only.

    The fold kernel is the interval merge in the NB-EGM step, which reaches the
    planner's width through the axis's `width_keyword` like any other streamed
    core. Its state is one fixed-shape envelope winner per query carrying that
    winner's global stored-link index, and ownership is decided by comparison
    arithmetic over that stable total order rather than by a rounded sum, so the
    merge is partition invariant.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the interval-envelope contract."""
        return ("interval-envelope", 1)

    @property
    def exactness(self) -> Literal["exact"]:
        """Return `"exact"`: ownership decisions are order independent."""
        return "exact"


# `initialize`, `merge` and `finalize` never form a weighted term, so the
# subnormal statement cannot reach them; this binding carries the three of them
# rather than each asserting a rule it does not use.
_STATEMENT_FREE_STEPS = BoundWeightedExpectationReduction(
    subnormal_is_accounted_for=False
)

WEIGHTED_EXPECTATION_REDUCTION = WeightedExpectationReduction()
# Shared weighted-expectation reduction specification.

HARD_MAX_WITH_CARRY_REDUCTION = HardMaxWithCarryReduction()
# Shared hard-max-with-carry declaration.

INTERVAL_ENVELOPE_REDUCTION = IntervalEnvelopeReduction()
# Shared interval-envelope declaration.
