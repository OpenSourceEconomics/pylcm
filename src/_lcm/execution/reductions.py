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
from typing import Literal, Protocol, runtime_checkable

from lcm.typing import FloatND

__all__ = [
    "EXACTNESS_VALUES",
    "HARD_MAX_WITH_CARRY_REDUCTION",
    "INTERVAL_ENVELOPE_REDUCTION",
    "WEIGHTED_EXPECTATION_REDUCTION",
    "HardMaxWithCarryReduction",
    "IntervalEnvelopeReduction",
    "ReductionDeclaration",
    "ReductionSemantics",
    "WeightedExpectationReduction",
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

    `values` is the only block input every family shares; each names whatever else
    its arithmetic needs alongside it. The hard maxes take the block's `feasible`
    mask and its canonical global `action_ids`, and break a tie toward the smallest
    global identity, which is what makes them exact under any partition. The
    collective one adds the block's per-stakeholder values and reads them at the
    one identity the household objective selects. The smoothed one takes the
    `scale` its exponential mass is rescaled by; `bind(*, scale)` fixes that scale
    once for a whole fold so partial steps cannot disagree about it. A family whose
    state carries an extra trailing axis names its own template — the collective
    reduction initializes from a `stakeholder_template`.
    """

    def initialize(self, *, value_template: FloatND) -> object:
        """Return the empty state, shaped and typed like `value_template`."""
        ...

    def add(self, *, accumulator: object, values: FloatND) -> object:
        """Fold one block of `values` into `accumulator` and return the new state."""
        ...

    def merge(self, *, left: object, right: object) -> object:
        """Combine two partial states into the state covering both their blocks."""
        ...

    def finalize(self, *, accumulator: object) -> object:
        """Publish the reduced result of a complete state."""
        ...


@dataclass(frozen=True)
class WeightedExpectationReduction:
    """Weighted sum over stochastic nodes; blocks contribute partial sums.

    Summation order is the canonical node order within and across blocks, and
    zero-weight padding fills a partial tile, so results agree to rounding rather
    than bit for bit.
    """

    @property
    def semantic_key(self) -> tuple[str, int]:
        """Return the stable key for the weighted-expectation contract."""
        return ("weighted-expectation", 1)

    @property
    def exactness(self) -> Literal["tolerance_equivalent"]:
        """Return `"tolerance_equivalent"`: floating-point sums are order dependent."""
        return "tolerance_equivalent"


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


WEIGHTED_EXPECTATION_REDUCTION = WeightedExpectationReduction()
# Shared weighted-expectation declaration.

HARD_MAX_WITH_CARRY_REDUCTION = HardMaxWithCarryReduction()
# Shared hard-max-with-carry declaration.

INTERVAL_ENVELOPE_REDUCTION = IntervalEnvelopeReduction()
# Shared interval-envelope declaration.
