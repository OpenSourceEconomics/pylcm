"""User-facing outer branch aggregation configuration (re-export façade).

An `OuterContinuousMargin`'s `adjustment_cost` field declares the cost of
moving the outer margin, which is what decides how the keeper and adjuster
branch values combine:

- unset (`None`): adjusting is free at the margin, so the branches combine by
  the deterministic hard maximum, keeper winning exact ties. Spelling it
  `DeterministicOuterMaximum()` means the same thing.
- `UniformObservedFixedCost(...)`: a uniform i.i.d. fixed adjustment cost,
  observed before the branch choice and entering only the adjuster's fixed
  cost, integrated analytically — no solve-state grid for the shock, and
  the adjustment probability becomes an analytic moment.

These two are the whole set: the margin refuses any other concrete
`OuterBranchAggregator` where it is declared, so the base class names the slot
rather than opening an extension point.

The declaration is economic structure and therefore lives with the margin. A
solver states only whether it can execute the aggregation the declaration
implies — `UniformObservedFixedCost` needs `NNBEGM` with
`outer_search=AdaptiveOuterMesh(...)`, and is refused otherwise.

The configurations and the closed-form kernel are defined engine-side in
`_lcm.egm.branch_aggregation`; this module is a thin re-export.
"""

from _lcm.egm.branch_aggregation import (
    BranchAggregateResult,
    DeterministicOuterMaximum,
    OuterBranchAggregator,
    UniformObservedFixedCost,
)

__all__ = [
    "BranchAggregateResult",
    "DeterministicOuterMaximum",
    "OuterBranchAggregator",
    "UniformObservedFixedCost",
]
