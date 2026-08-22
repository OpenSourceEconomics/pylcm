"""User-facing outer branch aggregation configuration (re-export façade).

A nested solver's `branch_aggregator` field selects how the keeper and
adjuster branch values combine:

- `DeterministicOuterMaximum()`: the historical hard maximum, keeper
  winning exact ties.
- `UniformObservedFixedCost(...)`: a uniform i.i.d. fixed adjustment cost,
  observed before the branch choice and entering only the adjuster's fixed
  cost, integrated analytically — no solve-state grid for the shock, and
  the adjustment probability becomes an analytic moment.

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
