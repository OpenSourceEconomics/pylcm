"""User-facing solver configuration (re-export façade).

A regime's `solver` field selects the algorithm used for that regime during
backward induction:

- `GridSearch()` (the default): grid search over the full state-action product.
- `DCEGM(...)`: the endogenous grid method for discrete-continuous choice
  (Iskhakov, Jørgensen, Rust & Schjerning 2017, Quantitative Economics 8(2),
  317-365, [doi:10.3982/QE643](https://doi.org/10.3982/QE643)).
- `NEGM(...)`: nested EGM — an outer deterministic grid search over a
  durable/illiquid continuous margin with an inner 1-D `DCEGM` solve of the
  consumption-savings problem conditional on that margin (Druedahl 2021,
  Computational Economics 58(3), 747-775,
  [doi:10.1007/s10614-020-10045-x](https://doi.org/10.1007/s10614-020-10045-x)).
- `EGM(...)`: the plain endogenous grid method (Carroll 2006; see
  `docs/credits.md`) for a regime with one continuous (Euler) state and no
  discrete kinks. Inverting the Euler equation on the post-decision savings
  grid solves such a period exactly, so this is the specialization whose step
  needs no upper envelope at all.
- `NBEGM(...)`: the non-convex-budget endogenous grid method for a 1-D
  consumption-savings regime whose budget carries declared breakpoints — a
  means-tested cliff split into case pieces, or a piecewise-affine schedule of
  kinks, jumps, and floors. See `docs/methods/nonconvex_budgets.md`.
- `NNBEGM(...)`: the same outer keeper/adjuster search as `NEGM` with an inner
  `NBEGM` solve, so declared liquid kinks, jumps, and hard constraints keep
  their exact NB-EGM treatment inside every outer candidate.

A second continuous state is reached by nesting: `NEGM` and `NNBEGM` solve an
inner 1-D problem conditional on the outer margin, rather than inverting two
coupled first-order conditions jointly. `ConsumptionSavingsRegime` declares the
liquid state, consumption action, resources, and post-decision role names the
endogenous-grid solvers read, and `NestedConsumptionSavingsRegime` adds the outer
continuous margin `NEGM` and `NNBEGM` search over. The solvers themselves carry
numerical configuration only, so one of these regimes is required to use them;
plain `Regime` stays the form for `GridSearch`.

`DCEGM` defaults to `ExactEnvelope`, whose certified finite-candidate ownership
requires pylcm's installed exact-affine payload. Selecting it without a compatible
payload for the active JAX backend fails during `Model(...)`. Certified `NBEGM` uses the
same installed capability but requests it at the first certified envelope evaluation;
`envelope_arithmetic="ordinary"` is the explicit no-kernel route under its documented
approximation contract. Neither solver silently falls back. Solvers specialized to one
paper's accounting remain published alongside that paper rather than here.

The solvers are defined engine-side in per-solver modules under
`_lcm.solution`; this module is a thin re-export so user code (and
`lcm.regime`) can name them, and the `Solver` contract, without eagerly
importing the numerical engine. The engine dispatches polymorphically on the
solver instance (`solver.build_period_kernels(context)`), not on its type.
"""

from _lcm.solution.contract import (
    OneMarginSolver,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    TwoMarginSolver,
)
from _lcm.solution.dcegm import (
    DCEGM,
    EnvelopeConfig,
    ExactEnvelope,
    FUESEnvelope,
    LTMEnvelope,
    MSSEnvelope,
    RFCEnvelope,
)
from _lcm.solution.egm import EGM
from _lcm.solution.grid_search import GridSearch
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import NEGM
from _lcm.solution.nnbegm import NNBEGM
from lcm.branch_aggregation import (
    BranchAggregateResult,
    DeterministicOuterMaximum,
    OuterBranchAggregator,
    UniformObservedFixedCost,
)
from lcm.outer_search import (
    AdaptiveOuterMesh,
    FiniteOuterGrid,
    LegacyGoldenSection,
    OuterSearch,
)

__all__ = [
    "DCEGM",
    "EGM",
    "NBEGM",
    "NEGM",
    "NNBEGM",
    "AdaptiveOuterMesh",
    "BranchAggregateResult",
    "DeterministicOuterMaximum",
    "EnvelopeConfig",
    "ExactEnvelope",
    "FUESEnvelope",
    "FiniteOuterGrid",
    "GridSearch",
    "LTMEnvelope",
    "LegacyGoldenSection",
    "MSSEnvelope",
    "OneMarginSolver",
    "OuterBranchAggregator",
    "OuterSearch",
    "RFCEnvelope",
    "SolutionKernels",
    "Solver",
    "SolverBuildContext",
    "TwoMarginSolver",
    "UniformObservedFixedCost",
]
