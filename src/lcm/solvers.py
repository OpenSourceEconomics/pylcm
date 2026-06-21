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

The solvers are defined engine-side in `_lcm.solution.solvers`; this module is a
thin re-export so user code (and `lcm.regime`) can name them, and the `Solver`
contract, without eagerly importing the numerical engine. The engine dispatches
polymorphically on the solver instance (`solver.build_period_kernels(context)`),
not on its type.
"""

from _lcm.solution.contract import SolutionKernels, Solver, SolverBuildContext
from _lcm.solution.solvers import DCEGM, NEGM, GridSearch

__all__ = [
    "DCEGM",
    "NEGM",
    "GridSearch",
    "SolutionKernels",
    "Solver",
    "SolverBuildContext",
]
