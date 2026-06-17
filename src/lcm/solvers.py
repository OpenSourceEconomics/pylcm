"""User-facing solver configuration (re-export façade).

A regime's `solver` field selects the algorithm used for that regime during
backward induction:

- `GridSearch()` (the default): grid search over the full state-action product.
- `DCEGM(...)`: the endogenous grid method for discrete-continuous choice
  (Iskhakov, Jørgensen, Rust & Schjerning 2017, Quantitative Economics 8(2),
  317-365, [doi:10.3982/QE643](https://doi.org/10.3982/QE643)).

The solvers are defined engine-side in `_lcm.solution.solvers`; this module is a
thin re-export so user code (and `lcm.regime`) can name them, and the `Solver`
contract, without eagerly importing the numerical engine. The engine dispatches
polymorphically on the solver instance (`solver.build_period_kernels(context)`),
not on its type.
"""

from _lcm.solution.contract import Solver, SolverBuildContext, SolverKernels
from _lcm.solution.solvers import DCEGM, GridSearch

__all__ = [
    "DCEGM",
    "GridSearch",
    "Solver",
    "SolverBuildContext",
    "SolverKernels",
]
