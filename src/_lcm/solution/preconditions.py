"""Run the solver preconditions that need real parameter values.

A solver's scope condition is sometimes a property of the *evaluated* model
rather than of its structure: whether the budget is affine in the liquid state
between declared breakpoints, whether a carried state's law is constant within
an interval. Deciding either requires the tax schedules, tables, and
coefficients that only exist once the user supplies params, so the check cannot
run in `Solver.validate`, which the engine calls while building kernels at
`Model` construction.

Such a solver publishes its checks as `SolutionKernels.param_checks`. `Model`
calls `check_solver_params` for every solve because the evaluated functions can
change with every parameter draw; a precondition established for one draw says
nothing about another.
"""

from types import MappingProxyType

from _lcm.engine import Regime
from _lcm.typing import FlatParams, RegimeName


def check_solver_params(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
) -> None:
    """Run every regime solver's parameter-dependent preconditions.

    Each check receives the regime's *complete* parameter vector — the params
    supplied to `solve` overlaid on the ones fixed at model construction. A
    precondition is a statement about the model's functions, and a schedule the
    author declared fixed is still what the budget reads; handing over only the
    free params would leave the fixed ones to be synthesized, which is the
    situation running at solve time exists to avoid.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.

    Raises:
        Exception: Whatever a failing check raises — each solver owns its own
            error type and message.

    """
    complete = MappingProxyType(
        {
            name: MappingProxyType(
                {
                    **regime.resolved_fixed_params,
                    **flat_params.get(name, MappingProxyType({})),
                }
            )
            for name, regime in regimes.items()
        }
    )
    for regime in regimes.values():
        for check in regime.solution.param_checks:
            check(flat_params=complete)
