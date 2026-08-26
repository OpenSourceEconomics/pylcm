"""The margin-consuming solvers this release can build kernels for.

`OneMarginSolver` and `TwoMarginSolver` declare the binding operation a
margin-consuming solver implements, and a regime accepts any instance of them.
The engine, however, still dispatches on the concrete shipped classes where it
synthesizes the simulate-phase budget constraint and where it reads a carry
target's inner configuration. A solver written outside the shipped set would
pass regime construction and then be solved with those steps skipped, changing
the published policy without failing, so it is refused when the model is built.
"""

from _lcm.solution.contract import OneMarginSolver, Solver, TwoMarginSolver
from _lcm.solution.dcegm import DCEGM
from _lcm.solution.egm import EGM
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import NEGM
from _lcm.solution.nnbegm import NNBEGM
from _lcm.typing import RegimeName
from lcm.exceptions import ModelInitializationError

# The margin-consuming solver classes every dispatch site tests for.
SHIPPED_MARGIN_SOLVERS: tuple[type[Solver], ...] = (EGM, DCEGM, NEGM, NBEGM, NNBEGM)


def fail_if_solver_is_not_shipped(*, solver: Solver, regime_name: RegimeName) -> None:
    """Raise if `solver` consumes a margin but is not one of the shipped solvers.

    A subclass of a shipped solver is accepted: it inherits the concrete type
    every dispatch site tests for. A solver deriving straight from
    `OneMarginSolver` or `TwoMarginSolver` is not.

    Args:
        solver: The solver declared by the regime.
        regime_name: Name of the regime declaring it, for the message.

    Raises:
        ModelInitializationError: If the solver consumes a margin and is not an
            instance of one of `SHIPPED_MARGIN_SOLVERS`.

    """
    if not isinstance(solver, OneMarginSolver | TwoMarginSolver):
        return
    if isinstance(solver, SHIPPED_MARGIN_SOLVERS):
        return
    shipped = ", ".join(cls.__name__ for cls in SHIPPED_MARGIN_SOLVERS)
    msg = (
        f"Regime '{regime_name}' declares solver "
        f"'{type(solver).__name__}', which derives from a margin marker rather "
        f"than from one of the shipped solvers ({shipped}). Custom solvers are "
        "not supported right now: the engine dispatches on those concrete "
        "classes when it synthesizes the simulate-phase budget constraint and "
        "when it reads a carry target, so a solver outside the set would be "
        "solved with both steps skipped and would publish a policy that is "
        "wrong without failing. Use a shipped solver, or subclass one."
    )
    raise ModelInitializationError(msg)
