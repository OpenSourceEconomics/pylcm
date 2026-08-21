"""The checkable lower-bound declaration backing `lcm.post_decision_lower_bound`.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the user-facing vocabulary module and the
engine-internal validators can both import it without an import cycle.

An endogenous-grid solve enforces its borrowing limit through the savings
grid: the grid's lowest node *is* the solve-time limit. A regime that states its
own limit is therefore making a claim about that grid, and this declaration is
what makes the claim checkable — the instance carries the number, so validation
can compare it with the grid instead of taking an opaque predicate's word for
it. The same instance remains an executable constraint in simulation.
"""

import inspect

from _lcm.typing import FunctionName
from lcm.typing import BoolND, FloatND


class _PostDecisionLowerBound:
    """A declared lower bound on a post-decision state.

    Instances are produced by `lcm.post_decision_lower_bound`. The instance is
    an ordinary constraint callable — it evaluates `post_decision >= bound`, so
    it behaves like any other entry of `constraints`. Normalization turns it
    into the comparison it stands for, which is what lets validation prove the
    declaration against the solver's savings grid rather than infer it from a
    DAG. Nothing downstream recognises the class itself: a hand-written
    comparison of the same shape is proved and dropped identically, so the
    constructor is a way to spell one, not a privileged kind of one.
    """

    def __init__(self, *, post_decision: FunctionName, lower_bound: float) -> None:
        self.post_decision = post_decision
        """Name of the post-decision function the bound applies to."""
        self.lower_bound = lower_bound
        """The declared lower bound, compared exactly against the grid's node."""
        self.__name__ = f"{post_decision}_lower_bound"
        param = inspect.Parameter(
            post_decision,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=FloatND,
        )
        self.__signature__ = inspect.Signature([param], return_annotation=BoolND)
        self.__annotations__ = {post_decision: FloatND, "return": BoolND}

    def __call__(self, **kwargs: FloatND) -> BoolND:
        return kwargs[self.post_decision] >= self.lower_bound

    def __repr__(self) -> str:
        return f"<declared lower bound: {self.post_decision} >= {self.lower_bound!r}>"
