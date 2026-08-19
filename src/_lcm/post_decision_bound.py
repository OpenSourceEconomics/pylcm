"""The checkable lower-bound declaration backing `lcm.post_decision_lower_bound`.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the user-facing vocabulary module and the
engine-internal validators can both import it without an import cycle.

An endogenous-grid solve enforces its borrowing limit through the savings
grid: the grid's lowest node *is* the limit, and the simulate phase receives a
mask synthesized from it. A regime that states its own limit is therefore
making a claim about that grid, and this declaration is what makes the claim
checkable — the instance carries the number, so validation can compare it with
the grid instead of taking an opaque predicate's word for it.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType

from _lcm.typing import FunctionName
from lcm.typing import BoolND, FloatND, UserFunction


class _PostDecisionLowerBound:
    """A declared lower bound on a post-decision state.

    Instances are produced by `lcm.post_decision_lower_bound`. The instance is
    an ordinary constraint callable — it evaluates `post_decision >= bound`, so
    it behaves like any other entry of `constraints` — and additionally exposes
    the declared bound, which is what lets validation prove the declaration
    against the solver's savings grid rather than infer it from a DAG.
    """

    _is_post_decision_lower_bound: bool = True

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


def without_proved_lower_bounds(
    *,
    constraints: Mapping[FunctionName, UserFunction],
    grid_enforces_the_bound: bool,
) -> MappingProxyType[FunctionName, UserFunction]:
    """Drop declared lower bounds a savings grid already enforces.

    A declared bound is a claim about the savings grid, checked when the model
    is built. Once proved it carries no information the grid does not already
    carry: the solve enforces it by inverting on that grid, and the simulate
    phase enforces it through the mask synthesized from the same lowest node.
    Leaving it in the engine's constraint set would have it evaluated a second
    time — and the solve's feasibility predicate is built per discrete combo,
    which is not a place a continuous post-decision state can be read.

    A solver that does not invert on a savings grid — grid search is the case
    — enforces nothing implicitly, so for it the declaration is an ordinary
    constraint and must survive. The same declaration is therefore load-bearing
    in one regime and redundant in another, which is the point: one spelling
    that both arms of a model honour.

    Args:
        constraints: The regime's constraint mapping.
        grid_enforces_the_bound: Whether the regime's solver enforces the bound
            through its savings grid, making the declaration redundant.

    Returns:
        Immutable mapping of the constraints the engine evaluates.

    """
    if not grid_enforces_the_bound:
        return MappingProxyType(dict(constraints))
    return MappingProxyType(
        {
            name: value
            for name, value in constraints.items()
            if not getattr(value, "_is_post_decision_lower_bound", False)
        }
    )
