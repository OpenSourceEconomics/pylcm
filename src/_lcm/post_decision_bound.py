"""The checkable lower-bound declaration backing `lcm.post_decision_lower_bound`.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the user-facing vocabulary module and the
engine-internal validators can both import it without an import cycle.

A declaration retains the complete liquid margin and either a literal floor
or a callable producer. Normalization registers callable floors as named DAG
nodes. Model validation checks RHS dependencies independently of the comparison
LHS; ordinary constraints evaluate the same comparison in solve and simulation.
Literal endogenous-grid declarations are checked against the grid origin.

"""

import inspect
from dataclasses import dataclass
from typing import Literal, cast

from _lcm.grids import ContinuousGrid
from _lcm.typing import FunctionName
from lcm.typing import BoolND, FloatND, UserFunction


@dataclass(frozen=True, kw_only=True)
class PostDecisionBoundSpec:
    """One canonical floor, with liquid roles and RHS-only leaf dependencies."""

    post_decision: str
    """Bounded liquid post-decision target."""
    liquid_state: str
    """Current liquid state excluded from the RHS."""
    action: str
    """Current liquid action excluded from the RHS."""
    resources_target: str
    """Declared liquid resources target."""
    kind: Literal["literal", "dag"]
    """Whether the floor is a number or a named producer."""
    literal_value: float | None
    """Literal floor, when declared."""
    lower_target: str | None
    """Named RHS producer, when declared."""
    lower_inputs: tuple[str, ...]
    """Ordered transitive leaves of the RHS alone."""
    dependency_identity: tuple[tuple[tuple[str, ...], object], ...]
    """Canonical producer and leaf paths, excluding generated root spelling."""
    declaration_names: tuple[str, ...] = ()
    """User declarations represented by this candidate identity."""


def grid_origin_bound_spec(
    *,
    post_decision: str,
    liquid_state: str,
    action: str,
    resources_target: str,
    grid: ContinuousGrid,
) -> PostDecisionBoundSpec | None:
    """Bind a literal capability only when the grid supplies its numerical origin."""
    if getattr(grid, "pass_points_at_runtime", False):
        return None
    origin = float(grid.to_jax()[0])
    return PostDecisionBoundSpec(
        post_decision=post_decision,
        liquid_state=liquid_state,
        action=action,
        resources_target=resources_target,
        kind="literal",
        literal_value=origin,
        lower_target=None,
        lower_inputs=(),
        dependency_identity=(((), origin),),
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _PostDecisionLowerBound:
    """A declared lower bound on a post-decision state.

    Retains the complete liquid margin and evaluates its literal or named DAG
    floor. Contextual validation treats the equivalent handwritten comparison
    identically. A decomposed instance reads a named producer; a standalone
    instance calls its declared producer directly.
    """

    post_decision: FunctionName
    """Name of the post-decision function the bound applies to."""
    lower_bound: float | UserFunction
    """The literal floor or its declared DAG producer."""
    liquid_state: FunctionName
    """Current liquid state, excluded from callable floor inputs."""
    action: FunctionName
    """Liquid action, excluded from callable floor inputs."""
    resources_target: FunctionName
    """Declared liquid resources role."""
    lower_target: FunctionName | None = None
    """Named producer in a decomposed regime, absent for standalone calls."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__name__", f"{self.post_decision}_lower_bound")
        param = inspect.Parameter(
            self.post_decision,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=FloatND,
        )
        params = [param]
        if self.lower_target is not None:
            params.append(
                inspect.Parameter(
                    self.lower_target,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=FloatND,
                )
            )
        elif callable(self.lower_bound):
            params.extend(
                parameter
                for parameter in inspect.signature(self.lower_bound).parameters.values()
                if parameter.name != self.post_decision
            )
        signature = inspect.Signature(params, return_annotation=BoolND)
        object.__setattr__(self, "__signature__", signature)
        object.__setattr__(
            self,
            "__annotations__",
            {
                **{p.name: p.annotation for p in params},
                "return": BoolND,
            },
        )

    def __call__(self, **kwargs: FloatND) -> BoolND:
        lower = self.lower_bound
        if self.lower_target is not None:
            lower = kwargs[self.lower_target]
        elif callable(lower):
            lower = lower(
                **{name: kwargs[name] for name in inspect.signature(lower).parameters}
            )
        return kwargs[self.post_decision] >= cast("FloatND | float", lower)

    def __repr__(self) -> str:
        return f"<declared lower bound: {self.post_decision} >= {self.lower_bound!r}>"
