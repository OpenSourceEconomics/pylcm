"""Compile declarative liquid-state constraints into NBEGM boundary surfaces."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from dags.tree import qname_from_tree_path

from _lcm.axis_boundaries import (
    AxisBoundary,
    boundary_owner_for_feasible_region,
)
from _lcm.constraints.dispositions import (
    BoundaryProgram,
    CompileBoundary,
    ConstraintContext,
    Reject,
)
from _lcm.constraints.ir import Compare, Const, Ref
from _lcm.constraints.routes import BoundaryCompiler, BoundConstraint
from _lcm.typing import FunctionName
from lcm.typing import BoolND, FloatND, IntND, StateName


@dataclass(frozen=True)
class NBEGMFeasibilitySurface:
    """One threshold and the side of it admitted by a constraint."""

    threshold: Ref | Const
    """Literal or flat parameter defining the boundary value."""

    feasible_side: Literal["below", "above"]
    """Side of the threshold admitted by the comparison."""

    includes_boundary: bool
    """Whether equality belongs to the feasible side."""


@dataclass(frozen=True)
class NBEGMFeasibilityBoundaryProgram:
    """All conjunctive feasibility surfaces compiled from one constraint."""

    constraint_name: FunctionName
    """Name under which the constraint was declared."""

    liquid_state: StateName
    """Current liquid state whose axis the solver partitions."""

    surfaces: tuple[NBEGMFeasibilitySurface, ...]
    """Conjunctive boundary surfaces in declaration order."""


def build_nbegm_feasibility_boundary_compiler(
    *, liquid_state: StateName
) -> BoundaryCompiler:
    """Build the compiler for constraints on one resolved NBEGM liquid state.

    Args:
        liquid_state: Current liquid state whose axis the solver partitions.

    Returns:
        A route capability compiling supported comparisons or rejecting the
        unsupported declaration with an attributed diagnostic.

    """
    return _NBEGMFeasibilityBoundaryCompiler(liquid_state=liquid_state)


@dataclass(frozen=True, kw_only=True)
class _NBEGMFeasibilityBoundaryCompiler:
    """The `BoundaryCompiler` of one resolved NBEGM liquid state."""

    liquid_state: StateName
    """Current liquid state whose axis the solver partitions."""

    def __call__(
        self,
        *,
        bound: BoundConstraint,
        context: ConstraintContext,
    ) -> CompileBoundary | Reject:
        surfaces = bound.constraint.boundary_surfaces
        if surfaces is None:
            detail = (
                "the declaration is opaque and exposes no boundary structure"
                if bound.constraint.is_opaque
                else (
                    "the declaration must be a comparison or a conjunction of "
                    "comparisons; unions, complements, and implications do not "
                    "define one intersected feasible region"
                )
            )
            return _reject(bound=bound, context=context, detail=detail)

        compiled: list[NBEGMFeasibilitySurface] = []
        for surface in surfaces:
            result = _compile_surface(
                surface=surface,
                constraint_name=bound.constraint.name,
                liquid_state=self.liquid_state,
                flat_param_names=context.param_names,
            )
            if isinstance(result, str):
                return _reject(bound=bound, context=context, detail=result)
            compiled.append(result)

        payload = NBEGMFeasibilityBoundaryProgram(
            constraint_name=bound.constraint.name,
            liquid_state=self.liquid_state,
            surfaces=tuple(compiled),
        )
        return CompileBoundary(
            constraint=bound.constraint,
            program=BoundaryProgram(surfaces=surfaces, payload=payload),
        )


def _compile_surface(
    *,
    surface: Compare,
    constraint_name: FunctionName,
    liquid_state: StateName,
    flat_param_names: frozenset[str],
) -> NBEGMFeasibilitySurface | str:
    """Compile one ordered comparison or return its diagnostic detail."""
    if surface.op not in {"<", "<=", ">", ">="}:
        return (
            f"comparison {surface.op!r} does not select one feasible side; "
            "use <, <=, >, or >="
        )

    if isinstance(surface.left, Ref) and surface.left.name == liquid_state:
        threshold = surface.right
        feasible_side: Literal["below", "above"] = (
            "below" if surface.op in {"<", "<="} else "above"
        )
    elif isinstance(surface.right, Ref) and surface.right.name == liquid_state:
        threshold = surface.left
        feasible_side = "above" if surface.op in {"<", "<="} else "below"
    else:
        return (
            f"every comparison must place the liquid state {liquid_state!r} on "
            "exactly one side"
        )

    if not isinstance(threshold, Ref | Const):
        return "the threshold must be a literal or flat parameter"
    if isinstance(threshold, Ref) and threshold.name not in flat_param_names:
        qualified_name = qname_from_tree_path((constraint_name, threshold.name))
        if qualified_name not in flat_param_names:
            return (
                f"threshold reference {threshold.name!r} is not a flat parameter; "
                "the threshold must be a literal or flat parameter"
            )
        threshold = Ref(qualified_name)

    return NBEGMFeasibilitySurface(
        threshold=threshold,
        feasible_side=feasible_side,
        includes_boundary=surface.admits_equality,
    )


def feasibility_axis_boundaries(
    *,
    programs: tuple[NBEGMFeasibilityBoundaryProgram, ...],
    params: Mapping[str, FloatND | IntND | BoolND],
) -> tuple[AxisBoundary, ...]:
    """Resolve compiled literal and flat-parameter thresholds into axis sources."""
    return tuple(
        AxisBoundary(
            value=(
                surface.threshold.value
                if isinstance(surface.threshold, Const)
                else params[surface.threshold.name]
            ),
            owner=boundary_owner_for_feasible_region(
                feasible_side=surface.feasible_side,
                includes_boundary=surface.includes_boundary,
            ),
            effect="feasibility",
        )
        for program in programs
        for surface in program.surfaces
    )


def _reject(
    *,
    bound: BoundConstraint,
    context: ConstraintContext,
    detail: str,
) -> Reject:
    """Attribute one compiler refusal to its constraint and regime."""
    return Reject(
        constraint=bound.constraint,
        reason=(
            f"NBEGM cannot compile constraint {bound.constraint.name!r} of "
            f"regime {context.regime_name!r}: {detail}."
        ),
    )
