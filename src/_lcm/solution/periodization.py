"""Shared period grouping and concrete-node resolution for solver builders."""

from collections.abc import Hashable
from dataclasses import replace
from typing import cast

from _lcm.solution.contract import SolverBuildContext
from _lcm.transition_plans import TargetTransitionPlans
from _lcm.typing import (
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    RegimeName,
    TransitionFunctionsMapping,
)


def solver_period_group_key(
    *,
    context: SolverBuildContext,
    period: int,
    continuation_targets: tuple[RegimeName, ...],
    solver_path: tuple[str, ...],
) -> Hashable:
    """Return every period-varying input that makes a numerical core distinct."""
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        periodized_node_signature,
        periodized_tree_signature,
    )

    grid_signatures = context.period_to_regime_grid_signature
    current_grid_signature = (
        ()
        if grid_signatures is None
        else grid_signatures.get(period, {}).get(context.regime_name, ())
    )
    continuation_grid_signature = (
        ()
        if grid_signatures is None
        else tuple(
            (
                target,
                grid_signatures.get(period + 1, {}).get(target, ()),
            )
            for target in continuation_targets
        )
    )
    return (
        continuation_targets,
        periodized_tree_signature(context.functions, period),
        periodized_tree_signature(context.constraints, period),
        periodized_tree_signature(context.constraint_functions, period),
        periodized_tree_signature(context.transitions, period),
        periodized_tree_signature(context.transition_plans, period),
        periodized_node_signature(context.koopmans_aggregator, period),
        periodized_node_signature(context.compute_regime_transition_probs, period),
        current_grid_signature,
        continuation_grid_signature,
        solver_path,
        id(context.constraint_plan),
    )


def resolve_solver_build_context(
    *, context: SolverBuildContext, period: int
) -> SolverBuildContext:
    """Resolve every periodized numerical node before a solver constructs DAGs."""
    from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
        resolve_periodized_node,
        resolve_periodized_nodes,
        resolve_periodized_tree,
    )

    functions = cast(
        "EconFunctionsMapping", resolve_periodized_nodes(context.functions, period)
    )
    constraints = cast(
        "ConstraintFunctionsMapping",
        resolve_periodized_nodes(context.constraints, period),
    )
    constraint_functions = cast(
        "ConstraintFunctionsMapping",
        resolve_periodized_nodes(context.constraint_functions, period),
    )
    transitions = cast(
        "TransitionFunctionsMapping",
        resolve_periodized_tree(context.transitions, period),
    )
    transition_plans = cast(
        "TargetTransitionPlans",
        resolve_periodized_tree(context.transition_plans, period),
    )
    return replace(
        context,
        functions=functions,
        constraints=constraints,
        constraint_functions=constraint_functions,
        transitions=transitions,
        transition_plans=transition_plans,
        koopmans_aggregator=cast(
            "EconFunction | None",
            resolve_periodized_node(context.koopmans_aggregator, period),
        ),
        compute_regime_transition_probs=resolve_periodized_node(
            context.compute_regime_transition_probs, period
        ),
    )
