"""Shared period grouping and concrete-node resolution for solver builders."""

from collections.abc import Hashable
from dataclasses import replace
from types import MappingProxyType
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
        periodized_tree_signature(tree=context.functions, period=period),
        periodized_tree_signature(tree=context.constraints, period=period),
        periodized_tree_signature(tree=context.constraint_functions, period=period),
        periodized_tree_signature(tree=context.transitions, period=period),
        periodized_tree_signature(tree=context.transition_plans, period=period),
        periodized_node_signature(node=context.koopmans_aggregator, period=period),
        periodized_node_signature(
            node=context.compute_regime_transition_probs, period=period
        ),
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
        "EconFunctionsMapping",
        resolve_periodized_nodes(mapping=context.functions, period=period),
    )
    constraints = cast(
        "ConstraintFunctionsMapping",
        resolve_periodized_nodes(mapping=context.constraints, period=period),
    )
    constraint_functions = cast(
        "ConstraintFunctionsMapping",
        resolve_periodized_nodes(mapping=context.constraint_functions, period=period),
    )
    transitions = cast(
        "TransitionFunctionsMapping",
        resolve_periodized_tree(tree=context.transitions, period=period),
    )
    transition_plans = cast(
        "TargetTransitionPlans",
        resolve_periodized_tree(tree=context.transition_plans, period=period),
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
            resolve_periodized_node(node=context.koopmans_aggregator, period=period),
        ),
        compute_regime_transition_probs=resolve_periodized_node(
            node=context.compute_regime_transition_probs, period=period
        ),
    )


def restrict_solver_build_context_to_period_group(
    *, context: SolverBuildContext, periods: tuple[int, ...]
) -> SolverBuildContext:
    """Expose exactly one current-regime period group to a nested solver build.

    A representative-resolved context is valid only for the periods whose complete
    solver-group key selected that representative. Nested builders use
    ``regimes_to_active_periods`` to enumerate kernels, continuation plans, and
    parameter checks, so leaving the source regime's full lifecycle visible would
    pair the representative function pool with periods from other groups.

    Other regimes retain their complete active-period metadata: a source solver can
    still need it while constructing continuation reads into those targets.

    Args:
        context: Concrete build context resolved for the group's representative.
        periods: Nonempty, ordered subset of the current regime's active periods.

    Returns:
        A frozen copy whose current regime exposes exactly ``periods``.

    Raises:
        ValueError: If ``periods`` is empty, duplicated, reordered, or contains a
            period outside the source regime's active lifecycle.

    """
    active_periods = context.regimes_to_active_periods[context.regime_name]
    if periods == active_periods:
        return context

    selected = frozenset(periods)
    ordered_selection = tuple(period for period in active_periods if period in selected)
    if not periods or len(selected) != len(periods) or ordered_selection != periods:
        msg = (
            "A solver period group must be a nonempty ordered subset of the current "
            f"regime's active periods. Got {periods!r} for regime "
            f"{context.regime_name!r}, whose active periods are {active_periods!r}."
        )
        raise ValueError(msg)

    return replace(
        context,
        regimes_to_active_periods=MappingProxyType(
            {
                **context.regimes_to_active_periods,
                context.regime_name: periods,
            }
        ),
    )
