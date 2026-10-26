"""Opaque-resources ancestry for the complete admitted DC action problem."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.validation import (
    _dag_ancestors,
    _savings_stage_candidates,
    _transition_variants,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.dcegm import _BoundDCEGM
from lcm.typing import ReferenceName, UserFunction


@dataclass(frozen=True, kw_only=True)
class ResourcesDependencyProof:
    """A declaration proof and its actual remaining row-context dependency set."""

    context_inputs: frozenset[ReferenceName]
    """Primitive inputs retained after resources becomes an opaque leaf."""
    root_dependencies: Mapping[str, frozenset[ReferenceName]]
    """Complete semantic root census with the resolved opaque DAG ancestry."""
    bypasses: Mapping[str, frozenset[ReferenceName]]
    """Physical or supported-code ancestry outside the permitted resources cut."""
    unsupported: tuple[str, ...]
    """Unrepresented composition channels, never an implicit resources proof."""

    @property
    def admitted(self) -> bool:
        """Require both the full dependency proof and represented consumers."""
        return not self.bypasses and not self.unsupported


def prove_resources_only(
    *, context: SolverBuildContext, solver: _BoundDCEGM, plan: CurrentIIDPlan
) -> ResourcesDependencyProof:
    """Read canonical declarations without sampling a value, gradient or branch.

    Typed-query intersections not represented by this producer are named before
    their dependencies could be pruned out. Current law dependencies remain in
    the context independently of whether utility reads the corresponding shock.
    """
    user_regime = context.user_regimes[context.regime_name]
    law = context.laws[context.regime_name]
    unsupported = []
    if context.phase_variation_paths:
        unsupported.append("phase-varying current action problem")
    if (
        user_regime.exits
        or user_regime.same_period_refs
        or user_regime.value_constraints
    ):
        unsupported.append("typed exit or same-period value query")
    if law.arrivals or law.gated_edges or context.arrival_queries:
        unsupported.append("arrival or gated continuation query")
    if (
        context.dcegm_source_query_model is not None
        or (context.dcegm_query_models or {}).get(context.regime_name) is not None
    ):
        unsupported.append("conditional source-query authority")
    if context.certainty_equivalent is not None and not isinstance(
        context.certainty_equivalent, LinearExpectation
    ):
        unsupported.append("nonlinear certainty equivalent")

    # This mapping is normalized but unprocessed: physical/code ancestry and
    # original helper/parameter names are still visible before routing prunes it.
    functions = MappingProxyType(
        {
            name: function
            for name, function in context.solve_functions.items()
            if name != solver.resources
        }
    )
    roots: list[tuple[str, UserFunction]] = [
        ("utility", context.solve_functions["utility"])
    ]
    if "inverse_marginal_utility" in context.solve_functions:
        roots.append(
            (
                "inverse marginal utility",
                context.solve_functions["inverse_marginal_utility"],
            )
        )
    if context.koopmans_aggregator is not None:
        roots.append(("Bellman aggregator", context.koopmans_aggregator))
    roots.extend(
        (f"constraint {name}", function)
        for name, function in context.constraints.items()
    )
    roots.extend(
        (f"declared constraint {name}{variant}", function)
        for name, declaration in user_regime.decomposed_constraints.items()
        if declaration is not None
        for variant, function in _transition_variants(value=declaration)
    )
    # Includes every declared target of every state law and every regime weight,
    # rather than only the root selected by a reachable sampled candidate.
    roots.extend(
        (label, function)
        for _, label, function in _savings_stage_candidates(
            user_regime=user_regime, law=law, solver=solver
        )
    )
    primitive_physical = plan.physical_names | frozenset(
        f"{name}_code" for name in plan.physical_names
    )
    dependencies = {
        label: _dag_ancestors(functions=functions, target_func=function)
        for label, function in roots
    }
    bypasses = {
        label: dependencies & primitive_physical
        for label, dependencies in dependencies.items()
        if dependencies & primitive_physical
    }
    internal: frozenset[ReferenceName] = frozenset(
        {
            *functions,
            solver.resources,
            solver.continuous_action,
            solver.post_decision_function,
            "utility",
            "CE",
            "marginal_continuation",
        }
    )
    row_inputs = frozenset().union(*dependencies.values()) - internal
    row_inputs |= plan.state_inputs
    if solver.continuous_state in plan.state_inputs:
        # This remains required; a zero-weight/physical-law derivative channel
        # must exist before this implementation can consume liquid-dependent laws.
        unsupported.append("liquid-dependent current law or rule derivative")
    return ResourcesDependencyProof(
        context_inputs=row_inputs,
        root_dependencies=MappingProxyType(dependencies),
        bypasses=MappingProxyType(bypasses),
        unsupported=tuple(unsupported),
    )
