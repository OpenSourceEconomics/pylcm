"""Bind current U to the genuine finite NNBEGM owner and original NB banks."""

import inspect
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.certainty_equivalent import aggregates_nonlinearly
from _lcm.current_iid import CurrentIIDPlan
from _lcm.egm.branch_aggregation import DeterministicOuterMaximum
from _lcm.egm.current_iid_nnbegm import NNBEGMCaseUPoint, NNBEGMInnerSource
from _lcm.egm.current_iid_pullback import certify_resources_map
from _lcm.egm.nbegm_query_model import NBEGMIntervalRecipe
from _lcm.egm.outer_inversion import certify_declared_outer_inverse
from _lcm.egm.outer_search import FiniteOuterGrid
from _lcm.egm.preferences import concatenate_regime_function
from _lcm.egm.validation import _dag_ancestors
from _lcm.execution.core_program import CoreBuildContext
from _lcm.regime_building.age_normalization import resolve_periodized_nodes
from _lcm.solution.continuation_target import union_fixed_params, union_free_params
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.current_iid_case_r_build import (
    _PositionalResources,
    _ResourcesStructureCheck,
)
from _lcm.solution.current_iid_negm import PreparedNEGMCaseUPoint
from _lcm.solution.nbegm import (
    _NBEGMQueryIntervalEvaluators,
    _NBEGMTiledCore,
    _RideAlongNBEGMPeriodKernel,
)
from _lcm.solution.negm import _with_no_adjustment_outer_function
from _lcm.solution.nnbegm import _BoundNNBEGM, _FiniteNNBEGMPeriodKernel
from _lcm.typing import EconFunctionsMapping, FlatParams, PytreeValue
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import Discrete


def prepare_original_nnbegm_u_point(
    *,
    context: SolverBuildContext,
    solver: _BoundNNBEGM,
    call: CoreBuildContext,
    fixed_flat_params: FlatParams,
    period: int,
) -> PreparedNEGMCaseUPoint:
    """Keep the original nested builder, both actual NB cores and host checks."""
    plan = context.current_iid_plans.get(period)
    space = call.state_action_space
    if (
        plan is None
        or space is None
        or call.period != period
        or call.ages is not context.ages
        or solver is not context.user_regimes[context.regime_name].solver
    ):
        raise TypeError("NNBEGM U requires its original dated call-owned source.")
    _admit_finite_nnbegm_u(context=context, solver=solver, plan=plan)
    if (
        type(solver.outer_search) is not FiniteOuterGrid
        or type(solver.branch_aggregator) is not DeterministicOuterMaximum
    ):
        raise UnsupportedOperationError(
            "NNBEGM U requires its original finite deterministic outer sweep."
        )
    solver.validate_build(context=context)
    built = solver.build_period_kernels(context=replace(context, enable_jit=False))
    owner = built.period_kernels[period]
    if type(owner) is not _FiniteNNBEGMPeriodKernel:
        raise TypeError("NNBEGM U lost its genuine finite nested owner.")
    sources = []
    for kernel in (owner.keeper_kernel, owner.adjuster_kernel):
        if type(kernel) is not _RideAlongNBEGMPeriodKernel:
            raise TypeError("NNBEGM U lost its genuine inner ride kernel.")
        core = kernel.core_programs()["main"].function
        if type(core) is not _NBEGMTiledCore:
            raise TypeError("NNBEGM U lost its original inner numerical core.")
        evaluator = _NBEGMQueryIntervalEvaluators(
            sources=core.schedule_spec.sources,
            liquid_name=solver.inner.continuous_state,
        )
        inputs = frozenset(
            name
            for source in core.schedule_spec.sources
            for name in (
                source.threshold_param_name,
                *source.derived_param_names,
                *source.derived_state_names,
                *(
                    ()
                    if source.threshold_index_state is None
                    else (source.threshold_index_state,)
                ),
            )
        )
        sources.append(
            NNBEGMInnerSource(
                kernel=kernel,
                core=core,
                borrowing_limit=core.envelope_build.bind_cell.savings_grid[0],
                recipe=NBEGMIntervalRecipe(
                    source_identity=core.schedule_spec.sources,
                    input_names=inputs,
                    equal_above=tuple(
                        source.equality_owner == "above"
                        for source in core.schedule_spec.sources
                    ),
                    evaluate_preimages=evaluator.preimages,
                    evaluate_sides=evaluator.sides,
                ),
            )
        )
    keeper, adjuster = sources
    source_grid = space.states[solver.inner.continuous_state]
    outer_grid = space.states[solver.outer_state]
    targets = tuple(context.transitions)
    actual_pool: dict[str, PytreeValue] = {
        **union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=context.regime_name,
            transition_target_names=targets,
        ),
        **union_free_params(
            flat_params=MappingProxyType(call.flat_params),
            regime_name=context.regime_name,
            transition_target_names=targets,
        ),
        **{name: values[0] for name, values in space.states.items()},
        solver.outer_post_decision: owner.outer_grid_values[0],
        solver.outer_action: outer_grid[0],
        solver.inner.continuous_action: source_grid[0],
        "period": jnp.int32(period),
        "age": context.ages.values[period],
    }
    physical_dtypes = {
        name: jnp.dtype(jnp.int32) if isinstance(law, Discrete) else source_grid.dtype
        for _, binding, _ in plan.bindings
        for name, law in binding.group.shocks.items()
    }
    checks = []
    coefficients = []
    for inner in (keeper, adjuster):
        original = inner.core.schedule_spec.coh_of_liquid_dag
        names = tuple(inspect.signature(original).parameters)
        function = _PositionalResources(function=original, names=names)
        arguments = tuple(
            jax.ShapeDtypeStruct(value.shape, value.dtype)
            if isinstance(value := actual_pool.get(name), jax.Array)
            else jax.ShapeDtypeStruct((), physical_dtypes.get(name, source_grid.dtype))
            for name in names
        )
        proof = certify_resources_map(
            func=function,
            arg_names=names,
            abstract_args=arguments,
            state_name=solver.inner.continuous_state,
        )
        if proof.violation is not None or proof.coefficient is None:
            raise UnsupportedOperationError(
                "NNBEGM U requires original positive affine resource maps."
            )
        coefficients.append(proof.coefficient)
        checks.append(
            _ResourcesStructureCheck(
                function=function,
                state=solver.inner.continuous_state,
                physical_dtypes=physical_dtypes,
                regime=context.regime_name,
                targets=targets,
                coefficient=proof.coefficient,
                dtype=source_grid.dtype,
            )
        )
    if coefficients[0] != coefficients[1]:
        raise UnsupportedOperationError("NNBEGM U requires a common resource Jacobian.")
    functions = cast(
        "EconFunctionsMapping",
        resolve_periodized_nodes(mapping=context.functions, period=period),
    )
    keeper_functions = _with_no_adjustment_outer_function(
        functions=functions,
        durable_state=solver.outer_state,
        outer_post_decision=solver.outer_post_decision,
        no_adjustment_func=(
            None
            if solver.outer_no_adjustment_candidate is None
            else functions[solver.outer_no_adjustment_candidate]
        ),
    )
    outer_function = concatenate_functions(
        functions=functions,
        targets=[solver.outer_post_decision],
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    names = tuple(inspect.signature(outer_function).parameters)
    inverse = certify_declared_outer_inverse(
        func=outer_function,
        arg_names=names,
        abstract_args=tuple(
            jax.ShapeDtypeStruct(value.shape, value.dtype)
            if isinstance(value := actual_pool.get(name), jax.Array)
            else jax.ShapeDtypeStruct((), physical_dtypes.get(name, source_grid.dtype))
            for name in names
        ),
        outer_action_name=solver.outer_action,
        outer_post_decision_name=solver.outer_post_decision,
        outer_state_domain=(float(outer_grid[0]), float(outer_grid[-1])),
        regime_name=context.regime_name,
    )
    return PreparedNEGMCaseUPoint(
        point=NNBEGMCaseUPoint(
            context=context,
            plan=plan,
            period=period,
            owner=owner,
            keeper=keeper,
            adjuster=adjuster,
            source_grid=source_grid,
            savings_nodes=keeper.savings_nodes,
            borrowing_limit=keeper.borrowing_limit,
            outer_nodes=owner.outer_grid_values,
            coefficient=coefficients[0],
            outer_state=solver.outer_state,
            outer_action=solver.outer_action,
            keeper_target=concatenate_regime_function(
                functions=keeper_functions, target=solver.outer_post_decision
            ),
            outer_function=outer_function,
            outer_inverse=inverse,
        ),
        param_checks=(*built.param_checks, *checks),
        state_names=tuple(space.states),
    )


def _admit_finite_nnbegm_u(
    *, context: SolverBuildContext, solver: _BoundNNBEGM, plan: CurrentIIDPlan
) -> None:
    """Refuse unsupported intersections before the genuine original inner build."""
    user = context.user_regimes[context.regime_name]
    law = context.laws[context.regime_name]
    if (
        set(context.state_action_space.states)
        != {solver.inner.continuous_state, solver.outer_state}
        or context.state_action_space.discrete_actions
        or context.has_taste_shocks
        or context.sharded_state_names
        or context.co_map_state_names
        or context.action_partitions != 1
        or context.phase_variation_paths
        or aggregates_nonlinearly(context.certainty_equivalent)
        or context.arrival_queries
        or law.arrivals
        or law.gated_edges
        or user.exits
        or user.same_period_refs
        or user.value_constraints
        or (
            solver.inner.post_decision_bound is not None
            and solver.inner.post_decision_bound.kind != "literal"
        )
        or any(binding.state_inputs for _name, binding, _nodes in plan.bindings)
    ):
        raise UnsupportedOperationError(
            "Finite NNBEGM U requires unsharded liquid/durable additive ordinary "
            "branches, a literal floor and a current law independent of pre-state. "
            "Adaptive/CE/arrival/query/moving-floor intersections remain required."
        )
    opaque = {
        name: function
        for name, function in context.solve_functions.items()
        if name != solver.inner.budget_target
    }
    roots = [context.solve_functions["utility"]]
    if "inverse_marginal_utility" in opaque:
        roots.append(opaque["inverse_marginal_utility"])
    if "discount_factor" in opaque:
        roots.append(opaque["discount_factor"])
    dependencies = tuple(
        _dag_ancestors(functions=opaque, target_func=function) for function in roots
    )
    if any(
        solver.inner.continuous_state in inputs or solver.inner.budget_target in inputs
        for inputs in dependencies
    ):
        raise UnsupportedOperationError(
            "Finite NNBEGM U direct-liquid/resource utility, inverse and discount "
            "directions remain required."
        )
