"""Close actual deterministic Euler parents over one-axis NB U producers."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.current_iid_resources_proof import prove_resources_only
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import CoreBuildContext
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.reachability import PhaseReachability
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_dated import DatedBalancedKernels
from _lcm.solution.current_iid_query_exclusions import (
    require_transition_only_projected_reads,
)
from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
from _lcm.solution.nbegm import _BoundNBEGM
from _lcm.typing import FlatParams
from lcm.exceptions import UnsupportedOperationError
from lcm.regime import Regime as UserRegime
from lcm.typing import RegimeName

_ONE_PASSIVE_STATE_COUNT = 2


def nd_parent_capacity_bounds(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    reachability: PhaseReachability,
    materializer: DatedGenerationMaterializer | None,
    projected_targets: frozenset[RegimeName],
) -> Mapping[RegimeName, int]:
    """Bound mandatory original parent coordinates before child allocation."""
    if materializer is not None:
        # The original cold ND guard still refuses before any producer.
        return MappingProxyType({})
    result: dict[RegimeName, int] = {}
    for source, period in call_contexts:
        solver = user_regimes[source].solver
        if type(solver) is not _BoundDCEGM:
            continue
        for target in reachability.targets(period=period, source=source):
            child = user_regimes[target].solver
            child_call = call_contexts.get((target, period + 1))
            if (
                target not in projected_targets
                or type(child) is not _BoundNBEGM
                or child_call is None
                or child_call.state_action_space is None
                or len(child_call.state_action_space.states) != _ONE_PASSIVE_STATE_COUNT
            ):
                continue
            # Reuse the original grid factory. The exact final coordinates are
            # still produced by ParentResourcesDemand from runtime parameters.
            width = solver.savings_grid.to_jax().shape[0]
            result[target] = result.get(target, 0) + width
    return MappingProxyType(result)


def prepare_nb_parent_kernels(  # noqa: C901 - original dated parent/source admission before preparation
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    nb_solvers: Mapping[RegimeName, _BoundNBEGM],
    parents: Mapping[RegimeName, _BoundDCEGM],
    declarations: Mapping[RegimeName, ProjectedContinuationSpec],
    ordinary_kernels: Mapping[RegimeName, SolutionKernels],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer | None,
) -> DatedBalancedKernels:
    """Prepare original parent pieces, close child demands, then expose programs."""
    from _lcm.solution.current_iid_nbegm import (  # noqa: PLC0415 - preserve original declaration import order
        prepare_scalar_nbegm_u_kernels,
    )

    if materializer is not None:
        raise UnsupportedOperationError(
            "Cold incoming ND U admission remains required."
        )
    if len(parents) != 1 or len(nb_solvers) != 1:
        raise UnsupportedOperationError(
            "ND U closure requires one actual parent/child route."
        )
    for target, solver in nb_solvers.items():
        context = contexts[target]
        periods = context.regimes_to_active_periods[target]
        if (
            len(periods) != 1
            or len(context.state_action_space.states) != _ONE_PASSIVE_STATE_COUNT
        ):
            raise UnsupportedOperationError(
                "ND U closure requires one passive child axis/date."
            )
        plan = context.current_iid_plans.get(periods[0])
        if (
            plan is None
            or not prove_resources_only(
                context=context, solver=solver, plan=plan
            ).bypasses
        ):
            raise UnsupportedOperationError(
                "ND incoming resources-only/noncurrent NB closure remains required."
            )
    builds: dict[RegimeName, EGMStepBuild] = {}
    demands: dict[tuple[RegimeName, int], list[ParentResourcesDemand]] = {}
    for source, solver in parents.items():
        context = contexts[source]
        user = context.user_regimes[source]
        periods = context.regimes_to_active_periods[source]
        if (
            len(periods) != 1
            or context.state_action_space.discrete_actions
            or len(context.state_action_space.states) != 1
            or any(plan is not None for plan in context.current_iid_plans.values())
            or context.has_taste_shocks
            or context.sharded_state_names
            or context.co_map_state_names
            or context.phase_variation_paths
            or context.arrival_queries
            or user.exits
            or user.same_period_refs
            or user.value_constraints
        ):
            raise UnsupportedOperationError(
                "ND U parents require their original scalar deterministic Euler path."
            )
        (period,) = periods
        targets = context.solution_reachability.targets(period=period, source=source)
        if len(targets) != 1 or targets[0] not in nb_solvers:
            raise UnsupportedOperationError(
                "ND U parent requires one original NB child at the next date."
            )
        (target,) = targets
        require_transition_only_projected_reads(target=target, contexts=contexts)
        build = solver.prepare_period_steps(
            context=context, periods=periods, projected_dates=declarations
        )
        step = build.steps[period]
        if not isinstance(step, _EGMStep):
            raise UnsupportedOperationError(
                "ND U parent lost its original Euler pieces."
            )
        builds[source] = build
        demands.setdefault((target, period + 1), []).append(
            ParentResourcesDemand(
                source=source,
                period=period,
                target=target,
                spec=declarations[target],
                pieces=step.pieces,
                transition_targets=tuple(context.transitions),
            )
        )
    prepared = prepare_scalar_nbegm_u_kernels(
        contexts=contexts,
        solvers=nb_solvers,
        declarations=declarations,
        ordinary_kernels=ordinary_kernels,
        call_contexts=call_contexts,
        fixed_flat_params=fixed_flat_params,
        materializer=None,
        parent_demands=MappingProxyType(
            {key: tuple(value) for key, value in demands.items()}
        ),
    )
    actual_contexts = dict(prepared.contexts)
    specs = {name: prepared.kernels[name].continuation_spec for name in nb_solvers}
    if any(specs[name] is not declarations[name] for name in nb_solvers):
        raise TypeError("ND closure replaced the original parent-bound declarations.")
    kernels = dict(prepared.kernels)
    for source, solver in parents.items():
        context = replace(
            contexts[source], projected_continuation_declarations=declarations
        )
        actual_contexts[source] = context
        kernels[source] = solver.finish_period_kernels(
            context=context, build=builds[source]
        )
    return DatedBalancedKernels(
        kernels=MappingProxyType(kernels),
        contexts=MappingProxyType(actual_contexts),
        steps=MappingProxyType(builds),
        solvers=MappingProxyType(dict(parents)),
    )
