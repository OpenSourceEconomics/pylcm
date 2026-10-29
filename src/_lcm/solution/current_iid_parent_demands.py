"""Authenticate complete scalar Euler consumers before exact native publication."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.egm.step import _EGMStep
from _lcm.egm.step_core import _EgmKernelPieces
from _lcm.execution.core_program import CoreBuildContext
from _lcm.solution.continuation_target import union_fixed_params
from _lcm.solution.contract import PeriodKernel, SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_r import (
    CaseRArguments,
    CaseRProgramFunction,
    NativeCaseRPeriodKernel,
)
from _lcm.solution.current_iid_case_r_build import PreparedNativeCaseRDate
from _lcm.solution.current_iid_query_exclusions import (
    require_transition_only_projected_reads,
)
from _lcm.solution.dcegm import EGMStepBuild, _DCEGMPeriodKernel
from _lcm.typing import FlatParams
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import RegimeName


def _parent_pieces(*, kernel: PeriodKernel | None) -> _EgmKernelPieces:
    """Recover the actual consumer without unwrapping or rebuilding its kernels."""
    if isinstance(kernel, _DCEGMPeriodKernel):
        if not isinstance(kernel.demand_step, _EGMStep):
            raise UnsupportedOperationError(
                "Exact projected demands require an original scalar EGM step."
            )
        return kernel.demand_step.pieces
    if isinstance(kernel, NativeCaseRPeriodKernel):
        function = kernel.programs["main"].function
        if not isinstance(function, CaseRProgramFunction):
            raise TypeError("Native parent lost its original program.")
        return function.core.pieces
    raise UnsupportedOperationError(
        "An incoming projected consumer has no closed scalar Euler declaration."
    )


def _incoming_parent_dates(
    *,
    target: RegimeName,
    target_period: int,
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> tuple[tuple[RegimeName, int], ...]:
    """Enumerate actual incoming edges without consulting numerical probabilities."""
    parent_period = target_period - 1
    if parent_period < 0:
        return ()
    return tuple(
        (source, parent_period)
        for source, context in contexts.items()
        if target
        in context.solution_reachability.targets(period=parent_period, source=source)
    )


def _prepared_parent_pieces(
    *,
    context: SolverBuildContext,
    period: int,
    steps: Mapping[RegimeName, EGMStepBuild],
) -> _EgmKernelPieces:
    """Refuse an unrepresented actual source before looking up its period core."""
    build = steps.get(context.regime_name)
    step = None if build is None else build.steps.get(period)
    if not isinstance(step, _EGMStep):
        raise UnsupportedOperationError(
            f"Incoming projected consumer {context.regime_name!r} at period "
            f"{period} lacks original scalar Euler pieces."
        )
    return step.pieces


def close_prepared_case_r_parent_demands(
    *,
    prepared: Mapping[tuple[RegimeName, int], PreparedNativeCaseRDate],
    steps: Mapping[RegimeName, EGMStepBuild],
    specs: Mapping[RegimeName, ProjectedContinuationSpec],
    contexts: Mapping[RegimeName, SolverBuildContext],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
) -> Mapping[tuple[RegimeName, int], PreparedNativeCaseRDate]:
    """Close actual consumers onto frozen producer schemas before CorePrograms."""
    result = {}
    for (target, target_period), producer in prepared.items():
        spec = specs[target]
        if producer.date is not spec.date_at(period=target_period):
            raise TypeError("Prepared producer lost its final aggregate date identity.")
        require_transition_only_projected_reads(target=target, contexts=contexts)
        declarations = []
        for source, parent_period in _incoming_parent_dates(
            target=target, target_period=target_period, contexts=contexts
        ):
            parent_pieces = _prepared_parent_pieces(
                context=contexts[source], period=parent_period, steps=steps
            )
            declarations.append(
                ParentResourcesDemand(
                    source=source,
                    period=parent_period,
                    target=target,
                    spec=spec,
                    pieces=(
                        prepared[source, parent_period].core.pieces
                        if (source, parent_period) in prepared
                        else parent_pieces
                    ),
                    transition_targets=tuple(contexts[source].transitions),
                )
            )
        demands = tuple(declarations)
        pools = tuple(
            union_fixed_params(
                fixed_flat_params=fixed_flat_params,
                regime_name=demand.source,
                transition_target_names=demand.transition_targets,
            )
            for demand in demands
        )
        arguments = replace(
            producer.arguments, parent_demands=demands, fixed_demand_pools=pools
        )
        core = replace(producer.core, parent_demands=demands)
        call = call_contexts[target, target_period]
        if call.period != target_period or call.ages is not contexts[target].ages:
            raise TypeError("Demand closure lost its original date/clock.")
        # The frozen template declares the program's output roles; lowering
        # checks the closed core's output against them, so no shape re-trace.
        result[target, target_period] = replace(
            producer, core=core, arguments=arguments
        )
    return MappingProxyType(result)


def close_case_r_parent_demands(
    *,
    kernels: Mapping[RegimeName, SolutionKernels],
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> Mapping[RegimeName, SolutionKernels]:
    """Bind only original prepared readers; reject any unrepresented consumer.

    This runs after all producers exist and before native programs are lowered.
    Every actual incoming source/date is visited even when its probability or
    discount is zero. No caller-supplied probe list can authorize publication.
    """
    result = dict(kernels)
    if set(kernels) != set(contexts):
        raise TypeError("Exact projected closure requires every source kernel set.")
    for target, target_kernels in kernels.items():
        if not any(
            isinstance(kernel, NativeCaseRPeriodKernel)
            for kernel in target_kernels.period_kernels.values()
        ):
            continue
        require_transition_only_projected_reads(target=target, contexts=contexts)
        period_kernels = dict(target_kernels.period_kernels)
        for target_period, target_kernel in period_kernels.items():
            if not isinstance(target_kernel, NativeCaseRPeriodKernel):
                continue
            if target_kernel.period != target_period:
                raise TypeError("Projected closure cannot redate its producer kernel.")
            declarations = []
            for source, parent_period in _incoming_parent_dates(
                target=target, target_period=target_period, contexts=contexts
            ):
                context = contexts[source]
                parent_kernel = kernels[source].period_kernels.get(parent_period)
                pieces = _parent_pieces(kernel=parent_kernel)
                declarations.append(
                    ParentResourcesDemand(
                        source=source,
                        period=parent_period,
                        target=target,
                        spec=target_kernel.spec,
                        pieces=pieces,
                        transition_targets=tuple(context.transitions),
                    )
                )
            period_kernels[target_period] = _with_parent_demands(
                kernel=target_kernel, demands=tuple(declarations)
            )
        result[target] = replace(
            target_kernels, period_kernels=MappingProxyType(period_kernels)
        )
    return MappingProxyType(result)


def _with_parent_demands(
    *, kernel: NativeCaseRPeriodKernel, demands: tuple[ParentResourcesDemand, ...]
) -> NativeCaseRPeriodKernel:
    """Bind the authenticated population into the original native graph variants."""
    first = kernel.programs["main"].function
    if not isinstance(first, CaseRProgramFunction):
        raise TypeError("Projected producer lost its original native graph.")
    if first.core.balanced is not None:
        existing = first.core.parent_demands
        if (
            existing is None
            or len(existing) != len(demands)
            or any(
                old.source != new.source
                or old.period != new.period
                or old.target != new.target
                or old.spec is not new.spec
                or old.pieces is not new.pieces
                or old.transition_targets != new.transition_targets
                for old, new in zip(existing, demands, strict=True)
            )
        ):
            raise TypeError("Final declaration differs from preclosed parent demands.")
        return kernel
    programs = {}
    for name, program in kernel.programs.items():
        function = program.function
        arguments = program.argument_builder
        if not isinstance(function, CaseRProgramFunction) or not isinstance(
            arguments, CaseRArguments
        ):
            raise TypeError("Projected producer lost its declared native graph.")
        programs[name] = replace(
            program,
            function=replace(
                function, core=replace(function.core, parent_demands=demands)
            ),
            argument_builder=replace(arguments, parent_demands=demands),
        )
    return replace(kernel, programs=programs)
