"""Private dated BALANCED preparation using actual canonical call-owned operands.

The public Model admission guard remains closed. This module neither supplies
runtime parameter schemas nor fabricates an economic continuation: templates
are the genuine builders' shape inputs, and actual calls retain their statuses.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import jax

from _lcm.continuation import (
    ContinuationPayload,
    ProjectedContinuationSpec,
    ProjectedProducerDate,
)
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_envelope import (
    DatedCaseUStepAuthority,
    StateUtilityObligations,
)
from _lcm.egm.current_iid_euler_grid import EulerGridObligations, EulerGridValidations
from _lcm.egm.step import _EGMStep
from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.core_program import CoreBuildContext
from _lcm.grids import (
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution.continuation_target import union_fixed_params
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.solution.current_iid_case_r_build import (
    PreparedNativeCaseRDate,
    build_native_case_r_kernels,
    prepare_balanced_case_r_date,
    prepare_native_case_r_date,
)
from _lcm.solution.current_iid_case_u_build import CaseUArguments
from _lcm.solution.current_iid_case_u_native import (
    CaseUExactArguments,
    CaseUExactFunction,
    NativeCaseUPeriodKernel,
    PreparedCaseUDate,
    build_case_u_kernels,
    is_case_u_source,
    prepare_case_u_date,
)
from _lcm.solution.current_iid_case_u_parents import close_case_u_parent_demands
from _lcm.solution.current_iid_parent_demands import (
    close_prepared_case_r_parent_demands,
)
from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
from _lcm.typing import FlatParams
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import RegimeName

_logger = logging.getLogger("lcm")


@dataclass(frozen=True, kw_only=True)
class DatedBalancedKernels:
    """Final original programs and the contexts sharing their final declarations."""

    kernels: Mapping[RegimeName, SolutionKernels]
    contexts: Mapping[RegimeName, SolverBuildContext]
    steps: Mapping[RegimeName, EGMStepBuild]
    solvers: Mapping[RegimeName, _BoundDCEGM]
    euler_grid_obligations: EulerGridObligations = MappingProxyType({})
    euler_grid_validations: EulerGridValidations = MappingProxyType({})
    state_utility_obligations: StateUtilityObligations = MappingProxyType({})

    def validate_euler_grids(self, *, expected: EulerGridObligations) -> None:
        """Require the installed generation's exact complete original validation."""
        roster = {
            (name, period)
            for name, obligation in expected.items()
            for period in obligation.periods
        }
        if (
            self.euler_grid_obligations is not expected
            or set(self.euler_grid_validations) != roster
        ):
            raise TypeError(
                "Runtime Euler validation lost its installed obligation roster."
            )
        for (name, period), resolved in self.euler_grid_validations.items():
            obligation = expected[name]
            context = self.contexts[name]
            if (
                resolved.obligation is not obligation
                or resolved.call.period != period
                or resolved.call.ages is not context.ages
                or self.solvers[name] is not obligation.user_regime.solver
                or context.regimes_to_active_periods[name] != obligation.periods
            ):
                raise TypeError(
                    "Runtime Euler validation lost its actual installed date/solver."
                )
            resolved.require()

    def validate_state_utility(
        self,
        *,
        installed_regimes: Mapping[RegimeName, Regime] | None = None,
        fixed_flat_params: FlatParams = MappingProxyType({}),
    ) -> None:
        """Require original U authority through the actual fixed-parameter binding."""
        for name, obligation in self.state_utility_obligations.items():
            context = self.contexts[name]
            for period in obligation.periods:
                original = self.kernels[name].period_kernels[period]
                if not isinstance(original, NativeCaseUPeriodKernel):
                    raise UnsupportedOperationError(
                        "Direct-state utility requires its original U producer."
                    )
                kernel = (
                    original
                    if installed_regimes is None
                    else installed_regimes[name].solution.period_kernels[period]
                )
                step = self.steps[name].steps[period]
                if not isinstance(kernel, NativeCaseUPeriodKernel) or not isinstance(
                    step, _EGMStep
                ):
                    raise UnsupportedOperationError(
                        "Direct-state utility requires its installed U envelope."
                    )
                prepared = kernel.prepared
                declaration = context.projected_continuation_declarations[name]
                function = prepared.program.function
                if (
                    prepared.context is not context
                    or prepared.build is not self.steps[name]
                    or prepared.period != period
                    or prepared.date is not declaration.date_at(period=period)
                    or kernel.spec is not declaration
                    or not isinstance(function, CaseUExactFunction)
                    or function.point.pieces is not step.pieces
                    or function is not original.prepared.program.function
                    or prepared.template is not original.prepared.template
                ):
                    raise TypeError(
                        "Installed U envelope substituted original dated authority "
                        "or Euler pieces."
                    )
                if installed_regimes is not None:
                    _validate_bound_state_utility_arguments(
                        kernel=kernel,
                        original=original,
                        fixed_flat_params=fixed_flat_params,
                    )


def _validate_bound_state_utility_arguments(
    *,
    kernel: NativeCaseUPeriodKernel,
    original: NativeCaseUPeriodKernel,
    fixed_flat_params: FlatParams,
) -> None:
    """Validate the original argument projection after fixed-parameter binding."""
    arguments = kernel.prepared.program.argument_builder
    previous = original.prepared.program.argument_builder
    if not isinstance(arguments, CaseUExactArguments) or not isinstance(
        previous, CaseUExactArguments
    ):
        raise TypeError("Bound U envelope lost its argument projection.")
    point, before = arguments.point, previous.point
    if not isinstance(point, CaseUArguments) or not isinstance(before, CaseUArguments):
        raise TypeError(
            "Direct-state utility lost its original DC argument projection."
        )
    if (
        point.base is not before.base
        or point.plan is not before.plan
        or point.pieces is not before.pieces
        or point.period != before.period
        or point.ages is not before.ages
        or point.fixed_flat_params is not fixed_flat_params
        or arguments.demands is not previous.demands
    ):
        raise TypeError("Bound U envelope changed its original inputs.")
    for demand, pool in zip(arguments.demands, arguments.fixed_pools, strict=True):
        expected = union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=demand.source,
            transition_target_names=demand.transition_targets,
        )
        if pool.keys() != expected.keys() or any(
            pool[key] is not value for key, value in expected.items()
        ):
            raise TypeError("Bound U parent changed original operands.")


def _validate_state_utility_sources(
    *,
    obligations: StateUtilityObligations,
    contexts: Mapping[RegimeName, SolverBuildContext],
    solvers: Mapping[RegimeName, _BoundDCEGM],
    declarations: Mapping[RegimeName, ProjectedContinuationSpec],
) -> None:
    """Refuse undischargable source dates before any original Euler factory runs."""
    for name, obligation in obligations.items():
        if name not in contexts or name not in solvers or name not in declarations:
            raise UnsupportedOperationError(
                "Direct-state utility requires every actual dated U source."
            )
        context = contexts[name]
        if solvers[name] is not context.user_regimes[name].solver:
            raise TypeError("U envelope source substituted its original solver.")
        periods = context.regimes_to_active_periods[name]
        obligation.require(
            regime_name=name,
            user_regime=context.user_regimes[name],
            functions=context.solve_functions,
            periods=periods,
        )
        if set(declarations[name].dates_by_period) != set(periods):
            raise UnsupportedOperationError(
                "Direct-state utility has incomplete producer dates."
            )
        for period in periods:
            if context.current_iid_plans.get(period) is None or not is_case_u_source(
                context=context, solver=solvers[name], period=period
            ):
                raise UnsupportedOperationError(
                    "Direct-state utility requires U at every demanded date."
                )


def _same_schema(*, left: EGMCarry, right: EGMCarry) -> bool:
    """Require the original single-schema ordinary carry convention."""
    return jax.tree.structure(left) == jax.tree.structure(right) and tuple(
        (leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(left)
    ) == tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(right))


def _merge_steps(*, dates: Mapping[int, EGMStepBuild]) -> EGMStepBuild:
    """Join original dated preparations without reconstructing numerical pieces."""
    first = next(iter(dates.values()))
    if any(
        not _same_schema(left=first.carry_template, right=build.carry_template)
        or (
            first.row_discrete_state_names,
            first.row_passive_state_names,
            first.row_discrete_action_names,
            first.cell_extent,
            first.savings_point_extent,
            first.euler_point_extent,
            first.envelope_cell_extent,
        )
        != (
            build.row_discrete_state_names,
            build.row_passive_state_names,
            build.row_discrete_action_names,
            build.cell_extent,
            build.savings_point_extent,
            build.euler_point_extent,
            build.envelope_cell_extent,
        )
        for build in dates.values()
    ):
        raise UnsupportedOperationError(
            "Date-varying EGM carry geometry is not installed."
        )
    representatives = {}
    for period in sorted(dates):
        key = dates[period].step_group_keys[period]
        representatives.setdefault(key, dates[period].steps[period])
    return replace(
        first,
        native_case_u_sources=MappingProxyType(
            {
                period: source
                for build in dates.values()
                for period, source in build.native_case_u_sources.items()
            }
        ),
        steps=MappingProxyType(
            {
                period: representatives[dates[period].step_group_keys[period]]
                for period in sorted(dates)
            }
        ),
        step_group_keys=MappingProxyType(
            {period: dates[period].step_group_keys[period] for period in sorted(dates)}
        ),
        stochastic_node_axes_by_period=MappingProxyType(
            {
                period: dates[period].stochastic_node_axes_by_period[period]
                for period in sorted(dates)
            }
        ),
    )


def prepare_dated_balanced_kernels(  # noqa: C901, PLR0912, PLR0915 - one immutable dated preparation pass
    *,
    contexts: Mapping[RegimeName, SolverBuildContext],
    solvers: Mapping[RegimeName, _BoundDCEGM],
    declarations: Mapping[RegimeName, ProjectedContinuationSpec],
    ordinary_templates: Mapping[tuple[RegimeName, int], ContinuationPayload],
    ordinary_kernels: Mapping[RegimeName, SolutionKernels],
    euler_grid_obligations: EulerGridObligations = MappingProxyType({}),
    euler_grid_validations: EulerGridValidations = MappingProxyType({}),
    state_utility_obligations: StateUtilityObligations = MappingProxyType({}),
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
    materializer: DatedGenerationMaterializer | None = None,
) -> DatedBalancedKernels:
    """Prepare the date DAG, close real consumers, then assemble native programs.

    Call contexts must come from canonical parameter/support preparation. No
    scalar shape is inferred from a template annotation or a callback sample.
    The caller owns ordinary terminal kernels/templates from their actual
    publisher. Other non-DCEGM consumers remain an explicit private refusal.
    """
    if set(contexts) != set(solvers) | set(ordinary_kernels):
        raise TypeError("Dated preparation requires every original source context.")
    if set(solvers) & set(ordinary_kernels):
        raise TypeError("A source cannot have duplicate numerical build authorities.")
    if not set(declarations) <= set(solvers):
        raise TypeError("Projected producer lacks its actual DCEGM solver.")
    for name, context in contexts.items():
        if context.regime_name != name:
            raise TypeError("A prepared context belongs to another source.")
    required = {
        (name, period)
        for name in solvers
        for period in contexts[name].regimes_to_active_periods[name]
    }
    if set(call_contexts) != required:
        raise UnsupportedOperationError(
            "Dated BALANCED requires exact call-owned parameter/support schemas; "
            f"missing source dates={sorted(required - set(call_contexts))!r}; "
            f"unexpected source dates={sorted(set(call_contexts) - required)!r}. "
            "Declaration annotations cannot supply those schemas."
        )
    _validate_state_utility_sources(
        obligations=state_utility_obligations,
        contexts=contexts,
        solvers=solvers,
        declarations=declarations,
    )
    case_u_copier = None
    if materializer is not None and any(
        is_case_u_source(context=contexts[name], solver=solvers[name], period=period)
        for name in declarations
        for period in contexts[name].regimes_to_active_periods[name]
    ):
        from _lcm.simulation.entry_allocations import (  # noqa: PLC0415 - actual call-owned constructor boundary
            SimulationEntryAllocations,
        )

        if type(materializer) is not SimulationEntryAllocations:
            raise UnsupportedOperationError(
                "Cold Case-U construction requires its original entry owner."
            )
        concrete_grids = (
            LinSpacedGrid,
            LogSpacedGrid,
            IrregSpacedGrid,
            PiecewiseLinSpacedGrid,
            PiecewiseLogSpacedGrid,
        )
        for context in contexts.values():
            if (
                context.sharded_state_names
                or context.action_partitions != 1
                or len(materializer.devices) != 1
                or placed_devices_for_ids(
                    submesh_device_ids=context.submesh_device_ids,
                    visible_device_ids=tuple(d.id for d in materializer.devices),
                )
                != materializer.devices
                or any(
                    type(grid) not in concrete_grids for grid in context.grids.values()
                )
            ):
                raise UnsupportedOperationError(
                    "Cold Case-U support requires original concrete unsharded "
                    "grids on its entry device; process and custom support "
                    "producers need separate admission."
                )
        case_u_copier = materializer.copy_solution_leaf
    dates: dict[tuple[RegimeName, int], ProjectedProducerDate] = {}
    templates = dict(ordinary_templates)
    step_builds: dict[RegimeName, dict[int, EGMStepBuild]] = {
        name: {} for name in solvers
    }
    native: dict[tuple[RegimeName, int], PreparedNativeCaseRDate] = {}
    case_u: dict[tuple[RegimeName, int], PreparedCaseUDate] = {}
    source_classes: dict[RegimeName, bool] = {}
    capacity_warned: set[RegimeName] = set()
    materialization_contexts = {}
    # Strict next-date edges also handle a regime's self transition.
    for name, period in sorted(required, key=lambda item: (-item[1], item[0])):
        context = contexts[name]
        solver = solvers[name]
        call = call_contexts[name, period]
        if call.period != period or call.ages is not context.ages:
            raise TypeError("Dated preparation lost its call-owned clock.")
        if call.state_action_space is None:
            raise TypeError("Dated preparation requires actual resolved state support.")
        declared_states = context.state_action_space.states
        actual_states = call.state_action_space.states
        if tuple(declared_states) != tuple(actual_states) or any(
            (actual_states[state].shape, actual_states[state].dtype)
            != (declared_states[state].shape, declared_states[state].dtype)
            for state in declared_states
        ):
            raise UnsupportedOperationError(
                "Call-owned support changed the prepared source geometry."
            )
        targets = (
            ()
            if period == context.solution_reachability.n_periods - 1
            else context.solution_reachability.targets(period=period, source=name)
        )
        child_dates = {}
        child_templates = {}
        for target in targets:
            key = (target, period + 1)
            if target in declarations:
                if key not in dates:
                    raise TypeError("Projected child date was not genuinely prepared.")
                child_dates[target] = dates[key]
                child_templates[target] = dates[key].template
            elif key in templates:
                child_templates[target] = templates[key]
            else:
                raise UnsupportedOperationError(
                    f"Original ordinary child template is unavailable at {key!r}."
                )
        is_u = name in declarations and is_case_u_source(
            context=context,
            solver=solver,
            period=period,
        )
        plan = context.current_iid_plans.get(period)
        case_u_source = None
        if is_u:
            if plan is None:
                raise TypeError(
                    "Native U preparation requires its actual current plan."
                )
            case_u_source = DatedCaseUStepAuthority(
                context=context,
                solver=solver,
                period=period,
                date=declarations[name].date_at(period=period),
                plan=plan,
                state_utility_obligation=state_utility_obligations.get(name),
            )
        resolved_euler_grid = euler_grid_validations.get((name, period))
        if name in euler_grid_obligations and (
            resolved_euler_grid is None
            or resolved_euler_grid.obligation is not euler_grid_obligations[name]
            or resolved_euler_grid.call is not call
        ):
            raise TypeError("Dated pieces lost their actual validated Euler support.")
        build = solver.prepare_period_steps(
            resolved_euler_grid=resolved_euler_grid,
            case_u_source=case_u_source,
            context=context,
            periods=(period,),
            projected_dates=child_dates,
            materializer=materializer,
        )
        step_builds[name][period] = build
        call = replace(
            call,
            next_regime_to_continuation=MappingProxyType(child_templates),
        )
        materialization_contexts[name, period] = call
        if name not in declarations:
            templates[name, period] = build.carry_template
            continue
        if name in source_classes and source_classes[name] != is_u:
            raise UnsupportedOperationError(
                "Date-varying R/U publication families remain required."
            )
        source_classes[name] = is_u
        if is_u:
            producer_u = prepare_case_u_date(
                context=context,
                solver=solver,
                build=build,
                period=period,
                date=declarations[name].date_at(period=period),
                call=call,
                fixed_flat_params=fixed_flat_params,
                materializer=materializer,
            )
            case_u[name, period] = producer_u
            dates[name, period] = producer_u.date
            continue
        if (
            name not in capacity_warned
            and plan is not None
            and _warn_on_likely_short_projection_capacity(
                regime_name=name,
                n_nodes=int(plan.indices.shape[0]),
                n_savings=build.savings_point_extent,
                capacity=declarations[name].date_at(period=period).authority.capacity,
            )
        ):
            capacity_warned.add(name)
        producer = prepare_native_case_r_date(
            context=context,
            solver=solver,
            build=build,
            period=period,
            date=declarations[name].date_at(period=period),
            call_context=call,
            fixed_flat_params=fixed_flat_params,
            materializer=materializer,
        )
        producer = prepare_balanced_case_r_date(
            prepared=producer,
            call_context=call,
            fixed_flat_params=fixed_flat_params,
            materializer=materializer,
        )
        native[name, period] = producer
        dates[name, period] = producer.date

    final_specs = {}
    for name, declaration in declarations.items():
        actual_dates = {
            period: dates[name, period] for period in declaration.dates_by_period
        }
        first = next(iter(actual_dates.values()))
        final_specs[name] = replace(
            declaration,
            template=first.template,
            templates_by_period={
                period: date.template for period, date in actual_dates.items()
            },
            dates_by_period=actual_dates,
            guard_programs={
                period: date.guard_program
                for period, date in actual_dates.items()
                if date.guard_program is not None
            },
        )
    final_specs = MappingProxyType(final_specs)
    final_contexts = MappingProxyType(
        {
            name: replace(context, projected_continuation_declarations=final_specs)
            for name, context in contexts.items()
        }
    )
    builds = {
        name: _merge_steps(dates=periods) for name, periods in step_builds.items()
    }
    # U's installed program retains each exact original dated numerical step.
    for name in {target for target, _ in case_u}:
        builds[name] = replace(
            builds[name],
            steps=MappingProxyType(
                {
                    period: dated.steps[period]
                    for period, dated in step_builds[name].items()
                }
            ),
        )
    # Equality cannot substitute for the actual dated authority captured by a reader.
    for build in builds.values():
        for period, step in build.steps.items():
            if not isinstance(step, _EGMStep):
                raise TypeError("Dated preparation lost its original Euler pieces.")
            for target, read in step.pieces.continuation_plan.child_reads.items():
                if read.projected_date is not None and (
                    target not in final_specs
                    or read.projected_date
                    is not final_specs[target].date_at(period=period + 1)
                ):
                    raise TypeError(
                        "Prepared parent borrowed another child-date binding."
                    )
    closed_native = close_prepared_case_r_parent_demands(
        prepared=native,
        steps=builds,
        specs=final_specs,
        contexts=final_contexts,
        call_contexts=materialization_contexts,
        fixed_flat_params=fixed_flat_params,
    )
    closed_case_u = close_case_u_parent_demands(
        prepared=case_u,
        steps=builds,
        specs=final_specs,
        contexts=final_contexts,
        call_contexts=materialization_contexts,
        fixed_flat_params=fixed_flat_params,
        solvers=solvers,
        materializer=materializer,
    )
    kernels = dict(ordinary_kernels)
    for name, solver in solvers.items():
        context = final_contexts[name]
        if source_classes.get(name, False):
            kernels[name] = build_case_u_kernels(
                array_copier=case_u_copier,
                context=context,
                spec=final_specs[name],
                prepared={
                    period: closed_case_u[name, period] for period in builds[name].steps
                },
                build=builds[name],
                future_specs={
                    target: kernel.continuation_spec
                    for target, kernel in ordinary_kernels.items()
                    if kernel.continuation_spec is not None
                },
            )
        elif name in final_specs:
            kernels[name] = build_native_case_r_kernels(
                materializer=materializer,
                context=context,
                solver=solver,
                build=builds[name],
                prepared_dates={
                    period: closed_native[name, period] for period in builds[name].steps
                },
            )
        else:
            kernels[name] = solver.finish_period_kernels(
                context=context, build=builds[name], materializer=materializer
            )
    result = DatedBalancedKernels(
        kernels=MappingProxyType(kernels),
        contexts=final_contexts,
        steps=MappingProxyType(builds),
        solvers=MappingProxyType(dict(solvers)),
        euler_grid_obligations=euler_grid_obligations,
        euler_grid_validations=euler_grid_validations,
        state_utility_obligations=state_utility_obligations,
    )
    for (name, period), closed in closed_case_u.items():
        if name in state_utility_obligations:
            kernel = kernels[name].period_kernels[period]
            if not isinstance(kernel, NativeCaseUPeriodKernel) or (
                kernel.prepared.program is not closed.program
                or kernel.prepared.template is not closed.template
                or kernel.prepared.date is not closed.date
            ):
                raise TypeError("U envelope lost its actual closed producer binding.")
    result.validate_euler_grids(expected=euler_grid_obligations)
    result.validate_state_utility()
    return result


def _warn_on_likely_short_projection_capacity(
    *, regime_name: RegimeName, n_nodes: int, n_savings: int, capacity: int
) -> bool:
    """Warn when a Case-R row's estimated record count exceeds its capacity.

    The estimate `4 * n_nodes**2 * n_savings` tracks measured record counts once
    the node count is moderate; at small node counts fixed overheads dominate.
    It never refuses: the exact check runs after each solved period.

    Returns:
        Whether the warning was emitted.
    """
    estimate = 4 * n_nodes**2 * n_savings
    if estimate <= capacity:
        return False
    _logger.warning(
        "Regime '%s' may exhaust its current-IID projection capacity: an "
        "estimated %d records (4 * %d**2 shock nodes * %d savings points) exceed "
        "its capacity of %d. This is an estimate; the exact check runs after each "
        "of its periods is solved. Raise `current_iid_projection_capacity` on the "
        "solver of regime '%s' if that check refuses.",
        regime_name,
        estimate,
        n_nodes,
        n_savings,
        capacity,
        regime_name,
    )
    return True
