"""Borrow already declared native inputs for a dated arrival reconstruction."""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from _lcm.arrival_continuation import (
    ARRIVAL_CONTINUATION,
    ArrivalContinuation,
    ArrivalContinuationSpec,
    ArrivalOperand,
)
from _lcm.continuation import ContinuationPayload
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ValueRead,
    core_program_graph,
)
from _lcm.execution.output_layout import VALUE, StateAxesLeading
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.solution.contract import PeriodKernel, SolverBuildContext
from _lcm.time import TimeAxis
from _lcm.typing import (
    ArrivalQueryArguments,
    FlatParams,
    PytreeValue,
    ShapeDtypePytree,
    ValueTemplate,
)
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.solver_api import ContinuationReader, KernelOutput
from lcm.typing import FloatND, IntND, RegimeName

if TYPE_CHECKING:
    from _lcm.arrival_queries import ArrivalQueryPlan
    from _lcm.engine import Regime, StateActionSpace
else:
    # Plans are built by `_lcm.arrival_queries`, which imports this module.
    type ArrivalQueryPlan = object
    # Canonical construction imports the engine before installing these adapters.
    type Regime = object
    type StateActionSpace = object


_PUBLICATION_OPERANDS = "_lcm_arrival_publication_operands"


@dataclass(frozen=True, kw_only=True)
class _PublicationArguments:
    """Expose only missing original leaves at the genuine Full publication core."""

    base: CoreArgumentBuilder
    origins: Mapping[str, ValueArtifactAddress]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", MappingProxyType(dict(self.origins)))

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        arguments = dict(self.base(context))
        if _PUBLICATION_OPERANDS in arguments:
            raise ExecutionPlanningError(
                "Arrival publication operands collide with a native argument."
            )
        # As in the ordinary value argument builder, this mapping is wholly
        # concrete at dispatch and wholly abstract during materialization.
        arguments[_PUBLICATION_OPERANDS] = cast(
            "PytreeValue | ShapeDtypePytree",
            MappingProxyType(
                _borrow_operands(
                    origins=self.origins,
                    period=context.period,
                    values=context.next_regime_to_V_arr,
                    continuations=context.next_regime_to_continuation,
                )
            ),
        )
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True, eq=False)
class _PublicationCore:
    """Retain genuine Full outputs; borrowed storage publishes outside arithmetic."""

    base: Callable[..., PytreeValue]

    def __call__(self, **arguments: PytreeValue) -> PytreeValue:
        if _PUBLICATION_OPERANDS not in arguments:
            raise ExecutionPlanningError(
                "Declared arrival custody operands are missing."
            )
        return self.base(
            **{
                name: value
                for name, value in arguments.items()
                if name != _PUBLICATION_OPERANDS
            }
        )


@dataclass(frozen=True, kw_only=True)
class _PublicationCall:
    """Call-local explicit dynamic inputs, never captured by the numerical callable."""

    compiled: Callable[..., PytreeValue]
    operands: Mapping[str, ArrivalOperand]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", MappingProxyType(dict(self.operands)))

    def __call__(self, **arguments: PytreeValue) -> PytreeValue:
        if _PUBLICATION_OPERANDS in arguments:
            raise ExecutionPlanningError(
                "Arrival custody operands were supplied twice."
            )
        return self.compiled(**arguments, **{_PUBLICATION_OPERANDS: self.operands})


def _publication_programs(
    *,
    graph: Mapping[str, CoreProgram],
    origins: Mapping[str, ValueArtifactAddress],
    owner: RegimeName,
    period: int,
) -> Mapping[str, CoreProgram]:
    """Count additional custody reads on existing genuine VALUE publishers only."""
    native = {
        read.target
        for program in graph.values()
        for read in program.requirements.value_reads
        if read.source.source_period == period and read.source.source_regime == owner
    }
    missing = {name: origin for name, origin in origins.items() if origin not in native}
    if not missing:
        return graph
    publishers = {
        name: program
        for name, program in graph.items()
        if any(role is VALUE for role in jax.tree.leaves(program.output_roles))
    }
    if not publishers:
        raise UnsupportedOperationError(
            "Arrival custody requires the genuine native Full VALUE publisher."
        )
    updated = dict(graph)
    for name, program in publishers.items():
        reads = []
        for operand, origin in missing.items():
            if origin.period != period + 1 or origin.kind not in (
                ValueArtifactKind.REGIME_VALUE,
                ValueArtifactKind.CONTINUATION_LEAF,
            ):
                raise UnsupportedOperationError(
                    "Additional arrival custody needs its next-date stored original."
                )
            reads.append(
                ValueRead(
                    target=origin,
                    source=ValueConsumerAddress(
                        source_period=period,
                        source_regime=owner,
                        core_key=name,
                        channel=ValueInputChannel.NEXT_REGIME_VALUE
                        if origin.kind is ValueArtifactKind.REGIME_VALUE
                        else ValueInputChannel.CONTINUATION_LEAF,
                        argument=_PUBLICATION_OPERANDS,
                        path=(operand,),
                    ),
                )
            )
        updated[name] = replace(
            program,
            function=_PublicationCore(base=program.function),
            argument_builder=_PublicationArguments(
                base=program.argument_builder, origins=missing
            ),
            requirements=replace(
                program.requirements,
                value_reads=(*program.requirements.value_reads, *reads),
            ),
            checked_variant=None
            if program.checked_variant is None
            else replace(
                program.checked_variant,
                function=_PublicationCore(base=program.checked_variant.function),
            ),
        )
    return MappingProxyType(updated)


_SURFACES = "_arrival_surfaces"


@dataclass(frozen=True, kw_only=True)
class _SurfaceGrids:
    """The target grid nodes each stored arrival surface is tabulated on."""

    grids: Mapping[str, Mapping[str, FloatND | IntND]]

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        del context
        return MappingProxyType(
            {"grids": cast("PytreeValue", MappingProxyType(dict(self.grids)))}
        )


@dataclass(frozen=True, kw_only=True)
class _SurfaceArguments:
    """The publisher's own arrival operands, as its action problem receives them."""

    arguments: CoreArgumentBuilder
    fixed_flat_params: FlatParams = MappingProxyType({})

    def __call__(
        self, context: CoreBuildContext
    ) -> Mapping[str, PytreeValue | ShapeDtypePytree]:
        from _lcm.policy_queries import _query_flat_params  # noqa: PLC0415

        return self.arguments(
            replace(
                context,
                flat_params=_query_flat_params(
                    flat_params=cast("FlatParams", context.flat_params),
                    fixed_flat_params=self.fixed_flat_params,
                ),
            )
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class _ArrivalSurfaces:
    """Tabulate each arrival's value on its target's grid nodes.

    At every node the arrival plan rebuilds the target's action problem exactly,
    as a reader at the landing does. A node whose answer fails its integrity
    check is NaN, so a reader interpolating from it is NaN too.
    """

    plans: Mapping[str, ArrivalQueryPlan]
    width: int

    def __call__(
        self,
        *,
        grids: Mapping[str, Mapping[str, FloatND | IntND]],
        arrival_arguments: ArrivalQueryArguments,
    ) -> dict[str, FloatND]:
        by_target = arrival_arguments.by_target
        surfaces = {}
        for name, plan in self.plans.items():
            states = grids[name]
            arguments = by_target[plan.target]

            # keyword-only-exempt: library-callback=jax.lax.map
            def answer(
                point: Mapping[str, FloatND | IntND],
                *,
                plan: ArrivalQueryPlan = plan,
                arguments: Mapping[str, Mapping[str, PytreeValue]] = arguments,
            ) -> FloatND:
                result = plan.queries(states=point, arguments=arguments)
                return jnp.where(result[3], result[0], jnp.nan)

            if not states:
                surfaces[name] = answer({})
                continue
            extents = tuple(nodes.size for nodes in states.values())
            mesh = jnp.meshgrid(*states.values(), indexing="ij")
            points = {
                state: coordinate.reshape(-1)
                for state, coordinate in zip(states, mesh, strict=True)
            }
            size = math.prod(extents)
            values = jax.lax.map(answer, points, batch_size=min(self.width, size))
            surfaces[name] = values.reshape(*extents, *values.shape[1:])
        return surfaces


def _surface_program(
    *,
    base: PeriodKernel,
    owner: RegimeName,
    period: int,
    plans: Mapping[str, ArrivalQueryPlan],
    grids: Mapping[str, Mapping[str, FloatND | IntND]],
    templates: Mapping[str, ArrivalOperand],
    width: int,
) -> CoreProgram:
    """Declare the native program computing the publisher's stored surfaces."""
    from _lcm.arrival_queries import (  # noqa: PLC0415 - canonical query graphs import this module
        ArrivalSourceArguments,
        arrival_source_value_reads,
    )

    by_target = {plan.target: plan for plan in plans.values()}
    schedules = [
        program.requirements.internal_schedule
        for program in core_program_graph(kernel=base).values()
    ]
    if any(schedule != schedules[0] for schedule in schedules):
        raise ExecutionPlanningError(
            "An arrival surface needs its publisher's one closed schedule."
        )
    return CoreProgram(
        name=_SURFACES,
        function=_ArrivalSurfaces(plans=MappingProxyType(dict(plans)), width=width),
        argument_builder=_SurfaceArguments(
            arguments=ArrivalSourceArguments(
                base=_SurfaceGrids(grids=MappingProxyType(dict(grids))),
                plans=MappingProxyType(by_target),
            ),
            fixed_flat_params=getattr(base, "fixed_flat_params", MappingProxyType({})),
        ),
        requirements=CoreExecutionRequirements(
            value_reads=tuple(
                replace(read, source=replace(read.source, core_key=_SURFACES))
                for read in arrival_source_value_reads(
                    plans=by_target, source=owner, period=period
                )
            ),
            internal_schedule=schedules[0],
        ),
        output_roles={
            name: StateAxesLeading(
                state_names=(), shape=template.shape, dtype=template.dtype
            )
            for name, template in templates.items()
        },
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason=(
            "Arrival surfaces are tabulated over the target grid in a bounded "
            "sequential map; one plan answer per node."
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _BorrowedArrivalKernel:
    """Publish exact input identities only after the genuine Full solve succeeds."""

    base: PeriodKernel
    template: ArrivalContinuation
    origins: Mapping[str, ValueArtifactAddress]
    surfaces: Mapping[str, ArrivalQueryPlan] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """The publisher's own arrival plan behind each stored surface leaf."""
    surface_grids: Mapping[str, Mapping[str, FloatND | IntND]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Target grid nodes of each stored surface leaf, at the landing date."""
    surface_width: int = 1
    """Grid nodes tabulated at once."""
    _core_programs: Mapping[str, CoreProgram] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", MappingProxyType(dict(self.origins)))
        programs = dict(
            _publication_programs(
                graph=core_program_graph(kernel=self.base),
                origins=self.origins,
                owner=self.template.regime,
                period=self.template.period,
            )
        )
        if self.surfaces:
            if _SURFACES in programs:
                raise ExecutionPlanningError(
                    "Arrival surfaces collide with a native publication program."
                )
            programs[_SURFACES] = _surface_program(
                base=self.base,
                owner=self.template.regime,
                period=self.template.period,
                plans=self.surfaces,
                grids=self.surface_grids,
                templates={
                    name: self.template.operands[name] for name in self.surfaces
                },
                width=self.surface_width,
            )
        object.__setattr__(self, "_core_programs", MappingProxyType(programs))

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Retain the genuine producer's numerical graph and all native readers."""
        return self._core_programs

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _BorrowedArrivalKernel:
        return replace(
            self, base=self.base.with_fixed_params(fixed_flat_params=fixed_flat_params)
        )

    def narrow_workspace(self) -> PeriodKernel | None:
        """Preserve the genuine producer's existing bounded-workspace alternatives."""
        narrow = getattr(self.base, "narrow_workspace", None)
        if not callable(narrow):
            return None
        candidate = narrow()
        return None if candidate is None else replace(self, base=candidate)

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelOutput:
        if period != self.template.period:
            raise ExecutionPlanningError("Arrival publisher received the wrong date.")
        operands = _borrow_operands(
            origins=self.origins,
            period=period,
            values=next_regime_to_V_arr,
            continuations=next_regime_to_continuation,
        )
        optional = {}
        if same_period_regime_to_V_arr is not None:
            optional["same_period_regime_to_V_arr"] = same_period_regime_to_V_arr
        if edge_regime_to_V_arr is not None:
            optional["edge_regime_to_V_arr"] = edge_regime_to_V_arr
        programs = self.core_programs()
        dynamic_cores = {}
        for name, compiled in compiled_cores.items():
            builder = programs[name].argument_builder
            dynamic_cores[name] = (
                _PublicationCall(
                    compiled=compiled,
                    operands=cast(
                        "Mapping[str, ArrivalOperand]",
                        {operand: operands[operand] for operand in builder.origins},
                    ),
                )
                if isinstance(builder, _PublicationArguments)
                else compiled
            )
        output = self.base(
            compiled_cores=dynamic_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
            **optional,
        )
        if output.continuations:
            raise UnsupportedOperationError(
                "Combining arrival and another continuation publisher "
                "is an S10 dependency."
            )
        if self.surfaces:
            surfaces = cast(
                "Mapping[str, FloatND]",
                compiled_cores[_SURFACES](
                    **programs[_SURFACES].argument_builder(
                        CoreBuildContext(
                            state_action_space=state_action_space,
                            next_regime_to_V_arr=next_regime_to_V_arr,
                            next_regime_to_continuation=next_regime_to_continuation,
                            flat_params=flat_params,
                            period=period,
                            ages=ages,
                            **optional,
                        )
                    )
                ),
            )
            for name, surface in surfaces.items():
                template = self.template.operands[name]
                operands[name] = (
                    surface
                    if surface.sharding == template.sharding
                    else jax.device_put(surface, template.sharding)
                )
        payload = ArrivalContinuation(
            regime=self.template.regime,
            period=period,
            schema=self.template.schema,
            operands=cast("Mapping[str, ArrivalOperand]", operands),
        )
        if any(payload.operands[name] is not value for name, value in operands.items()):
            raise ExecutionPlanningError(
                "Arrival publication must retain exact inputs."
            )
        return replace(output, continuations={ARRIVAL_CONTINUATION: payload})


def _borrow_operands(
    *,
    origins: Mapping[str, ValueArtifactAddress],
    period: int,
    values: Mapping[RegimeName, ValueTemplate],
    continuations: Mapping[RegimeName, ContinuationPayload],
) -> dict[str, ValueTemplate]:
    operands = {}
    for name, origin in origins.items():
        if origin.period != period + 1:
            raise UnsupportedOperationError(
                "An arrival publisher needs its actual next-date stored input."
            )
        if origin.kind is ValueArtifactKind.REGIME_VALUE:
            operands[name] = values[origin.regime]
        elif origin.kind is ValueArtifactKind.CONTINUATION_LEAF:
            payload = continuations[origin.regime]
            if not isinstance(payload, ContinuationReader) or (
                payload.artifact_key != origin.artifact_key
            ):
                raise ExecutionPlanningError(
                    "An arrival publisher received the wrong continuation schema."
                )
            if isinstance(payload, ArrivalContinuation) and (
                payload.period != origin.period or payload.regime != origin.regime
            ):
                raise ExecutionPlanningError(
                    "An arrival publisher received a wrong-date continuation."
                )
            operands[name] = payload.leaves()[origin.leaf_path]
        else:
            raise UnsupportedOperationError(
                "Arrival custody requires a declared value or continuation leaf."
            )
    return operands


def install_arrival_publishers(
    *,
    canonical_regimes: Mapping[RegimeName, Regime],
    contexts: Mapping[RegimeName, SolverBuildContext],
) -> MappingProxyType[RegimeName, Regime]:
    """Admit borrowed schemas against the actual producer's declared native reads.

    A regime whose arrival surfaces a later reader needs also tabulates them
    in its own period program and publishes them in the same dated payload.
    """
    updated = dict(canonical_regimes)
    for owner, context in contexts.items():
        if not context.arrival_publications:
            continue
        regime = updated[owner]
        if regime.solution.continuation_spec is not None:
            raise UnsupportedOperationError(
                "Combining arrival custody with an existing continuation "
                "is an S10 dependency."
            )
        kernels = dict(regime.solution.period_kernels)
        for period, publication in context.arrival_publications.items():
            plans = context.arrival_queries.get(period, MappingProxyType({}))
            from _lcm.arrival_queries import landing_grid  # noqa: PLC0415

            publisher = _BorrowedArrivalKernel(
                base=kernels[period],
                template=publication.template,
                origins=publication.origins,
                surfaces=MappingProxyType(
                    {
                        name: plans[target]
                        for name, target in publication.surfaces.items()
                    }
                ),
                surface_grids=MappingProxyType(
                    {
                        name: landing_grid(context=contexts[target], period=period + 1)
                        for name, target in publication.surfaces.items()
                    }
                ),
                surface_width=context.axis_widths.get("query_cell", 64),
            )
            graph = publisher.core_programs()
            native = {
                read.target
                for program in graph.values()
                for read in program.requirements.value_reads
                if read.source.source_period == period
                and read.source.source_regime == owner
            }
            if any(origin not in native for origin in publication.origins.values()):
                raise UnsupportedOperationError(
                    "An arrival publisher needs an explicit native read for every "
                    "additional comparison origin."
                )
            kernels[period] = publisher
        dated = {
            period: publication.template
            for period, publication in context.arrival_publications.items()
        }
        spec = ArrivalContinuationSpec(
            template=dated[min(dated)],
            artifact_key=ARRIVAL_CONTINUATION,
            templates_by_period=dated,
            origins_by_period={
                period: publication.origins
                for period, publication in context.arrival_publications.items()
            },
        )
        updated[owner] = replace(
            regime,
            solution=replace(
                regime.solution,
                period_kernels=MappingProxyType(kernels),
                continuation_spec=spec,
            ),
        )
    return MappingProxyType(updated)
