"""Borrow already declared native inputs for a dated arrival reconstruction."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from _lcm.arrival_continuation import (
    ARRIVAL_CONTINUATION,
    ArrivalContinuation,
    ArrivalContinuationSpec,
    ArrivalOperand,
)
from _lcm.continuation import ContinuationPayload
from _lcm.execution.core_program import CoreProgram, core_program_graph
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.solution.contract import PeriodKernel, SolverBuildContext
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue
from lcm.exceptions import ExecutionPlanningError, UnsupportedOperationError
from lcm.solver_api import ContinuationReader, KernelOutput
from lcm.typing import FloatND, RegimeName

if TYPE_CHECKING:
    from _lcm.engine import Regime, StateActionSpace
else:
    # Canonical construction imports the engine before installing these adapters.
    type Regime = object
    type StateActionSpace = object


@dataclass(frozen=True, kw_only=True)
class _BorrowedArrivalKernel:
    """Publish exact input identities only after the genuine Full solve succeeds."""

    base: PeriodKernel
    template: ArrivalContinuation
    origins: Mapping[str, ValueArtifactAddress]

    def __post_init__(self) -> None:
        object.__setattr__(self, "origins", MappingProxyType(dict(self.origins)))

    def core_programs(self) -> Mapping[str, CoreProgram]:
        """Retain the genuine producer's numerical graph and all native readers."""
        return core_program_graph(kernel=self.base)

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
        output = self.base(
            compiled_cores=compiled_cores,
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
    values: Mapping[RegimeName, FloatND],
    continuations: Mapping[RegimeName, ContinuationPayload],
) -> dict[str, ArrivalOperand]:
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
    """Admit borrowed schemas against the actual producer's declared native reads."""
    templates: dict[RegimeName, dict[int, ArrivalContinuation]] = {}
    origins: dict[RegimeName, dict[int, Mapping[str, ValueArtifactAddress]]] = {}
    for context in contexts.values():
        for plans in context.arrival_queries.values():
            for plan in plans.values():
                if plan.transport_template is None:
                    continue
                templates.setdefault(plan.target, {})[plan.period] = (
                    plan.transport_template
                )
                origins.setdefault(plan.target, {})[plan.period] = (
                    plan.transport_origins
                )
    updated = dict(canonical_regimes)
    for target, dated in templates.items():
        regime = updated[target]
        if regime.solution.continuation_spec is not None:
            raise UnsupportedOperationError(
                "Combining arrival custody with an existing continuation "
                "is an S10 dependency."
            )
        kernels = dict(regime.solution.period_kernels)
        for period, template in dated.items():
            base = kernels[period]
            graph = core_program_graph(kernel=base)
            native = {
                read.target
                for program in graph.values()
                for read in program.requirements.value_reads
                if read.source.source_period == period
                and read.source.source_regime == target
            }
            if any(origin not in native for origin in origins[target][period].values()):
                raise UnsupportedOperationError(
                    "An arrival publisher needs an explicit native read for every "
                    "additional comparison origin."
                )
            kernels[period] = _BorrowedArrivalKernel(
                base=base, template=template, origins=origins[target][period]
            )
        spec = ArrivalContinuationSpec(
            template=dated[min(dated)],
            artifact_key=ARRIVAL_CONTINUATION,
            templates_by_period=dated,
            origins_by_period=origins[target],
        )
        updated[target] = replace(
            regime,
            solution=replace(
                regime.solution,
                period_kernels=MappingProxyType(kernels),
                continuation_spec=spec,
            ),
        )
    return MappingProxyType(updated)
