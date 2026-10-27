"""Close the genuine installed Case-U source's complete scalar parent roster."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import jax

from _lcm.continuation import ProjectedContinuationSpec
from _lcm.egm.current_iid_demands import ParentResourcesDemand
from _lcm.execution.core_program import CoreBuildContext
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.current_iid_case_u_native import (
    PreparedCaseUDate,
    prepare_case_u_date,
)
from _lcm.solution.current_iid_parent_demands import (
    _incoming_parent_dates,
    _prepared_parent_pieces,
)
from _lcm.solution.current_iid_query_exclusions import (
    require_transition_only_projected_reads,
)
from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
from _lcm.typing import FlatParams
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import RegimeName


def close_case_u_parent_demands(
    *,
    prepared: Mapping[tuple[RegimeName, int], PreparedCaseUDate],
    steps: Mapping[RegimeName, EGMStepBuild],
    specs: Mapping[RegimeName, ProjectedContinuationSpec],
    contexts: Mapping[RegimeName, SolverBuildContext],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    fixed_flat_params: FlatParams,
    solvers: Mapping[RegimeName, _BoundDCEGM],
    materializer: DatedGenerationMaterializer | None = None,
) -> Mapping[tuple[RegimeName, int], PreparedCaseUDate]:
    """Close all structural consumers before any final Case-U kernel is exposed."""
    result = {}
    for (target, period), producer in prepared.items():
        spec = specs[target]
        if spec.date_at(period=period) is not producer.date:
            raise TypeError("Exact Case-U closure substituted its original date.")
        require_transition_only_projected_reads(target=target, contexts=contexts)
        demands = []
        for source, parent_period in _incoming_parent_dates(
            target=target, target_period=period, contexts=contexts
        ):
            context = contexts[source]
            if context.current_iid_plans.get(parent_period) is not None:
                raise UnsupportedOperationError(
                    "Exact Case-U still requires current-shock parent demand closure."
                )
            call = call_contexts.get((source, parent_period))
            if (
                call is None
                or call.period != parent_period
                or call.ages is not context.ages
            ):
                raise UnsupportedOperationError(
                    "Exact Case-U parent lacks its original dated call/support."
                )
            demands.append(
                ParentResourcesDemand(
                    source=source,
                    period=parent_period,
                    target=target,
                    spec=spec,
                    pieces=_prepared_parent_pieces(
                        context=context, period=parent_period, steps=steps
                    ),
                    transition_targets=tuple(context.transitions),
                )
            )
        closed = prepare_case_u_date(
            previous=producer,
            context=contexts[target],
            solver=solvers[target],
            build=steps[target],
            period=period,
            date=producer.date,
            call=call_contexts[target, period],
            fixed_flat_params=fixed_flat_params,
            demands=tuple(demands),
            materializer=materializer,
        )
        if jax.tree.structure(closed.template) != jax.tree.structure(producer.template):
            raise TypeError("Parent closure changed the frozen U native schema.")
        if tuple((x.shape, x.dtype) for x in jax.tree.leaves(closed.template)) != tuple(
            (x.shape, x.dtype) for x in jax.tree.leaves(producer.template)
        ):
            raise TypeError("Parent closure changed the frozen U native leaf shape.")
        # Keep the exact original date captured by all actual parent readers.
        result[target, period] = replace(
            closed,
            date=producer.date,
            template=replace(
                closed.template,
                output=replace(
                    closed.template.output, projected=producer.date.template
                ),
            ),
        )
    return MappingProxyType(result)
