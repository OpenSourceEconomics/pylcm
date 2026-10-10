"""Lower one exact primary candidate through the production structural resolver."""

import hashlib
import logging
from collections.abc import Hashable, Mapping
from types import MappingProxyType

from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.regime_building.processing import Regime
from _lcm.solution.backward_induction import (
    _build_base_state_action_spaces,
    _build_continuation_templates,
    _CoreCandidate,
    _count_triples_per_lowering_key,
    _donated_arguments,
    _LazyCandidateFrontier,
    _lower_resolved_candidate,
    _prepare_solve_programs,
    _RankedCandidateSource,
    _reject_edge_fold_state_param_collisions,
    _trace_settings_key,
    _width_key,
)
from _lcm.solution.continuation_arguments import MARGINAL_ARGUMENT
from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.solution.lowering_descriptors import (
    LoweringDescriptor,
    capture_lowering_identity,
    describe_lowering_value,
)
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams
from lcm.exceptions import ExecutionPlanningError
from lcm.lowering import LoweredPeriodCandidate, PeriodCandidate
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.typing import RegimeName


def lower_period_candidate(
    *,
    candidate: PeriodCandidate,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: TimeAxis,
    execution: ResolvedExecution,
    retention: ResultRetention,
    persistable_artifact_refs: frozenset[ArtifactRef],
    program_fingerprint: str,
    authority: Mapping[str, LoweringDescriptor],
    logger: logging.Logger,
) -> LoweredPeriodCandidate:
    """Resolve the unchanged graph, then lower only the exact requested primary.

    Structural preparation retains ordinary solve's zero-template behavior. No
    residency inventory, compiler submission, admission or dispatch is performed.
    """
    identities = capture_lowering_identity()
    spaces = _build_base_state_action_spaces(
        regimes=regimes, flat_params=flat_params, process_grid_resolver=None
    )
    _reject_edge_fold_state_param_collisions(
        regimes=regimes, base_state_action_spaces=spaces, flat_params=flat_params
    )
    values, continuations, edges = _build_continuation_templates(
        regimes=regimes,
        flat_params=flat_params,
        device_ids=execution.device_ids,
        process_grid_resolver=None,
    )
    (
        layouts,
        keys,
        programs,
        internal,
        _liveness,
        donations,
        _metadata,
        frontier,
        _all_programs,
        _fallback_programs,
    ) = _prepare_solve_programs(
        regimes=regimes,
        program_fingerprint=program_fingerprint,
        flat_params=flat_params,
        ages=ages,
        next_regime_to_V_arr=values,
        next_regime_to_continuation=continuations,
        next_edge_to_V_arr=edges,
        enable_jit=True,
        execution=execution,
        retain_replay=retention is ResultRetention.VALUES_AND_REPLAY,
        retain_all_artifacts=retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
        persistable_artifact_refs=persistable_artifact_refs,
        logger=logger,
    )
    triple = candidate.regime, candidate.period, candidate.core
    rank = _find_candidate_rank(candidate=candidate, frontier=frontier)
    selected = frontier.candidate(triple=triple, position=rank)
    resolved = programs[selected]
    donated = _donated_arguments(donations=donations[selected])
    fanout = _initial_wave_fanout(
        rank=rank,
        execution=execution,
        frontier=frontier,
        keys=keys,
        selected=selected,
    )
    primary_key = _semantic_fingerprint(describe_lowering_value(keys[selected]))
    context = {
        **authority,
        **identities,
        "primary_key": primary_key,
        "regime": candidate.regime,
        "period": candidate.period,
        "core": candidate.core,
        "widths": describe_lowering_value(resolved.tile_widths),
        "rank": rank,
        "primary_donated": donated,
        "selected_artifact_keys": describe_lowering_value(
            frozenset(
                ref.key
                for ref in persistable_artifact_refs
                if ref.regime == candidate.regime and ref.period == candidate.period
            )
        ),
        "continuation": describe_lowering_value(
            resolved.arguments.get(MARGINAL_ARGUMENT, MappingProxyType({}))
        ),
        "requirements": describe_lowering_value(resolved.requirements),
        "specialization": describe_lowering_value(resolved.specialization_key),
        "transfer_plan": describe_lowering_value(resolved.input_transfer_plan),
        "compiler_options": describe_lowering_value(resolved.compiler_options),
        "trace_settings": describe_lowering_value(_trace_settings_key()),
        "execution": describe_lowering_value(execution),
        "variant": "primary",
        "dedup_key": primary_key,
        "dedup_fanout": fanout,
        "schema": 1,
        "retention": retention.name,
        "inputs": describe_lowering_value(resolved.arguments),
        "internal_inputs": describe_lowering_value(internal[selected]),
        "static_kwargs": describe_lowering_value(resolved.static_kwargs),
        "output_roles": describe_lowering_value(resolved.output_roles),
        "output_layout": describe_lowering_value(layouts[triple]),
        "compiled": False,
        "dispatched": False,
        "admission": "not_evaluated",
        "optimized_hlo": None,
        "buffer_assignment": None,
        "compiler_memory": None,
        "physical_residency": None,
    }
    lowered = _lower_resolved_candidate(
        resolved=resolved,
        layout=layouts[triple],
        donated=donated,
        internal_templates=internal[selected],
        label=f"{candidate.regime} {candidate.core} (period {candidate.period})",
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo")).encode("utf-8")
    return LoweredPeriodCandidate(
        manifest=MappingProxyType(
            {
                **context,
                "ir_sha256": hashlib.sha256(stablehlo).hexdigest(),
            }
        ),
        stablehlo=stablehlo,
    )


def _find_candidate_rank(
    *, candidate: PeriodCandidate, frontier: _LazyCandidateFrontier
) -> int:
    """Find exact ranked membership without inventing a width or a fallback."""
    triple = candidate.regime, candidate.period, candidate.core
    bound = frontier.candidates_by_triple.get(triple)
    if bound is None:
        raise ExecutionPlanningError(
            "Candidate is absent from the selected solve graph."
        )
    widths = _width_key(widths=candidate.widths)
    for rank, entry in enumerate(bound):
        if entry[1] == widths:
            return rank
    remaining = frontier.frontiers.get(triple)
    if remaining is not None:
        for rank, entry_widths in enumerate(remaining.widths):
            if _width_key(widths=entry_widths) == widths:
                return rank
    raise ExecutionPlanningError(
        "Candidate widths are absent from the ranked frontier."
    )


def _initial_wave_fanout(
    *,
    rank: int,
    execution: ResolvedExecution,
    frontier: _LazyCandidateFrontier,
    keys: Mapping[_CoreCandidate, Hashable],
    selected: _CoreCandidate,
) -> int | None:
    """Count the provable first unbudgeted wave, otherwise report unavailable.

    Without a budget production includes every triple and uses the ranked source
    at position zero. These are already-bound primaries from shared preparation.
    Its pending/candidate mapping is therefore the identical initial wave map;
    no admission or residency computation is needed. Later/budgeted waves depend
    on compiled refusals and cannot be inferred from the structural frontier.
    """
    if rank != 0 or execution.device_memory_bytes is not None:
        return None
    source = _RankedCandidateSource(
        frontier=frontier, positions=dict.fromkeys(frontier.candidates_by_triple, 0)
    )
    wave_candidates = {
        triple: source.candidate(triple=triple) for triple in source.pending()
    }
    wave_lowering_keys = {
        candidate: keys[candidate] for candidate in wave_candidates.values()
    }
    return _count_triples_per_lowering_key(lowering_keys=wave_lowering_keys)[
        keys[selected]
    ]
