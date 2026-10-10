"""Keyword arguments received by solution publication observers."""

from collections.abc import Mapping
from typing import NotRequired, TypedDict, TypeGuard

import jax
import numpy as np

from _lcm.execution import core_program, internal_outputs, workspace_planning
from _lcm.solution import backward_induction
from lcm.solver_api import ArtifactAuthority, ArtifactKey, KernelOutput
from lcm.typing import RegimeName, UserParams


class ConsumeOutputKwargs(TypedDict):
    output: KernelOutput
    continuation_key: ArtifactKey | None
    regime_name: RegimeName
    period: int
    artifact_authorities: NotRequired[Mapping[ArtifactKey, ArtifactAuthority]]


class PlanningKwargs(TypedDict):
    all_programs: backward_induction.Mapping[
        backward_induction._CoreTriple, backward_induction.CoreProgram
    ]
    regimes: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.Regime
    ]
    program_fingerprint: str
    flat_params: backward_induction.FlatParams
    ages: backward_induction.TimeAxis
    next_regime_to_V_arr: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.FloatND
    ]
    next_regime_to_continuation: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.ContinuationPayload
    ]
    next_edge_to_V_arr: backward_induction.MappingProxyType[
        backward_induction._EdgeKey, backward_induction.FloatND
    ]
    budget_bytes: int | None
    execution_widths: backward_induction.ResolvedExecution
    enable_jit: bool
    continuous_sharded_state: NotRequired[backward_induction.StateName | None]
    donate_buffers: NotRequired[bool]
    retain_all_artifacts: bool
    persistable_artifact_refs: frozenset[backward_induction.ArtifactRef]
    process_grid_resolver: NotRequired[backward_induction.ProcessGridResolver | None]
    structural_blueprints: NotRequired[
        backward_induction.StructuralBlueprintCache[
            backward_induction._StructuralBlueprint
        ]
        | None
    ]
    base_state_action_spaces: NotRequired[
        backward_induction.Mapping[
            backward_induction.RegimeName, backward_induction.StateActionSpace
        ]
        | None
    ]
    logger: NotRequired[backward_induction.logging.Logger | None]


type PlanningResult = tuple[
    dict[backward_induction._CoreTriple, backward_induction.ResolvedOutputLayout],
    dict[backward_induction._CoreCandidate, backward_induction.Hashable],
    dict[backward_induction._CoreCandidate, backward_induction.ResolvedCoreProgram],
    dict[
        backward_induction._CoreCandidate,
        backward_induction.Mapping[
            backward_induction.ReferenceName, backward_induction.ShapeDtypePytree
        ],
    ],
    backward_induction.PlannedInputLiveness[
        backward_induction._InputDispatch, backward_induction.ValueArtifactAddress
    ],
    dict[
        backward_induction._CoreCandidate,
        tuple[backward_induction.ResolvedDonation, ...],
    ],
    backward_induction.MappingProxyType[
        backward_induction._CoreTriple, backward_induction._ProgramExecutionMetadata
    ],
    backward_induction._LazyCandidateFrontier,
]


class CompileFunctionsKwargs(TypedDict):
    regimes: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.Regime
    ]
    program_fingerprint: str
    flat_params: backward_induction.FlatParams
    ages: backward_induction.TimeAxis
    next_regime_to_V_arr: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.FloatND
    ]
    next_regime_to_continuation: backward_induction.MappingProxyType[
        backward_induction.RegimeName, backward_induction.ContinuationPayload
    ]
    next_edge_to_V_arr: backward_induction.MappingProxyType[
        backward_induction._EdgeKey, backward_induction.FloatND
    ]
    enable_jit: bool
    execution: backward_induction.ResolvedExecution
    retain_replay: bool
    retain_all_artifacts: bool
    persistable_artifact_refs: frozenset[backward_induction.ArtifactRef]
    max_compilation_workers: int | None
    logger: backward_induction.logging.Logger
    call_id: NotRequired[backward_induction.CallId | None]
    fixed_input_arrays: NotRequired[backward_induction.PytreeByPeriod]
    process_grid_resolver: NotRequired[backward_induction.ProcessGridResolver | None]
    gather_checks: NotRequired[backward_induction.GatherChecks | None]
    executable_cache: NotRequired[backward_induction.ExecutableCache | None]
    capture_periods: NotRequired[tuple[tuple[str, int], ...]]
    structural_blueprints: NotRequired[
        backward_induction.StructuralBlueprintCache | None
    ]
    base_state_action_spaces: NotRequired[
        backward_induction.Mapping[
            backward_induction.RegimeName, backward_induction.StateActionSpace
        ]
        | None
    ]


class LowerWaveKwargs(TypedDict):
    new_lowerings: backward_induction.Mapping[
        backward_induction.Hashable, backward_induction._CoreCandidate
    ]
    resolved_programs: backward_induction.Mapping[
        backward_induction._CoreCandidate, backward_induction.ResolvedCoreProgram
    ]
    all_layouts: backward_induction.Mapping[
        backward_induction._CoreTriple, backward_induction.ResolvedOutputLayout
    ]
    internal_templates: backward_induction.Mapping[
        backward_induction._CoreCandidate,
        backward_induction.Mapping[
            backward_induction.ReferenceName, backward_induction.ShapeDtypePytree
        ],
    ]
    donations: backward_induction.Mapping[
        backward_induction._CoreCandidate,
        tuple[backward_induction.ResolvedDonation, ...],
    ]
    ages: backward_induction.TimeAxis
    n_triples_per_lowering: backward_induction.Mapping[backward_induction.Hashable, int]
    log_kernel_memory: bool
    n_workers: int
    logger: backward_induction.logging.Logger
    compiled: dict[backward_induction.Hashable, backward_induction.jax.stages.Compiled]
    labels: dict[backward_induction.Hashable, str]


class MeasureVariantKwargs(TypedDict):
    variant_key: backward_induction.Hashable
    candidate: backward_induction._CoreCandidate
    triple: backward_induction._CoreTriple
    compiled: backward_induction.Mapping[
        backward_induction.Hashable, backward_induction.jax.stages.Compiled
    ]
    labels: backward_induction.Mapping[backward_induction.Hashable, str]
    memory_by_lowering_key: dict[
        backward_induction.Hashable, backward_induction.CompilerMemoryReservation
    ]
    resolved_programs: backward_induction.Mapping[
        backward_induction._CoreCandidate, backward_induction.ResolvedCoreProgram
    ]
    internal_templates: backward_induction.Mapping[
        backward_induction._CoreCandidate,
        backward_induction.Mapping[
            backward_induction.ReferenceName, backward_induction.ShapeDtypePytree
        ],
    ]
    resident_inventory: backward_induction.Mapping[
        backward_induction._CoreTriple, backward_induction.ResidentInventory
    ]
    logger: backward_induction.logging.Logger


class AbstractArgumentsKwargs(TypedDict):
    arguments: backward_induction.Mapping[str, backward_induction.ArgumentTree]


class ExecutionMetadataKwargs(TypedDict):
    programs: backward_induction.Mapping[
        backward_induction._CoreTriple, backward_induction.ResolvedCoreProgram
    ]


type ExecutionMetadataResult = backward_induction.MappingProxyType[
    backward_induction._CoreTriple, backward_induction._ProgramExecutionMetadata
]


class CoreCandidatesKwargs(TypedDict):
    program: core_program.MaterializedCoreProgram
    tile_widths: tuple[core_program.Mapping[str, int] | None, ...]
    input_transfer_plan: NotRequired[tuple[core_program.ResolvedValueTransfer, ...]]
    abstract_inputs: NotRequired[bool]


class ResolveProducerKwargs(TypedDict):
    program: internal_outputs.ResolvedCoreProgram
    templates: internal_outputs.Mapping[
        internal_outputs.ReferenceName, internal_outputs.ShapeDtypePytree
    ]


class WidthCandidatesKwargs(TypedDict):
    axes: tuple[
        workspace_planning.ReducedAxis | workspace_planning.TiledOutputAxis, ...
    ]
    fixed_widths: NotRequired[workspace_planning.Mapping[str, int]]
    budget_bytes: NotRequired[int | None]
    width_ceilings: NotRequired[workspace_planning.Mapping[str, int]]
    covered_axes: NotRequired[workspace_planning.Collection[str]]


def filled_toy_params[Input](params: Input) -> TypeGuard[UserParams]:
    """A toy parameter mapping contains filled numeric leaves at every depth."""
    return isinstance(params, dict) and all(
        isinstance(name, str)
        and (
            isinstance(value, (int, float, jax.Array, np.ndarray))
            or filled_toy_params(value)
        )
        for name, value in params.items()
    )
