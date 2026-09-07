"""Execution planning for solve kernels.

Solve kernels name the logical roles of their outputs here so the engine can
resolve their final device layouts before lowering.
"""

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    CoreProgramGraphAware,
    InternalInputRef,
    InternalOutputSpec,
    MaterializedCoreProgram,
    ReducedAxis,
    ResolvedCoreProgram,
    TiledOutputAxis,
    ValueRead,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.reductions import (
    EXACTNESS_VALUES,
    HARD_MAX_WITH_CARRY_REDUCTION,
    INTERVAL_ENVELOPE_REDUCTION,
    WEIGHTED_EXPECTATION_REDUCTION,
    HardMaxWithCarryReduction,
    IntervalEnvelopeReduction,
    ReductionDeclaration,
    ReductionSemantics,
    WeightedExpectationReduction,
)
from _lcm.execution.value_transfer import (
    TransferCost,
    TransferOperationClass,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)

__all__ = [
    "EXACTNESS_VALUES",
    "HARD_MAX_WITH_CARRY_REDUCTION",
    "INTERVAL_ENVELOPE_REDUCTION",
    "WEIGHTED_EXPECTATION_REDUCTION",
    "CoreBuildContext",
    "CoreExecutionDisposition",
    "CoreExecutionRequirements",
    "CoreProgram",
    "CoreProgramGraphAware",
    "HardMaxWithCarryReduction",
    "InternalInputRef",
    "InternalOutputSpec",
    "IntervalEnvelopeReduction",
    "MaterializedCoreProgram",
    "ReducedAxis",
    "ReductionDeclaration",
    "ReductionSemantics",
    "ResolvedCoreProgram",
    "TiledOutputAxis",
    "TransferCost",
    "TransferOperationClass",
    "ValueArtifactAddress",
    "ValueArtifactKind",
    "ValueConsumerAddress",
    "ValueInputChannel",
    "ValueRead",
    "ValueTransferKind",
    "WeightedExpectationReduction",
    "core_program_graph",
    "materialize_core_program",
    "resolve_core_program",
]
