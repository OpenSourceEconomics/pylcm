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
    ReductionSemantics,
    ResolvedCoreProgram,
    StreamableProductAxis,
    ValueRead,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
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
    "CoreBuildContext",
    "CoreExecutionDisposition",
    "CoreExecutionRequirements",
    "CoreProgram",
    "CoreProgramGraphAware",
    "InternalInputRef",
    "InternalOutputSpec",
    "MaterializedCoreProgram",
    "ReductionSemantics",
    "ResolvedCoreProgram",
    "StreamableProductAxis",
    "TransferCost",
    "TransferOperationClass",
    "ValueArtifactAddress",
    "ValueArtifactKind",
    "ValueConsumerAddress",
    "ValueInputChannel",
    "ValueRead",
    "ValueTransferKind",
    "core_program_graph",
    "materialize_core_program",
    "resolve_core_program",
]
