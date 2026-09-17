"""Resolving what a model declares into where and how wide its solve work runs.

Solve kernels name the logical roles of their outputs here, so the engine can
resolve final device layouts before lowering. The package owns the rest of that
resolution too: submesh placement per regime (`placement`), the transfer
catalogue that turns each declared value read into exactly one operator
(`value_transfer`), remaining-consumer accounting and legal donation
(`liveness`, `donation`), the wave schedule and physical buffer lifetime
(`scheduler`), normalization of optional compiler memory reports
(`compiler_memory`, `compiler_inputs`), and the device-memory budget and the
workspace widths that fit inside it (`execution_plan`, `workspace_planning`).
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
from lcm._solver_api.capabilities import SolverExecutionCapabilities

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
    "SolverExecutionCapabilities",
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
