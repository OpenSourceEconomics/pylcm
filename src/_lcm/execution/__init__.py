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
    MaterializedCoreProgram,
    ReductionSemantics,
    ResolvedCoreProgram,
    StreamableProductAxis,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)

__all__ = [
    "CoreBuildContext",
    "CoreExecutionDisposition",
    "CoreExecutionRequirements",
    "CoreProgram",
    "CoreProgramGraphAware",
    "MaterializedCoreProgram",
    "ReductionSemantics",
    "ResolvedCoreProgram",
    "StreamableProductAxis",
    "core_program_graph",
    "materialize_core_program",
    "resolve_core_program",
]
