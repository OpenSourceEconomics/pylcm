"""Execution planning for solve kernels.

Solve kernels name the logical roles of their outputs here so the engine can
resolve their final device layouts before lowering.
"""

from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    CoreProgram,
    CoreProgramAware,
    ReductionSemantics,
    ResolvedCoreProgram,
    StreamableProductAxis,
    resolve_core_program,
)

__all__ = [
    "CoreExecutionRequirements",
    "CoreProgram",
    "CoreProgramAware",
    "ReductionSemantics",
    "ResolvedCoreProgram",
    "StreamableProductAxis",
    "resolve_core_program",
]
