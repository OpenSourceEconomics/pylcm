"""Retain and admit existing caller payloads before simulation normalizes inputs.

This boundary observes existing JAX arrays without conversion or solution getters.
It does not authorize the additional storage used by normalization, padding, foreign
snapshots or an automatic solve; those operations need their own allocation profiles.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import pandas as pd

from _lcm.engine import placed_devices_for_ids
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.params.mapping_leaf import LeafEntry, UserMappingLeaf
from _lcm.params.sequence_leaf import UserSequenceLeaf
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
)
from _lcm.solution.retained_buffers import retained_solution_buffers
from lcm.typing import (
    ActionName,
    StateName,
    UserInitialConditions,
    UserParams,
    UserParamsNode,
)

if TYPE_CHECKING:
    from lcm._solver_api.result import SolutionResult

    type SolutionResultBoundary = SolutionResult
else:
    # The caller's result reaches this module unvalidated; `retained_solution_buffers`
    # refuses any other class with its own error, before any field is read.
    type SolutionResultBoundary = object

# The caller inputs a budgeted public call owns: a simulation's initial
# conditions, or a policy lookup's states and action grids.
type EntryCallerInputs = (
    UserInitialConditions
    | pd.DataFrame
    | tuple[Mapping[StateName, jax.Array], Mapping[ActionName, jax.Array] | None]
)

# A value the caller-array walk reaches: a parameter node, an entry input, the
# data of a mapping or sequence leaf, or an absent action-grid mapping.
type _CallerValue = UserParamsNode | EntryCallerInputs | LeafEntry | None


@dataclass(frozen=True, kw_only=True)
class SimulationEntryInputs:
    """Call-local owners of source arrays replaced during input normalization."""

    arrays: tuple[jax.Array, ...]

    def footprint(
        self, *, solution: SolutionResultBoundary | None
    ) -> DeviceBufferFootprint:
        """Observe source and solution footprints while their owners remain live."""
        return measure_buffer_footprint(
            tree=(
                self.arrays,
                ()
                if solution is None
                else retained_solution_buffers(solution=solution),
            )
        )


def capture_simulation_entry_inputs(
    *,
    execution: ResolvedExecution,
    params: UserParams,
    initial_conditions: EntryCallerInputs,
    solution: SolutionResultBoundary | None,
) -> SimulationEntryInputs | None:
    """Refuse excess existing residency before any input conversion or snapshot.

    The owner stays local to ``Model.simulate`` through its return. Without a budget,
    neither extra traversal nor extra source retention changes the eager path.
    """
    if execution.device_memory_bytes is None:
        return None
    inputs = SimulationEntryInputs(
        arrays=_caller_arrays(values=(params, initial_conditions))
    )
    require_transfer_headroom(
        live=inputs.footprint(solution=solution),
        destination_bytes={},
        scratch_bytes={},
        budget_bytes=execution.device_memory_bytes,
        devices=placed_devices_for_ids(
            submesh_device_ids=(), visible_device_ids=execution.device_ids
        ),
    )
    return inputs


def _caller_arrays(
    *, values: tuple[UserParams, EntryCallerInputs]
) -> tuple[jax.Array, ...]:
    """Read the accepted input containers, including unregistered Mapping classes.

    Generic PyTree traversal treats ``UserDict`` and wrapper subclasses as opaque.
    Host numerical leaves and DataFrames have no existing JAX payload to inventory.
    """
    arrays: list[jax.Array] = []
    pending: list[_CallerValue] = list(values)
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, jax.Array):
            arrays.append(value)
        elif isinstance(value, UserMappingLeaf | UserSequenceLeaf):
            pending.append(value.data)
        elif isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, tuple | list):
            pending.extend(value)
    return tuple(arrays)
