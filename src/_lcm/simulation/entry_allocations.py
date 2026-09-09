"""Admit canonical numeric uploads while owning the complete live entry state.

Host validation and target-dtype conversion precede explicit first-selected-device
staging. Upload admission uses the existing destination-payload plus declared
transfer-scratch convention; it does not invent a zero-workspace device cast.
Pandas labels and scattered values are assembled on the host before the same numeric
upload boundary. Foreign eager values use exact-layout compiled copies with cumulative
ownership.
Trusted native value uploads use the same writer after archive verification and
before private cache publication; their detached copies use the foreign-copy owner.
Artifact and arbitrary lazy materializers require separate allocation profiles.
"""

import dataclasses
import weakref
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.process_grids import SimulationProcessGrids
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    union_buffer_footprints,
)
from _lcm.simulation.solution_copies import copy_solution_leaf
from _lcm.typing import InitialConditions


@dataclasses.dataclass(kw_only=True, eq=False)
class SimulationEntryAllocations:
    """Call-local mutable owners; none is retained in an executable cache."""

    original_inputs: SimulationEntryInputs | None
    """Original caller buffers retained until the public call returns."""
    solution: object | None
    """Currently resolved value, policy and replay-artifact owners."""
    model_roots: tuple[object, ...]
    """Already materialized model grids, fixed parameters, IDs and ages."""
    devices: tuple[jax.Device, ...]
    """Actual ordered execution devices; the first owns entry staging."""
    budget_bytes: int
    """Per-device modeled payload and workspace ceiling."""
    operations: ProfiledSimulationOperations = dataclasses.field(
        default_factory=ProfiledSimulationOperations
    )
    """Abstract signatures and executable profiles, without array owners."""
    _stages: dict[str, object] = dataclasses.field(default_factory=dict, init=False)
    """Complete parameter and initial-condition mappings at the current stage."""
    _pending: list[jax.Array] = dataclasses.field(default_factory=list, init=False)
    """Ready outputs not yet handed to a complete stage mapping."""
    _resolved_inputs: tuple[object, ...] = dataclasses.field(default=(), init=False)
    """Validated value/policy/flag/reader trees beside the original solution."""
    _foreign_copies: list[jax.Array] = dataclasses.field(
        default_factory=list, init=False
    )
    """Ready private copies retained until foreign resolution commits or fails."""
    process_grid_resolver: SimulationProcessGrids = dataclasses.field(init=False)
    """Call-owned runtime support admitted before any authority or solve reads it."""

    def __post_init__(self) -> None:
        """Bind grid admission to the complete entry ownership inventory."""
        self.process_grid_resolver = SimulationProcessGrids(
            live_footprint=_EntryFootprint(owner=weakref.ref(self)),
            devices=self.devices,
            budget_bytes=self.budget_bytes,
        )

    def snapshot(self) -> DeviceBufferFootprint:
        """Observe original, completed and intermediate owners without allocating."""
        if self.original_inputs is None:
            raise RuntimeError("Simulation entry allocation owner is closed.")
        return union_buffer_footprints(
            footprints=(
                self.original_inputs.footprint(solution=self.solution),
                measure_buffer_footprint(
                    tree=(
                        self.model_roots,
                        tuple(self._stages.values()),
                        tuple(self._pending),
                        self._resolved_inputs,
                        tuple(self._foreign_copies),
                        self.process_grid_resolver.array_roots,
                    )
                ),
            )
        )

    def solve_input_roots(self) -> tuple[object, ...]:
        """Keep original and normalized inputs charged during an automatic solve.

        These are actual array owners, not a byte total: the solve inventory must
        union them with its own parameters and grids by physical storage. The
        returned tuple belongs to this invocation and never enters a code cache.
        """
        if self.original_inputs is None:
            raise RuntimeError("Simulation entry allocation owner is closed.")
        return (
            self.original_inputs.arrays,
            self.model_roots,
            tuple(self._stages.values()),
            tuple(self._pending),
            self._resolved_inputs,
            tuple(self._foreign_copies),
            self.process_grid_resolver.array_roots,
        )

    def copy_solution_leaf(self, *, leaf: jax.Array, label: str) -> jax.Array:
        """Admit and own a private copy in its original source layout."""
        del label  # The caller owns its exact copy-validation error labels.
        result = copy_solution_leaf(
            leaf=leaf,
            operations=self.operations,
            live_footprint=self.snapshot,
            budget_devices=self.devices,
            budget_bytes=self.budget_bytes,
        )
        self._foreign_copies.append(result)
        return result

    def release_foreign_copies(self) -> None:
        """Release temporary copy owners after resolution hands off or fails."""
        self._foreign_copies.clear()

    def __call__(
        self, *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
    ) -> jax.Array:
        """Admit one validated leaf before creating its canonical device payload."""
        # Integer/float helpers already supplied their host-validated arrays.
        # Boolean JAX leaves retain the existing no-host-roundtrip fast path.
        canonical = (
            value
            if isinstance(value, jax.Array) and value.dtype == dtype
            else np.asarray(value, dtype=dtype)
        )
        live = self.snapshot()
        budget_devices = tuple(dict.fromkeys((*self.devices, *live.spans)))
        placed = place_simulation_arguments(
            arguments=MappingProxyType({name: canonical}),
            subject_arg_names=(),
            value_reads=(),
            devices=(self.devices[0],),
            budget_bytes=self.budget_bytes,
            live_footprint=live,
            budget_devices=budget_devices,
        )
        result = cast("jax.Array", placed[name])
        self._pending.append(result)
        return result

    def publish(self, *, stage: Literal["params", "initial"], tree: object) -> None:
        """Hand completed outputs to their fully constructed mapping owner."""
        self._stages[stage] = tree
        self._pending.clear()

    def pad(
        self, *, initial_conditions: InitialConditions, multiple: int
    ) -> tuple[InitialConditions, int]:
        """Profile exact last-row padding before allocating each completed leaf.

        The canonical input mapping and earlier padded leaves remain owned for
        the whole loop. The caller publishes the returned mapping at handoff.
        """
        original_n_subjects = len(next(iter(initial_conditions.values())))
        if multiple <= 1 or original_n_subjects % multiple == 0:
            return initial_conditions, original_n_subjects
        pad = multiple - original_n_subjects % multiple
        self._stages["initial"] = initial_conditions
        padded: dict[str, jax.Array] = {}
        for name, array in initial_conditions.items():
            live = self.snapshot()
            result = cast(
                "jax.Array",
                self.operations.dispatch(
                    function=_pad_initial_leaf,
                    arguments={"array": array},
                    static_arguments={"pad": pad},
                    subject_arg_names=(),
                    devices=(self.devices[0],),
                    live_footprint=self.snapshot,
                    budget_devices=tuple(dict.fromkeys((*self.devices, *live.spans))),
                    budget_bytes=self.budget_bytes,
                ),
            )
            self._pending.append(result)
            padded[name] = result
        return cast("InitialConditions", MappingProxyType(padded)), original_n_subjects

    def update_solution(
        self, *, solution: object | None, resolved_inputs: tuple[object, ...]
    ) -> None:
        """Observe newly retained result views without claiming their admission."""
        self.solution = solution
        self._resolved_inputs = resolved_inputs

    def close(self) -> None:
        """Release auxiliary owners after the receiving scope owns their trees."""
        self._stages.clear()
        self._pending.clear()
        self.solution = None
        self.model_roots = ()
        self._resolved_inputs = ()
        self._foreign_copies.clear()
        self.original_inputs = None
        self.process_grid_resolver.close()


def _pad_initial_leaf(*, array: jax.Array, pad: int) -> jax.Array:
    """Duplicate the last row with the existing repeat and concatenate operations."""
    pad_block = jnp.repeat(array[-1:], pad, axis=0)
    return jnp.concatenate([array, pad_block], axis=0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _EntryFootprint:
    """Read current residency without retaining the entry scope through a cycle."""

    owner: weakref.ReferenceType[SimulationEntryAllocations]

    def __call__(self) -> DeviceBufferFootprint:
        owner = self.owner()
        if owner is None:
            raise RuntimeError("Simulation entry allocation owner is closed.")
        return owner.snapshot()
