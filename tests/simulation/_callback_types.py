"""Declared keyword maps forwarded by simulation observers."""

from collections.abc import Callable, Collection, Mapping
from typing import Literal, NotRequired, TypedDict

import jax

import _lcm.simulation.chunk_profile_inventory as inventory
import _lcm.simulation.initial_conditions as initial_module
import _lcm.simulation.simulate as simulation
import _lcm.solution.validate_V as validation
import _lcm.utils.logging as lcm_logging
import lcm.model as model_module
from _lcm.execution.core_program import (
    CoreProgram,
    ReducedAxis,
    TiledOutputAxis,
    ValueRead,
)
from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.simulation import (
    chunk_admission,
    chunk_profiles,
    entry_inputs,
    host_operations,
    memory,
)
from _lcm.simulation.operand_placement import _OperandValue
from _lcm.simulation.residency import DeviceBufferFootprint
from _lcm.simulation.runtime import SimulationDispatchContext
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm.typing import ReferenceName


class RuntimePreparation(TypedDict):
    program: CoreProgram
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
    period: int
    n_subjects: int
    widths: Mapping[str, int]


class RuntimeDispatch(TypedDict):
    program: CoreProgram
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
    period: int
    n_subjects: int
    residency: NotRequired[SimulationDispatchContext | None]


class WidthProfileInputs(TypedDict):
    n_subjects: int
    widths: Mapping[str, int]


class WorkspacePlanning[Compiled](TypedDict):
    axes: tuple[ReducedAxis | TiledOutputAxis, ...]
    compile_candidate: Callable[[Mapping[str, int]], Compiled]
    fixed_widths: NotRequired[Mapping[str, int]]
    width_ceilings: NotRequired[Mapping[str, int]]
    budget_bytes: NotRequired[int | None]
    memory_for: NotRequired[Callable[[Compiled], CompilerMemoryReservation] | None]
    resident_bytes: NotRequired[int]
    resident_bytes_for: NotRequired[Callable[[Compiled], int] | None]
    covered_axes: NotRequired[Collection[str]]


class OperandPlacement[Operand: _OperandValue](TypedDict):
    arguments: Mapping[ReferenceName, Operand]
    subject_arg_names: tuple[ReferenceName, ...]
    value_reads: tuple[ValueRead, ...]
    devices: tuple[jax.Device, ...]
    budget_bytes: NotRequired[int | None]
    live_footprint: NotRequired[DeviceBufferFootprint | None]
    argument_footprint: NotRequired[DeviceBufferFootprint | None]
    budget_devices: NotRequired[tuple[jax.Device, ...]]


class SimulationChunkInputs(TypedDict):
    initial_states: dict[
        simulation.StateOrActionName, simulation.Float1D | simulation.IntND
    ]
    initial_regime_ids: simulation.Int1D
    starting_periods: simulation.Int1D
    n_subjects: int
    subject_slice: slice | simulation.SubjectRows
    code: NotRequired[int | None]
    stored_codes: NotRequired[tuple[int, ...] | None]
    original_n_subjects: NotRequired[int | None]
    regimes: simulation.MappingProxyType[simulation.RegimeName, simulation.Regime]
    regime_names_to_ids: simulation.RegimeNamesToIds
    regime_ids_to_names: simulation.RegimeIdsToNames
    period_to_regime_to_V_arr: simulation.MappingProxyType[
        int, simulation.MappingProxyType[simulation.RegimeName, simulation.FloatND]
    ]
    period_to_regime_to_dissolution_flags: simulation.MappingProxyType[
        int, simulation.MappingProxyType[simulation.RegimeName, simulation.BoolND]
    ]
    flat_params: simulation.FlatParams
    ages: simulation.TimeAxis
    seed: int
    logger: simulation.logging.Logger
    initial_own_stakeholder: simulation.Int1D
    period_to_regime_to_sim_policy: NotRequired[
        simulation.PeriodToRegimeToSimulationPolicy | None
    ]
    period_to_regime_to_replay_reader: NotRequired[
        simulation._PeriodToRegimeToReplayReader
    ]
    device_ids: NotRequired[tuple[int, ...]]
    memory: NotRequired[simulation.SimulationMemory | None]
    call_inputs: NotRequired[simulation.SimulationCallInputs | None]
    taste_shock_seed: NotRequired[int | None]
    taste_addresses: NotRequired[
        simulation.Mapping[tuple[int, simulation.RegimeName], tuple[int, ...]]
    ]
    process_grid_resolver: NotRequired[simulation.ProcessGridResolver | None]


class ChunkProfileInputs(TypedDict):
    runtime: chunk_profiles.SimulationRuntime
    regimes: chunk_profiles.Mapping[chunk_profiles.RegimeName, chunk_profiles.Regime]
    flat_params: chunk_profiles.FlatParams
    base_spaces: chunk_profiles.Mapping[str, chunk_profiles.StateActionSpace]
    values: chunk_profiles.Mapping[
        int, chunk_profiles.Mapping[str, chunk_profiles.jax.Array]
    ]
    flags: NotRequired[
        chunk_profiles.Mapping[
            int, chunk_profiles.Mapping[str, chunk_profiles.jax.Array]
        ]
    ]
    ages: chunk_profiles.TimeAxis
    initial_conditions: chunk_profiles.Mapping[str, chunk_profiles.jax.Array]
    regime_names_to_ids: chunk_profiles.RegimeNamesToIds
    n_subjects: int
    population: int
    original_population: int
    widths: chunk_profiles.Mapping[str, int]
    independent_taste: bool
    log_level: chunk_profiles.LogLevel
    policies: NotRequired[
        chunk_profiles.Mapping[
            int,
            chunk_profiles.Mapping[
                chunk_profiles.RegimeName, chunk_profiles.SimulationPolicy
            ],
        ]
        | None
    ]
    max_compilation_workers: NotRequired[int | None]
    group_sizes: NotRequired[tuple[int, ...] | None]


class ChunkProfileKeyInputs(TypedDict):
    runtime: chunk_admission.SimulationRuntime
    regimes: chunk_admission.Mapping[chunk_admission.RegimeName, chunk_admission.Regime]
    call_inputs: chunk_admission.SimulationCallInputs
    values: chunk_admission.Mapping[
        int, chunk_admission.Mapping[str, chunk_admission.jax.Array]
    ]
    flags: chunk_admission.Mapping[
        int, chunk_admission.Mapping[str, chunk_admission.jax.Array]
    ]
    policies: (
        chunk_admission.Mapping[
            int,
            chunk_admission.Mapping[
                chunk_admission.RegimeName, chunk_admission.SimulationPolicy
            ],
        ]
        | None
    )
    ages: chunk_admission.TimeAxis
    initial_conditions: chunk_admission.Mapping[str, chunk_admission.jax.Array]
    regime_names_to_ids: chunk_admission.RegimeNamesToIds
    n_subjects: int
    population: int
    original_population: int
    widths: chunk_admission.Mapping[str, int]
    independent_taste: bool
    log_level: chunk_admission.LogLevel
    group_sizes: NotRequired[tuple[int, ...] | None]


class HostDispatch(TypedDict):
    function: host_operations.Callable[..., host_operations.PytreeValue]
    arguments: host_operations.Mapping[
        host_operations.ReferenceName, host_operations.PytreeValue
    ]
    subject_arg_names: tuple[host_operations.ReferenceName, ...]
    devices: tuple[host_operations.jax.Device, ...]
    live_footprint: host_operations.Callable[[], host_operations.DeviceBufferFootprint]
    budget_devices: tuple[host_operations.jax.Device, ...]
    budget_bytes: int
    static_arguments: NotRequired[
        host_operations.Mapping[str, host_operations.StaticArgument]
    ]
    subject_outputs: NotRequired[bool]


class MemoryRun[T: PytreeValue](TypedDict):
    function: memory.Callable[..., T]
    arguments: memory.Mapping[memory.ReferenceName, memory.PytreeValue]
    subject_arg_names: NotRequired[tuple[memory.ReferenceName, ...]]
    static_arguments: NotRequired[memory.Mapping[str, memory.StaticArgument]]
    subject_outputs: NotRequired[bool]


class InventoryCompiled(TypedDict):
    name: str
    executable: inventory.jax.stages.Compiled
    memory: inventory.CompilerMemoryReservation
    arguments: inventory.Mapping[inventory.ReferenceName, inventory.ShapeDtypePytree]
    devices: NotRequired[tuple[inventory.jax.Device, ...] | None]


class EntryCapture(TypedDict):
    execution: entry_inputs.ResolvedExecution
    params: entry_inputs.UserParams
    initial_conditions: entry_inputs.EntryCallerInputs
    solution: entry_inputs.SolutionResultBoundary | None


class ChunkPreparation(TypedDict):
    regimes: chunk_admission.MappingProxyType[
        chunk_admission.RegimeName, chunk_admission.Regime
    ]
    flat_params: chunk_admission.FlatParams
    values: chunk_admission.Mapping[
        int, chunk_admission.Mapping[str, chunk_admission.jax.Array]
    ]
    flags: chunk_admission.Mapping[
        int, chunk_admission.Mapping[str, chunk_admission.jax.Array]
    ]
    ages: chunk_admission.TimeAxis
    initial_conditions: chunk_admission.Mapping[str, chunk_admission.jax.Array]
    regime_names_to_ids: chunk_admission.RegimeNamesToIds
    original_population: int
    retained_footprint: chunk_admission.DeviceBufferFootprint
    independent_taste: bool
    log_level: chunk_admission.LogLevel
    policies: NotRequired[
        chunk_admission.Mapping[
            int,
            chunk_admission.Mapping[
                chunk_admission.RegimeName, chunk_admission.SimulationPolicy
            ],
        ]
        | None
    ]
    process_grid_resolver: NotRequired[chunk_admission.ProcessGridResolver | None]
    max_compilation_workers: NotRequired[int | None]
    group_sizes: NotRequired[tuple[int, ...] | None]


class RuntimePrepare(TypedDict):
    program: CoreProgram
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
    period: int
    n_subjects: int


type ChunkResults = dict[
    simulation.RegimeName, dict[int, simulation.PeriodRegimeSimulationData]
]
type EntryCaptureResult = entry_inputs.SimulationEntryInputs | None
type ChunkPreparationResult = chunk_admission.PreparedSimulationChunks


class ProducerAdmission(TypedDict):
    function: host_operations.Callable[..., host_operations.PytreeValue]
    arguments: host_operations.Mapping[
        host_operations.ReferenceName, host_operations.ShapeDtypePytree
    ]
    devices: tuple[host_operations.jax.Device, ...]
    output_sharding: host_operations.jax.sharding.Sharding | None
    budget_bytes: int
    resident_bytes: int


class MemoryCreation(TypedDict):
    budget_bytes: int
    devices: tuple[memory.jax.Device, ...]
    subject_devices: tuple[memory.jax.Device, ...]
    operations: memory.ProfiledSimulationOperations
    inputs: memory.DeviceBufferFootprint
    producers: NotRequired[memory.ProfiledSimulationOperations]
    axis_widths: NotRequired[memory.MappingProxyType[str, int]]
    outputs: NotRequired[memory.DeviceBufferFootprint]
    chunk_inputs: NotRequired[memory.DeviceBufferFootprint]
    unit_inputs: NotRequired[memory.ArrayTree]
    derived: NotRequired[memory.PytreeValue]
    period_owner: NotRequired[memory.PeriodSimulationReads | None]
    ledger: NotRequired[memory.OwnerLedger]


class PeriodValidation(TypedDict):
    logger: simulation.logging.Logger
    age: simulation.ScalarInt | simulation.ScalarFloat
    period_results: tuple[
        tuple[simulation.RegimeName, simulation.PeriodRegimeSimulationData], ...
    ]
    memory: NotRequired[simulation.SimulationMemory | None]
    time_kind: NotRequired[Literal["age", "period"]]


class ValueValidation(TypedDict):
    value: simulation.FloatND
    subject_ids_in_regime: simulation.BoolND
    age: simulation.ScalarInt | simulation.ScalarFloat
    regime_name: simulation.RegimeName
    logger: simulation.logging.Logger
    memory: NotRequired[simulation.SimulationMemory | None]
    time_kind: NotRequired[Literal["age", "period"]]


class TransitionCountsValidation(TypedDict):
    logger: lcm_logging.logging.Logger
    prev_regime_ids: lcm_logging.Int1D
    new_regime_ids: lcm_logging.Int1D
    regime_ids_to_names: lcm_logging.RegimeIdsToNames
    counts_factory: NotRequired[lcm_logging.Callable[[], list[list[int]]] | None]


class DiagnosticEnrichment(TypedDict):
    exc: validation.InvalidValueFunctionError
    compute_intermediates: validation.Callable[..., validation._Reductions]
    state_action_space: validation.StateActionSpace
    next_regime_to_V_arr: (
        validation.MappingProxyType[validation.RegimeName, validation.FloatND] | None
    )
    flat_params: validation.FlatRegimeParams | None
    regime_name: validation.RegimeName
    age: float
    period: int | None
    time_kind: NotRequired[Literal["age", "period"]]


class DiagnosticInputs(TypedDict, total=False):
    logger: lcm_logging.logging.Logger
    age: simulation.ScalarInt | simulation.ScalarFloat
    period_results: tuple[
        tuple[simulation.RegimeName, simulation.PeriodRegimeSimulationData], ...
    ]
    memory: simulation.SimulationMemory | None
    time_kind: Literal["age", "period"]
    value: simulation.FloatND
    subject_ids_in_regime: simulation.BoolND
    regime_name: simulation.RegimeName
    prev_regime_ids: lcm_logging.Int1D
    new_regime_ids: lcm_logging.Int1D
    regime_ids_to_names: lcm_logging.RegimeIdsToNames
    counts_factory: lcm_logging.Callable[[], list[list[int]]] | None


class SolutionResolution(TypedDict):
    solution: model_module._SolutionResultBoundary
    flat_params: model_module.FlatParams
    entry_allocations: NotRequired[model_module.SimulationEntryAllocations | None]
    process_grid_resolver: NotRequired[model_module.ProcessGridResolver | None]


class InputValidation(TypedDict):
    initial_conditions: initial_module.InitialConditions
    regimes: initial_module.MappingProxyType[
        initial_module.RegimeName, initial_module.Regime
    ]
    regime_names_to_ids: initial_module.RegimeNamesToIds
    flat_params: initial_module.FlatParams
    ages: initial_module.TimeAxis
    logger: initial_module.logging.Logger
    execution: NotRequired[initial_module.ResolvedExecution | None]
    process_grid_resolver: NotRequired[initial_module.ProcessGridResolver | None]
    producers: NotRequired[initial_module.ProfiledSimulationOperations | None]


class WholeInputValidation(InputValidation):
    retained_footprint: NotRequired[DeviceBufferFootprint | None]
