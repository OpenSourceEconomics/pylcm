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
from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_replay_inputs import CurrentIIDReplayInputs
from _lcm.engine import Regime
from _lcm.execution.core_program import core_program_graph
from _lcm.grids import Grid
from _lcm.grids.materialization import admitted_grid_extent, materialize_original_grid
from _lcm.params.edges import EDGES, flat_namespaces
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.projected_forward import ProjectedForwardInputs
from _lcm.simulation.entry_inputs import SimulationEntryInputs, SolutionResultBoundary
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.process_grids import SimulationProcessGrids
from _lcm.simulation.replay_inputs import PreparedReplayReader
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resolve_budget_devices,
    union_buffer_footprints,
)
from _lcm.simulation.solution_copies import copy_solution_leaf
from _lcm.typing import (
    FlatEdgeParams,
    FlatParams,
    FlatRegimeParams,
    FootprintTree,
    InitialConditions,
    PytreeByPeriod,
    PytreeValue,
    RegimeName,
    SimulationPolicy,
)
from lcm.typing import BoolND, Float1D, FloatND, Int1D

# The replay inputs resolved from a solution, keyed by period and regime: values,
# simulation policies, dissolution flags and replay readers, each `None` when the
# solution provides none; empty before any solution is resolved.
type _ResolvedInputs = (
    tuple[
        Mapping[int, Mapping[RegimeName, FloatND]] | None,
        Mapping[int, Mapping[RegimeName, SimulationPolicy]] | None,
        Mapping[int, Mapping[RegimeName, BoolND]] | None,
        Mapping[int, Mapping[RegimeName, PreparedReplayReader]] | None,
        CurrentIIDReplayInputs | None,
        ProjectedForwardInputs | None,
    ]
    | tuple[()]
)


@dataclasses.dataclass(kw_only=True, eq=False)
class SimulationEntryAllocations:
    """Call-local mutable owners; none is retained in an executable cache."""

    original_inputs: SimulationEntryInputs | None
    """Original caller buffers retained until the public call returns."""
    solution: SolutionResultBoundary | None
    """Currently resolved value, policy and replay-artifact owners."""
    model_roots: tuple[PytreeValue, ...]
    """Already materialized model grids, fixed parameters, IDs and ages."""
    devices: tuple[jax.Device, ...]
    """Actual ordered execution devices; the first owns entry staging."""
    budget_bytes: int
    """Per-device modeled payload and workspace ceiling."""
    operations: ProfiledSimulationOperations = dataclasses.field(
        default_factory=ProfiledSimulationOperations
    )
    """Abstract signatures and executable profiles, without array owners."""
    retained_generation_roots: Callable[[], tuple[PytreeValue, ...]] | None = None
    """Live initial/cache/owned-view generations, including evicted result owners."""
    _stages: dict[str, PytreeValue] = dataclasses.field(
        default_factory=dict, init=False
    )
    """Complete parameter and initial-condition mappings at the current stage."""
    _pending: list[jax.Array] = dataclasses.field(default_factory=list, init=False)
    """Ready outputs not yet handed to a complete stage mapping."""
    _resolved_inputs: _ResolvedInputs = dataclasses.field(default=(), init=False)
    """Validated value/policy/flag/reader trees beside the original solution."""
    _foreign_copies: list[jax.Array] = dataclasses.field(
        default_factory=list, init=False
    )
    """Ready private copies retained until foreign resolution commits or fails."""
    process_grid_resolver: SimulationProcessGrids = dataclasses.field(init=False)
    """Call-owned runtime support admitted before any authority or solve reads it."""

    _grid_roots: list[jax.Array] = dataclasses.field(default_factory=list, init=False)
    """Actual original declaration operands retained before any grid dispatch."""
    _grid_bindings: dict[tuple[int, str, bool], tuple[Grid, Float1D | Int1D]] = (
        dataclasses.field(default_factory=dict, init=False)
    )
    """Strong original declarations and their call-owned completed supports."""

    _validation_capture_roots: list[jax.Array] = dataclasses.field(
        default_factory=list, init=False
    )
    """Actual referenced callback buffers, distinct from array-free cached code."""
    _validation_stage_roots: FootprintTree = dataclasses.field(default=(), init=False)
    """The complete active original recipe environment before each stage."""

    solve_parameter_views: Mapping[RegimeName, FlatParams] = dataclasses.field(
        default_factory=lambda: MappingProxyType({}), init=False
    )
    """Call-owned root-consumer params shared by query lowering and dispatch."""

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
                        self._retained_generation_roots(),
                        tuple(self._stages.values()),
                        tuple(self._pending),
                        self._resolved_inputs,
                        tuple(self._foreign_copies),
                        tuple(self._grid_roots),
                        tuple(value for _, value in self._grid_bindings.values()),
                        tuple(self._validation_capture_roots),
                        self._validation_stage_roots,
                        self.process_grid_resolver.array_roots,
                    )
                ),
            )
        )

    def _retained_generation_roots(self) -> tuple[PytreeValue, ...]:
        """Refresh root ownership before any existing admitted producer runs."""
        return (
            ()
            if self.retained_generation_roots is None
            else self.retained_generation_roots()
        )

    def solve_input_roots(self) -> tuple[tuple[PytreeByPeriod, ...], ...]:
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
            self._retained_generation_roots(),
            tuple(self._stages.values()),
            tuple(self._pending),
            self._resolved_inputs,
            tuple(self._foreign_copies),
            tuple(self._grid_roots),
            tuple(value for _, value in self._grid_bindings.values()),
            tuple(self._validation_capture_roots),
            tuple(
                value
                for value in jax.tree.leaves(self._validation_stage_roots)
                if isinstance(value, jax.Array)
            ),
            self.process_grid_resolver.array_roots,
        )

    def place_solve_parameters(
        self, *, flat_params: FlatParams, regimes: Mapping[RegimeName, Regime]
    ) -> FlatParams:
        """Admit replicated parameters on each consuming solve regime's devices.

        Entry staging and caller inputs remain owned alongside every completed
        regime copy. Each transfer reserves destination payload and scratch before
        allocation; the automatic solve inventories these same physical owners.
        """
        placed: dict[RegimeName, FlatRegimeParams | FlatEdgeParams] = {}
        placed_edges: dict[RegimeName, FlatRegimeParams] = {}
        self._stages["solve_params"] = placed
        # A source's edge slots are read by the source's kernels, so they go
        # where the source runs.
        for path, parameters in flat_namespaces(flat_params):
            phases = (
                (regimes[path[-1]].solution,)
                if path[-1] in regimes
                else tuple(
                    regime.solution
                    for regime in regimes.values()
                    if any(
                        name == path[-1]
                        for plan in regime.solution.current_iid_plans.values()
                        if plan is not None
                        for name, _, _ in plan.bindings
                    )
                )
            )
            devices = tuple(
                dict.fromkeys(
                    device
                    for phase in phases
                    for device in (
                        phase.placed_devices()
                        if phase.sharded_state_names
                        else phase.placed_devices()[:1]
                    )
                )
            )
            if not devices:
                raise RuntimeError(
                    f"Solve parameter root {path!r} has no declared consumer."
                )
            live = self.snapshot()
            arguments = place_simulation_arguments(
                arguments={"solve_params": parameters},
                subject_arg_names=(),
                value_reads=(),
                devices=devices,
                budget_bytes=self.budget_bytes,
                live_footprint=live,
                budget_devices=tuple(dict.fromkeys((*self.devices, *live.spans))),
            )
            on_devices = arguments["solve_params"]
            if path[0] == EDGES:
                # A live view, so each placed source counts toward the next
                # placement's footprint.
                placed.setdefault(EDGES, MappingProxyType(placed_edges))
                placed_edges[path[-1]] = on_devices
            else:
                placed[path[0]] = on_devices
        flat = MappingProxyType(placed)
        self._place_query_solve_parameters(flat_params=flat, regimes=regimes)
        return flat

    def _place_query_solve_parameters(
        self, *, flat_params: FlatParams, regimes: Mapping[RegimeName, Regime]
    ) -> None:
        """Admit only actual query consumers' copies, retaining every source."""
        from _lcm.policy_queries import query_parameter_namespaces  # noqa: PLC0415

        views: dict[RegimeName, FlatParams] = {}
        self.solve_parameter_views = MappingProxyType(views)
        self._stages["query_solve_params"] = views
        for owner, regime in regimes.items():
            namespaces = frozenset(
                name
                for kernel in regime.solution.period_kernels.values()
                for program in core_program_graph(kernel=kernel).values()
                for name in query_parameter_namespaces(builder=program.argument_builder)
            )
            if not namespaces:
                continue
            parameters = dict(flat_params)
            namespace_values = dict(flat_namespaces(flat_params))
            for name in namespaces & regimes.keys():
                parameters[name] = MappingProxyType(
                    {
                        **regimes[name].solution.resolved_fixed_params,
                        **namespace_values.get((name,), {}),
                    }
                )
            edges = {
                path[-1]: values
                for path, values in namespace_values.items()
                if path[0] == EDGES
            }
            if edges:
                parameters[EDGES] = MappingProxyType(edges)
            view = MappingProxyType(parameters)
            # Publish the live containers before transfers. A later refusal keeps
            # originals and completed copies owned until the public call closes.
            views[owner] = view
            devices = regime.solution.placed_devices()
            if not regime.solution.sharded_state_names:
                devices = devices[:1]
            for path, values in flat_namespaces(view):
                if path[-1] not in namespaces:
                    continue
                live = self.snapshot()
                copied = place_simulation_arguments(
                    arguments={"query_solve_params": values},
                    subject_arg_names=(),
                    value_reads=(),
                    devices=devices,
                    budget_bytes=self.budget_bytes,
                    live_footprint=live,
                    budget_devices=tuple(dict.fromkeys((*self.devices, *live.spans))),
                )["query_solve_params"]
                if path[0] == EDGES:
                    edges[path[-1]] = copied
                else:
                    parameters[path[0]] = copied

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

    def materialize_grid(
        self, *, spec: Grid, placeholder: bool = False
    ) -> Float1D | Int1D:
        """Retain original operands before the same admitted grid factories run."""
        from _lcm.grids.materialization import _grid_placeholder  # noqa: PLC0415
        from _lcm.solution.generation import canonical_generation_roots  # noqa: PLC0415
        from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

        count = admitted_grid_extent(spec=spec)
        key = (id(spec), np.dtype(canonical_float_dtype()).str, placeholder)
        if key in self._grid_bindings:
            return self._grid_bindings[key][1]
        self._grid_roots.extend(canonical_generation_roots(declarations=(spec,)))
        if isinstance(spec, _ContinuousStochasticProcess):
            if not self.process_grid_resolver.supports(spec):
                raise ExecutionPlanningError(
                    "Cold dated process has no admitted original support route."
                )
            if not spec.is_fully_specified:
                result = self.produce(
                    function=partial(_grid_placeholder, n_points=count),
                    arguments={},
                )
            else:
                result = self.process_grid_resolver(
                    spec=spec,
                    parameters={},
                    required=jax.sharding.SingleDeviceSharding(self.devices[0]),
                )
        else:
            result = materialize_original_grid(
                owner=self, spec=spec, placeholder=placeholder
            )
        self._grid_bindings[key] = (spec, result)
        return result

    def evaluate_validation[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[str, PytreeValue],
    ) -> T:
        """Replay original pure callback stages with their real captures retained."""
        from _lcm.egm.validation_stages import (  # noqa: PLC0415
            ValidationCaptureHasher,
            prepare_validation_recipe,
            validation_identity,
        )
        from _lcm.simulation.eager_stages import run_eager_stage  # noqa: PLC0415
        from _lcm.simulation.process_grids import (  # noqa: PLC0415
            _aval_schema,
            _read_process_operand,
        )
        from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

        # This is the existing exact referenced-binding walk, not a module scan.
        recorder = ValidationCaptureHasher(
            array_recorder=self._validation_capture_roots.append
        )
        recorder.visit(value=function)
        recipe = prepare_validation_recipe(function=function, arguments=arguments)
        environment = list(recipe.initial_values)
        self._validation_stage_roots = tuple(environment)
        required = jax.sharding.SingleDeviceSharding(self.devices[0])
        try:
            for stage, operands, outputs in recipe.stages:
                result = run_eager_stage(
                    function=stage,
                    arguments={
                        "parameters": {
                            f"operand_{i}": _read_process_operand(
                                operand=operand, environment=environment
                            )
                            for i, operand in enumerate(operands)
                        }
                    },
                    static_arguments=MappingProxyType({}),
                    operations=self.operations,
                    devices=self.devices,
                    required=required,
                    budget_bytes=self.budget_bytes,
                    live_footprint=self.snapshot,
                    program_identity=stage.identity,
                    layout_identity=("original_validator_eager_stage", required),
                )
                if len(result) != len(outputs):
                    raise ExecutionPlanningError(
                        "Validator stage changed its output arity."
                    )
                for value, variable in zip(result, outputs, strict=True):
                    actual = jax.typeof(value)
                    if (
                        actual.shape,
                        str(actual.dtype),
                        actual.weak_type,
                    ) != _aval_schema(variable):
                        raise ExecutionPlanningError(
                            "Validator stage changed original shape, dtype "
                            "or weak type."
                        )
                environment.extend(result)
                self._validation_stage_roots = tuple(environment)
            leaves = tuple(
                _read_process_operand(operand=operand, environment=environment)
                for operand in recipe.output_operands
            )
            if any(not isinstance(value, jax.Array) for value in leaves):
                leaves = run_eager_stage(
                    function=validation_identity,
                    arguments={
                        "parameters": {
                            f"operand_{i}": value for i, value in enumerate(leaves)
                        }
                    },
                    static_arguments=MappingProxyType({}),
                    operations=self.operations,
                    devices=self.devices,
                    required=required,
                    budget_bytes=self.budget_bytes,
                    live_footprint=self.snapshot,
                    program_identity=validation_identity,
                    layout_identity=("original_validator_literal_outputs", required),
                )
            for value, expected in zip(leaves, recipe.output_contracts, strict=True):
                actual = jax.typeof(value)
                if (actual.shape, str(actual.dtype), actual.weak_type) != expected:
                    raise ExecutionPlanningError(
                        "Validator output changed original shape, dtype or weak type."
                    )
            result_tree = recipe.output_tree.unflatten(leaves)
            self._pending.extend(jax.tree.leaves(result_tree))
            return cast("T", result_tree)
        finally:
            self._validation_stage_roots = ()

    def produce[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[str, PytreeValue],
    ) -> T:
        """Materialize a pure array factory under the complete current inventory.

        Each numerical operand is explicit. Refusing numeric captures prevents
        the executable cache from retaining buffers outside this entry owner.
        The existing producer path admits the actual enclosing compiler record;
        the same entry retains the completed leaves before another factory runs.
        """
        from _lcm.constraints.dynamic_bound import (  # noqa: PLC0415
            _literal_bound_isfinite,
        )
        from _lcm.egm.current_iid_declarations import (  # noqa: PLC0415
            _projected_template,
        )
        from _lcm.egm.validation_materialization import (  # noqa: PLC0415
            _validation_bound_disagrees,
            _validation_grid_first,
            _validation_grid_max,
        )
        from _lcm.grids.materialization import (  # noqa: PLC0415
            _grid_concatenate,
            _grid_index,
            _grid_linspace,
            _grid_log,
            _grid_logspace,
            _grid_placeholder,
            _grid_set_endpoint,
        )
        from _lcm.simulation.memory import (  # noqa: PLC0415
            SimulationMemory,
            run_simulation_producer,
        )
        from _lcm.solution.current_iid_case_r import (  # noqa: PLC0415
            _native_case_r_array_template,
            _zeros_for_template,
        )
        from _lcm.solution.generation import canonical_generation_roots  # noqa: PLC0415
        from lcm.exceptions import ExecutionPlanningError  # noqa: PLC0415

        source = function.func if type(function) is partial else function
        if not any(
            source is candidate
            for candidate in (
                _projected_template,
                _native_case_r_array_template,
                _zeros_for_template,
                _grid_linspace,
                _grid_log,
                _grid_logspace,
                _grid_set_endpoint,
                _grid_index,
                _grid_concatenate,
                _grid_placeholder,
                _literal_bound_isfinite,
                _validation_grid_max,
                _validation_grid_first,
                _validation_bound_disagrees,
            )
        ):
            raise ExecutionPlanningError(
                "Entry materialization accepts only original dated templates, "
                "grids and declaration checks; user economics require their "
                "model-owned producer route."
            )
        if canonical_generation_roots(declarations=(function,)):
            raise ExecutionPlanningError(
                "Dated materialization requires explicit numerical operands, "
                "not captured device buffers."
            )
        live = self.snapshot()
        memory = SimulationMemory(
            budget_bytes=self.budget_bytes,
            devices=resolve_budget_devices(execution_devices=self.devices, live=live),
            subject_devices=(self.devices[0],),
            operations=self.operations,
            producers=self.operations,
            inputs=live,
        )
        result = run_simulation_producer(
            memory=memory, function=function, arguments=arguments
        )
        self._pending.extend(jax.tree.leaves(result))
        return result

    def publish(
        self, *, stage: Literal["params", "initial"], tree: PytreeValue
    ) -> None:
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
        self,
        *,
        solution: SolutionResultBoundary | None,
        resolved_inputs: _ResolvedInputs,
    ) -> None:
        """Observe newly retained result views without claiming their admission."""
        self.solution = solution
        self._resolved_inputs = resolved_inputs

    def close(self) -> None:
        """Release auxiliary owners after the receiving scope owns their trees."""
        self._stages.clear()
        self.solve_parameter_views = MappingProxyType({})
        self._pending.clear()
        self.solution = None
        self.model_roots = ()
        self.retained_generation_roots = None
        self._resolved_inputs = ()
        self._foreign_copies.clear()
        self._grid_roots.clear()
        self._grid_bindings.clear()
        self._validation_capture_roots.clear()
        self._validation_stage_roots = ()
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
