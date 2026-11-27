"""Plan, lower and dispatch the declared forward-simulation programs."""

import dataclasses
import math
import threading
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future
from functools import partial
from types import MappingProxyType
from typing import cast

import jax

from _lcm.execution.compiler_inputs import compiler_input_paths
from _lcm.execution.core_program import (
    CoreArgumentBuilder,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    TiledOutputAxis,
    _value_read_argument_leaf,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.value_transfer import (
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.execution.workspace_planning import (
    CompilerMemoryReservation,
    _admissible_width,
    bootstrap_widths,
    compiler_memory_reservation,
    plan_workspace,
)
from _lcm.simulation.agent_parallel import (
    declared_agent_shard_arg_names,
    shard_agent_function,
)
from _lcm.simulation.chunk_profile_cache import ProfileCacheToken
from _lcm.simulation.host_operations import (
    ProfiledSimulationOperations,
    _abstract_operand,
)
from _lcm.simulation.operand_placement import (
    AgentArgumentNames,
    place_simulation_arguments,
)
from _lcm.simulation.program_types import (
    AGENT_AXIS,
    AGENT_WIDTH_KEYWORD,
    SimulationBuildContext,
    SimulationPrograms,
)
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    require_transfer_headroom,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.solution.backward_induction import (
    CompilationWave,
    _assert_lowered_output_tree,
    _func_dedup_key,
    _lowering_key,
    _trace_settings_key,
)
from _lcm.solution.solve_phase_records import nested_phase
from _lcm.typing import ArgumentTree, HostArray, PytreeValue, ShapeDtypePytree
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import ReferenceName

# Narrowest inner tile an unbudgeted agent axis is lowered at, and the width it
# keeps when one agent is too heavy for the byte cap below to afford more.
_DEFAULT_UNBUDGETED_AGENT_WIDTH = 4096

# Largest argument slice one unbudgeted agent tile aims for, so the live block
# stays bounded by a fixed number of bytes on every backend whatever the model.
# With the standing per-agent weight below this admits 262,144 agents, the
# widest tile the A40 width sweep measured and the widest it needed: every
# simulation there reaches its full population extent, or that tile.
_UNBUDGETED_AGENT_BLOCK_BYTES = 16 * 1024 * 1024

# Standing per-agent weight, and the floor under the weight read off a
# program's own operands.  Declared operands are only the visible part of a
# agent's working set -- the sweep measured roughly thirty times their bytes
# once temporaries and outputs are counted -- so a light program is still
# charged this much, and one that reads no per-agent operand at all is charged
# it rather than being admitted at an unbounded width.
_MIN_AGENT_ARGUMENT_BYTES = 64


def _empty_widths() -> MappingProxyType[str, int]:
    """Supply an immutable empty specialization for an unbound compiler result."""
    return MappingProxyType({})


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompiledSimulationProgram:
    """One selected executable and its static argument bindings."""

    executable: Callable[..., PytreeValue]
    """The exact executable selected by workspace planning."""

    static_kwargs: MappingProxyType[str, int]
    """Bindings used only by eager execution; compiled programs already bind them."""

    widths: MappingProxyType[str, int] = dataclasses.field(
        default_factory=_empty_widths
    )
    """Concrete compiler specialization, never a cached budget admission."""

    memory: CompilerMemoryReservation | None = None
    """Complete compiler accounting for this exact executable, cached once at
    compilation. `None` for an eager or host-driven callable, which never
    reaches budgeted dispatch."""

    def __call__(self, **arguments: PytreeValue | ShapeDtypePytree) -> PytreeValue:
        """Execute with live arrays; retain no call arguments on the cache entry."""
        return self.executable(**arguments, **self.static_kwargs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimulationDispatchContext:
    """Call-scoped live metadata; its provider never enters executable caches.

    The outer owner must complete represented transfers before returning a
    snapshot. A snapshot remains valid only while its actual buffer owners live.
    """

    live_footprint: Callable[[], DeviceBufferFootprint]
    budget_devices: tuple[jax.Device, ...]
    axis_widths: MappingProxyType[str, int] = dataclasses.field(
        default_factory=_empty_widths
    )
    """One common chunk specialization, clamped to each program's own extent."""

    def __post_init__(self) -> None:
        """Require named positive widths, separate from cached code identity."""
        if any(
            type(name) is not str or not name or type(width) is not int or width <= 0
            for name, width in self.axis_widths.items()
        ):
            raise ExecutionPlanningError(
                "Reserved simulation widths must be positive integers with axis names."
            )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SimulationRuntime:
    """Share one executable cache between lazy dispatch and prewarming."""

    execution: ResolvedExecution
    """Model-owned width and device configuration."""

    enable_jit: bool
    """Whether to compile the selected body or invoke its eager form."""

    agent_devices: tuple[jax.Device, ...]
    """Actual ordered devices evaluating agents, independent of solve placement."""

    operations: ProfiledSimulationOperations = dataclasses.field(
        default_factory=ProfiledSimulationOperations, repr=False
    )
    """Pure host-operation executable profiles, without call-owned arrays."""

    cache: dict[Hashable, CompiledSimulationProgram] = dataclasses.field(
        default_factory=dict, repr=False
    )
    """Executable identities containing abstract metadata only."""

    prepared: dict[tuple[int, str, Hashable], CompiledSimulationProgram] = (
        dataclasses.field(default_factory=dict, repr=False)
    )
    """Selected executable for each period and declared program identity."""

    routes: dict[Hashable, _PreparedRoute] = dataclasses.field(
        default_factory=dict, repr=False
    )
    """Static preparation an exact repeat of one abstract signature reuses.

    Keyed by the complete abstract signature of the *caller's* arguments, so the
    probe can precede materialization; see `_prepared_route_key`. Entries hold
    immutable abstract description and compiled code only — never a caller
    array, never an admission or validation verdict. The runtime's own
    configuration (`execution`, `enable_jit`, `agent_devices`) is immutable for
    the life of one `SimulationRuntime` and therefore constant across this dict,
    so it is deliberately absent from the key: a different execution config,
    device order, explicit width or JIT disposition is a different runtime with
    its own empty `routes`.
    """

    in_flight: dict[Hashable, Future[CompiledSimulationProgram]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    """Transient shared results for keys whose candidates are being compiled."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)
    """Protect cache ownership and publication while compilation runs outside it."""

    profile_cache_token: ProfileCacheToken = dataclasses.field(
        default_factory=ProfileCacheToken, repr=False
    )
    """Private identity distinguishing this runtime's abstract chunk profiles.

    A bare marker object, never a caller array or closure; it lets the
    process-wide `ChunkProfileCacheRegistry` (see `chunk_profile_cache.py`)
    share one bounded LRU across every model runtime while keeping entries
    from different runtimes (different execution configs, devices or JIT
    dispositions) from colliding. Its lifetime is this runtime's: the token is
    reachable only from here, so once this runtime is dropped the registry
    observes the token's death and releases every profile keyed under it,
    rather than holding compiled executables until LRU eviction. Each rebuild
    of `Model._simulate_runtime_regimes` (a fresh compile-batch shape, or a
    fresh runtime after unpickling) constructs a new `SimulationRuntime` and
    therefore a new token.
    """

    def dispatch(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
        n_agents: int,
        residency: SimulationDispatchContext | None = None,
    ) -> PytreeValue:
        """Invoke the selected executable with this call's dynamic arguments.

        An unbudgeted repeat of an exact abstract signature takes the prepared
        route: the record is probed *before* anything is materialized, and a hit
        binds this call's own leaves onto the cached static preparation (see
        `_bind_prepared_route`). Every other call — budgeted, first-seen, or one
        whose freshly bound operands no longer match the record — takes the full
        validated materialize/plan route below, which then publishes the record
        an exact repeat may reuse.
        """
        self._require_budget_context(program=program, residency=residency)
        route_key = (
            _prepared_route_key(
                program=program,
                arguments=arguments,
                period=period,
                n_agents=n_agents,
            )
            if self.execution.device_memory_bytes is None and residency is None
            else None
        )
        if route_key is not None:
            bound = self._bind_prepared_route(
                key=route_key, program=program, arguments=arguments, period=period
            )
            if bound is not None:
                compiled, bound_arguments = bound
                return compiled(**bound_arguments)
        materialized = self._materialize(
            program=program,
            arguments=arguments,
            period=period,
            n_agents=n_agents,
            residency=residency,
        )
        compiled = self._prepare_materialized(
            program=materialized, n_agents=n_agents, residency=residency
        )
        if route_key is not None:
            self._publish_prepared_route(
                key=route_key,
                program=program,
                materialized=materialized,
                compiled=compiled,
            )
        return compiled(**materialized.arguments)

    def _bind_prepared_route(
        self,
        *,
        key: Hashable,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
    ) -> (
        tuple[
            CompiledSimulationProgram,
            Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        ]
        | None
    ):
        """Bind this call's live leaves onto a cached static preparation.

        Returns the selected executable together with freshly built and freshly
        placed operands, or `None` to send the call down the full validated
        route. Nothing is carried over from the previous call except immutable
        abstract description and compiled code: the argument tree is rebuilt by
        the declared builder and placed again, so a same-shaped new array is a
        fresh value, never a reason to reuse the old object.

        Three guards keep the reuse exact. The record must have been published
        for this very `CoreProgram` object, so a rebuilt period program cannot
        inherit a predecessor's preparation. The builder's result must still be
        a mapping, the check `materialize_core_program` performs. And the bound,
        placed operands must carry exactly the abstract signature the cached
        executable was compiled for — which also settles the donation-candidate
        check, since that reads only the argument names the signature pins. Any
        other outcome is a refusal, never a repair.
        """
        with self.lock:
            route = self.routes.get(key)
        if route is None or route.source_program is not program:
            return None
        built = route.argument_builder(
            _build_context(arguments=arguments, period=period)
        )
        if not isinstance(built, Mapping):
            return None
        placed = place_simulation_arguments(
            arguments=built,
            agent_arg_names=route.agent_arg_names,
            value_reads=route.requirements.value_reads,
            devices=self.agent_devices,
            budget_bytes=None,
            live_footprint=None,
            budget_devices=(),
        )
        if _operand_signature(arguments=placed) != route.operand_signature:
            return None
        return route.compiled, placed

    def _publish_prepared_route(
        self,
        *,
        key: Hashable,
        program: CoreProgram,
        materialized: MaterializedCoreProgram,
        compiled: CompiledSimulationProgram,
    ) -> None:
        """Record the static preparation an exact repeat of this call may reuse."""
        builder = program.argument_builder
        if not isinstance(builder, AgentArgumentNames):
            return
        signature = _operand_signature(arguments=materialized.arguments)
        if signature is None:
            return
        with self.lock:
            self.routes[key] = _PreparedRoute(
                source_program=program,
                argument_builder=builder,
                agent_arg_names=builder.agent_arg_names,
                requirements=materialized.requirements,
                widths=compiled.widths,
                operand_signature=signature,
                compiled=compiled,
            )

    def prepare(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
        n_agents: int,
    ) -> CompiledSimulationProgram:
        """Return the shared lazy/AOT executable without executing its body."""
        if self.execution.device_memory_bytes is not None:
            raise ExecutionPlanningError(
                "Budgeted prewarming is deferred until forward resident-buffer "
                "accounting is available at live dispatch."
            )
        materialized = self._materialize(
            program=program, arguments=arguments, period=period, n_agents=n_agents
        )
        compiled = self._prepare_materialized(program=materialized, n_agents=n_agents)
        with self.lock:
            self.prepared[
                (period, program.name, _func_dedup_key(func=program.function))
            ] = compiled
        return compiled

    def _materialize(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
        n_agents: int,
        residency: SimulationDispatchContext | None = None,
    ) -> MaterializedCoreProgram:
        """Build and place one argument tree before any planner candidate runs."""
        builder = program.argument_builder
        if not isinstance(builder, AgentArgumentNames):
            raise TypeError("Simulation argument builders must declare agent names.")
        materialized = materialize_core_program(
            program=_with_agent_extent(program=program, n_agents=n_agents),
            context=_build_context(arguments=arguments, period=period),
        )
        live = (
            union_buffer_footprints(
                footprints=(
                    residency.live_footprint(),
                    measure_buffer_footprint(tree=materialized.arguments),
                )
            )
            if residency is not None and self.execution.device_memory_bytes is not None
            else None
        )
        return dataclasses.replace(
            materialized,
            arguments=place_simulation_arguments(
                arguments=materialized.arguments,
                agent_arg_names=builder.agent_arg_names,
                value_reads=materialized.requirements.value_reads,
                devices=self.agent_devices,
                budget_bytes=self.execution.device_memory_bytes,
                live_footprint=live,
                budget_devices=() if residency is None else residency.budget_devices,
            ),
        )

    def prepare_abstract(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
        n_agents: int,
        widths: Mapping[str, int],
    ) -> CompiledSimulationProgram:
        """Compile an explicitly placed abstract candidate without admitting a call.

        The caller supplies only shape/dtype/layout descriptors. The shared
        compiler cache stores code; neither concrete buffers nor a budget verdict
        are captured. Actual execution still requires its live residency context.
        """
        return self.compile_candidate(
            program=_materialize_abstract(
                runtime=self,
                program=program,
                arguments=arguments,
                period=period,
                n_agents=n_agents,
            ),
            n_agents=n_agents,
            widths=widths,
            abstract_inputs=True,
        )

    def lower_abstract(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
        period: int,
        n_agents: int,
        widths: Mapping[str, int],
        wave: CompilationWave,
        label: str,
        wait: bool = False,
    ) -> ShapeDtypePytree:
        """Lower an abstract candidate into `wave` and return its output descriptors.

        The wave compiles it off the calling thread and publishes the executable
        to the shared compiler cache when it closes. A candidate already in the
        cache is not lowered again. With `wait`, the descriptors are the compiled
        executable's, carrying the shardings the compiler chose.
        """
        materialized = _materialize_abstract(
            runtime=self,
            program=program,
            arguments=arguments,
            period=period,
            n_agents=n_agents,
        )
        key = _simulation_lowering_key(
            runtime=self, program=materialized, n_agents=n_agents, widths=widths
        )
        with self.lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cast("jax.stages.Compiled", cached.executable).out_info
        compiler = _SimulationCandidateCompiler(
            program=materialized,
            enable_jit=self.enable_jit,
            abstract_inputs=True,
            agent_devices=self.agent_devices,
            shard_agents=self.execution.simulation_sharding == "agents",
            agent_width=min(
                self.execution.axis_widths.get(AGENT_AXIS, n_agents), n_agents
            ),
        )
        return wave.lower(
            lowering_key=key,
            label=label,
            lower=partial(compiler.lower, widths),
            publish=partial(self._publish, key=key, widths=widths),
            wait=wait,
        )

    def _prepare_materialized(
        self,
        *,
        program: MaterializedCoreProgram,
        n_agents: int,
        residency: SimulationDispatchContext | None = None,
    ) -> CompiledSimulationProgram:
        """Recheck live residency while reusing compiled candidates by width."""
        resident = 0
        resident_lookup = None
        budget = self.execution.device_memory_bytes
        if budget is not None:
            if residency is None:
                raise ExecutionPlanningError(
                    "Budgeted dispatch requires a live residency context."
                )
            arguments = measure_buffer_footprint(tree=program.arguments)
            live = union_buffer_footprints(
                footprints=(residency.live_footprint(), arguments)
            )
            require_transfer_headroom(
                live=live,
                destination_bytes={},
                scratch_bytes={},
                budget_bytes=budget,
                devices=residency.budget_devices,
                budget_note=self.execution.device_memory_cap_note(),
            )
            external = resident_bytes_by_device(
                live=live, arguments=arguments, devices=self.agent_devices
            )
            resident = max(external.values())
            resident_lookup = _SimulationResidentBytes(
                live=live, arguments=program.arguments, devices=self.agent_devices
            )
        plan = plan_workspace(
            axes=program.requirements.axes,
            fixed_widths=_dispatch_widths(
                program=program,
                configured=self.execution.axis_widths,
                residency=residency,
                width_ceilings=self.execution.axis_width_ceilings,
            ),
            width_ceilings=self.execution.axis_width_ceilings,
            compile_candidate=_CachedSimulationCandidateCompiler(
                runtime=self, program=program, n_agents=n_agents
            ),
            budget_bytes=budget,
            resident_bytes=resident,
            resident_bytes_for=resident_lookup,
            memory_for=_simulation_memory,
        )
        return plan.compiled

    def _require_budget_context(
        self, *, program: CoreProgram, residency: SimulationDispatchContext | None
    ) -> None:
        """Refuse unprofiled adapters before materialization can allocate operands."""
        if self.execution.device_memory_bytes is None:
            return
        if residency is None:
            raise ExecutionPlanningError(
                "Budgeted dispatch requires a live residency context."
            )
        if not set(self.agent_devices).issubset(residency.budget_devices):
            raise ExecutionPlanningError("The live budget context omits agent devices.")
        if (
            not self.enable_jit
            or program.disposition is CoreExecutionDisposition.HOST_DRIVEN
        ):
            raise ExecutionPlanningError(
                "Budgeted simulation requires a compiled planned program; "
                "eager and host-driven adapters have no profiled workspace bound."
            )

    def compile_candidate(
        self,
        *,
        program: MaterializedCoreProgram,
        n_agents: int,
        widths: Mapping[str, int],
        abstract_inputs: bool = False,
    ) -> CompiledSimulationProgram:
        """Own one compilation per concrete width without retaining live arguments."""
        key = _simulation_lowering_key(
            runtime=self, program=program, n_agents=n_agents, widths=widths
        )
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            future = self.in_flight.get(key)
            owns_compilation = future is None
            if future is None:
                future = Future()
                self.in_flight[key] = future
        if not owns_compilation:
            return future.result()
        try:
            compile_candidate = _SimulationCandidateCompiler(
                program=program,
                enable_jit=self.enable_jit,
                abstract_inputs=abstract_inputs,
                agent_devices=self.agent_devices,
                shard_agents=self.execution.simulation_sharding == "agents",
                agent_width=min(
                    self.execution.axis_widths.get(AGENT_AXIS, n_agents),
                    n_agents,
                ),
            )
            with nested_phase(name="simulation_compilation"):
                compiled = dataclasses.replace(
                    compile_candidate(widths), widths=MappingProxyType(dict(widths))
                )
            if isinstance(compiled.executable, jax.stages.Compiled):
                compiled = _with_compiler_memory(compiled=compiled)
        except BaseException as error:
            with self.lock:
                del self.in_flight[key]
                future.set_exception(error)
            raise
        with self.lock:
            self.cache[key] = compiled
            del self.in_flight[key]
            future.set_result(compiled)
        return compiled

    def _publish(
        self,
        *,
        executable: jax.stages.Compiled,
        key: Hashable,
        widths: Mapping[str, int],
    ) -> None:
        """Cache an executable compiled by a wave under its lowering key."""
        compiled = _with_compiler_memory(
            compiled=CompiledSimulationProgram(
                executable=executable,
                static_kwargs=MappingProxyType({}),
                widths=MappingProxyType(dict(widths)),
            )
        )
        with self.lock:
            self.cache.setdefault(key, compiled)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PreparedRoute:
    """One signature's static preparation, separated from its per-call binding.

    Everything here is fixed by the abstract signature alone and is therefore
    safe to share between calls. The values a call actually computes on —
    the built argument tree and its placement — are rebuilt every time by
    `SimulationRuntime._bind_prepared_route`, which also refuses the record
    unless the freshly bound operands reproduce `operand_signature` exactly.
    """

    source_program: CoreProgram
    """The exact declared program this preparation was published for."""

    argument_builder: CoreArgumentBuilder
    """The declared binder invoked afresh on every call; it owns no arrays."""

    agent_arg_names: tuple[ReferenceName, ...]
    """Operands partitioned across agents, as the builder declares them."""

    requirements: CoreExecutionRequirements
    """Agent-extent descriptor: planner axes and addressed value reads."""

    widths: MappingProxyType[str, int]
    """The selected candidate's width map, reused instead of re-derived.

    Pinned from the compiled candidate itself, so an explicit `ExecutionConfig`
    width and a derived unbudgeted agent width are both carried exactly as the
    planner selected them; nothing here re-runs `workspace_width_candidates`.
    """

    operand_signature: Hashable
    """Abstract identity the bound operands must reproduce to be admitted."""

    compiled: CompiledSimulationProgram
    """The selected executable, already owned by the shared compiler cache."""


def _operand_signature(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
) -> Hashable | None:
    """Return one argument tree's complete abstract identity.

    Tree structure — which also pins the argument names, so an added or dropped
    optional column is a different signature — together with every leaf's shape,
    dtype, weak type and ordered device layout, and the exact value of a typed
    static leaf. `None` reports a tree this runtime cannot key on, which sends
    the call down the full validated route rather than guessing.
    """
    leaves, treedef = jax.tree_util.tree_flatten(dict(arguments))
    signature = (treedef, tuple(_abstract_operand(leaf) for leaf in leaves))
    try:
        hash(signature)
    except TypeError:
        return None
    return cast("Hashable", signature)


def _prepared_route_key(
    *,
    program: CoreProgram,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    period: int,
    n_agents: int,
) -> Hashable | None:
    """Return the caller-side key a prepared route is probed with.

    Built from the caller's own arguments before anything is materialized, so a
    warm hit reaches the record without constructing a static descriptor or a
    width frontier first. The trace context is part of the key, because the
    record holds compiled code. `None` means no route can be keyed for this call.
    """
    signature = _operand_signature(arguments=arguments)
    if signature is None:
        return None
    return (
        period,
        program.name,
        _func_dedup_key(func=program.function),
        program.disposition,
        program.compiler_options,
        n_agents,
        signature,
        _trace_settings_key(),
    )


def _simulation_lowering_key(
    *,
    runtime: SimulationRuntime,
    program: MaterializedCoreProgram,
    n_agents: int,
    widths: Mapping[str, int],
) -> Hashable:
    """Return the full compiler-identity key shared by compilation and dispatch.

    This is the single source of the exact-signature key: program/function
    identity, resolved argument PyTree structure and leaf shape/dtype/weak
    type/sharding, output roles, disposition, device placement and compiler
    options. Reused unchanged by `compile_candidate` (populating the cache) and
    by `compile_candidate`'s own cache probe, so the identity a candidate is
    stored under and the identity it is found under can never diverge.
    """
    return _lowering_key(
        program_identity=_func_dedup_key(func=program.function),
        arguments=jax.tree.map(_abstract_operand, program.arguments),
        specialization_key=(
            n_agents,
            tuple(widths.items()),
            runtime.enable_jit,
            runtime.execution.simulation_sharding,
            tuple((device.platform, device.id) for device in runtime.agent_devices),
        ),
        output_roles=program.output_roles,
        layout_key=program.disposition,
        placement_key=runtime.execution.device_ids,
        compiler_options=program.compiler_options,
    )


def _materialize_abstract(
    *,
    runtime: SimulationRuntime,
    program: CoreProgram,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    period: int,
    n_agents: int,
) -> MaterializedCoreProgram:
    """Bind placed shape descriptors to a compiled program at a population extent."""
    if not runtime.enable_jit or (
        program.disposition is CoreExecutionDisposition.HOST_DRIVEN
    ):
        raise ExecutionPlanningError(
            "Abstract profiling requires a compiled simulation program."
        )
    _require_abstract_arguments(arguments=arguments)
    if not isinstance(program.argument_builder, AgentArgumentNames):
        raise TypeError("Simulation argument builders must declare agent names.")
    materialized = materialize_core_program(
        program=_with_agent_extent(program=program, n_agents=n_agents),
        context=_build_context(arguments=arguments, period=period),
    )
    _require_abstract_arguments(arguments=materialized.arguments)
    # Match dispatch's container structure without placing any concrete leaf.
    return dataclasses.replace(
        materialized,
        arguments=place_simulation_arguments(
            arguments=materialized.arguments,
            agent_arg_names=program.argument_builder.agent_arg_names,
            value_reads=materialized.requirements.value_reads,
            devices=runtime.agent_devices,
        ),
    )


def _with_compiler_memory(
    *, compiled: CompiledSimulationProgram
) -> CompiledSimulationProgram:
    """Attach the executable's compiler report, read once per executable.

    Budgeted dispatch then reuses it rather than rereading it on every
    `plan_workspace` call that admits the executable.
    """
    return dataclasses.replace(
        compiled,
        memory=compiler_memory_reservation(
            compiled=cast("jax.stages.Compiled", compiled.executable),
            widths=compiled.widths,
        ),
    )


def _unbudgeted_agent_width(
    *, program: MaterializedCoreProgram, axis: TiledOutputAxis
) -> int:
    """Return the widest agent tile a constant byte cap admits without a budget.

    Without a declared budget the runtime still derives a bound: the argument
    slice of one agent tile may not exceed `_UNBUDGETED_AGENT_BLOCK_BYTES`.
    The per-agent weight comes from the operands this materialized program
    already holds abstractly — every leaf whose leading dimension is the agent
    extent — so the result depends on nothing but the model's shapes, dtypes and
    the population size, and is identical on every backend and in a certificate.

    The proposal never falls below `_DEFAULT_UNBUDGETED_AGENT_WIDTH`, so no
    population is lowered narrower than the fixed default alone would have
    lowered it, and it passes through `_admissible_width`, so the axis
    alignment, its floor and the extent clamp are unchanged. Width is a lowering
    specialization only: a wider tile moves no value and no RNG stream.
    """
    per_agent = max(
        sum(
            _agent_slice_bytes(leaf=leaf)
            for leaf in jax.tree.leaves(program.arguments)
            if getattr(leaf, "shape", ())[:1] == (axis.extent,)
        ),
        _MIN_AGENT_ARGUMENT_BYTES,
    )
    proposal = max(
        _DEFAULT_UNBUDGETED_AGENT_WIDTH,
        _UNBUDGETED_AGENT_BLOCK_BYTES // per_agent,
    )
    return _admissible_width(axis=axis, width=min(proposal, axis.extent))


def _agent_slice_bytes(*, leaf: jax.Array | HostArray | jax.ShapeDtypeStruct) -> int:
    """Size one agent's share of a leading-axis operand, extended dtypes included."""
    shape = tuple(leaf.shape)
    count = math.prod(shape)
    if isinstance(leaf, jax.Array):
        # nbytes also sizes extended PRNG-key dtypes, which are not NumPy dtypes.
        item_bytes = leaf.nbytes // count if count else 0
    else:
        item_bytes = jax.typeof(leaf).dtype.itemsize
    return item_bytes * math.prod(shape[1:])


def _dispatch_widths(
    *,
    program: MaterializedCoreProgram,
    configured: Mapping[str, int],
    residency: SimulationDispatchContext | None,
    width_ceilings: Mapping[str, int] = MappingProxyType({}),
) -> MappingProxyType[str, int]:
    """Resolve explicit, budgeted, or derived inner simulation widths.

    An unbudgeted simulation keeps the complete population in one outer chunk.
    Its inner agent tile is derived by `_unbudgeted_agent_width`, which
    bounds the block by bytes rather than pinning a model-blind cell count, so
    device programs do enough work per dispatch. Explicit widths and the
    budgeted outer plan remain authoritative.
    """
    fixed = dict(configured)
    if residency is None:
        if AGENT_AXIS not in fixed:
            agent_axis = next(
                (
                    axis
                    for axis in program.requirements.tiled_axes
                    if axis.name == AGENT_AXIS
                ),
                None,
            )
            if agent_axis is not None:
                fixed[AGENT_AXIS] = _unbudgeted_agent_width(
                    program=program, axis=agent_axis
                )
        return MappingProxyType(fixed)
    # Admission normalizes pins against the axis policy and its ceiling. Compare
    # the reservation with that same effective pin, not the original request.
    # Keep the equality guard: an actually different reservation is still an
    # incompatible specialization, even when it would fit under the ceiling.
    effective_pins = bootstrap_widths(
        axes=tuple(
            axis for axis in program.requirements.axes if axis.name in configured
        ),
        fixed_widths=configured,
        width_ceilings=width_ceilings,
    )
    for axis in program.requirements.axes:
        if axis.name not in residency.axis_widths:
            continue
        selected = min(residency.axis_widths[axis.name], axis.extent)
        if axis.name in configured and effective_pins[axis.name] != selected:
            raise ExecutionPlanningError(
                f"Reserved width for {axis.name!r} conflicts with "
                "explicit ExecutionConfig."
            )
        fixed[axis.name] = selected
    return MappingProxyType(fixed)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CachedSimulationCandidateCompiler:
    """A transient planner callback; persistent entries retain only compiled code."""

    runtime: SimulationRuntime
    program: MaterializedCoreProgram
    n_agents: int

    def __call__(self, widths: Mapping[str, int]) -> CompiledSimulationProgram:
        return self.runtime.compile_candidate(
            program=self.program, n_agents=self.n_agents, widths=widths
        )


def _simulation_memory(
    compiled: CompiledSimulationProgram,
) -> CompilerMemoryReservation:
    """Return the exact executable's compiler accounting, cached at compilation.

    Budgeted dispatch only ever reaches a compiled candidate (see
    `SimulationRuntime._require_budget_context`), so a `None` record here means
    the workspace planner asked for memory of an eager or host-driven
    executable outside that contract.
    """
    if compiled.memory is None:
        raise ExecutionPlanningError(
            "Budgeted simulation dispatch requires a compiled executable with "
            "cached compiler memory accounting; got an uncompiled candidate."
        )
    return compiled.memory


def _require_abstract_arguments(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
) -> None:
    """Require shape-only leaves with explicit layouts before abstract lowering."""
    if any(
        not isinstance(leaf, jax.ShapeDtypeStruct)
        or not isinstance(leaf.sharding, jax.sharding.Sharding)
        for leaf in jax.tree.leaves(arguments)
    ):
        raise ExecutionPlanningError(
            "Abstract simulation arguments require ShapeDtypeStruct leaves with "
            "explicit JAX shardings."
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SimulationResidentBytes:
    """Call-owned inventory outside this candidate's compiler-counted operands."""

    live: DeviceBufferFootprint
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree]
    devices: tuple[jax.Device, ...]

    def __call__(self, compiled: CompiledSimulationProgram) -> int:
        """Retain eliminated operands and uncovered parts of overlapping owners."""
        kept = compiler_input_paths(
            compiled=cast("jax.stages.Compiled", compiled.executable),
            arguments=self.arguments,
        )
        with_paths, _ = jax.tree_util.tree_flatten_with_path(dict(self.arguments))
        arguments = measure_buffer_footprint(
            tree=tuple(leaf for path, leaf in with_paths if path in kept)
        )
        return max(
            resident_bytes_by_device(
                live=self.live, arguments=arguments, devices=self.devices
            ).values()
        )


def execute_simulation_program(
    *,
    programs: SimulationPrograms,
    family: str,
    period: int,
    arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree],
    n_agents: int,
) -> PytreeValue:
    """Dispatch one family's published program for the current period.

    The `"decision"` family is the decision forward simulation dispatches,
    which is the type-local one on the grouped route.
    """
    if programs.executor is None:
        msg = "Simulation programs require a call-local executor before dispatch."
        raise ExecutionPlanningError(msg)
    families = {
        "decision": programs.forward_decision,
        "transition": programs.transition,
        "route": programs.route,
        "gate_fold": programs.gate_fold,
        "gate_route": programs.gate_route,
        "policy_prepare": programs.policy_prepare,
        "policy_rank": programs.policy_rank,
        # Frozen identities: the family names enter program identities and keep
        # their spelling.
        "exit_full": programs.exit_full,
        "exit_participating": programs.exit_participating,
    }
    return programs.executor.dispatch(
        program=families[family][period],
        arguments=arguments,
        period=period,
        n_agents=n_agents,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SimulationCandidateCompiler:
    """Compile each proposed width against the same materialized program."""

    program: MaterializedCoreProgram
    """The declared function and this invocation's complete arguments."""

    enable_jit: bool
    """Whether the selected callable is compiled."""

    agent_width: int
    """Width of a singleton or host-driven agent loop."""

    agent_devices: tuple[jax.Device, ...]
    """Ordered execution devices, including when no numerical input is kept."""

    abstract_inputs: bool = False
    """Whether the prepared operands describe required layouts without arrays."""

    shard_agents: bool = False
    """Place the independent agent loop inside the selected device partitions."""

    def __call__(self, widths: Mapping[str, int]) -> CompiledSimulationProgram:
        """Return the executable for exactly these proposed static widths."""
        if not self.enable_jit:
            function, _, static_kwargs = self._bound(widths)
            return CompiledSimulationProgram(
                executable=function, static_kwargs=MappingProxyType(static_kwargs)
            )
        return CompiledSimulationProgram(
            executable=self.lower(widths).compile(), static_kwargs=MappingProxyType({})
        )

    def lower(self, widths: Mapping[str, int]) -> jax.stages.Lowered:
        """Trace and lower the program at exactly these proposed static widths.

        Lowering binds the program to a mesh over the agent devices, so its
        placement belongs to the selected program even if dead-code elimination
        removes every operand. Auto axes retain the compiler's inferred output
        partitions.
        """
        function, arguments, static_kwargs = self._bound(widths)
        mesh = jax.make_mesh(
            (len(self.agent_devices),),
            ("X",),
            (jax.sharding.AxisType.Auto,),
            devices=self.agent_devices,
        )
        with jax.set_mesh(mesh):
            lowered = jax.jit(function, static_argnames=tuple(static_kwargs)).lower(
                **arguments, **static_kwargs
            )
        _assert_lowered_output_tree(
            output_roles=self.program.output_roles,
            output_info=lowered.out_info,
            label=self.program.name,
        )
        return lowered

    def _bound(
        self, widths: Mapping[str, int]
    ) -> tuple[Callable[..., PytreeValue], Mapping[str, ArgumentTree], dict[str, int]]:
        """Return the callable, its dynamic arguments and its static keywords."""
        if (
            self.shard_agents
            and len(self.agent_devices) > 1
            and (
                not self.enable_jit
                or self.program.disposition is CoreExecutionDisposition.HOST_DRIVEN
            )
        ):
            raise ExecutionPlanningError(
                "Agent sharding requires compiled, agent-local programs; "
                "host-driven or eager simulation is not supported by this mode."
            )
        if self.program.disposition is CoreExecutionDisposition.HOST_DRIVEN:
            function = self.program.function
            arguments = self.program.arguments
            static_kwargs = {AGENT_WIDTH_KEYWORD: self.agent_width}
        else:
            transfers = tuple(
                resolve_value_transfer(
                    target=read.target,
                    source=read.source,
                    kind=ValueTransferKind.ALIGNED_LOCAL,
                    stored_template=(
                        leaf := _value_read_argument_leaf(
                            program=self.program, read=read
                        )
                    ),
                    source_sharding=leaf.sharding,
                )
                for read in self.program.requirements.value_reads
            )
            resolved = resolve_core_program(
                program=self.program,
                tile_widths=widths,
                input_transfer_plan=transfers,
                abstract_inputs=self.abstract_inputs,
            )
            function = resolved.function
            arguments = resolved.arguments
            static_kwargs = dict(resolved.static_kwargs)
            static_kwargs.setdefault(AGENT_WIDTH_KEYWORD, self.agent_width)
        if self.shard_agents and len(self.agent_devices) > 1:
            # Read the declaration through any keyword binding the model layer
            # wrapped around the body -- a regime's fixed params arrive as a
            # `functools.partial`, which proxies no attribute -- and shard the
            # bound callable itself, so its binding identity stays the one the
            # dedup and lowering keys were taken from.
            agent_arg_names = declared_agent_shard_arg_names(function=function)
            if agent_arg_names is None:
                raise ExecutionPlanningError(
                    f"Simulation program {self.program.name!r} does not declare "
                    "independent leading-axis agent outputs."
                )
            if agent_arg_names:
                function = shard_agent_function(
                    function=function,
                    agent_arg_names=agent_arg_names,
                    arguments=arguments,
                    static_kwargs=static_kwargs,
                    devices=self.agent_devices,
                    agent_width_keyword=AGENT_WIDTH_KEYWORD,
                    output_roles=self.program.output_roles,
                )
                # The wrapper binds only widths and immutable layout metadata;
                # caller arrays remain dynamic inputs of the existing executable.
                static_kwargs = {}
        return function, arguments, static_kwargs


def _with_agent_extent(*, program: CoreProgram, n_agents: int) -> CoreProgram:
    """Bind a real population extent and omit a trivial singleton planner axis."""
    return dataclasses.replace(
        program,
        requirements=dataclasses.replace(
            program.requirements,
            tiled_axes=tuple(
                dataclasses.replace(axis, extent=n_agents)
                if axis.name == AGENT_AXIS
                else axis
                for axis in program.requirements.tiled_axes
                if axis.name != AGENT_AXIS or n_agents > 1
            ),
        ),
    )


def _build_context(
    *, arguments: Mapping[ReferenceName, PytreeValue | ShapeDtypePytree], period: int
) -> SimulationBuildContext:
    """Carry complete invocation operands without retaining a simulated population."""
    return SimulationBuildContext(
        state_action_space=None,
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params={},
        period=period,
        ages=None,
        call_arguments=MappingProxyType(dict(sorted(arguments.items()))),
    )
