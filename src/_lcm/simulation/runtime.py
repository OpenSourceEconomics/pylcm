"""Plan, lower and dispatch the declared forward-simulation programs."""

import dataclasses
import threading
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future
from types import MappingProxyType
from typing import cast

import jax

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreProgram,
    MaterializedCoreProgram,
    _value_read_argument_leaf,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.value_transfer import (
    ValueTransferKind,
    resolve_value_transfer,
)
from _lcm.execution.workspace_planning import compiler_peak_bytes, plan_workspace
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.operand_placement import (
    SubjectArgumentNames,
    place_simulation_arguments,
)
from _lcm.simulation.program_types import (
    SUBJECT_AXIS,
    SUBJECT_WIDTH_KEYWORD,
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
    _assert_lowered_output_tree,
    _func_dedup_key,
    _lowering_key,
)
from lcm.exceptions import ExecutionPlanningError


def _empty_widths() -> Mapping[str, int]:
    """Supply an immutable empty specialization for an unbound compiler result."""
    return MappingProxyType({})


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompiledSimulationProgram:
    """One selected executable and its static argument bindings."""

    executable: Callable[..., object]
    """The exact executable selected by workspace planning."""

    static_kwargs: Mapping[str, int]
    """Bindings used only by eager execution; compiled programs already bind them."""

    widths: Mapping[str, int] = dataclasses.field(default_factory=_empty_widths)
    """Concrete compiler specialization, never a cached budget admission."""

    def __call__(self, **arguments: object) -> object:
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


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SimulationRuntime:
    """Share one executable cache between lazy dispatch and prewarming."""

    execution: ResolvedExecution
    """Model-owned width and device configuration."""

    enable_jit: bool
    """Whether to compile the selected body or invoke its eager form."""

    subject_devices: tuple[jax.Device, ...]
    """Actual ordered devices evaluating subjects, independent of solve placement."""

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

    in_flight: dict[Hashable, Future[CompiledSimulationProgram]] = dataclasses.field(
        default_factory=dict, repr=False
    )
    """Transient shared results for keys whose candidates are being compiled."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)
    """Protect cache ownership and publication while compilation runs outside it."""

    def dispatch(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
        residency: SimulationDispatchContext | None = None,
    ) -> object:
        """Invoke the selected executable with this call's dynamic arguments."""
        self._require_budget_context(program=program, residency=residency)
        materialized = self._materialize(
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            residency=residency,
        )
        compiled = self._prepare_materialized(
            program=materialized, n_subjects=n_subjects, residency=residency
        )
        return compiled(**materialized.arguments)

    def prepare(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
    ) -> CompiledSimulationProgram:
        """Return the shared lazy/AOT executable without executing its body."""
        if self.execution.device_memory_bytes is not None:
            raise ExecutionPlanningError(
                "Budgeted prewarming is deferred until forward resident-buffer "
                "accounting is available at live dispatch."
            )
        materialized = self._materialize(
            program=program, arguments=arguments, period=period, n_subjects=n_subjects
        )
        compiled = self._prepare_materialized(
            program=materialized, n_subjects=n_subjects
        )
        with self.lock:
            self.prepared[
                (period, program.name, _func_dedup_key(func=program.function))
            ] = compiled
        return compiled

    def _materialize(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
        residency: SimulationDispatchContext | None = None,
    ) -> MaterializedCoreProgram:
        """Build and place one argument tree before any planner candidate runs."""
        builder = program.argument_builder
        if not isinstance(builder, SubjectArgumentNames):
            raise TypeError("Simulation argument builders must declare subject names.")
        materialized = materialize_core_program(
            program=_with_subject_extent(program=program, n_subjects=n_subjects),
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
                subject_arg_names=builder.subject_arg_names,
                value_reads=materialized.requirements.value_reads,
                devices=self.subject_devices,
                budget_bytes=self.execution.device_memory_bytes,
                live_footprint=live,
                budget_devices=() if residency is None else residency.budget_devices,
            ),
        )

    def is_prepared(self, *, program: CoreProgram, period: int) -> bool:
        """Return whether this period's declared body has a compiled selection."""
        with self.lock:
            selected = self.prepared.get(
                (period, program.name, _func_dedup_key(func=program.function))
            )
        return selected is not None and isinstance(
            selected.executable, jax.stages.Compiled
        )

    def _prepare_materialized(
        self,
        *,
        program: MaterializedCoreProgram,
        n_subjects: int,
        residency: SimulationDispatchContext | None = None,
    ) -> CompiledSimulationProgram:
        """Recheck live residency while reusing compiled candidates by width."""
        resident = 0
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
            )
            external = resident_bytes_by_device(
                live=live, arguments=arguments, devices=self.subject_devices
            )
            resident = max(external.values())
        plan = plan_workspace(
            axes=program.requirements.axes,
            fixed_widths=self.execution.axis_widths,
            compile_candidate=_CachedSimulationCandidateCompiler(
                runtime=self, program=program, n_subjects=n_subjects
            ),
            budget_bytes=budget,
            resident_bytes=resident,
            peak_bytes_for=_simulation_peak_bytes,
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
        if not set(self.subject_devices).issubset(residency.budget_devices):
            raise ExecutionPlanningError(
                "The live budget context omits subject devices."
            )
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
        n_subjects: int,
        widths: Mapping[str, int],
    ) -> CompiledSimulationProgram:
        """Own one compilation per concrete width without retaining live arguments."""
        key = _lowering_key(
            program_identity=_func_dedup_key(func=program.function),
            arguments=program.arguments,
            specialization_key=(
                n_subjects,
                tuple(widths.items()),
                self.enable_jit,
            ),
            output_roles=program.output_roles,
            layout_key=program.disposition,
            placement_key=self.execution.device_ids,
            compiler_options=program.compiler_options,
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
                subject_width=min(
                    self.execution.axis_widths.get(SUBJECT_AXIS, n_subjects),
                    n_subjects,
                ),
            )
            compiled = dataclasses.replace(
                compile_candidate(widths), widths=MappingProxyType(dict(widths))
            )
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CachedSimulationCandidateCompiler:
    """A transient planner callback; persistent entries retain only compiled code."""

    runtime: SimulationRuntime
    program: MaterializedCoreProgram
    n_subjects: int

    def __call__(self, widths: Mapping[str, int]) -> CompiledSimulationProgram:
        return self.runtime.compile_candidate(
            program=self.program, n_subjects=self.n_subjects, widths=widths
        )


def _simulation_peak_bytes(compiled: CompiledSimulationProgram) -> int:
    """Read the genuine underlying executable's compiler-reported memory peak."""
    return compiler_peak_bytes(compiled=compiled.executable, widths=compiled.widths)


def execute_simulation_program(
    *,
    programs: SimulationPrograms,
    family: str,
    period: int,
    arguments: Mapping[str, object],
    n_subjects: int,
) -> object:
    """Dispatch one family's published program for the current period."""
    if programs.executor is None:
        msg = "Simulation programs require a call-local executor before dispatch."
        raise ExecutionPlanningError(msg)
    families = {
        "decision": programs.decision,
        "transition": programs.transition,
        "route": programs.route,
    }
    return programs.executor.dispatch(
        program=families[family][period],
        arguments=arguments,
        period=period,
        n_subjects=n_subjects,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SimulationCandidateCompiler:
    """Compile each proposed width against the same materialized program."""

    program: MaterializedCoreProgram
    """The declared function and this invocation's complete arguments."""

    enable_jit: bool
    """Whether the selected callable is compiled."""

    subject_width: int
    """Width of a singleton or host-driven subject loop."""

    def __call__(self, widths: Mapping[str, int]) -> CompiledSimulationProgram:
        """Return the executable for exactly these proposed static widths."""
        if self.program.disposition is CoreExecutionDisposition.HOST_DRIVEN:
            function = self.program.function
            arguments = self.program.arguments
            static_kwargs = {SUBJECT_WIDTH_KEYWORD: self.subject_width}
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
                    source_sharding=cast("jax.sharding.Sharding", leaf.sharding),
                )
                for read in self.program.requirements.value_reads
            )
            resolved = resolve_core_program(
                program=self.program,
                tile_widths=widths,
                input_transfer_plan=transfers,
            )
            function = resolved.function
            arguments = resolved.arguments
            static_kwargs = dict(resolved.static_kwargs)
            static_kwargs.setdefault(SUBJECT_WIDTH_KEYWORD, self.subject_width)
        if not self.enable_jit:
            return CompiledSimulationProgram(
                executable=function, static_kwargs=MappingProxyType(static_kwargs)
            )
        lowered = jax.jit(function, static_argnames=tuple(static_kwargs)).lower(
            **arguments, **static_kwargs
        )
        _assert_lowered_output_tree(
            output_roles=self.program.output_roles,
            output_info=lowered.out_info,
            label=self.program.name,
        )
        return CompiledSimulationProgram(
            executable=lowered.compile(), static_kwargs=MappingProxyType({})
        )


def _with_subject_extent(*, program: CoreProgram, n_subjects: int) -> CoreProgram:
    """Bind a real population extent and omit a trivial singleton planner axis."""
    return dataclasses.replace(
        program,
        requirements=dataclasses.replace(
            program.requirements,
            tiled_axes=tuple(
                dataclasses.replace(axis, extent=n_subjects)
                if axis.name == SUBJECT_AXIS
                else axis
                for axis in program.requirements.tiled_axes
                if axis.name != SUBJECT_AXIS or n_subjects > 1
            ),
        ),
    )


def _build_context(
    *, arguments: Mapping[str, object], period: int
) -> SimulationBuildContext:
    """Carry complete invocation operands without retaining a simulated population."""
    return SimulationBuildContext(
        state_action_space=None,
        next_regime_to_V_arr={},
        next_regime_to_continuation={},
        flat_params={},
        period=period,
        ages=None,
        call_arguments=dict(sorted(arguments.items())),
    )
