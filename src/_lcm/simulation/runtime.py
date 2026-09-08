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
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.simulation.program_types import (
    SUBJECT_AXIS,
    SUBJECT_WIDTH_KEYWORD,
    SimulationBuildContext,
    SimulationPrograms,
)
from _lcm.solution.backward_induction import (
    _assert_lowered_output_tree,
    _func_dedup_key,
    _lowering_key,
)
from lcm.exceptions import ExecutionPlanningError


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompiledSimulationProgram:
    """One selected executable and its static argument bindings."""

    executable: Callable[..., object]
    """The exact executable selected by workspace planning."""

    static_kwargs: Mapping[str, int]
    """Bindings used only by eager execution; compiled programs already bind them."""

    def __call__(self, **arguments: object) -> object:
        """Execute with live arrays; retain no call arguments on the cache entry."""
        return self.executable(**arguments, **self.static_kwargs)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class SimulationRuntime:
    """Share one executable cache between lazy dispatch and prewarming."""

    execution: ResolvedExecution
    """Model-owned width and device configuration."""

    enable_jit: bool
    """Whether to compile the selected body or invoke its eager form."""

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
    ) -> object:
        """Invoke the selected executable with this call's dynamic arguments."""
        materialized = materialize_core_program(
            program=_with_subject_extent(program=program, n_subjects=n_subjects),
            context=_build_context(arguments=arguments, period=period),
        )
        compiled = self._prepare_materialized(
            program=materialized, n_subjects=n_subjects
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
        materialized = materialize_core_program(
            program=_with_subject_extent(program=program, n_subjects=n_subjects),
            context=_build_context(arguments=arguments, period=period),
        )
        compiled = self._prepare_materialized(
            program=materialized, n_subjects=n_subjects
        )
        with self.lock:
            self.prepared[
                (period, program.name, _func_dedup_key(func=program.function))
            ] = compiled
        return compiled

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
        self, *, program: MaterializedCoreProgram, n_subjects: int
    ) -> CompiledSimulationProgram:
        """Select an executable after the declared builder has run exactly once."""
        if self.execution.device_memory_bytes is not None:
            msg = (
                "Simulation memory budgets require forward resident-buffer "
                "accounting; this execution path does not yet support a "
                "device_memory_bytes budget."
            )
            raise ExecutionPlanningError(msg)
        key = _lowering_key(
            program_identity=_func_dedup_key(func=program.function),
            arguments=program.arguments,
            specialization_key=(
                n_subjects,
                tuple(self.execution.axis_widths.items()),
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
            plan = plan_workspace(
                axes=program.requirements.axes,
                fixed_widths=self.execution.axis_widths,
                compile_candidate=compile_candidate,
            )
        except BaseException as error:
            with self.lock:
                del self.in_flight[key]
                future.set_exception(error)
            raise
        with self.lock:
            self.cache[key] = plan.compiled
            del self.in_flight[key]
            future.set_result(plan.compiled)
        return plan.compiled


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
            static_kwargs = dict(resolved.static_kwargs)
            static_kwargs.setdefault(SUBJECT_WIDTH_KEYWORD, self.subject_width)
        if not self.enable_jit:
            return CompiledSimulationProgram(
                executable=function, static_kwargs=MappingProxyType(static_kwargs)
            )
        lowered = jax.jit(function, static_argnames=tuple(static_kwargs)).lower(
            **self.program.arguments, **static_kwargs
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
