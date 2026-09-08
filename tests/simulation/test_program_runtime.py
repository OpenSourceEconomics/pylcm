"""Simulation dispatches the programs declared by each canonical regime."""

import dataclasses
import logging
import threading
import weakref
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.compile as compile_module
import _lcm.simulation.runtime as runtime_module
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation.compile import _drain_compilations
from _lcm.simulation.program_types import (
    SUBJECT_WIDTH_KEYWORD,
    SimulationBuildContext,
    subject_axis,
)
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.runtime import CompiledSimulationProgram, SimulationRuntime
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm import AgeGrid, LinSpacedGrid, Model, categorical, fixed_transition
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.regime import Regime as UserRegime
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt, UserParams
from tests.test_models.processes import MultiRegimeId


@categorical(ordered=False)
class _WidthCollisionRegimeId:
    alive: ScalarInt
    done: ScalarInt


def _width_collision_utility(
    *, _lcm_subject_width: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    """Read a legal user action whose spelling resembles an execution keyword."""
    return _lcm_subject_width + wealth


def _width_collision_next_regime() -> ScalarInt:
    """Enter the terminal regime after one decision."""
    return _WidthCollisionRegimeId.done


def _width_collision_terminal_utility(*, wealth: ContinuousState) -> FloatND:
    """Return an action-free terminal value."""
    return wealth


@pytest.mark.parametrize("prewarm", [False, True])
def test_user_subject_width_name_remains_an_economic_action(*, prewarm: bool) -> None:
    """A legal user action cannot be consumed as an internal static tile width."""
    model = Model(
        regimes={
            "alive": UserRegime(
                transition=_width_collision_next_regime,
                active=lambda age: age == 0,
                functions={"utility": _width_collision_utility},
                actions={
                    "_lcm_subject_width": LinSpacedGrid(start=1, stop=2, n_points=2)
                },
            ),
            "done": UserRegime(
                transition=None,
                functions={"utility": _width_collision_terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_WidthCollisionRegimeId,
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
        state_transitions={"wealth": fixed_transition("wealth")},
        execution_config=ExecutionConfig(axis_widths={"subject": 1}),
        n_subjects=2 if prewarm else None,
    )
    params: UserParams = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    frame = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(2),
            "wealth": jnp.asarray([1.0, 2.0]),
            "regime_id": jnp.full(2, _WidthCollisionRegimeId.alive, dtype=jnp.int32),
        },
        log_level="off",
    ).to_dataframe(use_labels=False)
    np.testing.assert_array_equal(
        frame.query("period == 0")["_lcm_subject_width"], np.asarray([2.0, 2.0])
    )


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("family", ["decision", "transition", "route"])
def test_simulate_dispatches_the_declared_program_body(
    *, monkeypatch: pytest.MonkeyPatch, prewarm: bool, family: str
) -> None:
    """Both compilation modes execute each declared family on real subjects."""
    model, params, initial = WITNESSES["multi_regime"]()
    model = Model(
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=model.fixed_params,
    )
    model.n_subjects = 7 if prewarm else None
    solution = model.solve(params=params, log_level="off")
    body_ids = {
        id(
            program.function.func
            if isinstance(program.function, partial)
            else program.function
        )
        for regime in model._regimes.values()
        for program in getattr(regime.simulation.programs, family).values()
    }
    reached = []
    original = _SubjectTiled.__call__

    def record(self: _SubjectTiled, **kwargs: Any) -> object:
        if id(self) in body_ids:
            reached.append(id(self))
        return original(self, **kwargs)

    monkeypatch.setattr(_SubjectTiled, "__call__", record)
    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
    )
    assert reached


@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize("width", [1, 3, 7])
def test_public_widths_reach_the_live_subject_and_action_loops(
    *, monkeypatch: pytest.MonkeyPatch, prewarm: bool, width: int
) -> None:
    """The public request binds the static widths of the actual decision body."""
    model, params, initial = WITNESSES["multi_regime"](
        execution_config=ExecutionConfig(
            axis_widths={"subject": width, "action_product": width}
        ),
        n_subjects=7 if prewarm else None,
    )
    solution = model.solve(params=params, log_level="off")
    body_widths = {}
    for regime in model._regimes.values():
        for program in regime.simulation.programs.decision.values():
            body = program.function
            if isinstance(body, partial):
                body = body.func
            body_widths[id(body)] = {
                SUBJECT_WIDTH_KEYWORD: width,
                **{
                    axis.width_keyword: min(width, axis.extent)
                    for axis in program.requirements.reduced_axes
                },
            }
    observed = []
    original = _SubjectTiled.__call__

    def record(self: _SubjectTiled, **kwargs: Any) -> object:
        if id(self) in body_widths:
            expected = body_widths[id(self)]
            observed.append({name: kwargs[name] for name in expected} == expected)
        return original(self, **kwargs)

    monkeypatch.setattr(_SubjectTiled, "__call__", record)
    model.simulate(
        params=params, initial_conditions=initial, solution=solution, log_level="off"
    )
    assert (bool(observed), all(observed)) == (True, True)


def _increment_subject(*, state: FloatND) -> FloatND:
    """Advance the independently observed state by one unit."""
    return state + 1


def _program() -> CoreProgram:
    """Declare a single subject-valued program with a transparent scalar body."""
    return CoreProgram(
        name="simulate_transition",
        function=_SubjectTiled(func=_increment_subject, subject_arg_names=("state",)),
        argument_builder=_ArgumentsBoundAtDispatch(program_name="simulate_transition"),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),)
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _runtime(
    *, width: int = 3, enable_jit: bool = True, budget: int | None = None
) -> SimulationRuntime:
    """Build an executor with a fixed subject tile and an explicit budget policy."""
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({"subject": width}),
            device_memory_bytes=budget,
        ),
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("n_subjects", [1, 7])
@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_subject_tiles_preserve_every_output(
    *, n_subjects: int, width: int, enable_jit: bool
) -> None:
    """Singletons and incomplete final tiles return every subject in original order."""
    state = jnp.arange(n_subjects, dtype=float)
    output = _runtime(width=width, enable_jit=enable_jit).dispatch(
        program=_program(),
        arguments={"state": state},
        period=0,
        n_subjects=n_subjects,
    )
    np.testing.assert_array_equal(output, np.arange(n_subjects) + 1)


def test_dispatch_invokes_the_argument_builder_once() -> None:
    """One invocation applies the declared argument transformation exactly once."""
    calls = []

    def build(context: SimulationBuildContext) -> dict[str, object]:
        calls.append(context.period)
        return dict(context.call_arguments)

    _runtime().dispatch(
        program=dataclasses.replace(_program(), argument_builder=build),
        arguments={"state": jnp.arange(7.0)},
        period=0,
        n_subjects=7,
    )
    assert calls == [0]


def test_prepared_program_dispatch_uses_the_cached_executable() -> None:
    """Live values reuse the exact executable selected for equal abstract arguments."""
    runtime = _runtime()
    program = _program()
    first = runtime.prepare(
        program=program,
        arguments={"state": jnp.arange(7.0)},
        period=0,
        n_subjects=7,
    )
    second = runtime.prepare(
        program=program,
        arguments={"state": jnp.arange(7.0) + 10},
        period=1,
        n_subjects=7,
    )
    assert second is first


@pytest.mark.parametrize("enable_jit", [False, True])
def test_dispatch_executes_the_exact_planner_selection(
    *, monkeypatch: pytest.MonkeyPatch, enable_jit: bool
) -> None:
    """A selected executable wrapper is consumed with the live subject arrays."""
    selections = []
    calls = []
    original = runtime_module.plan_workspace

    def select(**kwargs: Any) -> object:
        plan = original(**kwargs)
        marker = object()
        selections.append(marker)

        def execute(**arguments: object) -> object:
            calls.append(marker)
            return plan.compiled(**arguments)

        return dataclasses.replace(
            plan,
            compiled=CompiledSimulationProgram(
                executable=execute, static_kwargs=MappingProxyType({})
            ),
        )

    monkeypatch.setattr(runtime_module, "plan_workspace", select)
    output = _runtime(enable_jit=enable_jit).dispatch(
        program=_program(),
        arguments={"state": jnp.arange(7.0)},
        period=0,
        n_subjects=7,
    )
    assert (bool(selections), calls == selections, np.asarray(output).tolist()) == (
        True,
        True,
        list(range(1, 8)),
    )


def test_compiler_specialization_metadata_separates_cache_entries() -> None:
    """Distinct fixed numerical compiler options cannot share a selected program."""
    runtime = _runtime()
    program = _program()
    for unroll in (1, 2):
        runtime.prepare(
            program=dataclasses.replace(
                program, compiler_options=(("scan_unroll", unroll),)
            ),
            arguments={"state": jnp.arange(7.0)},
            period=0,
            n_subjects=7,
        )
    assert len(runtime.cache) == 2


def test_independent_program_keys_compile_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent candidates can enter compilation together in a bounded pool."""
    runtime = _runtime()
    program = _program()
    barrier = threading.Barrier(2, timeout=5)
    original = runtime_module._SimulationCandidateCompiler.__call__

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def compile_candidate(
        self: runtime_module._SimulationCandidateCompiler, widths: Mapping[str, int], /
    ) -> CompiledSimulationProgram:
        barrier.wait()
        return original(self, widths)

    monkeypatch.setattr(
        runtime_module._SimulationCandidateCompiler, "__call__", compile_candidate
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                runtime.prepare,
                program=dataclasses.replace(
                    program, compiler_options=(("scan_unroll", option),)
                ),
                arguments={"state": jnp.arange(7.0)},
                period=0,
                n_subjects=7,
            )
            for option in (1, 2)
        ]
        results = [future.result() for future in futures]
    assert len({id(result) for result in results}) == 2


def test_concurrent_duplicate_program_keys_compile_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent requests for one abstract key share one selected candidate."""
    runtime = _runtime()
    program = _program()
    barrier = threading.Barrier(2, timeout=5)
    compilations = []
    original = runtime_module._SimulationCandidateCompiler.__call__

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def compile_candidate(
        self: runtime_module._SimulationCandidateCompiler, widths: Mapping[str, int], /
    ) -> CompiledSimulationProgram:
        compilations.append(1)
        return original(self, widths)

    def prepare() -> CompiledSimulationProgram:
        barrier.wait()
        return runtime.prepare(
            program=program,
            arguments={"state": jnp.arange(7.0)},
            period=0,
            n_subjects=7,
        )

    monkeypatch.setattr(
        runtime_module._SimulationCandidateCompiler, "__call__", compile_candidate
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare) for _ in range(2)]
        results = [future.result() for future in futures]
    assert (compilations, results[0] is results[1]) == ([1], True)


@pytest.mark.parametrize("workers", [1, 2])
def test_public_prewarming_uses_the_bounded_compile_pool(
    *, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    """Core candidates run in worker threads within the requested concurrency."""
    model, params, initial = WITNESSES["multi_regime"](n_subjects=7)
    solution = model.solve(params=params, log_level="off")
    thread_ids = set()
    original = runtime_module._SimulationCandidateCompiler.__call__

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def compile_candidate(
        self: runtime_module._SimulationCandidateCompiler, widths: Mapping[str, int], /
    ) -> CompiledSimulationProgram:
        thread_ids.add(threading.get_ident())
        return original(self, widths)

    monkeypatch.setattr(
        runtime_module._SimulationCandidateCompiler, "__call__", compile_candidate
    )
    model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        max_compilation_workers=workers,
    )
    assert (0 < len(thread_ids) <= workers, threading.get_ident() in thread_ids) == (
        True,
        False,
    )


def test_a_failed_candidate_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compilation error clears its transient key instead of poisoning retries."""
    runtime = _runtime()
    program = _program()
    original = runtime_module._SimulationCandidateCompiler.__call__
    attempts = []

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def compile_candidate(
        self: runtime_module._SimulationCandidateCompiler, widths: Mapping[str, int], /
    ) -> CompiledSimulationProgram:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("candidate compilation failed")
        return original(self, widths)

    monkeypatch.setattr(
        runtime_module._SimulationCandidateCompiler, "__call__", compile_candidate
    )
    with pytest.raises(RuntimeError, match="candidate compilation failed"):
        runtime.prepare(
            program=program,
            arguments={"state": jnp.arange(7.0)},
            period=0,
            n_subjects=7,
        )
    output = runtime.dispatch(
        program=program, arguments={"state": jnp.arange(7.0)}, period=0, n_subjects=7
    )
    assert (runtime.in_flight, np.asarray(output).tolist()) == ({}, list(range(1, 8)))


def test_prewarming_drains_successes_and_errors_before_raising() -> None:
    """All task references close when any selected candidate fails compilation."""
    success: Future[None] = Future()
    failure: Future[None] = Future()
    success.set_result(None)
    failure.set_exception(RuntimeError("candidate compilation failed"))
    futures = {success, failure}
    with pytest.raises(RuntimeError, match="candidate compilation failed"):
        _drain_compilations(futures=futures)
    assert futures == set()


class _TrackedArguments(dict[str, object]):
    """A weak-referenceable owner of a prewarming argument tree."""


def test_prewarming_bounds_live_argument_trees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow workers cannot leave a horizon of template trees queued in the producer."""
    model, params, _ = WITNESSES["multi_regime"](n_subjects=7)
    owners = []
    peak = [0]
    workers_started = threading.Semaphore(0)
    release_workers = threading.Event()
    third_owner_created = threading.Event()

    def track_builder(original: Any) -> Any:
        def build(**kwargs: Any) -> dict[str, object]:
            arguments = _TrackedArguments(original(**kwargs))
            owners.append(weakref.ref(arguments))
            alive = sum(owner() is not None for owner in owners)
            peak[0] = max(peak[0], alive)
            if alive > 2:
                third_owner_created.set()
            return arguments

        return build

    def slow_worker(*, arguments: Mapping[str, object], **_kwargs: object) -> None:
        workers_started.release()
        if not release_workers.wait(timeout=10):
            raise RuntimeError("controlled compilation worker was never released")
        tuple(arguments)

    for name in ("_build_argmax_args", "_build_next_state_args", "_build_crtp_args"):
        monkeypatch.setattr(
            compile_module, name, track_builder(getattr(compile_module, name))
        )
    monkeypatch.setattr(compile_module, "_prepare_and_log", slow_worker)
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(
            model._ensure_simulate_compiled,
            compile_batch_size=7,
            flat_params=model._process_params(params),
            max_compilation_workers=2,
            log=logging.getLogger("prewarm-lifetime-test"),
        )
        try:
            for _ in range(2):
                if not workers_started.acquire(timeout=5):
                    raise RuntimeError("controlled workers did not start")
            third_owner_created.wait(timeout=0.5)
        finally:
            release_workers.set()
        future.result()
    assert peak[0] <= 2


def test_simulation_budget_refuses_unaccounted_forward_residency() -> None:
    """An unsupported resident-memory schedule cannot be reported budget-feasible."""
    with pytest.raises(
        ExecutionPlanningError, match="forward resident-buffer accounting"
    ):
        _runtime(budget=1_000_000).prepare(
            program=_program(),
            arguments={"state": jnp.arange(7.0)},
            period=0,
            n_subjects=7,
        )
