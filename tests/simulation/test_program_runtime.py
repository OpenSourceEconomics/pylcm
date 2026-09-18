"""Simulation dispatches the programs declared by each canonical regime."""

import dataclasses
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.runtime as runtime_module
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.execution_plan import ResolvedExecution
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


def test_user_subject_width_name_remains_an_economic_action() -> None:
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


@pytest.mark.parametrize("family", ["decision", "transition", "route"])
def test_simulate_dispatches_the_declared_program_body(
    *, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """Runtime dispatch executes each declared family on real subjects."""
    model, params, initial = WITNESSES["multi_regime"]()
    model = Model(
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=model.fixed_params,
    )
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


@pytest.mark.parametrize("width", [1, 3, 7])
def test_public_widths_reach_the_live_subject_and_action_loops(
    *, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """The public request binds the static widths of the actual decision body."""
    model, params, initial = WITNESSES["multi_regime"](
        execution_config=ExecutionConfig(
            axis_widths={"subject": width, "action_product": width}
        ),
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
        argument_builder=_ArgumentsBoundAtDispatch(
            program_name="simulate_transition", subject_arg_names=("state",)
        ),
        requirements=CoreExecutionRequirements(
            tiled_axes=(subject_axis(state_names=("state",)),)
        ),
        output_roles="next_state",
        disposition=CoreExecutionDisposition.PLANNED,
        donation_candidates=(),
    )


def _runtime(
    *, width: int | None = 3, enable_jit: bool = True, budget: int | None = None
) -> SimulationRuntime:
    """Build an executor with an optional subject tile and budget policy."""
    return SimulationRuntime(
        execution=ResolvedExecution(
            device_ids=(0,),
            sharded_states=frozenset(),
            axis_widths=MappingProxyType({} if width is None else {"subject": width}),
            device_memory_bytes=budget,
        ),
        enable_jit=enable_jit,
        subject_devices=(jax.devices()[0],),
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


@pytest.mark.parametrize(("n_subjects", "expected_width"), [(7, 7), (4097, 4097)])
def test_unbudgeted_simulation_uses_the_subject_specific_inner_width(
    *,
    monkeypatch: pytest.MonkeyPatch,
    n_subjects: int,
    expected_width: int,
) -> None:
    """The inner tile is the widest the byte cap admits, the population complete.

    A scalar-per-subject program is far lighter than the standing per-subject
    weight, so both populations here fit one tile; the derived rule and its
    bounds are pinned in `test_unbudgeted_subject_width.py`.
    """
    selected: list[tuple[dict[str, int], int]] = []
    original = runtime_module.plan_workspace

    def observe(**kwargs: Any) -> Any:
        plan = original(**kwargs)
        selected.append((dict(plan.widths), kwargs["axes"][0].extent))
        return plan

    monkeypatch.setattr(runtime_module, "plan_workspace", observe)
    output = _runtime(width=None).dispatch(
        program=_program(),
        arguments={"state": jnp.arange(n_subjects, dtype=float)},
        period=0,
        n_subjects=n_subjects,
    )

    assert (selected, np.asarray(output).tolist()) == (
        [({"subject": expected_width}, n_subjects)],
        (np.arange(n_subjects) + 1).tolist(),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _RecordingArguments:
    """Record materializations while declaring the exact subject operand."""

    calls: list[int]
    subject_arg_names: tuple[str, ...] = ("state",)

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        if not isinstance(context, SimulationBuildContext):
            raise TypeError("This test builder requires simulation arguments.")
        self.calls.append(context.period)
        return dict(context.call_arguments)


def test_dispatch_invokes_the_argument_builder_once() -> None:
    """One invocation applies the declared argument transformation exactly once."""
    calls: list[int] = []

    _runtime().dispatch(
        program=dataclasses.replace(
            _program(), argument_builder=_RecordingArguments(calls=calls)
        ),
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
