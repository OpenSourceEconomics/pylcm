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
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from _lcm.simulation.program_types import (
    SUBJECT_WIDTH_KEYWORD,
    SimulationBuildContext,
    subject_axis,
)
from _lcm.simulation.programs import _ArgumentsBoundAtDispatch, _SubjectTiled
from _lcm.simulation.runtime import CompiledSimulationProgram, SimulationRuntime
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DiscreteGrid,
    InvariantBlockSchedule,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.test_models import independent_types
from tests.test_models.graph import with_fixture_graph
from tests.test_models.initial_nodes import initial_nodes_of
from tests.test_models.processes import MultiRegimeId


def _fixed_cost_of_work(*, age: float, reference_age: FloatND) -> FloatND:
    """Return the age-dependent cost using a namespaced fixed parameter."""
    return age - reference_age


def _fixed_cost_utility(
    *,
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    fixed_cost_of_work: FloatND,
) -> FloatND:
    """Consumption and preference type determine utility net of the work cost."""
    return consumption + pref_type - fixed_cost_of_work


def _fixed_cost_terminal(
    *, wealth: ContinuousState, pref_type: DiscreteState
) -> FloatND:
    """The terminal value retains both state axes."""
    return wealth + pref_type


@pytest.mark.parametrize(
    ("enable_jit", "subject_sharding"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "schedule",
    [None, InvariantBlockSchedule.PERIOD_MAJOR, InvariantBlockSchedule.BLOCK_MAJOR],
)
def test_simulation_preserves_nested_fixed_parameters(
    *,
    enable_jit: bool,
    subject_sharding: bool,
    combined: bool,
    schedule: InvariantBlockSchedule | None,
) -> None:
    """A nested fixed work cost reaches the chosen action and published value."""
    if subject_sharding and jax.local_device_count() < 2:
        pytest.skip("requires two actual devices for subject sharding")
    grid = LinSpacedGrid(start=0, stop=2, n_points=3)
    model = with_fixture_graph(
        regimes={
            "working": UserRegime(
                regime_transitions=_SupportedDeterministicTransition(
                    func=lambda: independent_types.RegimeId.terminal,
                    targets=("terminal",),
                ),
                states={
                    "wealth": grid,
                    "pref_type": DiscreteGrid(
                        category_class=independent_types.PrefType
                    ),
                },
                state_transitions={
                    "wealth": independent_types.next_wealth,
                    "pref_type": fixed_transition("pref_type"),
                },
                actions={"consumption": grid},
                functions={
                    "utility": _fixed_cost_utility,
                    "fixed_cost_of_work": _fixed_cost_of_work,
                },
                constraints={"affordable": independent_types.affordable},
            ),
            "terminal": UserRegime(
                regime_transitions=None,
                states={
                    "wealth": grid,
                    "pref_type": DiscreteGrid(
                        category_class=independent_types.PrefType
                    ),
                },
                functions={"utility": _fixed_cost_terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=independent_types.RegimeId,
        initial_nodes={0: "working"},
        enable_jit=enable_jit,
        fixed_params={"working": {"fixed_cost_of_work": {"reference_age": -1.0}}},
        execution_config=ExecutionConfig(
            devices=tuple(
                device.id
                for device in jax.local_devices()[: 2 if subject_sharding else 1]
            ),
            simulation_sharding="subjects" if subject_sharding else "legacy",
            invariant_block_widths={} if schedule is None else {"pref_type": 1},
            invariant_block_schedule=(
                InvariantBlockSchedule.PERIOD_MAJOR if schedule is None else schedule
            ),
            axis_widths={"subject": 2, "action_product": 2},
        ),
    )
    params: UserParams = {"working": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    solution = None if combined else model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.full(4, 2.0),
            "pref_type": jnp.asarray([1, 0, 1, 0], dtype=jnp.int32),
            "age": jnp.zeros(4),
            "regime_id": jnp.full(4, independent_types.RegimeId.working),
        },
        seed=17,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False).sort_values(["period", "subject_id"])
    np.testing.assert_array_equal(
        np.column_stack(
            (
                frame["regime_name"].map({"working": 0, "terminal": 1}),
                frame[["consumption", "wealth", "pref_type", "value"]],
            )
        ),
        np.asarray(
            [
                [0, 2, 2, 1, 2],
                [0, 2, 2, 0, 1],
                [0, 2, 2, 1, 2],
                [0, 2, 2, 0, 1],
                [1, np.nan, 0, 1, 1],
                [1, np.nan, 0, 0, 0],
                [1, np.nan, 0, 1, 1],
                [1, np.nan, 0, 0, 0],
            ]
        ),
    )


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
    model = with_fixture_graph(
        regimes={
            "alive": UserRegime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(
                            start=0, exclusive_stop=1
                        ): _SupportedDeterministicTransition(
                            func=_width_collision_next_regime, targets=("done",)
                        )
                    }
                ),
                functions={"utility": _width_collision_utility},
                actions={
                    "_lcm_subject_width": LinSpacedGrid(start=1, stop=2, n_points=2)
                },
            ),
            "done": UserRegime(
                regime_transitions=None,
                functions={"utility": _width_collision_terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_WidthCollisionRegimeId,
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
        state_transitions={"wealth": fixed_transition("wealth")},
        execution_config=ExecutionConfig(axis_widths={"subject": 1}),
        initial_nodes={0: "alive"},
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
        edges=model.graph.edges,
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=model.fixed_params,
        initial_nodes=initial_nodes_of(model=model),
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
