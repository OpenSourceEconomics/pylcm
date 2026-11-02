"""Chunk planning lowers forward programs on the caller and compiles them on a pool."""

import re
import threading
from collections import Counter
from collections.abc import Hashable, Mapping
from typing import Never, TypedDict, Unpack
from weakref import WeakKeyDictionary

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.execution.compiler_memory import CompilerMemoryReport
from _lcm.simulation.chunk_planning import SimulationChunkPlan
from _lcm.simulation.chunk_profile_inventory import ChunkProfileInventory
from _lcm.simulation.runtime import (
    CompiledSimulationProgram,
    SimulationRuntime,
    _SimulationCandidateCompiler,
)
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from lcm_examples import collective_household as household
from tests.execution._jax_callback_types import (
    CompileOptions,
    CompilerOptions,
    LowerOptions,
)
from tests.simulation._callback_types import (
    ChunkPreparation,
    ChunkPreparationResult,
    InventoryCompiled,
    RuntimePreparation,
)
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
_DEBUG_SECTIONS = frozenset(
    {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}
)
_WORKERS = (1, 2)


type _ChunkContract = tuple[
    int,
    int,
    tuple[tuple[str, int, int], ...],
    dict[jax.Device, int],
    dict[jax.Device, int],
    dict[jax.Device, int],
    dict[jax.Device, int],
]


class _Observation(TypedDict):
    lowerings: list[bool]
    compiles: list[tuple[bool, str]]
    published: dict[Hashable, str]
    contract: _ChunkContract
    panel: pd.DataFrame
    program_lowerings: list[tuple[str, bool]]
    program_compiles: list[tuple[str, bool, str]]
    publications: dict[Hashable, tuple[str, int, int]]
    prepared: list[tuple[Hashable, int, int]]
    report_reads: dict[int, int]
    inventory_exits: list[tuple[int, int]]
    raw_tree: str
    raw_arrays: tuple[tuple[tuple[int, ...], str, bytes], ...]


def _program_text(*, hlo: str) -> str:
    """Return compiled HLO without source-location metadata.

    The metadata names source files and wrapper addresses, which differ between
    two otherwise identical compilations.
    """
    blocks = (
        block
        for block in hlo.split("\n\n")
        if block.strip().split("\n", 1)[0] not in _DEBUG_SECTIONS
    )
    return re.sub(r", metadata=\{[^}]*\}", "", "\n\n".join(blocks))


def _readable_key(key: Hashable) -> tuple[Hashable, ...]:
    """Project a runtime lowering key onto its argument names and specialization.

    The program identity is an object address, so it is dropped; the argument
    names, subject extent and axis widths identify each forward program of the
    small model.
    """
    _, arguments, specialization, *_ = key  # ty: ignore[not-iterable]
    return (tuple(name for name, _ in arguments), *specialization[:2])


def _simulate_and_observe(  # noqa: C901, PLR0915 - keep scoped observers at the real public boundary
    *,
    monkeypatch: pytest.MonkeyPatch,
    max_compilation_workers: int,
    participation: bool = False,
) -> _Observation:
    """Simulate a fresh budgeted model, recording chunk planning's lowerings,
    compiles, published forward executables and plan."""
    if participation:
        model = household.get_model(
            n_periods=3,
            wealth_n_points=3,
            consumption_n_points=2,
            execution_config=ExecutionConfig(
                axis_widths={"query_cell": 1}, device_memory_bytes=2**31
            ),
        )
        params = household.get_params()
        initial = household.get_initial_conditions(n_subjects=4, model=model)
        assert all(
            model._regimes[name].simulation.exit_queries
            for name in ("couple", "single_f", "single_m")
        )
    else:
        model = _stateful_target_model()
        params = _PARAMS
        initial = {
            "wealth": jnp.asarray([1.0, 2.0, 3.0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
        }
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    caller = threading.get_ident()
    planning = [False]
    lock = threading.Lock()
    lowerings: list[bool] = []
    compiles: list[tuple[bool, str]] = []
    published: dict[Hashable, str] = {}
    program_lowerings: list[tuple[str, bool]] = []
    program_compiles: list[tuple[str, bool, str]] = []
    program_names: WeakKeyDictionary[jax.stages.Lowered, str] = WeakKeyDictionary()
    candidate_lower = _SimulationCandidateCompiler.lower
    publish = SimulationRuntime._publish
    prepare_abstract = SimulationRuntime.prepare_abstract
    analyze = jax.stages.Compiled.memory_analysis
    publications: dict[Hashable, tuple[str, int, int]] = {}
    prepared_entries: list[tuple[Hashable, int, int]] = []
    compiled_names: dict[int, str] = {}
    report_reads: Counter[int] = Counter()
    inventory_compiled = ChunkProfileInventory.compiled
    inventory_exits: list[tuple[int, int]] = []
    plans: list[SimulationChunkPlan] = []
    lower_body = jax.stages.Traced.lower
    compile_body = jax.stages.Lowered.compile
    prepare = model_module.prepare_simulation_chunks

    def observe_lower(
        self: jax.stages.Traced, *args: Never, **kwargs: Unpack[LowerOptions]
    ) -> jax.stages.Lowered:
        lowered = lower_body(self, *args, **kwargs)
        if planning[0]:
            with lock:
                lowerings.append(threading.get_ident() == caller)
        return lowered

    # keyword-only-exempt: library-callback=_SimulationCandidateCompiler.lower
    def observe_candidate_lower(
        self: _SimulationCandidateCompiler, widths: Mapping[str, int]
    ) -> jax.stages.Lowered:
        lowered = candidate_lower(self, widths)
        if planning[0]:
            with lock:
                program_names[lowered] = self.program.name
                program_lowerings.append(
                    (self.program.name, threading.get_ident() == caller)
                )
        return lowered

    # keyword-only-exempt: library-callback=jax.stages.Lowered.compile
    def observe_compile(
        self: jax.stages.Lowered,
        compiler_options: CompilerOptions | None = None,
        **kwargs: Unpack[CompileOptions],
    ) -> jax.stages.Compiled:
        compiled = compile_body(self, compiler_options, **kwargs)
        if planning[0]:
            with lock:
                text = _program_text(hlo=compiled.as_text() or "")
                on_caller = threading.get_ident() == caller
                compiles.append((on_caller, text))
                if self in program_names:
                    program_compiles.append((program_names[self], on_caller, text))
                    compiled_names[id(compiled)] = program_names[self]
        return compiled

    def observe_analysis(self: jax.stages.Compiled) -> CompilerMemoryReport | None:
        if planning[0] and id(self) in compiled_names:
            report_reads[id(self)] += 1
        return analyze(self)

    # keyword-only-exempt: library-callback=SimulationRuntime._publish
    def observe_publish(
        self: SimulationRuntime,
        *,
        executable: jax.stages.Compiled,
        key: Hashable,
        widths: Mapping[str, int],
    ) -> None:
        publish(self, executable=executable, key=key, widths=widths)
        if planning[0] and id(executable) in compiled_names:
            entry = self.cache[key]
            assert entry.memory is not None
            assert entry.memory.records
            assert entry.executable is executable
            assert dict(entry.widths) == dict(widths)
            assert key not in publications
            publications[key] = (
                compiled_names[id(executable)],
                id(executable),
                id(entry.memory),
            )

    def observe_abstract(
        self: SimulationRuntime, **call: Unpack[RuntimePreparation]
    ) -> CompiledSimulationProgram:
        entry = prepare_abstract(self, **call)
        if planning[0] and call["program"].name.startswith(
            ("simulate_exit_", "simulate_participation_entry:")
        ):
            keys = [key for key, cached in self.cache.items() if cached is entry]
            assert len(keys) == 1
            assert dict(entry.widths) == dict(call["widths"])
            prepared_entries.append((keys[0], id(entry.executable), id(entry.memory)))
        return entry

    def observe_inventory(
        self: ChunkProfileInventory, **call: Unpack[InventoryCompiled]
    ) -> ShapeDtypePytree:
        output = inventory_compiled(self, **call)
        if planning[0] and call["name"].startswith(
            ("core:exit_", "core:participation_entry:")
        ):
            inventory_exits.append((id(call["executable"]), id(call["memory"])))
        return output

    def observe_prepare(**call: Unpack[ChunkPreparation]) -> ChunkPreparationResult:
        runtime = next(iter(call["regimes"].values())).simulation.programs.executor
        assert isinstance(runtime, SimulationRuntime)
        before = set(runtime.cache)
        planning[0] = True
        try:
            prepared = prepare(**call)
        finally:
            planning[0] = False
        for key, entry in runtime.cache.items():
            if key not in before:
                assert isinstance(entry.executable, jax.stages.Compiled)
                assert entry.memory is not None
                assert entry.memory.records
                published[key] = _program_text(hlo=entry.executable.as_text() or "")
        plans.append(prepared.plan)
        return prepared

    with monkeypatch.context() as patch:
        patch.setattr(jax.stages.Traced, "lower", observe_lower)
        patch.setattr(_SimulationCandidateCompiler, "lower", observe_candidate_lower)
        patch.setattr(jax.stages.Lowered, "compile", observe_compile)
        patch.setattr(model_module, "prepare_simulation_chunks", observe_prepare)
        patch.setattr(SimulationRuntime, "_publish", observe_publish)
        patch.setattr(SimulationRuntime, "prepare_abstract", observe_abstract)
        patch.setattr(jax.stages.Compiled, "memory_analysis", observe_analysis)
        patch.setattr(ChunkProfileInventory, "compiled", observe_inventory)
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
            max_compilation_workers=max_compilation_workers,
        )
    (plan,) = plans
    profile = plan.profile
    contract = (
        profile.n_subjects,
        profile.padded_population,
        tuple(
            (stage.name, stage.peak_bytes, stage.reservation_bytes)
            for stage in profile.stages
        ),
        dict(profile.fixed_reservation),
        dict(profile.output_reservation),
        dict(profile.setup_reservation),
        dict(plan.required_bytes),
    )
    return {
        "lowerings": lowerings,
        "compiles": compiles,
        "published": published,
        "contract": contract,
        "panel": result.to_dataframe(),
        "program_lowerings": program_lowerings,
        "program_compiles": program_compiles,
        "publications": publications,
        "prepared": prepared_entries,
        "report_reads": dict(report_reads),
        "inventory_exits": inventory_exits,
        "raw_tree": str(jax.tree.structure(result.raw_results)),
        "raw_arrays": tuple(
            (tuple(array.shape), array.dtype.str, array.tobytes())
            for array in map(np.asarray, jax.tree.leaves(result.raw_results))
        ),
    }


@pytest.fixture(scope="module")
def observed() -> dict[int, _Observation]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        return {
            workers: _simulate_and_observe(
                monkeypatch=monkeypatch, max_compilation_workers=workers
            )
            for workers in _WORKERS
        }


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_lowers_every_program_on_the_calling_thread(
    *, observed: dict[int, _Observation], workers: int
) -> None:
    lowerings = observed[workers]["lowerings"]
    assert (len(lowerings) > 0, all(lowerings)) == (True, True)


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_compiles_every_forward_program_off_the_calling_thread(
    *, observed: dict[int, _Observation], workers: int
) -> None:
    pooled = {
        text for on_caller, text in observed[workers]["compiles"] if not on_caller
    }
    assert set(observed[workers]["published"].values()) - pooled == set()


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_publishes_one_executable_per_forward_program(
    *, observed: dict[int, _Observation], workers: int
) -> None:
    """The small model's four forward programs, each at three subjects."""
    step = ("age", "koopmans_aggregator__discount_factor", "period", "saving", "wealth")
    keys = sorted(map(_readable_key, observed[workers]["published"]), key=repr)
    assert keys == [
        # The alive decision, over its two-point action product.
        (
            (
                "age",
                "koopmans_aggregator__discount_factor",
                "next_regime_to_V_arr",
                "period",
                "saving",
                "wealth",
            ),
            3,
            (("action_product", 2), ("subject", 3)),
        ),
        # The alive transition and route, which read the same operands.
        (step, 3, (("subject", 3),)),
        (step, 3, (("subject", 3),)),
        # The terminal done decision.
        (("age", "next_regime_to_V_arr", "period", "wealth"), 3, (("subject", 3),)),
    ]


def test_chunk_planning_compiles_the_same_forward_programs_for_every_worker_count(
    observed: dict[int, _Observation],
) -> None:
    def programs(workers: int) -> list[str]:
        return sorted(observed[workers]["published"].values())

    assert (all("ENTRY" in text for text in programs(1)), programs(2)) == (
        True,
        programs(1),
    )


def test_chunk_planning_admits_the_same_chunk_contract_for_every_worker_count(
    observed: dict[int, _Observation],
) -> None:
    assert observed[2]["contract"] == observed[1]["contract"]


def test_chunk_planning_simulates_the_same_panel_for_every_worker_count(
    observed: dict[int, _Observation],
) -> None:
    pd.testing.assert_frame_equal(
        observed[2]["panel"], observed[1]["panel"], check_exact=True
    )


@pytest.fixture(scope="module")
def exit_observed() -> dict[int, _Observation]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        return {
            workers: _simulate_and_observe(
                monkeypatch=monkeypatch,
                max_compilation_workers=workers,
                participation=True,
            )
            for workers in _WORKERS
        }


@pytest.mark.slow
@pytest.mark.parametrize("workers", _WORKERS)
def test_exit_programs_compile_off_the_calling_thread(
    *, exit_observed: dict[int, _Observation], workers: int
) -> None:
    """Full, incoming and revealed decisions compile without blocking lowering."""
    observation = exit_observed[workers]
    expected = {
        "simulate_exit_full",
        "simulate_participation_entry",
        "simulate_exit_participating",
    }
    exits = [
        (name, on_caller, text)
        for name, on_caller, text in observation["program_compiles"]
        if name.split(":", 1)[0] in expected
    ]
    assert {name.split(":", 1)[0] for name, _, _ in exits} == expected
    assert all("ENTRY" in text for _, _, text in exits)
    assert all(on_caller for name, on_caller in observation["program_lowerings"])
    assert all(not on_caller for _, on_caller, _ in exits), [
        name for name, on_caller, _ in exits if on_caller
    ]
    assert {text for _, _, text in exits} <= set(observation["published"].values())


@pytest.mark.slow
@pytest.mark.parametrize("workers", _WORKERS)
def test_exit_profiles_reuse_the_exact_published_executable_and_report(
    *, exit_observed: dict[int, _Observation], workers: int
) -> None:
    """Reuse the exact published executable and report without rereading counters."""
    observed = exit_observed[workers]
    assert observed["prepared"]
    families = set()
    for key, executable, memory in observed["prepared"]:
        name, published_executable, published_memory = observed["publications"][key]
        assert (executable, memory) == (published_executable, published_memory)
        assert observed["report_reads"][executable] == 1
        families.add(name.split(":", 1)[0])
    assert Counter(observed["inventory_exits"]) == Counter(
        (executable, memory) for _, executable, memory in observed["prepared"]
    )
    assert families == {
        "simulate_exit_full",
        "simulate_exit_participating",
        "simulate_participation_entry",
    }


@pytest.mark.slow
def test_exit_profiles_keep_the_same_inventory_for_every_worker_count(
    exit_observed: dict[int, _Observation],
) -> None:
    assert exit_observed[2]["contract"] == exit_observed[1]["contract"]


@pytest.mark.slow
def test_exit_profiles_keep_every_raw_array_for_every_worker_count(
    exit_observed: dict[int, _Observation],
) -> None:
    expected, actual = exit_observed[1], exit_observed[2]
    assert expected["raw_arrays"]
    assert actual["raw_tree"] == expected["raw_tree"]
    # Includes every mask, role, state, shock, action and stakeholder value,
    # preserving dtype, signed zeros and NaN payloads as well as numeric values.
    assert actual["raw_arrays"] == expected["raw_arrays"]
    pd.testing.assert_frame_equal(actual["panel"], expected["panel"], check_exact=True)


class _ExitCompileError(RuntimeError):
    """A real exit program's compile boundary refused its lowered body."""


@pytest.mark.slow
@pytest.mark.parametrize("failure", ["compile", "report"])
def test_exit_frontiers_refuse_compile_and_report_failures(  # noqa: C901, PLR0915 - scope fault custody and observations together
    *, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failed original or revealed program is never adopted or dispatched."""
    caller = threading.get_ident()
    target = (
        "simulate_exit_full" if failure == "compile" else "simulate_exit_participating"
    )
    lowered_names: WeakKeyDictionary[jax.stages.Lowered, str] = WeakKeyDictionary()
    refused: set[int] = set()
    attempts: list[bool] = []
    adopted: list[str] = []
    dispatched: list[int] = []
    cached: list[int] = []
    lower = _SimulationCandidateCompiler.lower
    compile_body = jax.stages.Lowered.compile
    analysis = jax.stages.Compiled.memory_analysis
    inventory_compiled = ChunkProfileInventory.compiled
    publish = SimulationRuntime._publish
    execute = jax.stages.Compiled.__call__

    # keyword-only-exempt: library-callback=_SimulationCandidateCompiler.lower
    def record_lower(
        self: _SimulationCandidateCompiler, widths: Mapping[str, int]
    ) -> jax.stages.Lowered:
        lowered = lower(self, widths)
        lowered_names[lowered] = self.program.name
        return lowered

    # keyword-only-exempt: library-callback=jax.stages.Lowered.compile
    def refuse_compile(
        self: jax.stages.Lowered,
        compiler_options: CompilerOptions | None = None,
        **kwargs: Unpack[CompileOptions],
    ) -> jax.stages.Compiled:
        if lowered_names.get(self) == target and failure == "compile":
            attempts.append(threading.get_ident() != caller)
            raise _ExitCompileError("exit frontier compile failed")
        compiled = compile_body(self, compiler_options, **kwargs)
        if lowered_names.get(self) == target and failure == "report":
            refused.add(id(compiled))
        return compiled

    def refuse_report(self: jax.stages.Compiled) -> CompilerMemoryReport | None:
        if id(self) in refused:
            attempts.append(True)
            return None
        return analysis(self)

    def record_inventory(
        self: ChunkProfileInventory, **call: Unpack[InventoryCompiled]
    ) -> ShapeDtypePytree:
        output = inventory_compiled(self, **call)
        adopted.append(call["name"])
        return output

    # keyword-only-exempt: library-callback=SimulationRuntime._publish
    def record_cache(
        self: SimulationRuntime,
        *,
        executable: jax.stages.Compiled,
        key: Hashable,
        widths: Mapping[str, int],
    ) -> None:
        try:
            publish(self, executable=executable, key=key, widths=widths)
        finally:
            cached.extend(id(entry.executable) for entry in self.cache.values())

    def record_dispatch(
        self: jax.stages.Compiled, *args: PytreeValue, **kwargs: PytreeValue
    ) -> PytreeValue:
        if id(self) in refused:
            dispatched.append(id(self))
        return execute(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_SimulationCandidateCompiler, "lower", record_lower)
        patch.setattr(jax.stages.Lowered, "compile", refuse_compile)
        patch.setattr(jax.stages.Compiled, "memory_analysis", refuse_report)
        patch.setattr(jax.stages.Compiled, "__call__", record_dispatch)
        patch.setattr(ChunkProfileInventory, "compiled", record_inventory)
        patch.setattr(SimulationRuntime, "_publish", record_cache)
        if failure == "compile":
            with pytest.raises(
                _ExitCompileError, match="exit frontier compile failed"
            ) as raised:
                _simulate_and_observe(
                    monkeypatch=patch, max_compilation_workers=1, participation=True
                )
            assert any(
                "while compiling" in note and "exit_full" in note
                for note in raised.value.__notes__
            )
        else:
            with pytest.raises(
                ExecutionPlanningError, match="not an allocation record"
            ):
                _simulate_and_observe(
                    monkeypatch=patch, max_compilation_workers=1, participation=True
                )
            assert refused
            assert not refused.intersection(cached)
    assert attempts
    assert all(attempts)
    assert adopted
    assert f"core:{target.removeprefix('simulate_')}" not in adopted
    assert not dispatched
