"""Compiler memory reports are read once per executable, not once per dispatch.

`SimulationRuntime._prepare_materialized` calls `plan_workspace` on every budgeted
dispatch, which previously reread `compiler_memory_reservation` from
`_simulation_memory` on each call even though the underlying executable, and its
compiler report, had not changed. `CompiledSimulationProgram` now retains that
report at compilation, exactly as `_ProfiledOperation` already does for host
operations (see `host_operations._operation_memory`).
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jax
import pytest

from _lcm.execution import workspace_planning
from _lcm.simulation import chunk_admission, host_operations, runtime
from lcm import AgeGrid, ExecutionConfig, LinSpacedGrid, Model, NormalIIDProcess, Regime
from lcm.exceptions import ExecutionPlanningError
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.simulation.test_budget_lifecycle import _LifecycleRegimeId
from tests.simulation.test_normal_process_grid_admission import _inputs
from tests.simulation.test_process_grid_entry_admission import (
    _initial_age,
    _next_regime,
    _terminal_utility,
    _utility,
)


@contextmanager
def _count_memory_reads(*, monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """Count every real `compiler_memory_reservation` call without changing it.

    Patches every module-level name this function is imported under —
    `workspace_planning` (its defining module), `runtime` (per-dispatch
    `_simulation_memory`), and `host_operations` (pure host operations) — so
    the count is complete regardless of which cache layer a read comes from.
    Chunk-profile construction (`chunk_planning.SimulationStageProfile`,
    `chunk_profile_inventory.ChunkProfileInventory`) imports no such name at
    all after this change: it only ever adopts an already-computed record.
    """
    counts: Counter[str] = Counter()
    original = workspace_planning.compiler_memory_reservation

    def counted(**kwargs: Any) -> Any:
        counts["reservation_reads"] += 1
        return original(**kwargs)

    with monkeypatch.context() as observe:
        observe.setattr(workspace_planning, "compiler_memory_reservation", counted)
        observe.setattr(runtime, "compiler_memory_reservation", counted)
        observe.setattr(host_operations, "compiler_memory_reservation", counted)
        yield counts


@pytest.fixture
def budgeted_case() -> tuple[Any, Any, Any, Any]:
    """A small budgeted model whose forward simulation is actually compiled."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    return model, params, initial, solution


def _executable_cache_size(*, model: Any) -> int:
    """Read the number of distinct compiled executables every cache retains.

    A dispatch reads the compiler report from two independent caches: the
    core/forward-program cache on `SimulationRuntime` (`executor.cache`,
    `CompiledSimulationProgram.memory`) and the pure-host-operation cache on
    `ProfiledSimulationOperations` (`executor.operations.cache`,
    `_ProfiledOperation.memory`). Both are exact-executable-identity caches;
    summing their sizes gives the total distinct-executable count a complete
    cold dispatch reads memory for exactly once.
    """
    (regimes,) = model._simulate_runtime_regimes.values()
    regime = next(iter(regimes.values()))
    executor = regime.simulation.programs.executor
    return len(executor.cache) + len(executor.operations.cache)


def _simulate(
    *, model: Any, params: Any, initial: Any, solution: Any, seed: int
) -> Any:
    return model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        log_level="off",
        seed=seed,
    )


def test_warm_repeated_simulate_reads_the_report_per_executable(
    budgeted_case: tuple[Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second warm call reuses every executable, so no new report is read."""
    model, params, initial, solution = budgeted_case
    _simulate(model=model, params=params, initial=initial, solution=solution, seed=3)
    with _count_memory_reads(monkeypatch=monkeypatch) as first_counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=3
        )
    n_distinct_executables = _executable_cache_size(model=model)
    assert n_distinct_executables > 0
    with _count_memory_reads(monkeypatch=monkeypatch) as second_counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=5
        )
    assert first_counts["reservation_reads"] == 0
    assert second_counts["reservation_reads"] == 0


def test_first_dispatch_reads_the_report_once_per_distinct_executable(
    budgeted_case: tuple[Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold call compiles fresh code, so exactly one read per new executable."""
    model, params, initial, solution = budgeted_case
    with _count_memory_reads(monkeypatch=monkeypatch) as counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=7
        )
    n_distinct_executables = _executable_cache_size(model=model)
    assert n_distinct_executables > 0
    assert counts["reservation_reads"] == n_distinct_executables


def test_changed_shape_produces_a_fresh_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulating a different population size compiles new code and reads it fresh."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    small = _simulate(
        model=model, params=params, initial=initial, solution=solution, seed=1
    )
    del small
    wide_initial = {
        name: jax.numpy.concatenate([value, value]) for name, value in initial.items()
    }
    with _count_memory_reads(monkeypatch=monkeypatch) as counts:
        _simulate(
            model=model, params=params, initial=wide_initial, solution=solution, seed=1
        )
    assert counts["reservation_reads"] > 0


def test_malformed_memory_lookup_fails_loud() -> None:
    """A `None` cached record (uncompiled candidate) raises rather than reusing 0."""
    program = runtime.CompiledSimulationProgram(
        executable=lambda **kwargs: kwargs, static_kwargs={}, memory=None
    )
    with pytest.raises(ExecutionPlanningError, match="compiled executable"):
        runtime._simulation_memory(program)


def test_cached_memory_is_returned_without_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_simulation_memory` reads the retained record; it never recomputes it."""
    memory = synthetic_memory(2**20)
    program = runtime.CompiledSimulationProgram(
        executable=lambda **kwargs: kwargs, static_kwargs={}, memory=memory
    )

    def forbidden(**_kwargs: Any) -> Any:
        raise AssertionError("_simulation_memory recomputed a cached report.")

    monkeypatch.setattr(runtime, "compiler_memory_reservation", forbidden)
    assert runtime._simulation_memory(program) is memory


def test_dce_and_duplicate_alias_inputs_still_produce_one_report_per_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operand DCE removes, and a duplicated alias operand, are still one read."""
    model, params, initial = _inputs(budget=2**28, fixed=("mu", "sigma", "n_std"))
    solution = model.solve(params=params, log_level="off")
    aliased_initial = dict(initial)
    aliased_initial["income"] = initial["income"]  # same array object, duplicate leaf
    with _count_memory_reads(monkeypatch=monkeypatch) as counts:
        _simulate(
            model=model,
            params=params,
            initial=aliased_initial,
            solution=solution,
            seed=13,
        )
    n_distinct_executables = _executable_cache_size(model=model)
    assert counts["reservation_reads"] == n_distinct_executables


_CANDIDATE_SEARCH_POPULATION = 64


def _tight_budget_candidate_search_case(*, budget: int) -> tuple[Any, Any, Any]:
    """A 64-subject population under the given budget."""
    model, params, initial = _inputs(budget=budget)
    initial = {
        name: jax.numpy.concatenate([value] * _CANDIDATE_SEARCH_POPULATION)
        for name, value in initial.items()
    }
    return model, params, initial


def _find_multi_candidate_budget() -> int:
    """Probe live-process residency to find a budget that admits after visiting
    at least two distinct outer-chunk width candidates.

    The admitted band for this population is only a few kilobytes wide, and
    where it sits shifts with whatever device buffers earlier tests in this
    process still hold live. Rather than pin a value discovered offline
    (fragile to test order and to unrelated changes elsewhere in the suite),
    this probes the *current* process at test-run time and returns the
    smallest budget in a coarse geometric sweep that both admits and visited
    more than one width candidate before doing so.
    """
    original = chunk_admission.profile_simulation_chunk
    visited: list[int] = []

    def counted(**kwargs: Any) -> Any:
        visited.append(kwargs["n_subjects"])
        return original(**kwargs)

    chunk_admission.profile_simulation_chunk = counted
    try:
        for budget in (
            8_000,
            10_000,
            12_000,
            14_000,
            16_000,
            20_000,
            24_000,
            32_000,
            48_000,
        ):
            model, params, initial = _tight_budget_candidate_search_case(budget=budget)
            visited.clear()
            try:
                solution = model.solve(params=params, log_level="off")
                _simulate(
                    model=model,
                    params=params,
                    initial=initial,
                    solution=solution,
                    seed=1,
                )
            except ExecutionPlanningError:
                continue
            if len(set(visited)) > 1:
                return budget
        raise AssertionError(
            "No probed budget forced the outer chunk search to visit more than "
            "one distinct width candidate; widen the probed budget range."
        )
    finally:
        chunk_admission.profile_simulation_chunk = original


def _axis_width_case(*, axis_widths: dict[str, int]) -> tuple[Any, Any, Any]:
    """One model pinned to an explicit inner subject width, independent budget."""
    parameters = {"mu": 0.1415, "sigma": 1.876, "n_std": 3.2}
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=_initial_age,
                states={"income": NormalIIDProcess(n_points=5, gauss_hermite=False)},
                actions={"saving": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _utility},
            ),
            "done": Regime(transition=None, functions={"utility": _terminal_utility}),
        },
        regime_id_class=_LifecycleRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        execution_config=ExecutionConfig(
            device_memory_bytes=2**28, axis_widths=axis_widths
        ),
    )
    params = {
        "alive": {
            "income": parameters,
            "koopmans_aggregator": {"discount_factor": 0.9},
        },
        "done": {},
    }
    initial = {
        "income": jax.numpy.concatenate([jax.numpy.asarray([2.0])] * 8),
        "age": jax.numpy.concatenate([jax.numpy.asarray([0.0])] * 8),
        "regime_id": jax.numpy.concatenate(
            [jax.numpy.asarray([_LifecycleRegimeId.alive])] * 8
        ),
    }
    return model, params, initial


def test_candidate_search_reads_the_report_once_per_distinct_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk-profile construction (`ChunkProfileInventory`/
    `SimulationStageProfile`) visits several width candidates per dispatch, but
    the total reads across the whole cold candidate search still equal the
    number of distinct executables it compiled — never one read per candidate
    visit. A warm repeat, which revisits the same candidates, reads nothing."""
    budget = _find_multi_candidate_budget()
    model, params, initial = _tight_budget_candidate_search_case(budget=budget)
    solution = model.solve(params=params, log_level="off")
    with _count_memory_reads(monkeypatch=monkeypatch) as cold_counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=1
        )
    n_distinct_executables = _executable_cache_size(model=model)
    assert n_distinct_executables > 0
    assert cold_counts["reservation_reads"] == n_distinct_executables
    with _count_memory_reads(monkeypatch=monkeypatch) as warm_counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=1
        )
    assert warm_counts["reservation_reads"] == 0


def test_changed_width_map_produces_a_fresh_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two models pinned to different inner subject widths each compile and
    read their own distinct executables; neither one's cached report is
    reused for the other's width map."""
    narrow_model, narrow_params, narrow_initial = _axis_width_case(
        axis_widths={"subject": 4}
    )
    narrow_solution = narrow_model.solve(params=narrow_params, log_level="off")
    with _count_memory_reads(monkeypatch=monkeypatch) as narrow_counts:
        _simulate(
            model=narrow_model,
            params=narrow_params,
            initial=narrow_initial,
            solution=narrow_solution,
            seed=1,
        )
    assert narrow_counts["reservation_reads"] > 0

    wide_model, wide_params, wide_initial = _axis_width_case(axis_widths={"subject": 8})
    wide_solution = wide_model.solve(params=wide_params, log_level="off")
    with _count_memory_reads(monkeypatch=monkeypatch) as wide_counts:
        _simulate(
            model=wide_model,
            params=wide_params,
            initial=wide_initial,
            solution=wide_solution,
            seed=1,
        )
    assert wide_counts["reservation_reads"] > 0


def test_fresh_larger_retained_owner_still_causes_refusal(
    budgeted_case: tuple[Any, Any, Any, Any],
) -> None:
    """Admission still consults live residency; a cached memory report never
    masks it."""
    model, params, initial, solution = budgeted_case
    tiny_model, tiny_params, _tiny_initial = _inputs(budget=1)
    with pytest.raises(ExecutionPlanningError):
        tiny_model.solve(params=tiny_params, log_level="off")
    # Cached memory on a completely different runtime cannot leak across models.
    result = _simulate(
        model=model, params=params, initial=initial, solution=solution, seed=17
    )
    assert result is not None
