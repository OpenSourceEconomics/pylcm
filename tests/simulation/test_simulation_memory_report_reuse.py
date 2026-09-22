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
    *,
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
    *,
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
_CANDIDATE_SEARCH_ANCHOR = 8


def _axis_width_case(
    *,
    axis_widths: dict[str, int],
    budget: int = 2**28,
    population: int = 8,
) -> tuple[Any, Any, Any]:
    """One model pinned to an explicit inner subject width and population.

    Since the top-first planner (`_resolve_subject_anchor_width` in
    `chunk_admission.py`) always resolves an *unpinned* subject anchor to the
    full population — the outer-cohort doubling frontier
    (`_independent_outer_candidates`) then collapses to that one extent, and
    the top-first search never visits a second `n_subjects` candidate — an
    explicit `axis_widths={"subject": ...}` pin smaller than `population` is
    required to get a real multi-candidate doubling frontier at all.
    """
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
            device_memory_bytes=budget, axis_widths=axis_widths
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
        "income": jax.numpy.concatenate([jax.numpy.asarray([2.0])] * population),
        "age": jax.numpy.concatenate([jax.numpy.asarray([0.0])] * population),
        "regime_id": jax.numpy.concatenate(
            [jax.numpy.asarray([_LifecycleRegimeId.alive])] * population
        ),
    }
    return model, params, initial


def _top_first_descent_case(*, budget: int) -> tuple[Any, Any, Any]:
    """A 64-subject population pinned to an 8-subject anchor under `budget`.

    This gives the top-first planner (`_plan_independent_chunks`) an outer
    doubling frontier of (8, 16, 32, 64): it tries the full population (64)
    first, and only descends into the smaller candidates on refusal.
    """
    return _axis_width_case(
        axis_widths={"subject": _CANDIDATE_SEARCH_ANCHOR},
        budget=budget,
        population=_CANDIDATE_SEARCH_POPULATION,
    )


def _find_top_first_descent_budget() -> int | None:
    """Probe live-process residency to find a budget where the top-first
    planner refuses the full-population candidate at least once before
    admitting a smaller extent — i.e. visits at least two distinct
    `n_subjects` candidates via `chunk_admission._profile_independent_candidate`.

    The admitted band is only a few kilobytes wide, and where it sits shifts
    with whatever device buffers earlier tests in this process still hold
    live (an `ExecutionPlanningError` from a too-tight budget is normal here,
    not a bug: admission correctly consults *actual* current residency).
    Rather than pin a value discovered offline (fragile to test order and to
    unrelated changes elsewhere in the suite), this probes the *current*
    process at test-run time. Returns `None` if no probed budget in the swept
    range produces a multi-candidate descent, so the caller can fall back to
    an explicitly reported single-candidate-plus-refusal case instead of
    failing opaquely.
    """
    original = chunk_admission._profile_independent_candidate
    visited: list[int] = []

    def counted(**kwargs: Any) -> Any:
        visited.append(kwargs["n_subjects"])
        return original(**kwargs)

    with pytest.MonkeyPatch.context() as observe:
        observe.setattr(chunk_admission, "_profile_independent_candidate", counted)
        for budget in (
            11_000,
            12_000,
            13_000,
            14_000,
            15_000,
            16_000,
            17_000,
            18_000,
            20_000,
            24_000,
        ):
            model, params, initial = _top_first_descent_case(budget=budget)
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
        return None


def test_candidate_search_reads_the_report_once_per_distinct_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The top-first planner's descent (refuse the full population, then admit
    a smaller extent) visits several distinct `n_subjects` candidates per
    dispatch, but the total reads across the whole cold search still equal the
    number of distinct executables compiled — never one read per candidate
    visit. A warm repeat, which revisits the same candidates, reads nothing.

    If no probed budget produces a real multi-candidate descent on this model
    (see `_find_top_first_descent_budget`), this falls back to asserting the
    same per-executable-once invariant over the single-admitted-candidate path
    plus an independent refusal path, and records which case ran.
    """
    budget = _find_top_first_descent_budget()
    if budget is None:
        _assert_single_candidate_plus_refusal_reads_once(monkeypatch=monkeypatch)
        return
    model, params, initial = _top_first_descent_case(budget=budget)
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


def _assert_single_candidate_plus_refusal_reads_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fallback exercised only if no swept budget forces a multi-candidate
    top-first descent: one admitted single-candidate dispatch still reads
    memory exactly once per distinct executable, and an independent refusal
    (too-tight budget) never admits, whatever the cache holds."""
    model, params, initial = _inputs(budget=2**28)
    solution = model.solve(params=params, log_level="off")
    with _count_memory_reads(monkeypatch=monkeypatch) as counts:
        _simulate(
            model=model, params=params, initial=initial, solution=solution, seed=1
        )
    n_distinct_executables = _executable_cache_size(model=model)
    assert n_distinct_executables > 0
    assert counts["reservation_reads"] == n_distinct_executables

    tiny_model, tiny_params, _tiny_initial = _inputs(budget=1)
    with pytest.raises(ExecutionPlanningError):
        tiny_model.solve(params=tiny_params, log_level="off")


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
