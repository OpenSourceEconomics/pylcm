"""Tests for the combined ACA benchmark measurement."""

# ruff: noqa: SLF001

from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmarks.asv import _gpu_mem, bench_aca_baseline


class _FakeAcaBenchmark:
    def __init__(self) -> None:
        self.setup_calls = 0
        self.execution_calls = 0

    def setup_for_gpu_measurement(self) -> None:
        self.setup_calls += 1

    def execute_for_measurement(self) -> None:
        self.execution_calls += 1


def test_combined_measurement_uses_one_cold_and_one_warm_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One process collects all metrics without repeating the cold compile."""
    benchmark = _FakeAcaBenchmark()
    clock: Iterator[float] = iter((10.0, 13.0, 20.0, 22.0))
    monkeypatch.setattr(_gpu_mem.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(_gpu_mem, "_get_cpu_peak_bytes", lambda: 123_000)

    result = _gpu_mem._collect_combined_measurements(benchmark)

    assert benchmark.setup_calls == 1
    assert benchmark.execution_calls == 2
    assert result == {
        "compilation_time": 3.0,
        "execution_time": 2.0,
        "peak_cpu_mem": 123_000,
    }


def test_aca_asv_surface_contains_only_shared_result_trackers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASV sees timing/CPU plus three independently measured GPU phases."""
    measured = {
        "compilation_time": 3.0,
        "execution_time": 2.0,
        "peak_cpu_mem": 123_000,
    }
    profile = {
        _gpu_mem.AUTOMATIC_SOLVE_SIMULATE: 101_000,
        _gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE: 202_000,
        _gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE: 303_000,
    }
    combined_calls: list[str] = []
    profile_calls: list[str] = []

    def _measure(*, bench_module: str, bench_class: str) -> dict[str, float]:
        assert bench_module == "benchmarks.asv.bench_aca_baseline"
        combined_calls.append(bench_class)
        return measured

    def _measure_profile(*, bench_module: str, bench_class: str) -> dict[str, int]:
        assert bench_module == "benchmarks.asv.bench_aca_baseline"
        profile_calls.append(bench_class)
        return profile

    monkeypatch.setattr(_gpu_mem, "measure_combined", _measure)
    monkeypatch.setattr(_gpu_mem, "measure_gpu_memory_profile", _measure_profile)

    for cls in (
        bench_aca_baseline.AcaBaseline,
        bench_aca_baseline.AcaBaselineDebugLog,
    ):
        instance = cls()
        cache = instance.setup_cache()
        instance.setup(cache)
        assert instance.track_compilation_time() == 3.0
        assert instance.track_execution_time() == 2.0
        assert instance.track_peak_cpu_mem() == 123_000
        assert instance.track_peak_gpu_mem_automatic_solve_simulate() == 101_000
        assert instance.track_peak_gpu_mem_solve_save_all_persistable() == 202_000
        assert instance.track_peak_gpu_mem_load_supplied_solution_simulate() == 303_000

        metric_names = {
            name
            for name in dir(cls)
            if name.startswith(("time_", "peakmem_", "track_"))
        }
        assert metric_names == {
            "track_compilation_time",
            "track_execution_time",
            "track_peak_cpu_mem",
            "track_peak_gpu_mem_automatic_solve_simulate",
            "track_peak_gpu_mem_solve_save_all_persistable",
            "track_peak_gpu_mem_load_supplied_solution_simulate",
        }

    expected_calls = ["AcaBaseline", "AcaBaselineDebugLog"]
    assert combined_calls == expected_calls
    assert profile_calls == expected_calls


class _FakeSolution:
    def __init__(self) -> None:
        self.saved_paths: list[Path] = []

    def save(self, *, path: Path) -> None:
        self.saved_paths.append(path)


class _FakeModel:
    def __init__(self, solution: _FakeSolution) -> None:
        self.solution = solution
        self.solve_calls: list[dict[str, object]] = []
        self.simulate_calls: list[dict[str, object]] = []

    def solve(self, **kwargs: object) -> _FakeSolution:
        self.solve_calls.append(kwargs)
        return self.solution

    def simulate(self, **kwargs: object) -> object:
        self.simulate_calls.append(kwargs)
        return object()


def test_aca_gpu_memory_phases_dispatch_exact_workloads(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The three child phases exercise automatic, persistence, and replay paths."""
    saved = _FakeSolution()
    loaded = _FakeSolution()
    model = _FakeModel(saved)
    benchmark = bench_aca_baseline.AcaBaseline()
    benchmark.model = model
    benchmark.model_params = {"p": 1}
    benchmark.initial_conditions = {"state": object()}
    archive = tmp_path / "solution.h5"
    load_calls: list[Path] = []

    def _load_solution(*, path: Path) -> _FakeSolution:
        load_calls.append(path)
        return loaded

    monkeypatch.setattr("lcm.persistence.load_solution", _load_solution)

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.AUTOMATIC_SOLVE_SIMULATE,
        archive_path=archive,
    )
    assert model.simulate_calls[-1].get("solution") is None

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.SOLVE_SAVE_ALL_PERSISTABLE,
        archive_path=archive,
    )
    from lcm.solver_api import ResultRetention

    assert (
        model.solve_calls[-1]["retention"] is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
    )
    assert saved.saved_paths == [archive]

    benchmark.execute_gpu_memory_phase(
        phase=_gpu_mem.LOAD_SUPPLIED_SOLUTION_SIMULATE,
        archive_path=archive,
    )
    assert load_calls == [archive]
    assert model.simulate_calls[-1]["solution"] is loaded

    with pytest.raises(ValueError, match="Unknown GPU memory profile phase"):
        benchmark.execute_gpu_memory_phase(
            phase="not-a-phase",
            archive_path=archive,
        )
