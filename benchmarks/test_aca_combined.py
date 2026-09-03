"""Tests for the combined ACA benchmark measurement."""

# ruff: noqa: SLF001

from collections.abc import Iterator

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
    monkeypatch.setattr(_gpu_mem, "_get_gpu_peak_bytes", lambda: 456_000)

    result = _gpu_mem._collect_combined_measurements(benchmark)

    assert benchmark.setup_calls == 1
    assert benchmark.execution_calls == 2
    assert result == {
        "compilation_time": 3.0,
        "execution_time": 2.0,
        "peak_cpu_mem": 123_000,
        "peak_gpu_mem": 456_000,
    }


def test_aca_asv_surface_contains_only_shared_result_trackers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASV sees four cheap readers backed by one setup-cache measurement."""
    measured = {
        "compilation_time": 3.0,
        "execution_time": 2.0,
        "peak_cpu_mem": 123_000,
        "peak_gpu_mem": 456_000,
    }
    calls: list[str] = []

    def _measure(*, bench_module: str, bench_class: str) -> dict[str, float]:
        assert bench_module == "benchmarks.asv.bench_aca_baseline"
        calls.append(bench_class)
        return measured

    monkeypatch.setattr(_gpu_mem, "measure_combined", _measure)

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
        assert instance.track_peak_gpu_mem() == 456_000

        metric_names = {
            name
            for name in dir(cls)
            if name.startswith(("time_", "peakmem_", "track_"))
        }
        assert metric_names == {
            "track_compilation_time",
            "track_execution_time",
            "track_peak_cpu_mem",
            "track_peak_gpu_mem",
        }

    assert calls == ["AcaBaseline", "AcaBaselineDebugLog"]
