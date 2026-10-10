"""Tests for the combined ACA benchmark measurement."""

# ruff: noqa: SLF001

import sys
from collections.abc import Iterator
from typing import Never

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


def test_aca_timing_asv_surface_contains_only_the_combined_trackers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ACA timing classes run only the combined timing subprocess."""
    measured = {
        "compilation_time": 3.0,
        "execution_time": 2.0,
        "peak_cpu_mem": 123_000,
    }
    combined_calls: list[str] = []

    def _measure(*, bench_module: str, bench_class: str) -> dict[str, float]:
        assert bench_module == "benchmarks.asv.bench_aca_baseline"
        combined_calls.append(bench_class)
        return measured

    def _refuse_profile[T](**_: T) -> dict[str, int]:
        pytest.fail("selecting the timing class must not run the GPU-memory profile")

    monkeypatch.setattr(_gpu_mem, "measure_combined", _measure)
    monkeypatch.setattr(_gpu_mem, "measure_gpu_memory_profile", _refuse_profile)

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

        metric_names = {
            name
            for name in dir(cls)
            if name.startswith(("time_", "peakmem_", "track_"))
        }
        assert metric_names == {
            "track_compilation_time",
            "track_execution_time",
            "track_peak_cpu_mem",
        }

    assert combined_calls == ["AcaBaseline", "AcaBaselineDebugLog"]


def test_aca_module_defines_no_gpu_memory_profile() -> None:
    """The ACA benchmarks report CPU peak memory only; no GPU-memory profile exists."""
    profiles = [
        name
        for name, value in vars(bench_aca_baseline).items()
        if isinstance(value, type) and issubclass(value, _gpu_mem.GpuPeakMemProfile)
    ]
    assert profiles == []


def test_aca_asv_version_identifies_fixed_forward_simulation_seed() -> None:
    """ACA timing observations identify the deterministic series."""
    assert bench_aca_baseline.AcaBaseline.version == "2"
    assert bench_aca_baseline.AcaBaselineDebugLog.version == "2"


class _BuildStoppedError(Exception):
    pass


def test_aca_build_uses_the_default_execution_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ACA benchmark passes no execution policy, so the model factory's own
    default, which production runs under, derives the device-memory budget."""
    import types

    import lcm
    from _lcm.execution import execution_plan

    captured: dict[str, str] = {}

    def _create_benchmark_model(**kwargs: str) -> Never:
        captured.update(kwargs)
        raise _BuildStoppedError

    preferences = types.ModuleType("aca_model.agent.preferences")
    preferences.BenchmarkPrefType = object  # ty: ignore[unresolved-attribute]
    benchmark_module = types.ModuleType("aca_model.benchmark")
    benchmark_module.create_benchmark_model = _create_benchmark_model  # ty: ignore[unresolved-attribute]
    benchmark_module.get_benchmark_initial_conditions = None  # ty: ignore[unresolved-attribute]
    benchmark_module.get_benchmark_params = None  # ty: ignore[unresolved-attribute]
    for name, module in (
        ("aca_model", types.ModuleType("aca_model")),
        ("aca_model.agent", types.ModuleType("aca_model.agent")),
        ("aca_model.agent.preferences", preferences),
        ("aca_model.benchmark", benchmark_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(lcm, "DiscreteGrid", lambda **_: "pref_type_grid")
    # A device reporting a pool limit, as a GPU does, so a budget could be derived.
    monkeypatch.setattr(
        execution_plan, "visible_device_pool_limits", lambda: {0: 16_000_000_000}
    )

    with pytest.raises(_BuildStoppedError):
        bench_aca_baseline._build()

    assert captured == {"pref_type_grid": "pref_type_grid"}
