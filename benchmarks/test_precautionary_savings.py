"""Tests for the shared-measurement precautionary-savings benchmarks."""

import pytest

from benchmarks.asv import _gpu_mem, bench_precautionary_savings

_CLASSES = [
    bench_precautionary_savings.PrecautionarySavingsSolve,
    bench_precautionary_savings.PrecautionarySavingsSimulate,
    bench_precautionary_savings.PrecautionarySavingsSimulateWithSolve,
    bench_precautionary_savings.PrecautionarySavingsSimulateWithSolveIrreg,
]


@pytest.mark.parametrize("benchmark_class", _CLASSES)
def test_each_class_reads_its_shared_cache_for_all_three_trackers(
    *, monkeypatch: pytest.MonkeyPatch, benchmark_class: type
) -> None:
    """setup_cache is the sole producer; the three trackers only read it."""
    calls: list[tuple[str, str]] = []

    def _fake_measure_combined_with_warm_samples(
        *, bench_module, bench_class, warm_samples
    ):
        calls.append((bench_module, bench_class))
        return {
            "compilation_time": 1.5,
            "peak_cpu_mem": 42.0,
            "warm_samples": [1.0, 2.0, 3.0][:warm_samples],
        }

    monkeypatch.setattr(
        _gpu_mem,
        "measure_combined_with_warm_samples",
        _fake_measure_combined_with_warm_samples,
    )

    instance = benchmark_class()
    cache = instance.setup_cache()
    instance.setup(cache)

    assert calls == [
        ("benchmarks.asv.bench_precautionary_savings", benchmark_class.__name__)
    ]
    assert instance.track_execution_time(cache) == 2.0
    assert instance.track_peak_cpu_mem(cache) == 42.0
    assert instance.track_compilation_time(cache) == 1.5
    assert benchmark_class.version == "2"
    assert not hasattr(benchmark_class, "time_execution")
    assert not hasattr(benchmark_class, "peakmem_execution")
