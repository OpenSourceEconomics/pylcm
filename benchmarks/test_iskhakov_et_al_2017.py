"""Tests for the shared-measurement Iskhakov et al. (2017) benchmarks."""

import pytest

from benchmarks.asv import _gpu_mem, bench_iskhakov_et_al_2017

_CLASSES = [
    (bench_iskhakov_et_al_2017.IskhakovEtAl2017Solve, "2"),
    (bench_iskhakov_et_al_2017.IskhakovEtAl2017DCEGMSolve, "3"),
    (bench_iskhakov_et_al_2017.IskhakovEtAl2017Simulate, "2"),
    (bench_iskhakov_et_al_2017.IskhakovEtAl2017DCEGMSimulate, "2"),
]


@pytest.mark.parametrize(("benchmark_class", "expected_version"), _CLASSES)
def test_each_class_reads_its_shared_cache_for_all_three_trackers(
    *, monkeypatch: pytest.MonkeyPatch, benchmark_class: type, expected_version: str
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
        ("benchmarks.asv.bench_iskhakov_et_al_2017", benchmark_class.__name__)
    ]
    assert instance.track_execution_time(cache) == 2.0
    assert instance.track_peak_cpu_mem(cache) == 42.0
    assert instance.track_compilation_time(cache) == 1.5
    assert benchmark_class.version == expected_version
    assert not hasattr(benchmark_class, "time_execution")
    assert not hasattr(benchmark_class, "peakmem_execution")
