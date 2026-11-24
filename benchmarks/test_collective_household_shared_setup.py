"""Tests for the shared-measurement collective-household construct/solve benchmarks."""

import pytest
from asv_runner.benchmarks._base import Benchmark

from benchmarks.asv import _gpu_mem, bench_collective_household

_CLASSES = [
    bench_collective_household.CollectiveHouseholdConstruct,
    bench_collective_household.CollectiveHouseholdSolve,
]


@pytest.mark.parametrize("benchmark_class", _CLASSES)
def test_each_class_reads_its_shared_cache(
    *,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_class: type[
        bench_collective_household.CollectiveHouseholdConstruct
        | bench_collective_household.CollectiveHouseholdSolve
    ],
) -> None:
    """setup_cache is the sole producer; the trackers only read it."""
    calls: list[tuple[str, str]] = []

    def _fake_measure_combined_with_warm_samples(
        *, bench_module: str, bench_class: str, warm_samples: int
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
        ("benchmarks.asv.bench_collective_household", benchmark_class.__name__)
    ]
    assert instance.track_execution_time(cache) == 2.0
    assert instance.track_peak_cpu_mem(cache) == 42.0
    assert benchmark_class.version == "3"
    assert not hasattr(benchmark_class, "time_execution")
    assert not hasattr(benchmark_class, "peakmem_execution")


def test_construct_has_no_compilation_time_tracker() -> None:
    """Construction has no compile phase distinct from the cold call."""
    construct_class = bench_collective_household.CollectiveHouseholdConstruct
    assert not hasattr(construct_class, "track_compilation_time")


def test_solve_still_tracks_compilation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _gpu_mem,
        "measure_combined_with_warm_samples",
        lambda **_: {
            "compilation_time": 1.5,
            "peak_cpu_mem": 42.0,
            "warm_samples": [1.0, 2.0, 3.0],
        },
    )
    instance = bench_collective_household.CollectiveHouseholdSolve()
    cache = instance.setup_cache()
    instance.setup(cache)
    assert instance.track_compilation_time(cache) == 1.5


def test_every_simulate_metric_reads_one_measurement_per_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each cohort is measured once; no metric runs a simulation of its own."""
    measured: list[tuple[str, str]] = []

    def _fake_measure(
        *, bench_module: str, bench_class: str, warm_samples: int
    ) -> dict[str, float | list[float]]:
        measured.append((bench_module, bench_class))
        n_subjects = getattr(bench_collective_household, bench_class).n_subjects
        return {
            "compilation_time": 10.0 * n_subjects,
            "peak_cpu_mem": 20.0 * n_subjects,
            "peak_gpu_mem_automatic_solve_simulate": 30.0 * n_subjects,
            "warm_samples": [3.0 * n_subjects, 1.0 * n_subjects, 2.0 * n_subjects][
                :warm_samples
            ],
        }

    monkeypatch.setattr(
        _gpu_mem, "measure_combined_with_warm_samples_and_gpu_peak", _fake_measure
    )
    simulate = bench_collective_household.CollectiveHouseholdSimulate()
    gpu_peak = bench_collective_household.CollectiveHouseholdSimulateGpuPeakMem()

    cache = simulate.setup_cache()

    module = "benchmarks.asv.bench_collective_household"
    assert measured == [(module, "_Simulate1000"), (module, "_Simulate100000")]
    assert simulate.params == [1_000, 100_000]
    for n_subjects in simulate.params:
        assert simulate.track_compilation_time(cache, n_subjects) == 10.0 * n_subjects
        assert simulate.track_peak_cpu_mem(cache, n_subjects) == 20.0 * n_subjects
        assert simulate.track_execution_time(cache, n_subjects) == 2.0 * n_subjects
    assert gpu_peak.track_gpu_peak_mem(cache) == 30.0 * 100_000
    assert not hasattr(simulate, "peakmem_execution")


def test_the_simulate_classes_share_one_cache() -> None:
    """ASV keys a cache by its producer, so one producer means one measurement."""
    keys = {
        Benchmark(
            "track", benchmark_class.setup_cache, [benchmark_class()]
        ).setup_cache_key
        for benchmark_class in (
            bench_collective_household.CollectiveHouseholdSimulate,
            bench_collective_household.CollectiveHouseholdSimulateGpuPeakMem,
        )
    }

    assert len(keys) == 1
    assert None not in keys
