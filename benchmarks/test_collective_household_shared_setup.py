"""Tests for the shared-measurement collective-household construct/solve benchmarks."""

# ruff: noqa: SLF001

from collections.abc import Callable

import pytest

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
    assert benchmark_class.version == "2"
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


@pytest.mark.parametrize(
    ("benchmark_class", "param"),
    [
        (bench_collective_household.CollectiveHouseholdSimulate, 1_000),
        (bench_collective_household.ReferenceChainSolve, 1),
    ],
)
def test_parameterized_classes_track_the_median_warm_call(
    *,
    monkeypatch: pytest.MonkeyPatch,
    benchmark_class: type[
        bench_collective_household.CollectiveHouseholdSimulate
        | bench_collective_household.ReferenceChainSolve
    ],
    param: int,
) -> None:
    """Execution time is the median of the shared warm-call timer, as elsewhere."""
    calls = []

    def _fake_warm_call_seconds(
        *, execute: Callable[[], None], warm_samples: int
    ) -> list[float]:
        calls.append((execute, warm_samples))
        return [3.0, 1.0, 2.0]

    monkeypatch.setattr(_gpu_mem, "warm_call_seconds", _fake_warm_call_seconds)
    instance = benchmark_class()

    assert instance.track_execution_time(param) == 2.0
    assert calls == [
        (instance.execute_for_measurement, bench_collective_household._WARM_SAMPLES)
    ]
    assert benchmark_class.track_execution_time.unit == "seconds"
    assert not hasattr(benchmark_class, "time_execution")
