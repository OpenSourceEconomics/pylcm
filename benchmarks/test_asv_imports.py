"""ASV workers resolve Mahler metadata without the repository on their path."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parent / "asv"
_MAHLER_IDENTITIES = {
    "bench_mahler_yum.MahlerYumBudgetedGpu.peakmem_execution",
    "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution",
    "bench_mahler_yum.MahlerYumBudgetedGpu.track_compilation_time",
    (
        "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
        "track_peak_gpu_mem_automatic_solve_simulate"
    ),
    (
        "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
        "track_peak_gpu_mem_load_supplied_solution_simulate"
    ),
    (
        "bench_mahler_yum.MahlerYumBudgetedGpuPeakMem."
        "track_peak_gpu_mem_solve_save_all_persistable"
    ),
}


def test_asv_setup_cache_imports_mahler_in_its_temporary_working_directory(
    tmp_path: Path,
) -> None:
    """The real ASV cache worker imports a timing identity without running a model."""
    spec = importlib.util.find_spec("asv")
    assert spec is not None
    assert spec.origin is not None
    runner = Path(spec.origin).with_name("benchmark.py")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "ASV_PYTHONPATH"}
    }
    result = subprocess.run(
        [
            sys.executable,
            str(runner),
            "setup_cache",
            str(_BENCHMARK_ROOT),
            "bench_mahler_yum.MahlerYumBudgetedGpu.time_execution",
            "{}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "cache.pickle").read_bytes() == pickle.dumps(None)


def test_fresh_asv_worker_resolves_all_mahler_identities_without_gpu_imports(
    tmp_path: Path,
) -> None:
    """All six identities resolve directly before any other suite module loads."""
    code = """
import json
import sys
from asv_runner.discovery import get_benchmark_from_name

root, names_json = sys.argv[1:]
benchmarks = [get_benchmark_from_name(root, name) for name in json.loads(names_json)]
print(json.dumps({
    "names": [benchmark.name for benchmark in benchmarks],
    "jax_imported": "jax" in sys.modules,
    "lcm_imported": "lcm" in sys.modules,
    "repository_package_imported": "benchmarks" in sys.modules,
    "suite_package": sys.modules["asv.bench_mahler_yum"].__package__,
}))
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "ASV_PYTHONPATH"}
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(_BENCHMARK_ROOT),
            json.dumps(sorted(_MAHLER_IDENTITIES)),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert set(receipt["names"]) == _MAHLER_IDENTITIES
    assert len(receipt["names"]) == 6
    assert receipt["suite_package"] == "asv"
    assert receipt["repository_package_imported"] is False
    assert receipt["jax_imported"] is False
    assert receipt["lcm_imported"] is False
