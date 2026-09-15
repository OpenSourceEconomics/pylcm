"""ASV workers resolve Mahler metadata without the repository on their path."""

import json
import os
import subprocess
import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parent / "asv"
_MAHLER_IDENTITIES = {
    "bench_mahler_yum.MahlerYumBudgetedGpu.track_execution_time",
    "bench_mahler_yum.MahlerYumBudgetedGpu.track_peak_cpu_mem",
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

# `test_asv_setup_cache_imports_mahler_in_its_temporary_working_directory` was
# retired: it exercised the real ASV `setup_cache` subcommand against
# `MahlerYumBudgetedGpu.time_execution`, an identity whose class had no
# `setup_cache` at all, specifically so the check could assert a no-op
# `cache.pickle == pickle.dumps(None)` without touching a GPU. Sharing the
# heavy timing measurement across `track_execution_time`/`track_peak_cpu_mem`/
# `track_compilation_time` gave `MahlerYumBudgetedGpu` a real `setup_cache`
# (see bench_mahler_yum.py), so no current Mahler identity has that no-op
# property any more; the deferred-import/forkserver-safety property this test
# also covered remains checked below via direct resolution, without running
# `setup_cache` for real.


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
