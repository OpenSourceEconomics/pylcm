"""Measure GPU peak memory for a benchmark in an isolated subprocess.

ASV runs all benchmarks in one process, so ``peak_bytes_in_use`` accumulates
across runs and warm-up calls.  This module spawns a fresh Python process that
builds the model, runs it once cold (compilation + execution), and reports the
true peak.

Usage from a benchmark class::

    from benchmarks._gpu_mem import measure_gpu_peak

    def track_gpu_peak_mem(self):
        return measure_gpu_peak("benchmarks.bench_mahler_yum", "MahlerYum")

The subprocess calls ``setup_for_gpu_measurement()`` (model + params only, no
warm-up) followed by ``time_execution()`` (cold = compile + run), then prints
``peak_bytes_in_use``.
"""

import os
import subprocess
import sys


def measure_gpu_peak(bench_module: str, bench_class: str) -> int:
    """Run a benchmark in a subprocess and return peak GPU bytes.

    Args:
        bench_module: Dotted module path (e.g. ``"benchmarks.bench_mahler_yum"``).
        bench_class: Class name within the module (e.g. ``"MahlerYum"``).

    Returns:
        Peak GPU memory in bytes (including compilation).

    """
    # Disable JAX pre-allocation in the subprocess so it does not compete with the
    # parent ASV process for GPU memory.  peak_bytes_in_use still reflects actual
    # allocation peaks regardless of pre-allocation mode.
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks._gpu_mem", bench_module, bench_class],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return int(result.stdout.strip())


if __name__ == "__main__":
    import importlib

    module = importlib.import_module(sys.argv[1])
    cls = getattr(module, sys.argv[2])

    instance = cls()
    instance.setup_for_gpu_measurement()
    instance.time_execution()

    import jax

    stats = jax.local_devices()[0].memory_stats()
    print(stats["peak_bytes_in_use"])
