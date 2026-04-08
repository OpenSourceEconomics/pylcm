"""Measure GPU peak memory for a benchmark in an isolated subprocess.

ASV runs all benchmarks in one process, so ``peak_bytes_in_use`` accumulates
across runs and warm-up calls.  This module spawns a fresh Python process that
builds the model, runs it once cold (compilation + execution), and reports the
true peak.

The ``GpuPeakMem`` base class provides a ready-made ASV benchmark with a no-op
``setup`` so the parent process does not touch the GPU before spawning the
subprocess.  Subclass it and set ``bench_module`` / ``bench_class``::

    class MahlerYumGpuPeakMem(GpuPeakMem):
        bench_module = "benchmarks.bench_mahler_yum"
        bench_class = "MahlerYum"

The subprocess calls ``setup_for_gpu_measurement()`` (model + params only, no
warm-up) followed by ``time_execution()`` (cold = compile + run), then prints
``peak_bytes_in_use``.
"""

import os
import subprocess
import sys
from pathlib import Path

# Project root: the directory containing the benchmarks/ package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def measure_gpu_peak(bench_module: str, bench_class: str) -> int:
    """Run a benchmark in a subprocess and return peak GPU bytes.

    Args:
        bench_module: Dotted module path (e.g. ``"benchmarks.bench_mahler_yum"``).
        bench_class: Class name within the module (e.g. ``"MahlerYum"``).

    Returns:
        Peak GPU memory in bytes (including compilation).

    """
    # Remove MEM_FRACTION so the subprocess can use all available GPU memory.
    # The parent ASV process may limit itself (e.g. 0.3), but the subprocess
    # needs full access since it runs in isolation.
    env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_MEM_FRACTION"}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks._gpu_mem", bench_module, bench_class],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        msg = (
            f"GPU memory subprocess failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg)
    return int(result.stdout.strip())


def _track_gpu_peak_mem(self):
    return measure_gpu_peak(self.bench_module, self.bench_class)


_track_gpu_peak_mem.unit = "bytes"


class GpuPeakMem:
    """ASV benchmark base class for GPU peak memory measurement.

    Subclasses only need to set ``bench_module`` and ``bench_class``.  The
    ``setup`` is intentionally a no-op so the parent ASV process does not
    allocate GPU memory before the subprocess runs.

    The ``track_gpu_peak_mem`` method is injected into subclasses (not the base)
    so that ASV does not discover the base class as a runnable benchmark.
    """

    bench_module: str
    bench_class: str
    timeout = 1200

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.track_gpu_peak_mem = _track_gpu_peak_mem

    def setup(self):
        pass


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
