"""Measure GPU peak memory for a benchmark in an isolated subprocess.

ASV runs all benchmarks in one process, so ``peak_bytes_in_use`` accumulates
across runs and warm-up calls.  This module spawns a fresh Python process that
builds the model, runs it once cold (compilation + execution), and reports the
peak.  A single cold run in a fresh process is the production footprint: the
production launcher solves + simulates exactly once per process.

XLA autotuning is disabled in the subprocess (``--xla_gpu_autotune_level=0``).
Autotuning benchmarks candidate kernels at compile time, allocating large,
run-to-run-variable scratch buffers; for big models that transient dwarfs and
masks the execution working set, so the reported peak swings several-fold
between otherwise-identical runs. Turning it off makes the compile footprint
deterministic and matches how the model is run in production (the sbatch
already sets ``--xla_gpu_autotune_level=0``).

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
from collections.abc import Mapping
from pathlib import Path

# Project root: the directory containing the benchmarks/ package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Marks the peak-memory line on the subprocess's stdout. The subprocess imports
# lcm, whose beartype claw can emit diagnostics to stdout, so the parent locates
# this line instead of parsing stdout wholesale.
_PEAK_MARKER = "__PEAK_BYTES_IN_USE__"


def _subprocess_env(base_env: Mapping[str, str]) -> dict[str, str]:
    """Build the GPU-mem subprocess environment from a base mapping.

    - Drops ``XLA_PYTHON_CLIENT_MEM_FRACTION`` so the isolated subprocess can
      use all device memory (the parent ASV process may cap itself).
    - Disables preallocation so ``peak_bytes_in_use`` tracks real demand.
    - Appends ``--xla_gpu_autotune_level=0`` to ``XLA_FLAGS`` (preserving any
      existing flags) so the compile footprint is deterministic.
    """
    env = {k: v for k, v in base_env.items() if k != "XLA_PYTHON_CLIENT_MEM_FRACTION"}
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    autotune_off = "--xla_gpu_autotune_level=0"
    existing = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{existing} {autotune_off}".strip()
    return env


def measure_gpu_peak(bench_module: str, bench_class: str) -> int:
    """Run a benchmark in a subprocess and return peak GPU bytes.

    Args:
        bench_module: Dotted module path (e.g. ``"benchmarks.bench_mahler_yum"``).
        bench_class: Class name within the module (e.g. ``"MahlerYum"``).

    Returns:
        Peak GPU memory in bytes over a single cold run (compile + execute),
        with autotuning disabled.

    """
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks._gpu_mem", bench_module, bench_class],
        capture_output=True,
        text=True,
        check=False,
        cwd=_PROJECT_ROOT,
        env=_subprocess_env(os.environ),
    )
    if result.returncode != 0:
        msg = (
            f"GPU memory subprocess failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        raise RuntimeError(msg)
    for line in result.stdout.splitlines():
        if line.startswith(_PEAK_MARKER):
            return int(line.removeprefix(_PEAK_MARKER).strip())
    msg = (
        "GPU memory subprocess produced no peak-bytes line.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    raise RuntimeError(msg)


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
    # Stable version stamp so asv keeps continuity across benchmark-body
    # refactors that don't change what's measured.
    version = "1"
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
    print(f"{_PEAK_MARKER} {stats['peak_bytes_in_use']}")
