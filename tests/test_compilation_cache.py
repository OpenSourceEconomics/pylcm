"""The persistent compilation cache works for fast-compiling kernels.

pylcm models compile as many small XLA programs, most of which individually
compile in well under a second. JAX only writes an executable to the
persistent cache when its compile time exceeds
`jax_persistent_cache_min_compile_time_secs`, so with JAX's default threshold
the cache stays empty and every fresh process recompiles everything.
Importing `lcm` must therefore zero the threshold — regardless of whether
`jax` was imported first — while respecting an explicit user override via the
environment variable.
"""

import os
import subprocess
import sys
from pathlib import Path

_MIN_COMPILE_TIME_VAR = "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"


def _run_in_fresh_interpreter(code: str, *, env: dict[str, str]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_zeroes_min_compile_time_even_when_jax_is_imported_first():
    """Importing `lcm` zeroes the cache threshold, also after a prior `import jax`."""
    env = {k: v for k, v in os.environ.items() if k != _MIN_COMPILE_TIME_VAR}
    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; "
        "print(jax.config.jax_persistent_cache_min_compile_time_secs)",
        env=env,
    )
    assert float(stdout) == 0.0


def test_import_respects_user_min_compile_time():
    """A user-set compile-time threshold survives the `lcm` import."""
    env = {**os.environ, _MIN_COMPILE_TIME_VAR: "1.5"}
    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; "
        "print(jax.config.jax_persistent_cache_min_compile_time_secs)",
        env=env,
    )
    assert float(stdout) == 1.5


def test_fast_compilation_populates_the_persistent_cache(tmp_path: Path):
    """Compiling a sub-second kernel writes an entry to the persistent cache."""
    env = {k: v for k, v in os.environ.items() if k != _MIN_COMPILE_TIME_VAR} | {
        "JAX_COMPILATION_CACHE_DIR": str(tmp_path)
    }
    code = """
import jax
import lcm
import jax.numpy as jnp

jax.jit(lambda x: jnp.sin(x) + 1.0)(jnp.arange(3.0)).block_until_ready()
"""
    _run_in_fresh_interpreter(code, env=env)
    assert len(list(tmp_path.iterdir())) > 0
