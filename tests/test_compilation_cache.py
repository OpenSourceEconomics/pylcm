"""The persistent compilation cache works for fast-compiling kernels.

pylcm models compile as many small XLA programs, most of which individually
compile in well under a second. JAX only writes an executable to the
persistent cache when its compile time exceeds
`jax_persistent_cache_min_compile_time_secs`, so with JAX's default threshold
the cache stays empty and every fresh process recompiles everything.
Importing `lcm` must therefore zero the threshold — regardless of whether
`jax` was imported first — while respecting an explicit user override via the
environment variable.

Both halves of the cache have to survive that import order: the threshold and
the directory the cache lives in. JAX reads `JAX_COMPILATION_CACHE_DIR` once,
while defining its config options, so a process that reaches `import jax` before
`import lcm` — a test suite's `conftest`, a notebook, any library that pulls in
jax — will not see a directory exported afterwards, and the cache is then
disabled with no error. Importing `lcm` must leave the cache usable either way.
"""

import os
import subprocess
import sys
from pathlib import Path

_MIN_COMPILE_TIME_VAR = "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
_CACHE_DIR_VAR = "JAX_COMPILATION_CACHE_DIR"


def _env_with_home(tmp_path: Path) -> dict[str, str]:
    """Environment whose home directory is `tmp_path`, with the cache unset.

    `Path.home()` reads `USERPROFILE` on Windows and `HOME` everywhere else, so
    both have to point at the sandbox for the default cache directory to land
    inside it.
    """
    kept = {
        k: v
        for k, v in os.environ.items()
        if k not in (_MIN_COMPILE_TIME_VAR, _CACHE_DIR_VAR)
    }
    return kept | {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}


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


def test_import_sets_the_cache_dir_even_when_jax_is_imported_first(tmp_path: Path):
    """`lcm` supplies the default cache directory after a prior `import jax`.

    The directory is exported as an environment variable, which JAX only consults
    while defining its config options — so with `jax` already imported, exporting
    it is not enough and the value has to reach `jax.config` directly.
    """
    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)",
        env=_env_with_home(tmp_path),
    )
    assert stdout == str(tmp_path / ".cache" / "jax")


def test_default_cache_dir_populates_when_jax_is_imported_first(tmp_path: Path):
    """A kernel compiled after a jax-first import lands in the default cache.

    The end-to-end property the directory exists for: without it every fresh
    process recompiles the whole model, which is minutes on a production grid and
    is invisible because nothing reports it.
    """
    code = """
import jax
import lcm
import jax.numpy as jnp

jax.jit(lambda x: jnp.sin(x) + 1.0)(jnp.arange(3.0)).block_until_ready()
"""
    _run_in_fresh_interpreter(code, env=_env_with_home(tmp_path))
    assert len(list((tmp_path / ".cache" / "jax").iterdir())) > 0


def test_import_respects_a_user_supplied_cache_dir(tmp_path: Path):
    """An explicitly exported cache directory survives the `lcm` import."""
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    env = {**os.environ, _CACHE_DIR_VAR: str(chosen)}
    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)",
        env=env,
    )
    assert stdout == str(chosen)
