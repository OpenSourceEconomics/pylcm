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


def _run_in_fresh_interpreter(
    code: str, *, env: dict[str, str], cwd: Path | None = None
) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _project(root: Path, name: str) -> Path:
    """A directory that looks like a project root, i.e. carries a `pyproject.toml`."""
    project = root / name
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\n")
    return project


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
    project = _project(tmp_path, "some-project")
    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)",
        env=_env_with_home(tmp_path),
        cwd=project,
    )
    assert stdout == str(tmp_path / ".cache" / "jax" / "some-project")


def test_default_cache_dir_is_separated_per_project(tmp_path: Path):
    """Two projects get two cache directories, named after each project's own root.

    A single shared directory makes every project on the machine a concurrent
    writer of one cache, and concurrent writers are what corrupt its entries —
    reported at read time as an unreadable-entry warning, after which the work of
    the cache is silently lost. Entries are already keyed by a hash of the
    computation, so separating projects costs no reuse: it separates *writers*,
    not contents.
    """
    read_cache_dir = (
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)"
    )
    env = _env_with_home(tmp_path)

    first = _run_in_fresh_interpreter(
        read_cache_dir, env=env, cwd=_project(tmp_path, "aca-dev")
    )
    second = _run_in_fresh_interpreter(
        read_cache_dir, env=env, cwd=_project(tmp_path, "attanasio2018")
    )

    assert first == str(tmp_path / ".cache" / "jax" / "aca-dev")
    assert second == str(tmp_path / ".cache" / "jax" / "attanasio2018")


def test_default_cache_dir_is_the_same_from_any_subdirectory(tmp_path: Path):
    """One project keeps one cache directory wherever inside it the run starts.

    Naming the directory after the working directory alone would give a project's
    root and its `tests/` subdirectory two different caches, so a suite run would
    never reuse what a root-level run compiled — and two unrelated projects both
    invoked from a `tests/` directory would land back in one shared cache.
    """
    read_cache_dir = (
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)"
    )
    env = _env_with_home(tmp_path)
    project = _project(tmp_path, "my-project")
    nested = project / "tests" / "solution"
    nested.mkdir(parents=True)

    from_root = _run_in_fresh_interpreter(read_cache_dir, env=env, cwd=project)
    from_nested = _run_in_fresh_interpreter(read_cache_dir, env=env, cwd=nested)

    assert from_root == str(tmp_path / ".cache" / "jax" / "my-project")
    assert from_nested == from_root


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
    project = _project(tmp_path, "populating-project")
    _run_in_fresh_interpreter(code, env=_env_with_home(tmp_path), cwd=project)
    cache = tmp_path / ".cache" / "jax" / "populating-project"
    assert len(list(cache.iterdir())) > 0


def test_cache_name_can_be_set_explicitly(tmp_path: Path):
    """`LCM_COMPILATION_CACHE_NAME` names the leaf under the shared cache root.

    A repository holding several independent models that run concurrently — one
    package per paper — is a single project, so the enclosing project is too
    coarse a split and every model would share one cache again. Naming the leaf
    separates them without making each caller spell out an absolute path.
    """
    project = _project(tmp_path, "lcm-replications")
    paper = project / "src" / "lcm_reps" / "somePaper2016"
    paper.mkdir(parents=True)
    env = _env_with_home(tmp_path) | {
        "LCM_COMPILATION_CACHE_NAME": "somePaper2016",
    }

    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)",
        env=env,
        cwd=paper,
    )

    assert stdout == str(tmp_path / ".cache" / "jax" / "somePaper2016")


def test_an_explicit_cache_dir_outranks_an_explicit_cache_name(tmp_path: Path):
    """A directory given outright wins over a name; the two do not compose."""
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    env = {**os.environ, _CACHE_DIR_VAR: str(chosen)} | {
        "LCM_COMPILATION_CACHE_NAME": "ignored-when-a-dir-is-given",
    }

    stdout = _run_in_fresh_interpreter(
        "import jax; import lcm; print(jax.config.jax_compilation_cache_dir)",
        env=env,
    )

    assert stdout == str(chosen)


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
