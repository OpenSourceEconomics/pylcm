"""Where this project's slice of the persistent JIT compilation cache lives.

The cache is split per project rather than pooled in one directory. JAX keys its
entries by a hash of the computation, so splitting costs no reuse — what it
separates is concurrent *writers*. Concurrent writers are what leave unreadable
entries behind, which surface much later as a decompression warning at read time,
after which the cache's work is silently lost and every process recompiles.

Imported before `jax` is, so this module must stay free of heavy imports.
"""

import os
from pathlib import Path

_CACHE_NAME_VAR = "LCM_COMPILATION_CACHE_NAME"
_FALLBACK_NAME = "default"


def compilation_cache_name() -> str:
    """Return the leaf directory name for this project's slice of the cache.

    The enclosing project's directory — the nearest ancestor carrying a `.git` or
    a `pyproject.toml` — so a run started anywhere inside a project reaches the
    same cache, and a project's `tests/` subdirectory does not get a cache of its
    own. Falls back to the working directory's own name, and to `"default"` when
    neither yields one.

    `LCM_COMPILATION_CACHE_NAME` overrides the whole search. It exists for a
    repository holding several independent models that run concurrently — one
    package per paper, say — where the enclosing project is a single directory and
    so too coarse a split. It names a leaf under the shared root, not a path, so
    leading separators are stripped; to place the cache somewhere else entirely,
    set `JAX_COMPILATION_CACHE_DIR` instead.
    """
    if chosen := os.environ.get(_CACHE_NAME_VAR):
        return chosen.strip("/\\") or _FALLBACK_NAME
    try:
        working_dir = Path.cwd()
    except OSError:
        return _FALLBACK_NAME
    for directory in (working_dir, *working_dir.parents):
        if (directory / ".git").exists() or (directory / "pyproject.toml").is_file():
            return directory.name or _FALLBACK_NAME
    return working_dir.name or _FALLBACK_NAME
