"""Shared scaffolding for the plugins that reproduce a kernel capability state.

Both plugins repoint the same module-level library paths and reset the same
registration flag before any test imports the upper envelope. Keeping the
mechanics here leaves each plugin to state only the situation it reproduces,
and means a change to how capability is located is made once.

The file-presence predicates here are what the `requires_exact_affine_kernel`
marker and the capability plugins skip on. They read the kernel locations from
`ffi` at call time, so a repointed or monkeypatched location is what they see.
"""

import tempfile
from pathlib import Path

import jax

from _lcm.egm.upper_envelope._exact_affine import ffi


def nonexistent_directory(*, prefix: str) -> Path:
    """Return a directory path the operating system has just confirmed is free.

    A fresh temporary directory is created and immediately discarded, so
    nothing occupies the path. A fixed absolute path would silently stop
    reproducing absence on a machine that happens to have one.
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as created:
        path = Path(created)
    assert not path.exists()
    return path


def point_at(*, cpu_library: Path, cuda_library: Path) -> None:
    """Repoint the kernel locations and drop this process's registration."""
    ffi._CPU_LIBRARY = cpu_library
    ffi._CUDA_LIBRARY = cuda_library
    ffi._REGISTERED = False


def kernel_built() -> bool:
    """Return whether this platform has an exact-affine kernel at all.

    Answers from the file alone, deliberately: a platform that never built one
    is a different situation from a build that is present and cannot answer, and
    only the first is a reason to stop asking. The second is a broken build and
    belongs in front of whoever made it.
    """
    return ffi._CPU_LIBRARY.is_file()


def kernel_built_for_current_backend() -> bool:
    """Return whether a kernel file exists for the selected JAX backend.

    This is deliberately a file-presence predicate, not a loadability probe.
    A missing file is a supported capability absence and can justify skipping a
    test that explicitly declares an exact-kernel requirement. A file that is
    present but stale, unloadable, or missing a symbol is a broken build and must
    reach `_ensure_registered`, where it fails loudly.
    """
    if not kernel_built():
        return False
    return jax.default_backend() != "gpu" or ffi.cuda_kernel_built()
