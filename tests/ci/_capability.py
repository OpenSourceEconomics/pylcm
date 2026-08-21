"""Shared scaffolding for the plugins that reproduce a kernel capability state.

Both plugins repoint the same module-level library paths and reset the same
registration flag before any test imports the upper envelope. Keeping the
mechanics here leaves each plugin to state only the situation it reproduces,
and means a change to how capability is located is made once.
"""

import tempfile
from pathlib import Path

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
