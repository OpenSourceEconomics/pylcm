"""Pytest plugin reproducing a present exact library with a stale symbol table."""

from pathlib import Path

from _lcm.egm.upper_envelope._exact_affine import ffi
from tests.ci._capability import nonexistent_directory, point_at

# The file exists, so capability classification must say "built". Registration
# then reaches the stale-library failure mode: the shared object loads, but one
# of the handlers the running Python package requires is not exported. The
# request must fail rather than becoming a capability skip. The CUDA arm is out
# of scope — this reproduces a stale *CPU* library — so its path is one nothing
# occupies, which keeps the CUDA registration out of it on a GPU host.
point_at(
    cpu_library=Path(__file__).resolve(),
    cuda_library=nonexistent_directory(prefix="pylcm-kernel-absent-")
    / "libcertified_affine_ffi_cuda.so",
)


def _raise_missing_target(*, library: Path, platform: str) -> None:  # noqa: ARG001
    msg = "undefined symbol: ExactCellHullF64"
    raise AttributeError(msg)


# A deliberate stand-in for the real registration; ty compares against the
# original function type rather than its signature, so the swap needs naming.
ffi._register_platform = _raise_missing_target  # ty: ignore[invalid-assignment]

assert ffi.kernel_built() is True
assert ffi.kernel_built_for_current_backend() is True
