"""Pytest plugin reproducing a present exact library with a stale symbol table."""

from pathlib import Path

from _lcm.egm.upper_envelope._exact_affine import ffi

# The file exists, so capability classification must say "built". Registration
# then reaches the stale-library failure mode: the shared object loads, but one
# of the handlers the running Python package requires is not exported. The
# request must fail rather than becoming a capability skip.
_BROKEN_LIBRARY = Path(__file__).resolve()
ffi._CPU_LIBRARY = _BROKEN_LIBRARY
ffi._CUDA_LIBRARY = _BROKEN_LIBRARY
ffi.CUDA_AVAILABLE = False
ffi._REGISTERED = False


def _raise_missing_target(*, library: Path, platform: str) -> None:  # noqa: ARG001
    msg = "undefined symbol: ExactCellHullF64"
    raise AttributeError(msg)


# A deliberate stand-in for the real registration; ty compares against the
# original function type rather than its signature, so the swap needs naming.
ffi._register_platform = _raise_missing_target  # ty: ignore[invalid-assignment]

assert ffi.kernel_built() is True
assert ffi.kernel_built_for_current_backend() is True
