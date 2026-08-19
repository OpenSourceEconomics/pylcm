"""Pytest plugin reproducing a platform with no exact-affine kernel build."""

import tempfile
from pathlib import Path

from _lcm.egm.upper_envelope._exact_affine import ffi

# A fresh temporary directory is created and immediately discarded, so the
# paths below name a location the operating system has just confirmed nothing
# occupies. A fixed absolute path would silently stop reproducing absence on a
# machine that happens to have one.
with tempfile.TemporaryDirectory(prefix="pylcm-kernel-absent-") as _absent:
    _ABSENT = Path(_absent)

ffi._CPU_LIBRARY = _ABSENT / "libcertified_affine_ffi_cpu.so"
ffi._CUDA_LIBRARY = _ABSENT / "libcertified_affine_ffi_cuda.so"
ffi.CUDA_AVAILABLE = False
ffi._REGISTERED = False

assert not _ABSENT.exists()
assert ffi.kernel_built() is False
assert ffi.kernel_built_for_current_backend() is False
assert ffi.kernel_available_for_current_backend() is False
