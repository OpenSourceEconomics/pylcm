"""Pytest plugin reproducing a platform with no exact-affine kernel build."""

from _lcm.egm.upper_envelope._exact_affine import ffi
from tests.ci._capability import nonexistent_directory, point_at

_ABSENT = nonexistent_directory(prefix="pylcm-kernel-absent-")
point_at(
    cpu_library=_ABSENT / "libcertified_affine_ffi_cpu.so",
    cuda_library=_ABSENT / "libcertified_affine_ffi_cuda.so",
)

assert ffi.kernel_built() is False
assert ffi.kernel_built_for_current_backend() is False
assert ffi.kernel_available_for_current_backend() is False
