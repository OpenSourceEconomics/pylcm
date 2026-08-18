"""Pytest plugin reproducing a platform with no exact-affine kernel build."""

from pathlib import Path

from _lcm.egm.upper_envelope._exact_affine import ffi

ffi._CPU_LIBRARY = Path("/pylcm-kernel-absent/libcertified_affine_ffi_cpu.so")
ffi._CUDA_LIBRARY = Path("/pylcm-kernel-absent/libcertified_affine_ffi_cuda.so")
ffi.CUDA_AVAILABLE = False
ffi._REGISTERED = False

assert ffi.kernel_built() is False
assert ffi.kernel_built_for_current_backend() is False
assert ffi.kernel_available_for_current_backend() is False
