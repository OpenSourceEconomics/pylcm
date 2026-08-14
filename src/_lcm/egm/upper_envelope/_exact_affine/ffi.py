"""Exact affine comparison and publication over stored IEEE operands.

Both entry points here answer a question about the *stored* floating-point
operands, not about a floating evaluation of them. Every finite operand is
decoded as an exact signed dyadic $(-1)^s m 2^e$ and the arithmetic runs in
fixed-width integers, so subnormal bits are read as the values they are and no
backend rounding, flushing, or cancellation enters the decision.

Two properties follow, and both are load-bearing for the certified upper
envelope:

- an exact tie is returned only for exact equality of the two rational lines,
  never inferred from a difference that underflowed; and
- the lowered JAX program contains one `custom_call` per operation regardless of
  how a caller partitions its batch, so blocking a reduction changes launch
  grouping and nothing else.

The kernels live in shared objects built at install time from the C++ and CUDA
sources beside this module. They are loaded when a verdict is first requested,
not at import, so a platform that never asks for one needs no kernel; nothing
here ever compiles.
"""

import ctypes
from pathlib import Path

import jax
import jax.numpy as jnp

from lcm.exceptions import ExactAffineKernelUnavailableError
from lcm.typing import FloatND, IntND

# Returned where an operand is non-finite, a width is not positive, or an exact
# result overflows the target format. Callers must fail loud; nothing is known.
UNRESOLVED_STATUS: int = 2

_TARGETS = (
    "CertifiedAffineCompareF32",
    "CertifiedAffineCompareF64",
    "ExactAffineReadF32",
    "ExactAffineReadF64",
)

_DIRECTORY = Path(__file__).resolve().parent
_CPU_LIBRARY = _DIRECTORY / "libcertified_affine_ffi_cpu.so"
_CUDA_LIBRARY = _DIRECTORY / "libcertified_affine_ffi_cuda.so"


def _register_platform(*, library: Path, platform: str) -> None:
    """Register every FFI target exported by one shared object."""
    handle = ctypes.cdll.LoadLibrary(str(library))
    for name in _TARGETS:
        jax.ffi.register_ffi_target(
            name,
            jax.ffi.pycapsule(getattr(handle, name)),
            platform=platform,
        )


# CUDA is optional: without it the certified path runs on CPU only. Building it
# needs `nvcc` and the target architecture, e.g. NVCCFLAGS='-arch=sm_80'.
# Whether a kernel for the CUDA platform exists to be registered.
CUDA_AVAILABLE: bool = _CUDA_LIBRARY.is_file()

# Whether the targets have been registered with XLA in this process.
_REGISTERED: bool = False


def _ensure_registered() -> None:
    """Register the FFI targets with XLA, once per process.

    Registration is deferred to the first request for a verdict rather than run
    at import. A platform that never asks for one — and a checkout whose kernel
    has not been compiled yet — imports the upper envelope like any other
    module, so a build that is absent costs the caller who needs it and nobody
    else.

    Raises:
        ExactAffineKernelUnavailableError: If the CPU library is missing, or is
            present but cannot be loaded by this interpreter.

    """
    global _REGISTERED  # noqa: PLW0603
    if _REGISTERED:
        return

    if not _CPU_LIBRARY.is_file():
        msg = (
            f"The exact-affine kernel is not built: {_CPU_LIBRARY} is missing. It "
            "is compiled when pylcm is installed; after editing its C++ sources, "
            "or in a checkout installed before the kernel existed, rebuild it "
            "with `pixi run build-exact-affine`."
        )
        raise ExactAffineKernelUnavailableError(msg)

    try:
        _register_platform(library=_CPU_LIBRARY, platform="cpu")
        if CUDA_AVAILABLE:
            _register_platform(library=_CUDA_LIBRARY, platform="CUDA")
    except OSError as error:
        msg = (
            f"The exact-affine kernel at {_CPU_LIBRARY} exists but could not be "
            f"loaded: {error}. A library built by one toolchain and loaded by "
            "another commonly fails this way, missing the first toolchain's "
            "runtime. Rebuild it in this environment with "
            "`pixi run build-exact-affine`."
        )
        raise ExactAffineKernelUnavailableError(msg) from error

    _REGISTERED = True


def certified_affine_compare(
    *,
    a_x0: FloatND,
    a_x1: FloatND,
    a_v0: FloatND,
    a_v1: FloatND,
    b_x0: FloatND,
    b_x1: FloatND,
    b_v0: FloatND,
    b_v1: FloatND,
    x_query: FloatND,
) -> IntND:
    """Return the exact sign of `A(x_query) - B(x_query)`.

    `A` and `B` are the affine lines through the given endpoints, extended beyond
    them. The comparison is of the two exact rational lines determined by the
    stored operands, so its verdict is independent of the backend.

    Args:
        a_x0: Lower endpoint abscissa of the first link.
        a_x1: Upper endpoint abscissa of the first link; must exceed `a_x0`.
        a_v0: Value of the first link at `a_x0`.
        a_v1: Value of the first link at `a_x1`.
        b_x0: Lower endpoint abscissa of the second link.
        b_x1: Upper endpoint abscissa of the second link; must exceed `b_x0`.
        b_v0: Value of the second link at `b_x0`.
        b_v1: Value of the second link at `b_x1`.
        x_query: Abscissa at which the two lines are compared.

    Returns:
        `+1` where `A` is above `B`, `-1` where it is below, `0` only where the
        two rational lines are exactly equal at the query, and
        `UNRESOLVED_STATUS` where an operand is non-finite or a width is not
        strictly positive.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.
        TypeError: If the operands do not share one floating dtype, or that dtype
            is neither `float32` nor `float64`.

    """
    _ensure_registered()
    operands = _broadcast(a_x0, a_x1, a_v0, a_v1, b_x0, b_x1, b_v0, b_v1, x_query)
    target = _target_for(
        operands=operands,
        f32="CertifiedAffineCompareF32",
        f64="CertifiedAffineCompareF64",
    )
    result_shape = jax.ShapeDtypeStruct(operands[0].shape, jnp.int32)
    return jax.ffi.ffi_call(target, result_shape, vmap_method="broadcast_all")(
        *operands
    )


def exact_affine_read(
    *,
    x0: FloatND,
    x1: FloatND,
    v0: FloatND,
    v1: FloatND,
    x_query: FloatND,
) -> tuple[FloatND, IntND]:
    """Round the exact affine quotient of one link to the target IEEE format.

    The quotient $\\frac{v_0(x_1-q) + v_1(q-x_0)}{x_1-x_0}$ is formed as an exact
    signed rational from the stored operands and rounded once, to nearest with
    ties to even. Every representable result is published, the subnormal band
    included; a subnormal here is a value, not a refusal.

    Args:
        x0: Lower endpoint abscissa of the link.
        x1: Upper endpoint abscissa of the link; must exceed `x0`.
        v0: Channel value at `x0`.
        v1: Channel value at `x1`.
        x_query: Abscissa at which the link is read.

    Returns:
        Tuple of the rounded value and its status. The status is `0` where the
        value was published and `UNRESOLVED_STATUS` where an operand is
        non-finite, the width is not strictly positive, or the exact result
        overflows the format — in which case the value is NaN and callers must
        mask every channel together.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.
        TypeError: If the operands do not share one floating dtype, or that dtype
            is neither `float32` nor `float64`.

    """
    _ensure_registered()
    operands = _broadcast(x0, x1, v0, v1, x_query)
    target = _target_for(
        operands=operands, f32="ExactAffineReadF32", f64="ExactAffineReadF64"
    )
    result_shapes = (
        jax.ShapeDtypeStruct(operands[0].shape, operands[0].dtype),
        jax.ShapeDtypeStruct(operands[0].shape, jnp.int32),
    )
    published, status = jax.ffi.ffi_call(
        target, result_shapes, vmap_method="broadcast_all"
    )(*operands)
    return published, status


def _broadcast(*operands: FloatND) -> list[FloatND]:
    """Broadcast every operand to one common shape."""
    return jnp.broadcast_arrays(*[jnp.asarray(operand) for operand in operands])


def _target_for(*, operands: list[FloatND], f32: str, f64: str) -> str:
    """Return the FFI target name matching the operands' shared dtype."""
    dtype = operands[0].dtype
    if any(operand.dtype != dtype for operand in operands):
        msg = (
            "All operands must share one dtype, got "
            f"{sorted({str(operand.dtype) for operand in operands})}."
        )
        raise TypeError(msg)
    if dtype == jnp.float32:
        return f32
    if dtype == jnp.float64:
        return f64
    msg = f"Expected float32 or float64 operands, got {dtype}."
    raise TypeError(msg)
