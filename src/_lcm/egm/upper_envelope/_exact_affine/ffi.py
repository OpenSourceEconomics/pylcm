"""Exact affine comparison and publication over stored IEEE operands.

The entry points here answer questions about the *stored* floating-point
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
from lcm.typing import BoolND, FloatND, IntND

# Returned where input geometry is invalid, an operand is non-finite, or an
# exact result overflows the target format. Callers must fail loud; nothing is
# known.
UNRESOLVED_STATUS: int = 2

# Batched segment operands carry at least one batch axis ahead of the axis
# holding the segments themselves.
_MIN_BATCHED_RANK: int = 2

_TARGETS = (
    "CertifiedAffineCompareF32",
    "CertifiedAffineCompareF64",
    "ExactAffineReadF32",
    "ExactAffineReadF64",
    "ExactQueryWinnerF32",
    "ExactQueryWinnerF64",
    "ExactQueryWinnerBatchedF32",
    "ExactQueryWinnerBatchedF64",
    "ExactAffineHandoverF32",
    "ExactAffineHandoverF64",
    "ExactCellHullF32",
    "ExactCellHullF64",
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

# Backend names that mean "device compiles will look for the CUDA target".
_GPU_BACKENDS: frozenset[str] = frozenset({"gpu", "cuda"})


def _default_backend() -> str:
    """Return JAX's default backend name, as its own seam so tests can set it."""
    return jax.default_backend()


# Whether the targets have been registered with XLA in this process.
_REGISTERED: bool = False


def kernel_built() -> bool:
    """Return whether this platform has an exact-affine kernel at all.

    Answers from the file alone, deliberately: a platform that never built one
    is a different situation from a build that is present and cannot answer, and
    only the first is a reason to stop asking. The second is a broken build and
    belongs in front of whoever made it.
    """
    return _CPU_LIBRARY.is_file()


def kernel_built_for_current_backend() -> bool:
    """Return whether a kernel file exists for the selected JAX backend.

    This is deliberately a file-presence predicate, not a loadability probe.
    A missing file is a supported capability absence and can justify skipping a
    test that explicitly declares an exact-kernel requirement. A file that is
    present but stale, unloadable, or missing a symbol is a broken build and must
    reach :func:`_ensure_registered`, where it fails loudly.
    """
    if not kernel_built():
        return False
    return jax.default_backend() != "gpu" or _CUDA_LIBRARY.is_file()


def kernel_available() -> bool:
    """Return whether a verdict can be requested in this process.

    Answers rather than raises, so a caller that has something else to do — a
    test that would otherwise report a platform's missing build as a defect —
    can branch on it. Asking for a verdict remains the only way to get one.
    """
    try:
        _ensure_registered()
    except ExactAffineKernelUnavailableError:
        return False
    return True


def kernel_available_for_current_backend() -> bool:
    """Return whether the selected JAX backend has a loadable exact kernel."""
    if not kernel_available():
        return False
    return jax.default_backend() != "gpu" or CUDA_AVAILABLE


def _ensure_registered() -> None:
    """Register the FFI targets with XLA, once per process.

    Registration is deferred to the first request for a verdict rather than run
    at import. A platform that never asks for one — and a checkout whose kernel
    has not been compiled yet — imports the upper envelope like any other
    module, so a build that is absent costs the caller who needs it and nobody
    else.

    Raises:
        ExactAffineKernelUnavailableError: If the CPU library is missing, if the
            default backend is a GPU and the CUDA half was never built, or if a
            library is present but cannot be loaded by this interpreter.

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

    # The CPU library alone satisfies registration, so a solve on a CUDA backend
    # would otherwise reach the first compile needing the device target before
    # anything objects — and on a production mesh that compile is hours in. The
    # CUDA half is skipped whenever the build ran without `nvcc`, which is a
    # property of the environment rather than of the run, so it is knowable here.
    if not CUDA_AVAILABLE and _default_backend() in _GPU_BACKENDS:
        msg = (
            f"The exact-affine kernel has no CUDA half: {_CUDA_LIBRARY} is "
            "missing while JAX's default backend is a GPU, so every certified "
            "read would fail at its first device compile. The CUDA half is built "
            "only where `nvcc` is on PATH at install time; add it to this "
            "environment and rebuild with `pixi run build-exact-affine`."
        )
        raise ExactAffineKernelUnavailableError(msg)

    try:
        _register_platform(library=_CPU_LIBRARY, platform="cpu")
        if CUDA_AVAILABLE:
            _register_platform(library=_CUDA_LIBRARY, platform="CUDA")
    except (OSError, AttributeError) as error:
        msg = (
            f"The exact-affine kernel at {_CPU_LIBRARY} exists but could not be "
            f"loaded: {error}. Two builds commonly fail this way: one made by a "
            "different toolchain, which is missing that toolchain's runtime, and "
            "one made before a target existed, which loads but does not export "
            "it. Rebuild in this environment with `pixi run build-exact-affine`."
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


def exact_query_winner(
    *,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: BoolND,
    x_query: FloatND,
) -> tuple[IntND, IntND]:
    """Select the exact right-continuous owner of every query.

    Each query is compared with every live segment that brackets it. The native
    kernel orders the stored operands lexicographically by exact affine value,
    whether the segment extends strictly to the right, exact value slope, and
    finally the stable segment index. No candidate value is rounded before the
    winner is chosen.

    Segment operands are one-dimensional and shared by all elements of
    `x_query`, and one call resolves them. Under an outer `jax.vmap` each batch
    element carries its own segment set, which is what
    `exact_query_winner_batched` consumes: the transformed program holds one
    batched call for the whole batch, so a caller that vectorized its rows keeps
    that parallelism instead of trading it for a loop.

    Args:
        left_grid: Stored left abscissa of every segment.
        right_grid: Stored right abscissa of every segment.
        left_value: Stored value at `left_grid`.
        right_value: Stored value at `right_grid`.
        live: Whether each segment participates.
        x_query: Query abscissae, of any shape.

    Returns:
        Winner indices and one coupled status per query. Status is zero only
        where at least one valid segment brackets the query and the complete
        total order was resolved; otherwise it is `UNRESOLVED_STATUS`.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.
        TypeError: If floating operands do not share `float32` or `float64`.
        ValueError: If segment operands are not nonempty matching vectors.

    """
    _ensure_registered()
    floating = (
        jnp.asarray(left_grid),
        jnp.asarray(right_grid),
        jnp.asarray(left_value),
        jnp.asarray(right_value),
    )
    if any(array.ndim != 1 for array in floating):
        msg = (
            "exact-query segment operands must be one-dimensional, got "
            f"{[array.shape for array in floating]}."
        )
        raise ValueError(msg)
    shape = floating[0].shape
    if shape[0] == 0 or any(array.shape != shape for array in floating[1:]):
        msg = (
            "exact-query segment operands must be nonempty matching vectors, got "
            f"{[array.shape for array in floating]}."
        )
        raise ValueError(msg)
    query = jnp.asarray(x_query)
    live_array = jnp.asarray(live, dtype=jnp.int32)
    if live_array.shape != shape:
        msg = f"live must have segment shape {shape}, got {live_array.shape}."
        raise ValueError(msg)
    return _shared_segment_winner(*floating, live_array, query)


def exact_query_winner_batched(
    *,
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: BoolND,
    x_query: FloatND,
) -> tuple[IntND, IntND]:
    """Select the exact owner of every query against that query's own segments.

    Each batch element carries an independent segment set, so a microtile of
    pairs resolves in one custom call whose operand shapes, and therefore whose
    compilation key, do not depend on the batch size. Ownership follows the same
    total order as the shared-segment call: exact affine value, whether the
    segment extends strictly to the right, exact value slope, then the stable
    segment index.

    Args:
        left_grid: Stored left abscissa of every segment, with a trailing
            segment axis behind one or more batch axes.
        right_grid: Stored right abscissae, with the same shape.
        left_value: Values at `left_grid`, with the same shape.
        right_value: Values at `right_grid`, with the same shape.
        live: Whether each segment participates, with the same shape.
        x_query: Query abscissae, carrying the segment operands' batch shape
            ahead of its own trailing query axis.

    Returns:
        Winner indices and one coupled status per query, both shaped like
        `x_query`. An index counts from the start of its own batch element's
        segments. Status is zero only where at least one live segment of that
        element brackets the query and the complete total order was resolved;
        otherwise it is `UNRESOLVED_STATUS`.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.
        TypeError: If floating operands do not share `float32` or `float64`.
        ValueError: If the segment or query shapes are inconsistent.

    """
    _ensure_registered()
    floating = (
        jnp.asarray(left_grid),
        jnp.asarray(right_grid),
        jnp.asarray(left_value),
        jnp.asarray(right_value),
    )
    if any(array.ndim < _MIN_BATCHED_RANK for array in floating):
        msg = (
            "batched exact-query segment operands need a batch axis before the "
            f"segment axis, got {[array.shape for array in floating]}."
        )
        raise ValueError(msg)
    shape = floating[0].shape
    if shape[-1] == 0 or any(array.shape != shape for array in floating[1:]):
        msg = (
            "batched exact-query segment operands must be nonempty and share a "
            f"shape, got {[array.shape for array in floating]}."
        )
        raise ValueError(msg)
    query = jnp.asarray(x_query)
    if query.ndim != len(shape) or query.shape[:-1] != shape[:-1]:
        msg = (
            f"x_query must carry the batch shape {shape[:-1]} of the segment "
            f"operands ahead of its query axis, got {query.shape}."
        )
        raise ValueError(msg)
    live_array = jnp.asarray(live, dtype=jnp.int32)
    if live_array.shape != shape:
        msg = f"live must have segment shape {shape}, got {live_array.shape}."
        raise ValueError(msg)
    return _batched_segment_winner_impl(*floating, live_array, query)


def _batched_segment_winner_impl(
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: IntND,
    x_query: FloatND,
) -> tuple[IntND, IntND]:
    """Resolve every query against its own batch element's segment set."""
    target = _target_for(
        operands=(left_grid, right_grid, left_value, right_value, x_query),
        f32="ExactQueryWinnerBatchedF32",
        f64="ExactQueryWinnerBatchedF64",
    )
    result_shapes = (
        jax.ShapeDtypeStruct(x_query.shape, jnp.int32),
        jax.ShapeDtypeStruct(x_query.shape, jnp.int32),
    )
    winner, status = jax.ffi.ffi_call(
        target, result_shapes, vmap_method="broadcast_all"
    )(left_grid, right_grid, left_value, right_value, live, x_query)
    return winner, status


def _shared_segment_winner_impl(
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: IntND,
    x_query: FloatND,
) -> tuple[IntND, IntND]:
    """Resolve every query against the one segment set they all share."""
    target = _target_for(
        operands=(left_grid, right_grid, left_value, right_value, x_query),
        f32="ExactQueryWinnerF32",
        f64="ExactQueryWinnerF64",
    )
    result_shapes = (
        jax.ShapeDtypeStruct(x_query.shape, jnp.int32),
        jax.ShapeDtypeStruct(x_query.shape, jnp.int32),
    )
    winner, status = jax.ffi.ffi_call(target, result_shapes, vmap_method="sequential")(
        left_grid, right_grid, left_value, right_value, live, x_query
    )
    return winner, status


def _with_batch_axis(
    *, operand: FloatND | IntND, batched: bool, axis_size: int
) -> FloatND | IntND:
    """Give an operand shared by the whole batch the batch axis it lacks."""
    if batched:
        return operand
    return jnp.broadcast_to(operand, (axis_size, *operand.shape))


def _shared_segment_winner_vmap(
    axis_size: int,
    in_batched: list[bool],
    left_grid: FloatND,
    right_grid: FloatND,
    left_value: FloatND,
    right_value: FloatND,
    live: IntND,
    x_query: FloatND,
) -> tuple[tuple[IntND, IntND], tuple[bool, bool]]:
    """Resolve a whole batch in one call rather than a loop around a scalar one.

    Two shapes reach this rule, and each has a target that consumes it whole:

    - segments shared by every element, only the queries batched ⇒ the
      shared-segment target already accepts queries of any shape, so the batch
      axis folds into the query axis and no operand is replicated;
    - segments varying with the element ⇒ the batch-native target, which reads
      every element against its own segments. An operand the caller left
      unbatched is shared by every element and is stacked to say so.
    """
    *segments_batched, query_batched = in_batched
    query = _with_batch_axis(
        operand=x_query, batched=query_batched, axis_size=axis_size
    )
    if any(segments_batched):
        winner, status = _batched_segment_winner_impl(
            _with_batch_axis(
                operand=left_grid, batched=segments_batched[0], axis_size=axis_size
            ),
            _with_batch_axis(
                operand=right_grid, batched=segments_batched[1], axis_size=axis_size
            ),
            _with_batch_axis(
                operand=left_value, batched=segments_batched[2], axis_size=axis_size
            ),
            _with_batch_axis(
                operand=right_value, batched=segments_batched[3], axis_size=axis_size
            ),
            _with_batch_axis(
                operand=live, batched=segments_batched[4], axis_size=axis_size
            ),
            query.reshape(axis_size, -1),
        )
    else:
        winner, status = _shared_segment_winner_impl(
            left_grid, right_grid, left_value, right_value, live, query.reshape(-1)
        )
    published = (winner.reshape(query.shape), status.reshape(query.shape))
    return published, (True, True)


# Bound by assignment rather than by decorating the `def`: `custom_vmap` returns
# a callable instance, and a package-wide beartype claw rebinds such an instance
# to its own `__call__`, which would leave `def_vmap` unreachable at import.
_shared_segment_winner = jax.custom_batching.custom_vmap(_shared_segment_winner_impl)
_shared_segment_winner.def_vmap(_shared_segment_winner_vmap)


def exact_affine_handover(
    *,
    a_x0: FloatND,
    a_x1: FloatND,
    a_v0: FloatND,
    a_v1: FloatND,
    b_x0: FloatND,
    b_x1: FloatND,
    b_v0: FloatND,
    b_v1: FloatND,
    left: FloatND,
    right: FloatND,
) -> tuple[FloatND, IntND]:
    """Return the first representable state where the incoming line owns.

    The exact cross-multiplied line difference is affine. Its root is formed as
    a fixed-width integer ratio and rounded directly to IEEE; an exact sign check
    then moves at most one representable state to obtain the least float at or
    above the root.

    Args:
        a_x0: Lower endpoint abscissa of the outgoing line.
        a_x1: Upper endpoint abscissa of the outgoing line.
        a_v0: Outgoing-line value at `a_x0`.
        a_v1: Outgoing-line value at `a_x1`.
        b_x0: Lower endpoint abscissa of the incoming line.
        b_x1: Upper endpoint abscissa of the incoming line.
        b_v0: Incoming-line value at `b_x0`.
        b_v1: Incoming-line value at `b_x1`.
        left: Left edge of the interval the outgoing line owns from.
        right: Right edge of the interval in which the handover must lie.

    Returns:
        The first representable state the incoming line owns and a status. The
        status is nonzero when the operands are invalid, the lines do not have a
        unique increasing-slope handover, or that state cannot be represented.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.

    """
    _ensure_registered()
    operands = _broadcast(
        a_x0,
        a_x1,
        a_v0,
        a_v1,
        b_x0,
        b_x1,
        b_v0,
        b_v1,
        left,
        right,
    )
    target = _target_for(
        operands=operands,
        f32="ExactAffineHandoverF32",
        f64="ExactAffineHandoverF64",
    )
    result_shapes = (
        jax.ShapeDtypeStruct(operands[0].shape, operands[0].dtype),
        jax.ShapeDtypeStruct(operands[0].shape, jnp.int32),
    )
    handover, status = jax.ffi.ffi_call(
        target, result_shapes, vmap_method="broadcast_all"
    )(*operands)
    return handover, status


def exact_cell_hull(
    *,
    left: FloatND,
    right: FloatND,
    live: BoolND,
    low: IntND,
    high: IntND,
    endog_grid: FloatND,
    value: FloatND,
    max_runs: int,
) -> tuple[FloatND, IntND, IntND]:
    """Resolve one node cell's complete affine envelope in one custom call.

    All ownership and handover decisions are made from the stored IEEE endpoint
    operands by fixed-width integer arithmetic. The representation stays inside
    the FFI handler, so the lowered JAX program has one operation regardless of
    limb count or owner-walk length.

    Args:
        left: Left edge of every cell in the batch.
        right: Right edge of every cell in the batch.
        live: Run mask with shape `left.shape + (max_runs,)`.
        low: Lower candidate index of each run's covering link.
        high: Upper candidate index of each run's covering link.
        endog_grid: Candidate abscissae, with one trailing candidate axis.
        value: Candidate values, with the same shape as `endog_grid`.
        max_runs: Static owner capacity of every cell.

    Returns:
        Breakpoints, owner indices, and one status per cell. A nonzero status
        means no exact cell envelope was published.

    Raises:
        ExactAffineKernelUnavailableError: If the kernel is absent or unloadable.
        TypeError: If floating operands do not share `float32` or `float64`.
        ValueError: If the batch, run, or candidate shapes are inconsistent.

    """
    _ensure_registered()
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    endog_grid = jnp.asarray(endog_grid)
    value = jnp.asarray(value)
    floating = (left, right, endog_grid, value)
    target = _target_for(
        operands=floating,
        f32="ExactCellHullF32",
        f64="ExactCellHullF64",
    )
    if right.shape != left.shape:
        msg = f"left and right must share a shape, got {left.shape} and {right.shape}."
        raise ValueError(msg)
    expected_runs = (*left.shape, max_runs)
    live = jnp.asarray(live, dtype=jnp.int32)
    low = jnp.asarray(low, dtype=jnp.int32)
    high = jnp.asarray(high, dtype=jnp.int32)
    if (
        live.shape != expected_runs
        or low.shape != expected_runs
        or high.shape != expected_runs
    ):
        msg = (
            "live, low, and high must have shape left.shape + (max_runs,), got "
            f"{live.shape}, {low.shape}, and {high.shape}; expected {expected_runs}."
        )
        raise ValueError(msg)
    if endog_grid.shape != value.shape or endog_grid.ndim != left.ndim + 1:
        msg = (
            "endog_grid and value must share shape left.shape + (n_candidates,), "
            f"got {endog_grid.shape} and {value.shape}."
        )
        raise ValueError(msg)
    if endog_grid.shape[:-1] != left.shape:
        msg = (
            "endog_grid's batch prefix must match left.shape, got "
            f"{endog_grid.shape[:-1]} and {left.shape}."
        )
        raise ValueError(msg)
    result_shapes = (
        jax.ShapeDtypeStruct((*left.shape, max_runs + 1), left.dtype),
        jax.ShapeDtypeStruct((*left.shape, max_runs), jnp.int32),
        jax.ShapeDtypeStruct(left.shape, jnp.int32),
    )
    bounds, owners, status = jax.ffi.ffi_call(
        target, result_shapes, vmap_method="broadcast_all"
    )(left, right, live, low, high, endog_grid, value)
    return bounds, owners, status


def _broadcast(*operands: FloatND) -> tuple[FloatND, ...]:
    """Broadcast every operand to one common shape."""
    return jnp.broadcast_arrays(*[jnp.asarray(operand) for operand in operands])


def _target_for(*, operands: tuple[FloatND, ...], f32: str, f64: str) -> str:
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
