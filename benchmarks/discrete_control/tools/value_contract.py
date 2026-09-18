"""Unchanged source-pinned value comparator for the discrete control."""

import numpy as np
from numpy.typing import ArrayLike


def assert_agrees_to_ulp(
    *,
    got: ArrayLike,
    expected: ArrayLike,
    n_ulp: int,
    err_msg: str = "",
    operand_magnitude: float | None = None,
) -> None:
    """Assert two arrays name the same real number to within `n_ulp` of the format.

    The instrument for a knob that partitions a computation without changing it —
    a batch size, a block size. Such a knob changes the vmap width each block is
    compiled for, and XLA emits a differently vectorized kernel per width, so the
    two results can land on representable neighbours. Bounding the gap in units of
    the working format's spacing states exactly that, and states it once for both
    precisions: a partition-dependent *reduction*, the defect this guards against,
    moves a value by orders of magnitude more than a few ULP.

    The spacing is taken at each compared element's own magnitude unless
    `operand_magnitude` names the magnitude of the operands the value is formed
    from. A value born by cancellation — a flow utility plus a discounted
    continuation of opposite sign, say — carries the rounding of those operands,
    so no implementation locates it to within its own spacing; measuring the gap at
    the operands' spacing keeps the bound about code generation rather than about
    the cancellation.

    Args:
        got: Result under the partitioned computation.
        expected: Result under the unpartitioned one.
        n_ulp: Largest tolerated gap, in units of the spacing at the compared
            magnitude.
        err_msg: Context appended to the failure message.
        operand_magnitude: Magnitude whose spacing is the unit for every element,
            when the compared values are formed from operands of that magnitude;
            `None` measures each element at its own magnitude.

    """
    got_arr = np.asarray(got)
    expected_arr = np.asarray(expected)
    # Compare the non-finite entries as the exact values they are. ULP distance is
    # meaningless for them — `np.spacing(inf)` is NaN, so every comparison against
    # it is false and any mismatch would pass silently.
    finite = np.isfinite(expected_arr)
    np.testing.assert_array_equal(
        np.where(finite, 0.0, got_arr),
        np.where(finite, 0.0, expected_arr),
        err_msg=f"non-finite entries differ. {err_msg}",
    )
    np.testing.assert_array_equal(
        np.isfinite(got_arr), finite, err_msg=f"finiteness differs. {err_msg}"
    )
    gap = np.where(finite, np.abs(got_arr - expected_arr), 0.0)
    magnitude = np.maximum(np.abs(got_arr), np.abs(expected_arr))
    if operand_magnitude is not None:
        # Measure the spacing in the compared arrays' own format: promoting the
        # magnitude to a wider float would report the wider format's spacing.
        magnitude = np.maximum(
            magnitude, np.asarray(operand_magnitude, dtype=magnitude.dtype)
        )
    spacing = np.spacing(magnitude)
    in_ulp = np.divide(
        gap, spacing, out=np.zeros(gap.shape, dtype=float), where=gap > 0.0
    )
    worst = float(in_ulp.max(initial=0.0))
    if worst > n_ulp:
        where = np.unravel_index(int(np.argmax(in_ulp)), in_ulp.shape)
        msg = (
            f"Values differ by up to {worst:.1f} ULP, above the {n_ulp} allowed; "
            f"worst at {where}: {got_arr[where]!r} vs {expected_arr[where]!r}. "
            f"{err_msg}"
        )
        raise AssertionError(msg)
