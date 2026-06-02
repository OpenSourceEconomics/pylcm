"""Pick a subject batch size from compiled-program memory estimates.

`subject_batch_size="auto"` sizes the forward-simulation chunk to the device
rather than asking the user for a number. The device working set is affine in
the subject count — a fixed part (the resident value-function array and other
batch-independent arguments) plus a per-subject part — so two compile-only
`memory_analysis()` probes pin the line, and the largest batch under the memory
budget falls straight out. No execution, no out-of-memory risk; the estimate is
XLA's static buffer accounting, and the margin on the budget absorbs its slack.
"""

from collections.abc import Sequence
from typing import Protocol


class _MemoryStats(Protocol):
    """Subset of `jax.stages.Compiled.memory_analysis()` we read."""

    temp_size_in_bytes: int
    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int


def estimate_peak_bytes(stats: _MemoryStats) -> int:
    """Estimate a compiled program's peak device memory in bytes.

    Sums the temporary, argument, and output buffers and subtracts the aliased
    bytes — input buffers XLA reuses in place as output, which would otherwise
    be counted in both `argument_size` and `output_size`.

    Args:
        stats: The `memory_analysis()` result for one compiled program.

    Returns:
        Estimated peak bytes for that program.

    """
    return (
        stats.temp_size_in_bytes
        + stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        - stats.alias_size_in_bytes
    )


def pick_batch_size(
    *,
    probes: Sequence[tuple[int, int]],
    budget_bytes: int,
    max_batch: int,
) -> int:
    """Pick the largest subject batch whose estimated peak fits the budget.

    Models peak memory as affine in the batch size, `peak(b) = intercept +
    slope · b`, and solves `peak(b) = budget` for `b`. With two or more probes
    the line is fit through the lowest- and highest-batch measurements; with one
    probe the fixed overhead is taken as zero (proportional). The result is
    clamped to `[1, max_batch]`.

    Args:
        probes: Sequence of `(batch_size, peak_bytes)` measurements.
        budget_bytes: Memory the batch working set must fit within (already
            net of any safety margin).
        max_batch: Upper clamp — the (padded) population size; never batch
            larger than one pass.

    Returns:
        Subject batch size in `[1, max_batch]`.

    """
    ordered = sorted(probes)
    if len(ordered) == 1:
        batch, peak = ordered[0]
        slope = peak / batch
        intercept = 0.0
    else:
        (b_lo, p_lo), (b_hi, p_hi) = ordered[0], ordered[-1]
        slope = (p_hi - p_lo) / (b_hi - b_lo)
        intercept = p_hi - slope * b_hi
    if slope <= 0:
        return max_batch
    batch = int((budget_bytes - intercept) / slope)
    return max(1, min(max_batch, batch))
