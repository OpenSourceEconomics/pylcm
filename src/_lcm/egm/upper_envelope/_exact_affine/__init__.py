"""Exact arithmetic over stored IEEE operands for the certified upper envelope."""

from _lcm.egm.upper_envelope._exact_affine.ffi import (
    UNRESOLVED_STATUS,
    certified_affine_compare,
    cuda_kernel_built,
    exact_affine_handover,
    exact_affine_read,
    exact_cell_hull,
    exact_query_winner,
    exact_query_winner_batched,
)

__all__ = [
    "UNRESOLVED_STATUS",
    "certified_affine_compare",
    "cuda_kernel_built",
    "exact_affine_handover",
    "exact_affine_read",
    "exact_cell_hull",
    "exact_query_winner",
    "exact_query_winner_batched",
]
