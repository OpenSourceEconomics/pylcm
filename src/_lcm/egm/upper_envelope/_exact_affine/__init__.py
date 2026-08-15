"""Exact arithmetic over stored IEEE operands for the certified upper envelope."""

from _lcm.egm.upper_envelope._exact_affine.ffi import (
    CUDA_AVAILABLE,
    UNRESOLVED_STATUS,
    certified_affine_compare,
    exact_affine_handover,
    exact_affine_read,
    exact_cell_hull,
    exact_query_winner,
)

__all__ = [
    "CUDA_AVAILABLE",
    "UNRESOLVED_STATUS",
    "certified_affine_compare",
    "exact_affine_handover",
    "exact_affine_read",
    "exact_cell_hull",
    "exact_query_winner",
]
