"""The DS-2024 housing RFC-vs-NEGM comparison harness runs and is well-formed.

A smoke test: the harness benchmarks NEGM, RFC, and FUES on the DS-2024 housing
model and returns one well-formed row per method with finite, non-negative
timings and a finite accuracy against the VFI oracle. The headline ordering
(RFC faster than NEGM) is machine- and noise-dependent, so it is reported by the
harness rather than asserted here.
"""

import numpy as np

from benchmarks.ds_replication.ds2024_housing_comparison import (
    benchmark_ds2024_housing,
    format_comparison_table,
)


def test_ds2024_housing_comparison_runs_and_is_well_formed():
    """Benchmarking returns finite, well-formed rows for all three methods."""
    rows = benchmark_ds2024_housing(
        n_grid=8, n_housing=5, n_consumption=80, n_periods=4
    )
    assert [row.method for row in rows] == ["NEGM", "RFC", "FUES"]
    for row in rows:
        assert row.compile_seconds >= 0.0
        assert row.runtime_seconds > 0.0
        assert np.isfinite(row.accuracy_mean_abs)
        # Both methods solve the same economics up to grid resolution.
        assert row.accuracy_mean_abs < 1.0

    table = format_comparison_table(rows)
    assert "NEGM" in table
    assert "RFC" in table
