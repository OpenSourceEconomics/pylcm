"""The DS pension comparison harness emits a G2EGM accuracy/time row and a table.

The harness solves the DS pension model by G2EGM, times the solve, and pools the
working consumption Euler errors into the accuracy fields of a benchmark row — the
G2EGM column of the Dobrescu--Shanker pension comparison table. The row's accuracy
improves as the liquid grid refines, and `format_comparison_table` renders the rows.
"""

from _lcm.egm.ds_pension_benchmark import (
    MethodBenchmark,
    benchmark_g2egm_ds_pension,
    format_comparison_table,
)


def test_g2egm_benchmark_reports_sane_time_and_euler_errors():
    """The G2EGM row has a positive solve time and finite interior Euler errors."""
    row = benchmark_g2egm_ds_pension(n_liquid=12)
    assert row.method == "G2EGM"
    assert row.solve_seconds > 0.0
    assert row.euler_error_median_log10 < -1.0  # better than 10% on the interior
    assert row.euler_error_p90_log10 < row.euler_error_median_log10 + 1.0


def test_g2egm_benchmark_accuracy_improves_with_liquid_refinement():
    """Refining the liquid grid lowers the pooled Euler error."""
    coarse = benchmark_g2egm_ds_pension(n_liquid=12)
    fine = benchmark_g2egm_ds_pension(n_liquid=24)
    assert fine.euler_error_median_log10 < coarse.euler_error_median_log10 - 0.2


def test_format_comparison_table_renders_each_row():
    """The table renders a header and one line per method row."""
    rows = [
        MethodBenchmark(
            method="G2EGM",
            n_liquid=12,
            n_pension=10,
            solve_seconds=0.3,
            euler_error_median_log10=-1.4,
            euler_error_p90_log10=-0.9,
        )
    ]
    table = format_comparison_table(rows)
    assert "method" in table
    assert "G2EGM" in table
    assert table.count("\n") == 2  # header, rule, one row
