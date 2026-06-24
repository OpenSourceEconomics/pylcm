"""Reproduce the DS-2024 multidimensional pension comparison: G2EGM vs RFC.

Dobrescu & Shanker (2024) compare solution methods for the two-asset
(liquid + pension) discrete-continuous lifecycle model on solve time and the
distribution of consumption Euler errors. The pylcm P6/P7 stack provides both a
G2EGM (four-segment KKT envelope) and an RFC (combined-cloud rooftop cut) solver
on the shared multidimensional EGM foundation; this driver sweeps the liquid grid
resolution, benchmarks both methods at each, and writes the comparison table.

Run (CUDA env), e.g.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
        benchmarks.ds_replication.run_ds2024_pension --out ds_paper_tables \
        --n-liquid 20 40 80
"""

import argparse
import logging
from pathlib import Path

from _lcm.egm.ds_pension_benchmark import (
    benchmark_ds_pension_methods,
    format_comparison_table,
)

_logger = logging.getLogger("lcm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ds_paper_tables"))
    parser.add_argument("--n-liquid", type=int, nargs="+", default=[20, 40, 80])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "ds2024_pension_g2egm_vs_rfc.md"

    rows = []
    for n_liquid in args.n_liquid:
        _logger.info("=== DS-2024 pension n_liquid=%d ===", n_liquid)
        rows.extend(benchmark_ds_pension_methods(n_liquid=n_liquid))

    table = format_comparison_table(rows)
    text = (
        "# DS-2024 multidimensional pension — G2EGM vs RFC\n\n"
        "Solve time and consumption Euler-error accuracy per method, over the\n"
        "liquid-grid resolution. EE = log10 relative consumption Euler error on the\n"
        "unconstrained interior (more negative = more accurate).\n\n"
        "```\n" + table + "\n```\n"
    )
    path.write_text(text)
    _logger.info("wrote %s", path)
    print(text)


if __name__ == "__main__":
    main()
