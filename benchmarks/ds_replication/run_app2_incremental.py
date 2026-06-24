"""Run the App.2 housing Table 3 sweep incrementally, one (tau, NG) row at a time.

Each row is appended to the output markdown the instant it finishes, so a wall-clock
timeout on a short partition leaves every completed row on disk. Designed for the
A100 sgpu_devel partition where the full 2-D NEGM solve fits but the 1-hour limit may
not cover the whole 3-tau x 4-NG matrix in a single job.

Usage (CUDA env), e.g.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m \
        benchmarks.ds_replication.run_app2_incremental \
        --out ds_paper_tables --ngs 250 500 750 1000 --taus 0.05 0.07 0.12

`--liquid-batch-size > 0` chunks the liquid Euler grid to bound peak device memory.
"""

import argparse
import logging
import time
from pathlib import Path

from benchmarks.ds_replication.app2_fues_accuracy import (
    app2_fues_euler_error,
    app2_fues_timing,
)
from benchmarks.ds_replication.app2_housing_accuracy import (
    app2_negm_euler_error,
    app2_negm_timing,
)

_logger = logging.getLogger("lcm")

_HEADER = (
    "| NG | tau | FUES_eerr | NEGM_eerr | FUES_compile | FUES_run "
    "| NEGM_compile | NEGM_run |\n"
    "|---:|----:|----------:|----------:|-------------:|---------:"
    "|-------------:|--------:|\n"
)


def _safe(label, func, **kwargs):
    """Run one accuracy/timing measurement, returning None on a runtime failure.

    The FUES and NEGM columns are scored independently so a memory failure in one
    method (the dense NEGM outer argmax can exceed a single GPU at large `n_grid`)
    leaves the other method's number on the row instead of voiding the whole cell.
    """
    try:
        return func(**kwargs)
    except Exception:
        _logger.exception("%s failed", label)
        return None


def _fmt(value, spec):
    return format(value, spec) if value is not None else "OOM"


def _row(*, ng: int, tau: float, liquid_batch_size: int, outer_batch_size: int) -> str:
    fues_kw = {"n_grid": ng, "tau": tau, "liquid_batch_size": liquid_batch_size}
    negm_kw = {**fues_kw, "outer_batch_size": outer_batch_size}
    fues_t = _safe("FUES timing", app2_fues_timing, **fues_kw)
    negm_t = _safe("NEGM timing", app2_negm_timing, **negm_kw)
    fues_e = _safe("FUES euler", app2_fues_euler_error, **fues_kw)
    negm_e = _safe("NEGM euler", app2_negm_euler_error, **negm_kw)
    fc = fues_t["compile_time"] if fues_t else None
    fr = fues_t["runtime"] if fues_t else None
    nc = negm_t["compile_time"] if negm_t else None
    nr = negm_t["runtime"] if negm_t else None
    return (
        f"| {ng} | {tau} | {_fmt(fues_e, '.4f')} | {_fmt(negm_e, '.4f')} "
        f"| {_fmt(fc, '.3f')} | {_fmt(fr, '.4f')} "
        f"| {_fmt(nc, '.3f')} | {_fmt(nr, '.4f')} |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ds_paper_tables"))
    parser.add_argument("--ngs", type=int, nargs="+", default=[250, 500, 750, 1000])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.07, 0.12])
    parser.add_argument(
        "--liquid-batch-size",
        type=int,
        default=0,
        help="Chunk the liquid Euler grid to bound peak memory (0 = no chunking).",
    )
    parser.add_argument(
        "--outer-batch-size",
        type=int,
        default=0,
        help="Chunk the NEGM outer durable search (0 = solve all nodes at once).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "table3_app2_housing.md"
    if not path.exists():
        path.write_text("# App.2 housing — Table 3 (EGM-FUES vs NEGM)\n\n" + _HEADER)

    for tau in args.taus:
        for ng in args.ngs:
            start = time.perf_counter()
            _logger.info("=== App.2 NG=%d tau=%s ===", ng, tau)
            try:
                row = _row(
                    ng=ng,
                    tau=tau,
                    liquid_batch_size=args.liquid_batch_size,
                    outer_batch_size=args.outer_batch_size,
                )
            except Exception as exc:
                row = f"| {ng} | {tau} | FAILED: {type(exc).__name__} | | | | | |\n"
                _logger.exception("NG=%d tau=%s FAILED", ng, tau)
            with path.open("a") as handle:
                handle.write(row)
            elapsed = time.perf_counter() - start
            _logger.info("NG=%d tau=%s done in %.1fs -> %s", ng, tau, elapsed, path)
    _logger.info("all rows done -> %s", path)


if __name__ == "__main__":
    main()
