"""Produce the paper-scale DS-2026 comparison tables in one run.

Every model, method, oracle, and harness for Dobrescu & Shanker (2026) is built
and CPU-validated at small grids. The *paper-scale* table numbers (App.1 grids up
to 10000, App.2 NG up to 1000, App.3 W up to 4000) exceed local memory, so this
script runs the full sweeps on a CUDA-capable GPU.

Run with the CUDA env, e.g.

    XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        pixi run --environment tests-cuda12 \\
        python -m benchmarks.ds_replication.run_paper_scale --out /tmp/ds_tables

`--smoke` swaps the paper grids for tiny ones so the whole pipeline can be
validated end-to-end on CPU in a couple of minutes before the real run. `--tables`
selects a subset. Each table is written to its own markdown file under `--out` as
soon as it finishes, so a long run leaves partial results on interruption.

Timing is always compile-vs-runtime separated (the timing helpers clear the JAX
cache before the compile measurement).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from benchmarks.ds_replication.app1_retirement_accuracy import (
    PAPER_GRIDS,
    PAPER_TAUS,
    app1_accuracy_table,
    app1_timing_table,
)
from benchmarks.ds_replication.app2_fues_accuracy import (
    app2_fues_euler_error,
    app2_fues_timing,
)
from benchmarks.ds_replication.app2_housing_accuracy import (
    app2_negm_euler_error,
    app2_negm_timing,
)
from benchmarks.ds_replication.app3_discrete_housing_accuracy import (
    app3_timing,
    app3_vfi_euler_error,
)

_logger = logging.getLogger("lcm")

# Paper grids per application (exceed local memory; require a CUDA-capable GPU).
APP2_NG = (250, 500, 750, 1000)
APP2_TAUS = (0.05, 0.07, 0.12)
APP3_W = (1000, 2000, 4000)
APP3_H = (3, 5, 7)

# Tiny stand-ins for `--smoke` end-to-end validation on CPU.
SMOKE = {
    "app1_grids": (200, 400),
    "app1_taus": (1.0,),
    "app2_ng": (10, 14),
    "app2_taus": (0.07,),
    "app3_w": (30, 40),
    "app3_h": (3,),
}


def _write(out: Path, name: str, text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.md"
    path.write_text(text)
    _logger.info("wrote %s", path)


def table12_app1(*, out: Path, smoke: bool) -> None:
    """App.1 retirement Tables 1 (timing) + 2 (accuracy): FUES/RFC/LTM/MSS."""
    grids = SMOKE["app1_grids"] if smoke else PAPER_GRIDS
    taus = SMOKE["app1_taus"] if smoke else PAPER_TAUS
    methods = ("fues", "rfc", "ltm", "mss")
    accuracy = pd.concat(
        app1_accuracy_table(upper_envelope=ue, taus=taus, n_grids=grids)
        .rename(columns={f"{ue}_euler_error": "euler_error"})
        .assign(method=ue)
        for ue in methods
    )
    timing = app1_timing_table(upper_envelopes=methods, taus=(taus[0],), n_grids=grids)
    _write(
        out,
        "table1_2_app1_retirement",
        "# App.1 retirement — Table 2 (accuracy) + Table 1 (timing)\n\n"
        "## Table 2 — mean log10 Euler error\n\n"
        + accuracy.to_markdown(index=False)
        + "\n\n## Table 1 — compile vs runtime (s)\n\n"
        + timing.to_markdown(index=False)
        + "\n",
    )


def table3_app2(*, out: Path, smoke: bool) -> None:
    """App.2 housing Table 3: EGM-FUES vs NEGM, Euler + timing, over NG x tau."""
    ngs = SMOKE["app2_ng"] if smoke else APP2_NG
    taus = SMOKE["app2_taus"] if smoke else APP2_TAUS
    rows = []
    for tau in taus:
        for ng in ngs:
            fues_t = app2_fues_timing(n_grid=ng, tau=tau)
            negm_t = app2_negm_timing(n_grid=ng, tau=tau)
            rows.append(
                {
                    "NG": ng,
                    "tau": tau,
                    "FUES_eerr": app2_fues_euler_error(n_grid=ng, tau=tau),
                    "NEGM_eerr": app2_negm_euler_error(n_grid=ng, tau=tau),
                    "FUES_compile": fues_t["compile_time"],
                    "FUES_run": fues_t["runtime"],
                    "NEGM_compile": negm_t["compile_time"],
                    "NEGM_run": negm_t["runtime"],
                }
            )
    _write(
        out,
        "table3_app2_housing",
        "# App.2 housing — Table 3 (EGM-FUES vs NEGM)\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n",
    )


def table5_app3_taxed(*, out: Path, smoke: bool) -> None:
    """App.3 discrete housing with taxes Table 5: FUES vs VFI, Euler + timing."""
    ws = SMOKE["app3_w"] if smoke else APP3_W
    hs = SMOKE["app3_h"] if smoke else APP3_H
    n_periods = 6 if smoke else 20
    rows = []
    for h in hs:
        for w in ws:
            common = {
                "n_assets": w,
                "n_wage_nodes": h,
                "n_periods": n_periods,
                "use_taxes": True,
            }
            fues_t = app3_timing(variant="dcegm", upper_envelope="fues", **common)
            vfi_t = app3_timing(variant="brute", **common)
            rows.append(
                {
                    "W": w,
                    "H": h,
                    "FUES_eerr": app3_vfi_euler_error(
                        variant="dcegm", upper_envelope="fues", **common
                    ),
                    "VFI_eerr": app3_vfi_euler_error(variant="brute", **common),
                    "FUES_compile": fues_t["compile_time"],
                    "FUES_run": fues_t["runtime"],
                    "VFI_compile": vfi_t["compile_time"],
                    "VFI_run": vfi_t["runtime"],
                }
            )
    _write(
        out,
        "table5_app3_taxed",
        "# App.3 discrete housing with taxes — Table 5 (FUES vs VFI)\n\n"
        + pd.DataFrame(rows).to_markdown(index=False)
        + "\n",
    )


TABLES = {
    "table12": table12_app1,
    "table3": table3_app2,
    "table5": table5_app3_taxed,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ds_paper_tables"))
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=sorted(TABLES),
        default=sorted(TABLES),
        help="Which tables to produce (default: all).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny grids to validate the pipeline end-to-end on CPU.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for name in args.tables:
        _logger.info("=== producing %s ===", name)
        TABLES[name](out=args.out, smoke=args.smoke)
    _logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
