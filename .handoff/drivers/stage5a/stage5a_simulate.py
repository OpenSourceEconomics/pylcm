"""Stage 5A arms: grouped vs ungrouped simulate on reduced3 ACA, cold and warm.

Reuses the Stage 3 harness `stage3_arms.py` (a byte copy beside this file) for the
model build, GPU exclusivity probe, allocator stats, compile counters and phase
ledgers. One process builds two models from the same inputs:

- `ungrouped`: `ExecutionConfig()`, so simulate runs today's route;
- `grouped`: `invariant_block_widths={"pref_type": 1}`, so simulate groups subjects
  by `pref_type` when the simulate phase is certified invariant.

The ungrouped model solves once; the solution is archived and reloaded, so both arms
simulate from the identical input solution. Each arm simulates cold, then warm, in
the order given by `--order`. Parity: every arm/call must produce the same panel
(pickle sha256 of `to_dataframe()`) and the same raw results (sha256 over every
leaf's bytes). With `--end-to-end`, the grouped model also solves (blocked) and
simulates its own solution, which must match the ungrouped panel too.

Run from the pylcm checkout root under test:

    pixi run --frozen -e benchmarks-cuda12 python \
        .task-evidence/invariant-state/stage5a/drivers/stage5a_simulate.py \
        --out DIR --aca-slurm-src DIR [--fp32] [--n-subjects 4096] \
        [--order grouped-first|ungrouped-first] [--end-to-end]

Exit status is 1 when any parity check fails.
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_arms as harness  # noqa: E402

_SEED = 20_260_903


def _build(*, grouped: bool, args: argparse.Namespace):  # noqa: ANN202
    harness._BLOCK_WIDTHS.clear()  # noqa: SLF001
    if grouped:
        harness._BLOCK_WIDTHS["pref_type"] = 1  # noqa: SLF001
    try:
        return harness._build(  # noqa: SLF001
            workload=args.workload,
            aca_slurm_src=args.aca_slurm_src,
            n_subjects=args.n_subjects,
        )
    finally:
        harness._BLOCK_WIDTHS.clear()  # noqa: SLF001


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="reduced3", choices=("reduced3",))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aca-slurm-src", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--n-subjects", type=int, default=4096)
    parser.add_argument(
        "--order", default="grouped-first", choices=("grouped-first", "ungrouped-first")
    )
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--log-level", default="progress")
    args = parser.parse_args()

    args.out.mkdir(parents=True)
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    os.environ["JAX_COMPILATION_CACHE_DIR"] = f"/tmp/jax-cache-{job}-stage5a"
    x64 = not args.fp32
    os.environ["JAX_ENABLE_X64"] = "1" if x64 else "0"
    os.environ["ACA_JAX_ENABLE_X64"] = "1" if x64 else "0"

    harness._OWN_UUIDS.update(  # noqa: SLF001
        harness._run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).split()  # noqa: SLF001
    )
    exclusivity = harness._Exclusivity()  # noqa: SLF001
    if exclusivity.at_start:
        raise SystemExit(f"GPU not exclusive at start: {exclusivity.at_start}")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-lms",
            "200",
        ],
        stdout=(args.out / "nvml.csv").open("w"),
    )

    import dataclasses
    import gc
    import logging

    import jax

    jax.config.update("jax_enable_x64", val=x64)
    sys.path.insert(0, str(Path.cwd()))
    import lcm
    from benchmarks.asv._compile_counters import count_compile_requests
    from benchmarks.warm_solve_phases import parse_phase_records
    from lcm.solver_api import ResultRetention

    def git(*cmd: str) -> str:
        return harness._run(["git", *cmd]).strip()  # noqa: SLF001

    record: dict = {
        "argv": sys.argv,
        "pylcm_sha": git("rev-parse", "HEAD"),
        "pylcm_dirty": git("status", "--porcelain", "--untracked-files=no"),
        "lcm_file": lcm.__file__,
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "devices": [f"{d} {d.device_kind}" for d in jax.devices()],
        "simulation_seed": _SEED,
        "order": args.order,
        "calls": [],
        "parity": {},
    }

    def dump() -> None:
        record["gpu_exclusivity"] = exclusivity.record()
        (args.out / "result.json").write_text(json.dumps(record, indent=1, default=str))

    print("lcm.__file__", lcm.__file__, flush=True)
    dump()
    lcm_logger = logging.getLogger("lcm")

    start = time.perf_counter()
    ungrouped, params, initial, description = _build(grouped=False, args=args)
    grouped, grouped_params, _, grouped_description = _build(grouped=True, args=args)
    record["construction_seconds"] = time.perf_counter() - start
    record["model"] = {"ungrouped": description, "grouped": grouped_description}
    record["params_sha256"] = harness._sha256_of(params)  # noqa: SLF001
    if harness._sha256_of(grouped_params) != record["params_sha256"]:  # noqa: SLF001
        raise SystemExit("the two builds disagree on params")
    dump()

    begin = time.perf_counter()
    archive = ungrouped.solve(
        params=params, log_level=args.log_level, retention=ResultRetention.VALUES
    ).save(path=args.out / "solution")
    record["solve_and_save_seconds"] = time.perf_counter() - begin
    solution = lcm.load_solution(path=archive)
    dump()

    def simulate(*, label: str, model, solution) -> None:  # noqa: ANN001
        lines = harness._Lines()  # noqa: SLF001
        lcm_logger.addHandler(lines)
        entry: dict = {"label": label}
        gc.collect()
        with count_compile_requests() as counts:
            begin = time.perf_counter()
            result = model.simulate(
                params=params,
                initial_conditions=initial,
                seed=_SEED,
                solution=solution,
                log_level=args.log_level,
            )
            for array in jax.live_arrays():
                array.block_until_ready()
            entry["wall_seconds"] = time.perf_counter() - begin
        lcm_logger.removeHandler(lines)
        entry["compile_requests"] = dataclasses.asdict(counts)
        entry["phases"] = [
            {"call_id": c.call_id, "phases": [dataclasses.asdict(p) for p in c.phases]}
            for c in parse_phase_records(lines=lines.lines)
        ]
        summary = result.plan_summary
        entry["subject_grouping"] = None if summary is None else summary.subject_grouping
        entry["plan_summary"] = None if summary is None else summary.summary()
        entry["memory_stats_after"] = [d.memory_stats() for d in jax.local_devices()]
        frame = result.to_dataframe()
        entry["panel_rows"] = len(frame)
        entry["panel_sha256"] = hashlib.sha256(pickle.dumps(frame, protocol=5)).hexdigest()
        entry["raw_sha256"] = harness._sha256_of(result.raw_results)  # noqa: SLF001
        record["calls"].append(entry)
        dump()
        print(label, json.dumps({k: entry[k] for k in ("wall_seconds", "subject_grouping", "panel_sha256")}), flush=True)
        del result, frame

    arms = {"grouped": grouped, "ungrouped": ungrouped}
    order = ("grouped", "ungrouped") if args.order == "grouped-first" else ("ungrouped", "grouped")
    for arm in order:
        for call in ("cold", "warm"):
            simulate(label=f"{arm}_simulate_{call}", model=arms[arm], solution=solution)

    if args.end_to_end:
        del solution
        gc.collect()
        begin = time.perf_counter()
        blocked = grouped.solve(
            params=params, log_level=args.log_level, retention=ResultRetention.VALUES
        )
        record["blocked_solve_seconds"] = time.perf_counter() - begin
        simulate(label="end_to_end_grouped", model=grouped, solution=blocked)

    calls = {entry["label"]: entry for entry in record["calls"]}
    reference = calls["ungrouped_simulate_cold"]
    record["parity"] = {
        label: {
            "panel": entry["panel_sha256"] == reference["panel_sha256"],
            "raw": entry["raw_sha256"] == reference["raw_sha256"],
        }
        for label, entry in calls.items()
    }
    record["grouping_as_expected"] = all(
        (entry["subject_grouping"] == "pref_type") == label.startswith(("grouped", "end_to_end"))
        for label, entry in calls.items()
    )
    sampler.terminate()
    dump()
    ok = record["grouping_as_expected"] and all(
        check["panel"] and check["raw"] for check in record["parity"].values()
    )
    print("parity", "ok" if ok else "FAILED", json.dumps(record["parity"]), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
