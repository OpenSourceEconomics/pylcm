"""Stage 5B arm: one schedule per process on reduced3 ACA, unbudgeted.

Arms (one per process, so device and host peaks are the arm's own):

- `unblocked`: `ExecutionConfig(device_memory_bytes=None)`;
- `period_major`: Stage 3 blocked solve + Stage 5A grouped simulate;
- `block_major`: `invariant_block_schedule=BLOCK_MAJOR`.

Each arm runs, in order: combined `simulate()` cold, combined warm, then
`solve()` + `simulate(solution=...)` (split) warm. It records walls, compile
requests, phase ledgers, per-device `peak_bytes_in_use` after each call, host
peak RSS (getrusage), the block-major retention record (host bytes, D2H/H2D
bytes) and the sha256 of every published value's bytes and of the panel and
raw results. Compare arms with `compare_stage5b.py`.

    pixi run --frozen -e benchmarks-cuda12 python stage5b_lifetime.py \
        --arm block_major --out DIR --aca-slurm-src DIR [--fp32] [--n-subjects 4096]
"""

import argparse
import hashlib
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_arms as harness  # noqa: E402

_SEED = 20_260_903


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("unblocked", "period_major", "block_major"))
    parser.add_argument("--workload", default="reduced3", choices=("reduced3",))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aca-slurm-src", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--n-subjects", type=int, default=4096)
    parser.add_argument("--log-level", default="progress")
    args = parser.parse_args()

    args.out.mkdir(parents=True)
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    os.environ["JAX_COMPILATION_CACHE_DIR"] = f"/tmp/jax-cache-{job}-stage5b-{args.arm}"
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
        ["nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=(args.out / "nvml.csv").open("w"),
    )

    import dataclasses
    import gc
    import logging

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", val=x64)
    sys.path.insert(0, str(Path.cwd()))
    import lcm
    from benchmarks.asv._compile_counters import count_compile_requests
    from benchmarks.warm_solve_phases import parse_phase_records
    from lcm import InvariantBlockSchedule

    def configure(config):  # noqa: ANN001, ANN202
        widths = {} if args.arm == "unblocked" else {"pref_type": 1}
        schedule = (
            InvariantBlockSchedule.BLOCK_MAJOR
            if args.arm == "block_major"
            else InvariantBlockSchedule.PERIOD_MAJOR
        )
        return dataclasses.replace(
            config,
            invariant_block_widths=widths,
            invariant_block_schedule=schedule,
            device_memory_bytes=None,
        )

    harness._blocked = configure  # noqa: SLF001

    record: dict = {
        "argv": sys.argv,
        "arm": args.arm,
        "pylcm_sha": harness._run(["git", "rev-parse", "HEAD"]).strip(),  # noqa: SLF001
        "pylcm_dirty": harness._run(["git", "status", "--porcelain", "--untracked-files=no"]).strip(),  # noqa: SLF001
        "lcm_file": lcm.__file__,
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "devices": [f"{d} {d.device_kind}" for d in jax.devices()],
        "calls": [],
    }

    def dump() -> None:
        record["gpu_exclusivity"] = exclusivity.record()
        (args.out / "result.json").write_text(json.dumps(record, indent=1, default=str))

    lcm_logger = logging.getLogger("lcm")
    begin = time.perf_counter()
    try:
        model, params, initial, description = harness._build(  # noqa: SLF001
            workload=args.workload, aca_slurm_src=args.aca_slurm_src, n_subjects=args.n_subjects
        )
    except Exception as error:  # noqa: BLE001 - a refusal is a recorded outcome
        record["construction_error"] = f"{type(error).__name__}: {error}"
        dump()
        sampler.terminate()
        raise
    record["construction_seconds"] = time.perf_counter() - begin
    record["model"] = description
    dump()

    def value_digests(solution) -> dict:  # noqa: ANN001
        # One value at a time, so the digest never holds every value at once.
        return {
            f"{period}/{regime}": hashlib.sha256(np.asarray(solution.value(period=period, regime=regime)).tobytes()).hexdigest()
            for period in solution.values
            for regime in solution.values[period]
        }

    def call(label: str, function) -> object:  # noqa: ANN001
        lines = harness._Lines()  # noqa: SLF001
        lcm_logger.addHandler(lines)
        gc.collect()
        entry: dict = {"label": label}
        with count_compile_requests() as counts:
            begin = time.perf_counter()
            result = function()
            for array in jax.live_arrays():
                array.block_until_ready()
            entry["wall_seconds"] = time.perf_counter() - begin
        lcm_logger.removeHandler(lines)
        entry["compile_requests"] = dataclasses.asdict(counts)
        entry["phases"] = [
            {"call_id": c.call_id, "phases": [dataclasses.asdict(p) for p in c.phases]}
            for c in parse_phase_records(lines=lines.lines)
        ]
        entry["memory_stats_after"] = [d.memory_stats() for d in jax.local_devices()]
        entry["host_maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        solution = getattr(result, "solution", result)
        try:
            raw = solution.values._raw(period=next(iter(solution.values)), regime=next(iter(solution.values[next(iter(solution.values))])))  # noqa: SLF001
            owner = getattr(raw, "owner", None)
            entry["retention_record"] = None if owner is None else json.loads(owner.retention_record().to_json())
        except Exception as error:  # noqa: BLE001
            entry["retention_record"] = f"unavailable: {error}"
        if hasattr(result, "to_dataframe"):
            frame = result.to_dataframe()
            entry["panel_sha256"] = hashlib.sha256(pickle.dumps(frame, protocol=5)).hexdigest()
            entry["raw_sha256"] = harness._sha256_of(result.raw_results)  # noqa: SLF001
            entry["plan_summary"] = None if result.plan_summary is None else result.plan_summary.summary()
        entry["value_sha256"] = value_digests(solution)
        record["calls"].append(entry)
        dump()
        print(label, entry["wall_seconds"], flush=True)
        return result

    for label in ("combined_cold", "combined_warm"):
        result = call(label, lambda: model.simulate(params=params, initial_conditions=initial, seed=_SEED, log_level=args.log_level))
        del result
    solution = call("solve_warm", lambda: model.solve(params=params, log_level=args.log_level))
    result = call("split_simulate_warm", lambda: model.simulate(params=params, initial_conditions=initial, seed=_SEED, solution=solution, log_level=args.log_level))
    del result, solution
    sampler.terminate()
    dump()


if __name__ == "__main__":
    main()
