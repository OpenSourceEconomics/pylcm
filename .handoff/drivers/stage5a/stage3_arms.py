"""Stage 0 baseline arms: construction, cold/warm solve, simulate and output I/O.

Adapted from the production ACA harness `~/dense-aca/production6/dense_arms_aca.py`
(the #482 production-pair facility): same model build, exclusivity probe, NVML
sampler and per-period allocator stats; added construction timing, repeated warm
calls, warm simulate, output I/O, phase ledgers parsed by the checkout's own
`benchmarks.warm_solve_phases.parse_phase_records`, compile-request counters from
`benchmarks.asv._compile_counters`, and an environment record.

Run from a pylcm checkout root (its `src/` and `benchmarks/` are under test):

    pixi run --frozen -e benchmarks-cuda12 python stage0_arms.py \
        --workload production|reduced2|reduced3 --arm A1 --out DIR [--fp32] \
        [--warm-same 2] [--simulate] [--aca-slurm-src DIR] \
        [--log-level progress|debug] [--profile-ages 64,63,62] [--calls cold]

`--profile-ages` opens a CUDA profiler capture range (for `nsys --capture-range=
cudaProfilerApi`) while the solve announces one of those ages; it is for
diagnostic runs, never timed ones.
"""

import argparse
import contextlib
import ctypes
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

_CHANGED_PARAM = "bequest_shifter"
_JAX_FLOOR = (0, 11, 1)
_OWN_UUIDS: set[str] = set()
# Stage 3: set by --invariant-blocking; applied to every arm's ExecutionConfig.
_BLOCK_WIDTHS: dict[str, int] = {}
_ACTION_PARTITIONS: dict[str, int] = {}
_SHARDED_STATES: tuple[str, ...] | None = None


def _blocked(config):  # noqa: ANN001, ANN202
    overrides = {}
    if _BLOCK_WIDTHS:
        overrides["invariant_block_widths"] = dict(_BLOCK_WIDTHS)
    if _ACTION_PARTITIONS:
        overrides["action_partitions"] = dict(_ACTION_PARTITIONS)
    if _SHARDED_STATES is not None:
        overrides["sharded_states"] = _SHARDED_STATES
    return dataclasses.replace(config, **overrides) if overrides else config


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except FileNotFoundError as error:
        return f"unavailable: {error}"


def _foreign() -> dict[str, str]:
    found: dict[str, str] = {}
    apps = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"]
    )
    for line in apps.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        pid, name, uuid = parts
        if pid and int(pid) != os.getpid() and uuid in _OWN_UUIDS:
            found[f"gpu:{pid}"] = name
    for line in _run(["pgrep", "-a", "-x", "Runner.Worker"]).splitlines():
        pid, _, cmd = line.partition(" ")
        found[f"runner-job:{pid}"] = cmd[:120]
    return found


class _Exclusivity:
    def __init__(self) -> None:
        self.at_start = _foreign()
        self.seen: dict[str, str] = {}
        self.polls = 0
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while True:
            self.seen.update(_foreign())
            self.polls += 1
            time.sleep(2)

    def record(self) -> dict:
        return {
            "probe": "nvidia-smi compute apps != own pid on own GPU UUIDs; pgrep -x Runner.Worker",
            "foreign_at_start": self.at_start,
            "foreign_seen_during_run": dict(self.seen),
            "polls": self.polls,
            "exclusive": not self.at_start and not self.seen,
        }


class _MemStats(logging.Handler):
    """Append per-device allocator stats at every age header and period finish."""

    _KEYS = (
        "bytes_in_use",
        "peak_bytes_in_use",
        "largest_free_block_bytes",
        "bytes_reserved",
        "pool_bytes",
        "bytes_limit",
        "num_allocs",
    )

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.INFO)
        self.path = path
        self.label = ""

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not (message.startswith("Age ") or "finished in" in message):
            return
        import jax

        stats = [
            {key: (d.memory_stats() or {}).get(key) for key in self._KEYS}
            for d in jax.local_devices()
        ]
        with self.path.open("a") as handle:
            handle.write(
                json.dumps(
                    {"t": time.time(), "call": self.label, "msg": message.strip(), "devices": stats}
                )
                + "\n"
            )


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []
        self.plan_records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())
        plan = getattr(record, "core_plan_record", None)
        if plan is not None:
            self.plan_records.append(plan.to_json())


_AGE_HEADER = re.compile(r"^Age (?P<age>[0-9]+(?:\.[0-9]+)?) \([0-9]+ regimes\):$")


class _AgeWindow(logging.Filter):
    """cudaProfilerStart on the first in-window age header, Stop on the first after it."""

    def __init__(self, *, ages: frozenset[float]) -> None:
        super().__init__()
        self.ages = ages
        self.opened = self.closed = False
        self.marks: list[tuple[float, float]] = []
        self.cudart = _load_cudart()

    def filter(self, record: logging.LogRecord) -> bool:
        match = _AGE_HEADER.match(record.getMessage().strip())
        if match is None or self.closed:
            return True
        age = float(match["age"])
        if age in self.ages:
            if not self.opened:
                self.opened = True
                self.cudart.cudaProfilerStart()
            self.marks.append((age, time.time()))
        elif self.opened:
            self.close()
        return True

    def close(self) -> None:
        if self.opened and not self.closed:
            import jax

            for array in jax.live_arrays():
                array.block_until_ready()
            self.cudart.cudaProfilerStop()
            self.closed = True
            self.marks.append((-1.0, time.time()))


def _load_cudart() -> ctypes.CDLL:
    import nvidia.cuda_runtime as runtime  # the cuda12 env's runtime wheel

    lib_dir = Path(next(iter(runtime.__path__))) / "lib"
    candidates = sorted(lib_dir.glob("libcudart.so*"))
    if not candidates:
        raise SystemExit(f"no libcudart in {lib_dir}")
    return ctypes.CDLL(str(candidates[0]))


def _sha256_of(obj: object) -> str:
    import jax
    import numpy as np

    leaves = []

    def visit(path, leaf):  # noqa: ANN001, ANN202
        leaves.append((jax.tree_util.keystr(path), np.asarray(leaf).tobytes() if hasattr(leaf, "shape") or isinstance(leaf, (int, float)) else repr(leaf).encode()))
        return leaf

    jax.tree_util.tree_map_with_path(visit, obj)
    digest = hashlib.sha256()
    for key, payload in sorted(leaves):
        digest.update(key.encode())
        digest.update(payload)
    return digest.hexdigest()


def _scaled(tree, factor: float):  # noqa: ANN001, ANN202
    import jax

    hits = 0

    def scale(path, leaf):  # noqa: ANN001, ANN202
        nonlocal hits
        if any(getattr(key, "key", None) == _CHANGED_PARAM for key in path):
            hits += 1
            return leaf * factor
        return leaf

    out = jax.tree_util.tree_map_with_path(scale, tree)
    if hits == 0:
        raise SystemExit(f"no leaf named {_CHANGED_PARAM}")
    return out


def _build(*, workload: str, aca_slurm_src: Path | None, n_subjects: int):  # noqa: ANN202
    """Return (model, params, initial_conditions, description)."""
    if workload == "production":
        sys.path.insert(0, str(aca_slurm_src))
        from aca_model.aca.health_insurance import PolicyVariant
        from aca_slurm._simulate import (
            _load_inputs,
            _production_input_paths,
            build_aca_policy_model,
        )
        from aca_slurm._type_prediction import (
            replicate_for_draws,
            triple_initdist_by_pref_type,
        )
        from aca_slurm.config import (
            _GRID_CONFIG_BY_GPU,
            N_DRAWS_PER_INDIVIDUAL,
            make_execution_config,
        )

        config = _blocked(
            make_execution_config(solver="brute_force", continuous_sharding=True)
        )
        grid_config = _GRID_CONFIG_BY_GPU["nvidia_a100_sxm4_80gb"]
        model, params = build_aca_policy_model(
            policy=PolicyVariant.ACA,
            grid_config=grid_config,
            solver="brute_force",
            execution_config=config,
        )
        inputs = _load_inputs(**_production_input_paths())
        initial = replicate_for_draws(
            triple_initdist_by_pref_type(inputs.initdist_df),
            n_draws=N_DRAWS_PER_INDIVIDUAL,
        )
        return model, params, initial, {
            "builder": "aca_slurm._simulate.build_aca_policy_model(PolicyVariant.ACA, brute_force)",
            "grid_config": repr(grid_config),
            "execution_config": repr(config),
            "action_partitions": dict(config.action_partitions),
            "sharded_states": list(config.sharded_states),
            "initial_conditions": f"triple_initdist_by_pref_type + replicate_for_draws(n_draws={N_DRAWS_PER_INDIVIDUAL})",
            "pref_types": 3,
        }
    from aca_model.agent.preferences import BenchmarkPrefType, PrefType
    from aca_model.benchmark import (
        create_benchmark_model,
        get_benchmark_initial_conditions,
        get_benchmark_params,
    )
    from aca_model.config import BENCHMARK_GRID_CONFIG

    from lcm import DiscreteGrid, ExecutionConfig

    category = {"reduced2": BenchmarkPrefType, "reduced3": PrefType}[workload]
    config = _blocked(ExecutionConfig())
    if workload == "reduced2":
        model = create_benchmark_model(
            pref_type_grid=DiscreteGrid(category_class=category), execution_config=config
        )
    else:
        # `create_benchmark_model` with its derived categoricals' `pref_type`
        # switched to the 3-type grid; everything else is the benchmark's.
        from aca_model import benchmark as aca_benchmark

        fixed_params, wage_params, _ = get_benchmark_params(model=None)
        model = aca_benchmark.create_model(
            grid_config=BENCHMARK_GRID_CONFIG,
            fixed_params=fixed_params,
            wage_params=wage_params,
            derived_categoricals={
                **aca_benchmark._DERIVED_CATEGORICALS,
                "pref_type": DiscreteGrid(category_class=category),
            },
            pref_type_grid=DiscreteGrid(category_class=category),
            execution_config=config,
        )
    params = get_benchmark_params(model=model)[2]
    initial = get_benchmark_initial_conditions(model=model, n_subjects=n_subjects, seed=0)
    substituted: dict = {}
    if workload == "reduced3":
        # The frozen benchmark snapshot carries 2-entry preference-type vectors; a
        # 3-type grid would index past them (JAX clamps silently). Take the three
        # pref-type-indexed vectors from the production aca-data assembly instead.
        import jax.numpy as jnp
        import numpy as np

        sys.path.insert(0, str(aca_slurm_src))
        from aca_slurm._assemble_params import assemble_params
        from aca_slurm._simulate import _load_inputs, _production_input_paths

        inputs = _load_inputs(**_production_input_paths())
        production = assemble_params(
            pref_params=inputs.pref, base_wage_profile=inputs.wage["log_ft_wage_base"]
        )
        params = dict(params)
        for key in ("consumption_weights", "discount_factor_by_type", "coefficients_rra"):
            if np.asarray(params[key]).shape != (2,):
                raise SystemExit(f"benchmark {key} is not a 2-type vector: {params[key]!r}")
            vector = np.asarray(production[key], dtype=np.asarray(params[key]).dtype)
            if vector.shape != (3,):
                raise SystemExit(f"production {key} is not a 3-type vector: {vector!r}")
            params[key] = jnp.asarray(vector)
            substituted[key] = vector.tolist()
        rng = np.random.default_rng(seed=1)
        initial = dict(initial)
        initial["pref_type"] = jnp.asarray(rng.integers(0, 3, n_subjects).astype(np.int32))
    return model, params, initial, {
        "substituted_pref_type_params": substituted,
        "builder": f"aca_model.benchmark: create_model(BENCHMARK_GRID_CONFIG, benchmark fixed/wage params, pref_type=DiscreteGrid({category.__name__}))",
        "grid_config": repr(BENCHMARK_GRID_CONFIG),
        "execution_config": repr(config),
        "action_partitions": dict(config.action_partitions),
        "sharded_states": list(config.sharded_states),
        "initial_conditions": f"get_benchmark_initial_conditions(n_subjects={n_subjects}, seed=0)",
        "pref_types": len(DiscreteGrid(category_class=category).categories),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark workload and execution-layout controls."""
    global _SHARDED_STATES  # noqa: PLW0603
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, choices=("production", "reduced2", "reduced3"))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--warm-same", type=int, default=2)
    parser.add_argument("--no-warm-changed", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--skip-simulate", action="store_true")
    parser.add_argument("--n-subjects", type=int, default=4096)
    parser.add_argument("--aca-slurm-src", type=Path)
    parser.add_argument("--log-level", default="progress")
    parser.add_argument("--profile-ages", default="")
    parser.add_argument("--profile-call", default="warm_same_1")
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument(
        "--invariant-blocking",
        action="store_true",
        help="ExecutionConfig(invariant_block_widths={'pref_type': 1}) (Stage 3 arm)",
    )
    parser.add_argument(
        "--action-partitions", action="append", default=[], metavar="REGIME=COUNT",
        help="Share a named regime's actions over COUNT devices; repeat per regime",
    )
    parser.add_argument(
        "--sharded-states", nargs="*", default=None, metavar="STATE",
        help="Override state sharding; give no names for an action-only layout",
    )
    args = parser.parse_args(argv)
    _BLOCK_WIDTHS.clear()
    if args.invariant_blocking:
        _BLOCK_WIDTHS["pref_type"] = 1
    _ACTION_PARTITIONS.clear()
    for request in args.action_partitions:
        try:
            name, raw_count = request.split("=")
            count = int(raw_count)
        except ValueError:
            parser.error(f"--action-partitions expects REGIME=COUNT, got {request!r}")
        if not name.strip() or count < 1:
            parser.error(
                f"--action-partitions requires a regime and positive count: {request!r}"
            )
        if name in _ACTION_PARTITIONS:
            parser.error(f"--action-partitions repeats regime {name!r}")
        _ACTION_PARTITIONS[name] = count
    _SHARDED_STATES = (
        None if args.sharded_states is None else tuple(args.sharded_states)
    )
    if _SHARDED_STATES is not None and (
        any(not name.strip() for name in _SHARDED_STATES)
        or len(set(_SHARDED_STATES)) != len(_SHARDED_STATES)
    ):
        parser.error("--sharded-states needs distinct non-empty state names")
    return args


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = parse_args()

    args.out.mkdir(parents=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = f"/tmp/jax-cache-{os.environ.get('SLURM_JOB_ID', os.getpid())}-{args.arm}"
    x64 = not args.fp32
    os.environ["JAX_ENABLE_X64"] = "1" if x64 else "0"
    os.environ["ACA_JAX_ENABLE_X64"] = "1" if x64 else "0"

    _OWN_UUIDS.update(_run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).split())
    exclusivity = _Exclusivity()
    if exclusivity.at_start:
        raise SystemExit(f"GPU not exclusive at start: {exclusivity.at_start}")
    sampler = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=(args.out / "nvml.csv").open("w"),
    )

    import jax
    import jaxlib
    import numpy as np

    jax.config.update("jax_enable_x64", val=x64)
    if tuple(int(part) for part in jax.__version__.split(".")[:3]) < _JAX_FLOOR:
        raise SystemExit(f"jax {jax.__version__} is below the pyproject floor {_JAX_FLOOR}")
    sys.path.insert(0, str(Path.cwd()))
    import aca_model

    import lcm
    from benchmarks.asv._compile_counters import count_compile_requests
    from benchmarks.warm_solve_phases import parse_phase_records
    from lcm.solver_api import ResultRetention

    if bool(jax.config.jax_enable_x64) != x64:
        raise SystemExit(f"x64 is {jax.config.jax_enable_x64} after importing aca_model, expected {x64}")

    def git(*cmd: str) -> str:
        return _run(["git", *cmd]).strip()

    def dist_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    record: dict = {
        "arm": args.arm,
        "invariant_block_widths": dict(_BLOCK_WIDTHS),
        "action_partitions": dict(_ACTION_PARTITIONS),
        "workload": args.workload,
        "argv": sys.argv,
        "pylcm_sha": git("rev-parse", "HEAD"),
        "pylcm_dirty": git("status", "--porcelain", "--untracked-files=no"),
        "lcm_file": lcm.__file__,
        "aca_model_file": aca_model.__file__,
        "aca_model_direct_url": importlib.metadata.distribution("aca-model").read_text("direct_url.json"),
        "aca_slurm_src": None if args.aca_slurm_src is None else str(args.aca_slurm_src),
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jax_cuda12_plugin": dist_version("jax-cuda12-plugin"),
        "jax_cuda12_pjrt": dist_version("jax-cuda12-pjrt"),
        "jaxlib_git_revision": getattr(jaxlib.version, "__git_version__", None) if hasattr(jaxlib, "version") else None,
        "x64": bool(jax.config.jax_enable_x64),
        "devices": [f"{d} {d.device_kind}" for d in jax.devices()],
        "env": {
            key: os.environ.get(key)
            for key in (
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "TF_GPU_ALLOCATOR",
                "JAX_COMPILATION_CACHE_DIR",
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_PARTITION",
                "LCM_CAPTURE_PERIOD",
                "LCM_CAPTURE_DIR",
            )
        },
        "gpus": _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.applications.graphics", "--format=csv,noheader"]).strip(),
        "nvidia_smi_header": _run(["nvidia-smi"]).splitlines()[:4],
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "log_level": args.log_level,
        "retention": "ResultRetention.VALUES",
        "calls": [],
    }

    def dump() -> None:
        record["gpu_exclusivity"] = exclusivity.record()
        (args.out / "result.json").write_text(json.dumps(record, indent=1, default=str))

    dump()
    lcm_logger = logging.getLogger("lcm")
    memstats = _MemStats(args.out / "memstats.jsonl")
    lcm_logger.addHandler(memstats)

    start = time.perf_counter()
    model, params, initial_conditions, description = _build(
        workload=args.workload, aca_slurm_src=args.aca_slurm_src, n_subjects=args.n_subjects
    )
    record["construction_seconds"] = time.perf_counter() - start
    record["model"] = description
    record["action_partitions"] = description["action_partitions"]
    record["sharded_states"] = description["sharded_states"]
    record["params_sha256"] = _sha256_of(params)
    record["simulation_seed"] = 20_260_903
    dump()

    window = None
    if args.profile_ages:
        window = _AgeWindow(ages=frozenset(float(a) for a in args.profile_ages.split(",")))

    @contextlib.contextmanager
    def call(label: str):  # noqa: ANN202
        handler = logging.FileHandler(args.out / f"{label}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lines = _Lines()
        lcm_logger.addHandler(handler)
        lcm_logger.addHandler(lines)
        memstats.label = label
        entry: dict = {"label": label}
        gc.collect()
        with count_compile_requests() as counts:
            begin = time.perf_counter()
            try:
                yield entry
            finally:
                for array in jax.live_arrays():
                    array.block_until_ready()
                entry["wall_seconds"] = time.perf_counter() - begin
        lcm_logger.removeHandler(handler)
        lcm_logger.removeHandler(lines)
        handler.close()
        entry["compile_requests"] = dataclasses.asdict(counts)
        entry["phases"] = [
            {"call_id": c.call_id, "phases": [dataclasses.asdict(p) for p in c.phases]}
            for c in parse_phase_records(lines=lines.lines)
        ]
        entry["memory_stats_after"] = [d.memory_stats() for d in jax.local_devices()]
        if lines.plan_records:
            (args.out / f"{label}.plan_records.jsonl").write_text("\n".join(lines.plan_records) + "\n")
            entry["plan_records"] = len(lines.plan_records)
        record["calls"].append(entry)
        dump()

    def save_v(values, name: str) -> None:  # noqa: ANN001
        np.savez(
            args.out / f"V-{name}.npz",
            **{f"{period}__{regime}": np.asarray(array) for period in values.keys() for regime, array in values[period].items()},  # noqa: SIM118
        )

    labels = ["cold"] + ([] if args.cold_only else [f"warm_same_{i}" for i in range(1, args.warm_same + 1)])
    solution = None
    for label in labels:
        profiled = window is not None and label == args.profile_call
        if profiled:
            lcm_logger.addFilter(window)
        with call(label) as entry:
            solution = model.solve(params=params, log_level=args.log_level, retention=ResultRetention.VALUES)
        if profiled:
            window.close()
            lcm_logger.removeFilter(window)
        save_start = time.perf_counter()
        if label in {"cold", "warm_same_1"}:
            save_v(solution.values, label)
        entry["value_dtypes"] = sorted({str(np.asarray(a).dtype) for p in solution.values.keys() for a in solution.values[p].values()})  # noqa: SIM118
        entry["v_savez_seconds"] = time.perf_counter() - save_start
        dump()
        if label != labels[-1]:
            del solution
            solution = None
    if window is not None:
        window.close()
        record["profile_window_marks"] = window.marks
        dump()

    if not args.cold_only and not args.no_warm_changed:
        changed = _scaled(params, 1.01)
        with call("warm_changed"):
            changed_solution = model.solve(params=changed, log_level=args.log_level, retention=ResultRetention.VALUES)
        save_v(changed_solution.values, "warm_changed")
        del changed_solution

    if args.simulate and not args.skip_simulate and solution is not None:
        for label in ("simulate_cold", "simulate_warm"):
            with call(label) as entry:
                if args.workload == "production":
                    from aca_model.simulation import simulate_with_dense_index

                    result, _ids = simulate_with_dense_index(
                        model=model,
                        initial_conditions=initial_conditions,
                        params=params,
                        seed=record["simulation_seed"],
                        solution=solution,
                        log_level=args.log_level,
                    )
                else:
                    result = model.simulate(
                        params=params,
                        initial_conditions=initial_conditions,
                        seed=record["simulation_seed"],
                        solution=solution,
                        log_level=args.log_level,
                    )
            io_start = time.perf_counter()
            frame = result.to_dataframe()
            entry["to_dataframe_seconds"] = time.perf_counter() - io_start
            write_start = time.perf_counter()
            panel = args.out / f"panel-{label}.arrow"
            frame.reset_index().to_feather(panel)
            entry["feather_write_seconds"] = time.perf_counter() - write_start
            entry["panel_rows"] = len(frame)
            entry["panel_bytes"] = panel.stat().st_size
            entry["panel_sha256"] = hashlib.sha256(pickle.dumps(frame, protocol=5)).hexdigest()
            dump()
            del result, frame

    record["device_memory_stats"] = [d.memory_stats() for d in jax.local_devices()]
    sampler.terminate()
    peaks: dict[str, float] = {}
    for line in (args.out / "nvml.csv").read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            with contextlib.suppress(ValueError):
                peaks[parts[1]] = max(peaks.get(parts[1], 0.0), float(parts[3]))
    record["nvml_peak_used_mib_per_gpu"] = peaks
    dump()


if __name__ == "__main__":
    main()
