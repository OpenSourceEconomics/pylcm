"""Fresh-process worker for one side of a paired GridSearch measurement."""

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from benchmarks.grid_search_pair_scenarios import EXTERNAL_HARNESS_SOURCES

_MEMORY_FIELDS = (
    "generated_code_size_in_bytes",
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "peak_memory_in_bytes",
    "host_generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
)
_HLO_OPS = (
    "while",
    "reduce",
    "map",
    "gather",
    "dynamic-slice",
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-broadcast",
    "collective-permute",
    "ragged-all-to-all",
    "reduce-scatter",
)
_COLLECTIVE_OPS = (
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-broadcast",
    "collective-permute",
    "ragged-all-to-all",
    "reduce-scatter",
)

_VERSION_SHIM_MODULE = "_lcm.version"
_VERSION_SHIM_VERSION = "0+gridsearchpair"
_VERSION_SHIM_ORIGIN = "<pylcm-grid-search-pair-version-shim>"
_VERSION_SHIM_EXPORTS = (
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "__commit_id__",
    "commit_id",
)


def _version_shim_identity() -> dict[str, Any]:
    identity = {
        "module": _VERSION_SHIM_MODULE,
        "origin": _VERSION_SHIM_ORIGIN,
        "exports": list(_VERSION_SHIM_EXPORTS),
        "version": _VERSION_SHIM_VERSION,
        "version_tuple": [0, "gridsearchpair"],
        "commit_id": None,
    }
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity


class _VersionShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load exactly ``_lcm.version`` from immutable harness metadata."""

    # keyword-only-exempt: library-callback=importlib.abc.MetaPathFinder.find_spec
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> Any:
        del path, target
        if fullname != _VERSION_SHIM_MODULE:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=_VERSION_SHIM_ORIGIN,
        )

    def create_module(self, spec: Any) -> None:
        del spec

    def exec_module(self, module: ModuleType) -> None:
        version_tuple = (0, "gridsearchpair")
        module.__file__ = _VERSION_SHIM_ORIGIN
        module.__all__ = list(_VERSION_SHIM_EXPORTS)
        module.__version__ = _VERSION_SHIM_VERSION
        module.version = _VERSION_SHIM_VERSION
        module.__version_tuple__ = version_tuple
        module.version_tuple = version_tuple
        module.__commit_id__ = None
        module.commit_id = None


def _import_lcm_with_version_shim(
    *, target_src: Path
) -> tuple[ModuleType, dict[str, Any]]:
    """Import target ``lcm`` with identical build metadata for both revisions.

    Hatch's VCS hook generates ``src/_lcm/version.py`` while building the package,
    and the generated file is intentionally ignored by git.  An exact source
    checkout therefore need not contain it.  ``lcm.__init__`` only imports its
    ``__version__`` metadata; no measured kernel reads it.  A temporary exact-name
    finder makes source imports reliable and prevents an ignored local generation from
    distinguishing the base and head workers.
    """
    preloaded = sorted(
        name
        for name in sys.modules
        if name in {"lcm", "_lcm"} or name.startswith(("lcm.", "_lcm."))
    )
    if preloaded:
        raise RuntimeError(
            f"Target packages were imported before their shim: {preloaded!r}."
        )
    if not sys.path or Path(sys.path[0]).resolve() != target_src.resolve():
        raise RuntimeError("The target source root must be first on sys.path.")

    finder = _VersionShimFinder()
    sys.meta_path.insert(0, finder)
    try:
        lcm_module = importlib.import_module("lcm")
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError as error:
            raise RuntimeError(
                "The version-shim finder disappeared during import."
            ) from error

    private_module = sys.modules.get("_lcm")
    version_module = sys.modules.get(_VERSION_SHIM_MODULE)
    if not isinstance(private_module, ModuleType) or not isinstance(
        version_module, ModuleType
    ):
        raise TypeError("Target lcm did not import the version shim.")
    for name, module in (("lcm", lcm_module), ("_lcm", private_module)):
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            target_src.resolve()
        ):
            raise RuntimeError(
                f"Imported {name} from {module_file!r}, outside target source "
                f"{target_src}."
            )
    if getattr(private_module, "version", None) is not version_module:
        raise RuntimeError("The version shim is not bound as _lcm.version.")
    if getattr(lcm_module, "__version__", None) != _VERSION_SHIM_VERSION:
        raise RuntimeError(
            "Target lcm did not consume the revision-neutral version shim."
        )
    return lcm_module, _version_shim_identity()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-harness-revision", required=True)
    parser.add_argument("--expected-harness-digest", required=True)
    parser.add_argument("--expected-scenario-digest", required=True)
    parser.add_argument("--expected-lock-digest", required=True)
    parser.add_argument("--expected-pixi-environment", required=True)
    parser.add_argument("--expected-pixi-exe", required=True)
    parser.add_argument("--expected-pixi-project-root", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--precision", choices=("32", "64"), required=True)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-samples", type=int, default=3)
    return parser


def _git(*, checkout: Path, args: Iterable[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        msg = f"git {' '.join(args)} failed: {result.stderr.strip()}"
        raise RuntimeError(msg)
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(*, root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for raw_relative in sorted(relative_paths):
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Source path must be repo-relative: {raw_relative!r}.")
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256_file(root / relative)))
    return digest.hexdigest()


def _read_proc_memory() -> dict[str, int | None]:
    fields = {"VmRSS": None, "VmHWM": None}
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return {"rss_bytes": None, "hwm_bytes": None}
    for line in lines:
        name, separator, raw = line.partition(":")
        if separator and name in fields:
            value, unit = raw.split()
            if unit != "kB":
                msg = f"Unexpected /proc memory unit: {unit!r}."
                raise RuntimeError(msg)
            fields[name] = int(value) * 1024
    return {"rss_bytes": fields["VmRSS"], "hwm_bytes": fields["VmHWM"]}


def _memory_analysis(compiled: Any) -> dict[str, int | None] | None:
    try:
        stats = compiled.memory_analysis()
    except Exception:
        return None
    if stats is None:
        return None
    return {
        field: (
            None if getattr(stats, field, None) is None else int(getattr(stats, field))
        )
        for field in _MEMORY_FIELDS
    }


def _hlo_census(text: str) -> dict[str, Any]:
    lowered = text.lower()
    instruction_count = sum(
        1
        for line in lowered.splitlines()
        if " = " in line and not line.lstrip().startswith(("hlo_module", "entry"))
    )

    def count_form(op: str) -> int:
        return len(re.findall(rf"(?<![a-z0-9_-]){re.escape(op)}\s*\(", lowered))

    op_counts = {}
    async_communication_counts = {}
    for op in _HLO_OPS:
        if op not in _COLLECTIVE_OPS:
            op_counts[op] = count_form(op)
            continue
        synchronous = count_form(op)
        starts = count_form(f"{op}-start")
        dones = count_form(f"{op}-done")
        async_count = max(starts, dones)
        op_counts[op] = synchronous + async_count
        async_communication_counts[op] = {
            "synchronous": synchronous,
            "starts": starts,
            "dones": dones,
            "deduplicated_async": async_count,
        }
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_bytes": len(text.encode()),
        "line_count": len(text.splitlines()),
        "instruction_count": instruction_count,
        "op_counts": op_counts,
        "async_communication_counts": async_communication_counts,
        "communication_collective_count": sum(op_counts[op] for op in _COLLECTIVE_OPS),
    }


def _safe_stem(label: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-").lower()
    return stem[:100] or "core"


def _kernel_execution_metadata(kernel: Any) -> dict[str, Any]:
    """Describe one kernel through its native graph or the historical declaration."""
    graph_provider = getattr(kernel, "core_programs", None)
    if callable(graph_provider):
        programs = graph_provider()
        if tuple(programs) != ("main",):
            raise RuntimeError(
                "GridSearch benchmark expected exactly one native 'main' program."
            )
        program = programs["main"]
        raw_disposition = program.disposition
        disposition = getattr(raw_disposition, "value", raw_disposition)
        return {
            "streamed": disposition == "planned",
            "execution_disposition": disposition,
            "disposition_reason": program.disposition_reason,
        }

    streamed = getattr(kernel, "streamed_core", None) is not None
    return {
        "streamed": streamed,
        "execution_disposition": "planned" if streamed else "legacy-unplanned",
        "disposition_reason": None if streamed else "legacy_adapter",
    }


def _route_metadata(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    folded_regimes: list[str] = []
    distributed_regimes: list[str] = []
    collective_regimes: list[str] = []
    taste_shock_regimes: list[str] = []
    gs_vd_regimes: list[str] = []
    for regime_name, regime in sorted(model._regimes.items()):  # noqa: SLF001
        fold_names = tuple(regime.fold_state_names)
        action_names = tuple(regime.solution.action_names)
        action_extents = tuple(
            int(regime.solution.grids[name].to_jax().shape[0]) for name in action_names
        )
        collective = regime.stakeholders is not None
        has_taste_shocks = bool(regime.has_taste_shocks)
        if fold_names:
            folded_regimes.append(regime_name)
        if has_taste_shocks:
            taste_shock_regimes.append(regime_name)
        if collective:
            collective_regimes.append(regime_name)
        if regime.same_period_ref_regimes or regime.gated_edges:
            gs_vd_regimes.append(regime_name)
        if any(
            bool(getattr(grid, "distributed", False))
            for grid in regime.solution.grids.values()
        ):
            distributed_regimes.append(regime_name)
        for period, kernel in sorted(regime.solution.period_kernels.items()):
            execution = _kernel_execution_metadata(kernel)
            rows.append(
                {
                    "regime": regime_name,
                    "period": int(period),
                    "action_names": list(action_names),
                    "action_extents": list(action_extents),
                    **execution,
                    "collective": collective,
                    "has_taste_shocks": has_taste_shocks,
                    "fold_state_names": list(fold_names),
                }
            )
    return {
        "kernels": rows,
        "streamed_kernel_count": sum(row["streamed"] for row in rows),
        "folded_regimes": sorted(set(folded_regimes)),
        "distributed_regimes": sorted(set(distributed_regimes)),
        "collective_regimes": sorted(set(collective_regimes)),
        "taste_shock_regimes": sorted(set(taste_shock_regimes)),
        "gs_vd_regimes": sorted(set(gs_vd_regimes)),
    }


def _scenario_dimensions(model: Any) -> dict[str, Any]:
    return {
        "n_periods": int(model.n_periods),
        "regimes": {
            regime_name: {
                "active_periods": [int(period) for period in regime.active_periods],
                "solution_grid_extents": {
                    name: int(grid.to_jax().shape[0])
                    for name, grid in regime.solution.grids.items()
                },
            }
            for regime_name, regime in sorted(model._regimes.items())  # noqa: SLF001
        },
    }


def _assert_scenario_identity(*, spec: Any, routes: Mapping[str, Any]) -> None:
    has_fold = bool(routes["folded_regimes"])
    has_collective = bool(routes["collective_regimes"])
    has_distributed = bool(routes["distributed_regimes"])
    has_taste_shocks = bool(routes["taste_shock_regimes"])
    has_gs_vd = bool(routes["gs_vd_regimes"])
    if has_fold != spec.expected_folded:
        raise RuntimeError("Folded-route identity does not match its scenario spec.")
    if has_collective != spec.expected_collective:
        raise RuntimeError(
            "Collective-route identity does not match its scenario spec."
        )
    if has_distributed != spec.expected_distributed:
        raise RuntimeError(
            "Distributed-route identity does not match its scenario spec."
        )
    if spec.name == "singleton-ev1" and not has_taste_shocks:
        raise RuntimeError("The singleton EV1 row contains no taste-shock regime.")
    if spec.name != "singleton-ev1" and has_taste_shocks:
        raise RuntimeError("A non-EV1 row unexpectedly contains taste shocks.")
    if spec.name == "collective-gs-vd" and not has_gs_vd:
        raise RuntimeError(
            "The collective GS-VD row contains no value-dependent route."
        )
    if spec.name != "collective-gs-vd" and has_gs_vd:
        raise RuntimeError(
            "A non-GS-VD row unexpectedly contains a value-dependent route."
        )


def _flatten_arrays(*, result: Any) -> dict[str, Any]:
    if isinstance(result, tuple):
        values, dissolution_flags = result
    else:
        values = result.values
        dissolution_flags: dict[int, dict[str, Any]] = {}
        for ref, payload in result.replay_artifacts.items():
            if ref.key.type_id == "pylcm.collective.dissolution_flag":
                dissolution_flags.setdefault(ref.period, {})[ref.regime] = payload
    flattened: dict[str, Any] = {}
    for prefix, tree in (("value", values), ("dissolution", dissolution_flags)):
        for period, by_regime in sorted(tree.items()):
            for regime, array in sorted(by_regime.items()):
                key = f"{prefix}/{int(period)}/{regime}"
                flattened[key] = array
    return flattened


def _block_result(result: Any) -> None:
    """Wait for every published result leaf, including MappingProxyType trees."""
    for array in _flatten_arrays(result=result).values():
        array.block_until_ready()


def _partition_entry(value: Any) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return list(value)
    msg = f"Unsupported PartitionSpec entry: {value!r}."
    raise TypeError(msg)


def _sharding_descriptor(*, array: Any, jax: Any) -> dict[str, Any]:
    sharding = getattr(array, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        axis_names = [str(name) for name in sharding.mesh.axis_names]
        return {
            "kind": "NamedSharding",
            "mesh_axis_names": axis_names,
            "mesh_shape": {name: int(sharding.mesh.shape[name]) for name in axis_names},
            "partition_spec": [
                _partition_entry(entry) for entry in tuple(sharding.spec)
            ],
            "memory_kind": sharding.memory_kind,
        }
    if isinstance(sharding, jax.sharding.SingleDeviceSharding):
        devices = tuple(sharding.device_set)
        if len(devices) != 1:
            raise RuntimeError(
                "SingleDeviceSharding unexpectedly names more than one device."
            )
        device = devices[0]
        return {
            "kind": "SingleDeviceSharding",
            "platform": device.platform,
            "device_id": int(device.id),
            "memory_kind": sharding.memory_kind,
        }
    msg = f"Unsupported array sharding in paired artifact: {type(sharding).__name__}."
    raise TypeError(msg)


def _write_arrays(
    *, path: Path, arrays: Mapping[str, Any], jax: Any
) -> list[dict[str, Any]]:
    import numpy as np

    descriptors = {
        key: _sharding_descriptor(array=value, jax=jax) for key, value in arrays.items()
    }
    converted = {
        key: np.asarray(jax.device_get(value)) for key, value in arrays.items()
    }
    np.savez_compressed(path, **converted)
    return [
        {
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "sharding": descriptors[key],
        }
        for key, array in sorted(converted.items())
    ]


def _device_memory(jax: Any) -> list[dict[str, Any]]:
    result = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except Exception:
            stats = None
        peak = (
            None
            if not stats or stats.get("peak_bytes_in_use") is None
            else int(stats["peak_bytes_in_use"])
        )
        is_gpu = device.platform == "gpu"
        result.append(
            {
                "id": int(device.id),
                "platform": device.platform,
                "kind": device.device_kind,
                "peak_bytes_in_use": peak if is_gpu else None,
                "status": (
                    "measured"
                    if is_gpu and peak is not None
                    else "unavailable"
                    if is_gpu
                    else "not_applicable"
                ),
                "reason": (
                    None
                    if is_gpu and peak is not None
                    else "device.memory_stats() omitted peak_bytes_in_use"
                    if is_gpu
                    else "Device peak memory is only defined for GPU rows."
                ),
            }
        )
    return result


def _validate_pixi_identity(*, args: argparse.Namespace, harness_root: Path) -> None:
    expected = {
        "PIXI_ENVIRONMENT_NAME": args.expected_pixi_environment,
        "PIXI_EXE": str(Path(args.expected_pixi_exe).resolve()),
        "PIXI_PROJECT_ROOT": str(Path(args.expected_pixi_project_root).resolve()),
    }
    actual = {
        name: (
            None
            if os.environ.get(name) is None
            else str(Path(os.environ[name]).resolve())
            if name in {"PIXI_EXE", "PIXI_PROJECT_ROOT"}
            else os.environ[name]
        )
        for name in expected
    }
    if actual != expected:
        raise RuntimeError(
            f"Pixi worker identity mismatch: expected {expected}, got {actual}."
        )
    if Path(expected["PIXI_PROJECT_ROOT"]) != harness_root:
        raise RuntimeError("Pixi worker project root is not the external harness root.")


def _validate_identity(*, args: argparse.Namespace, harness_root: Path) -> Path:
    checkout = args.checkout.resolve()
    if not (checkout / ".git").exists() and not _git(
        checkout=checkout, args=("rev-parse", "--git-dir")
    ):
        raise RuntimeError(f"Not a git checkout: {checkout}")
    revision = _git(checkout=checkout, args=("rev-parse", "HEAD^{commit}"))
    if revision != args.expected_revision:
        raise RuntimeError(
            f"Revision mismatch: expected {args.expected_revision}, got {revision}."
        )
    dirty = _git(
        checkout=checkout,
        args=("status", "--porcelain", "--untracked-files=all"),
    )
    if dirty:
        raise RuntimeError(f"Target checkout is dirty:\n{dirty}")
    harness_revision = _git(checkout=harness_root, args=("rev-parse", "HEAD^{commit}"))
    if harness_revision != args.expected_harness_revision:
        raise RuntimeError(
            "External harness revision mismatch: "
            f"expected {args.expected_harness_revision}, got {harness_revision}."
        )
    harness_dirty = _git(
        checkout=harness_root,
        args=("status", "--porcelain", "--untracked-files=all"),
    )
    if harness_dirty:
        raise RuntimeError(f"External harness checkout is dirty:\n{harness_dirty}")
    _validate_pixi_identity(args=args, harness_root=harness_root)
    lock_digest = _sha256_file(checkout / "pixi.lock")
    if lock_digest != args.expected_lock_digest:
        raise RuntimeError(
            "Target pixi.lock digest changed after controller validation."
        )
    harness_digest = _sha256_files(
        root=harness_root,
        relative_paths=EXTERNAL_HARNESS_SOURCES,
    )
    if harness_digest != args.expected_harness_digest:
        raise RuntimeError("External harness digest mismatch.")
    scenario_path = harness_root / "benchmarks/grid_search_pair_scenarios.py"
    if _sha256_file(scenario_path) != args.expected_scenario_digest:
        raise RuntimeError("External scenario digest mismatch.")
    return checkout


def main(argv: list[str] | None = None) -> None:  # noqa: C901, PLR0912, PLR0915
    args = _parser().parse_args(argv)
    if args.warm_samples != 3:
        raise ValueError("The bounded F harness requires exactly three warm samples.")
    if args.output.exists():
        raise FileExistsError(f"Worker output already exists: {args.output}")

    harness_root = Path(__file__).resolve().parent.parent
    checkout = _validate_identity(args=args, harness_root=harness_root)
    target_src = (checkout / "src").resolve()
    sys.path.insert(0, str(target_src))

    from benchmarks.grid_search_pair_scenarios import SCENARIOS, build_scenario

    spec = SCENARIOS.get(args.scenario)
    if spec is None:
        raise KeyError(f"Unknown scenario: {args.scenario!r}.")

    import jax
    import jaxlib

    jax.config.update("jax_enable_x64", args.precision == "64")
    if spec.topology == "cpu-4":
        jax.config.update("jax_num_cpu_devices", 4)
        jax.config.update("jax_platform_name", "cpu")
    elif args.backend != "auto":
        jax.config.update("jax_platform_name", args.backend)

    lcm, version_shim = _import_lcm_with_version_shim(target_src=target_src)
    from _lcm.solution import backward_induction

    imported_lcm = Path(lcm.__file__).resolve()
    if not imported_lcm.is_relative_to(target_src):
        raise RuntimeError(
            f"Imported lcm from {imported_lcm}, outside target source {target_src}."
        )
    devices = jax.devices()
    if spec.topology == "cpu-4":
        if len(devices) != 4 or {device.platform for device in devices} != {"cpu"}:
            raise RuntimeError(
                f"The distributed row requires exactly 4 CPUs: {devices}"
            )
    elif args.backend != "auto" and {device.platform for device in devices} != {
        args.backend
    }:
        raise RuntimeError(f"Requested backend {args.backend!r}, got {devices}.")

    build_started = time.perf_counter_ns()
    model, params = build_scenario(name=cast("Any", spec.name))
    build_wall_ns = time.perf_counter_ns() - build_started
    routes = _route_metadata(model)
    _assert_scenario_identity(spec=spec, routes=routes)
    dimensions = _scenario_dimensions(model)

    compiled_refs: list[tuple[str, Any]] = []
    resolved_plans: list[Any] = []
    compile_phase_ns: list[int] = []
    capture_cold = True
    capture_lock = threading.Lock()

    original_compile_all = backward_induction._compile_all_functions  # noqa: SLF001
    original_log_memory = backward_induction._log_kernel_memory  # noqa: SLF001
    original_resolve_program = getattr(backward_induction, "resolve_core_program", None)

    def timed_compile_all(*call_args: Any, **call_kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return original_compile_all(*call_args, **call_kwargs)
        finally:
            compile_phase_ns.append(time.perf_counter_ns() - started)

    def retain_compiled(*, compiled: Any, label: str, logger: Any) -> None:  # noqa: ARG001
        if capture_cold:
            with capture_lock:
                compiled_refs.append((label, compiled))

    def retain_plan(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original_resolve_program(*call_args, **call_kwargs)
        if capture_cold:
            resolved_plans.append(result)
        return result

    solve_kwargs = {
        "params": params,
        "log_level": "off",
    }
    if "return_dissolution_flags" in inspect.signature(model.solve).parameters:
        solve_kwargs["return_dissolution_flags"] = True
    memory_before_solve = _read_proc_memory()
    with ExitStack() as stack:
        backward_induction._compile_all_functions = timed_compile_all  # noqa: SLF001
        stack.callback(
            setattr,
            backward_induction,
            "_compile_all_functions",
            original_compile_all,
        )
        backward_induction._log_kernel_memory = retain_compiled  # noqa: SLF001
        stack.callback(
            setattr,
            backward_induction,
            "_log_kernel_memory",
            original_log_memory,
        )
        if original_resolve_program is not None:
            backward_induction.resolve_core_program = retain_plan
            stack.callback(
                setattr,
                backward_induction,
                "resolve_core_program",
                original_resolve_program,
            )

        cold_started = time.perf_counter_ns()
        cold_result = model.solve(**solve_kwargs)
        _block_result(cold_result)
        cold_wall_ns = time.perf_counter_ns() - cold_started
        del cold_result
        capture_cold = False

        warm_wall_ns: list[int] = []
        final_warm_result = None
        for sample in range(args.warm_samples):
            warm_started = time.perf_counter_ns()
            candidate = model.solve(**solve_kwargs)
            _block_result(candidate)
            warm_wall_ns.append(time.perf_counter_ns() - warm_started)
            if sample == args.warm_samples - 1:
                final_warm_result = candidate
            del candidate

    memory_through_warm = _read_proc_memory()
    device_memory = _device_memory(jax)
    if final_warm_result is None:
        raise RuntimeError("The worker produced no final warm result.")
    arrays = _flatten_arrays(result=final_warm_result)
    args.output.mkdir(parents=True)
    array_manifest = _write_arrays(
        path=args.output / "values.npz", arrays=arrays, jax=jax
    )
    del arrays, final_warm_result

    tile_plans = [
        {
            "tile_widths": dict(result.tile_widths),
            "static_kwargs": dict(result.static_kwargs),
            "specialization_key": repr(result.specialization_key),
        }
        for result in resolved_plans
    ]

    hlo_dir = args.output / "hlo"
    hlo_dir.mkdir()
    core_records: list[dict[str, Any]] = []
    ordered_compiled = sorted(compiled_refs, key=lambda item: item[0])
    for ordinal, (label, compiled) in enumerate(ordered_compiled):
        text = compiled.as_text()
        census = _hlo_census(text)
        compiler_memory = _memory_analysis(compiled)
        filename = f"{ordinal:03d}-{_safe_stem(label)}-{census['sha256'][:12]}.hlo.txt"
        (hlo_dir / filename).write_text(text)
        core_records.append(
            {
                "label": label,
                "hlo_file": f"hlo/{filename}",
                "hlo": census,
                "compiler_memory": compiler_memory,
                "compiler_memory_status": (
                    "measured" if compiler_memory is not None else "unavailable"
                ),
                "compiler_memory_reason": (
                    None
                    if compiler_memory is not None
                    else "Executable memory_analysis() is unavailable."
                ),
            }
        )

    final_memory = _read_proc_memory()
    metrics = {
        "schema_version": "1.0",
        "scenario": spec.name,
        "revision": args.expected_revision,
        "harness_revision": args.expected_harness_revision,
        "checkout": str(checkout),
        "harness_digest": args.expected_harness_digest,
        "scenario_digest": args.expected_scenario_digest,
        "lock_digest": args.expected_lock_digest,
        "target_lcm_file": str(imported_lcm),
        "version_shim": version_shim,
        "precision": int(args.precision),
        "pixi": {
            "environment_name": args.expected_pixi_environment,
            "exe": str(Path(args.expected_pixi_exe).resolve()),
            "project_root": str(Path(args.expected_pixi_project_root).resolve()),
        },
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "python": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "devices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "kind": device.device_kind,
            }
            for device in devices
        ],
        "environment": {
            "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        },
        "timing_ns": {
            "build": build_wall_ns,
            "cold_solve": cold_wall_ns,
            "aot_compile_calls": compile_phase_ns,
            "warm_solve": warm_wall_ns,
        },
        "memory": {
            "before_solve": memory_before_solve,
            "through_warm": memory_through_warm,
            "after_hlo_and_serialization": final_memory,
            "device": device_memory,
        },
        "routes": routes,
        "dimensions": dimensions,
        "tile_plans": tile_plans,
        "compiled_cores": core_records,
        "arrays": array_manifest,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
