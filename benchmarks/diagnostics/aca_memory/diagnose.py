"""Observe one cold ACA benchmark solve with durable allocation receipts.

All receipts contain scalar metadata. Synchronous observation changes scheduling;
reported intervals are diagnostic call intervals, with no performance claim.
"""

import argparse
import dataclasses
import functools
import importlib
import inspect
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Never, cast

from observer import Observer
from sources import verify_sources

if TYPE_CHECKING:
    import jax

    from _lcm.execution.output_layout import PlannedCore
    from _lcm.execution.pending_work import PendingSolveWork
    from _lcm.execution.scheduler import BufferRegistry
    from lcm import Model

# Borrowed callables have heterogeneous signatures and return types.
type Hook = Callable[..., Any]

OWNER_NAMES = (
    "solution",
    "simulation_policies",
    "generated_replay_authorities",
    "dissolution_flags",
    "solver_diagnostics",
    "retained_continuations",
    "replay_artifacts",
    "auxiliary_artifacts",
    "input_templates",
    "base_state_action_spaces",
    "flat_params",
    "retained_input_arrays",
    "next_regime_to_V_arr",
    "next_regime_to_continuation",
    "next_edge_to_V_arr",
    "period_solution",
    "period_transfer_cache",
    "buffer_registry",
    "pending_work",
    "period_continuations",
    "period_simulation_policies",
    "period_generated_replay_authorities",
    "period_dissolution_flags",
    "period_solver_diagnostics",
    "period_retained_continuations",
    "period_replay_artifacts",
    "period_auxiliary_artifacts",
    "period_inputs",
    "period_pending_outputs",
    "dispatch_outputs",
    "donated_inputs",
    "output",
    "result",
    "V_arr",
)
_context: ContextVar[dict | None] = ContextVar("aca_dispatch", default=None)


class SolveCompleteError(Exception):
    """The single automatic solve completed; forward simulation is excluded."""


SolveComplete = SolveCompleteError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pylcm-root", type=Path, required=True)
    aca = parser.add_mutually_exclusive_group(required=True)
    aca.add_argument("--aca-root", type=Path)
    aca.add_argument("--aca-package-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--mode", choices=("observe", "synchronize"), default="synchronize"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    observer = Observer(args.output / "receipts.jsonl")
    try:
        _verify_sources(args=args, observer=observer)
        if args.prepare_only:
            observer.emit("prepared", remote_execution=False)
            return 0
        _configure(args)
        aca_model = importlib.import_module("aca_model")
        import jax

        import _lcm
        import lcm
        from benchmarks.asv import bench_aca_baseline
        from benchmarks.asv.bench_aca_baseline import _build

        _validate_runtime(
            modules=(
                (lcm, args.pylcm_root),
                (_lcm, args.pylcm_root),
                (bench_aca_baseline, args.pylcm_root),
                (aca_model, args.aca_root or args.aca_package_root),
            ),
            actual_x64=bool(jax.config.jax_enable_x64),
            precision=args.precision,
        )
        observer.emit(
            "environment",
            python=platform.python_version(),
            jax=jax.__version__,
            pylcm_origin=lcm.__file__,
            aca_origin=aca_model.__file__,
            precision=args.precision,
            mode=args.mode,
            env={
                k: os.environ.get(k)
                for k in (
                    "ACA_JAX_ENABLE_X64",
                    "JAX_ENABLE_X64",
                    "JAX_PLATFORMS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_FLAGS",
                    "JAX_COMPILATION_CACHE_DIR",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
            devices=_device_stats(),
            gpu=_gpu_snapshot(),
        )
        observer.emit("build_begin")
        from lcm.typing import UserParams

        model_raw, params_raw, initial_raw = _build()
        # The pinned benchmark factory declares object returns for ASV discovery.
        model = cast("lcm.Model", model_raw)
        params = cast("UserParams", params_raw)
        initial = cast("Mapping[str, jax.Array]", initial_raw)
        execution = model._execution  # noqa: SLF001 - read the actual planning budget
        observer.emit(
            "build_complete",
            subjects=1000,
            execution_budget=execution.device_memory_bytes,
            axis_widths=dict(execution.axis_widths),
            device_ids=list(execution.device_ids),
            initial_shapes={k: list(v.shape) for k, v in initial.items()},
            census=memory_census(),
        )
        with install_hooks(observer=observer, synchronize=args.mode == "synchronize"):
            try:
                model.simulate(
                    params=params,
                    initial_conditions=initial,
                    log_level="off",
                    log_path=None,
                )
            except SolveComplete:
                observer.emit(
                    "solve_complete", forward_simulation=False, census=memory_census()
                )
                return 0
        _require_solve_sentinel()
    except Exception as error:
        _emit_error(observer=observer, event="run_error", error=error)
        try:
            observer.emit("failure_gpu_snapshot", gpu=_gpu_snapshot())
        except Exception as snapshot_error:
            error.add_note(f"GPU failure snapshot unavailable: {snapshot_error}")
        raise


@contextmanager
def install_hooks(*, observer: Observer, synchronize: bool) -> Iterator[None]:
    """Restore all borrowed callables after the diagnostic run."""
    import jax

    from _lcm.execution import output_layout, pending_work
    from _lcm.solution import backward_induction as bi
    from lcm import Model

    patches: list[tuple[object, str, object]] = []
    compiler_cache: dict[int, dict | None] = {}

    def patch(*, owner: object, name: str, wrapper: Hook) -> None:
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, wrapper)

    # These private seams expose the exact solve dispatch and its budget decisions.
    period_call = bi._run_period_kernel  # noqa: SLF001
    select_call = bi._select_runtime_donation_cores  # noqa: SLF001
    solve_call = Model._solve_from_flat_params  # noqa: SLF001
    try:
        patch(
            owner=bi,
            name="plan_workspace",
            wrapper=_make_plan_hook(original=bi.plan_workspace, observer=observer),
        )
        patch(
            owner=bi,
            name="_select_runtime_donation_cores",
            wrapper=_make_select_hook(original=select_call, observer=observer),
        )
        patch(
            owner=bi,
            name="_run_period_kernel",
            wrapper=_make_period_hook(original=period_call),
        )
        patch(
            owner=output_layout.PlannedCore,
            name="__call__",
            wrapper=_make_core_hook(
                original=output_layout.PlannedCore.__call__,
                observer=observer,
                compiler_cache=compiler_cache,
            ),
        )
        patch(
            owner=jax.stages.Compiled,
            name="__call__",
            wrapper=_make_compiled_hook(
                original=jax.stages.Compiled.__call__,
                observer=observer,
                compiler_cache=compiler_cache,
                synchronize=synchronize,
            ),
        )
        patch(
            owner=pending_work.PendingSolveWork,
            name="before",
            wrapper=_make_before_hook(
                original=pending_work.PendingSolveWork.before, observer=observer
            ),
        )
        patch(
            owner=pending_work,
            name="apply_value_transfer_plan",
            wrapper=_make_transfer_hook(
                original=pending_work.apply_value_transfer_plan,
                observer=observer,
                synchronize=synchronize,
            ),
        )
        patch(
            owner=Model,
            name="_solve_from_flat_params",
            wrapper=_make_solve_hook(original=solve_call),
        )
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def _make_plan_hook(*, original: Hook, observer: Observer) -> Hook:
    @functools.wraps(original)
    def plan(**kwargs: Any) -> object:
        frame = inspect.currentframe()
        assert frame is not None
        frame = frame.f_back
        assert frame is not None
        try:
            triple = frame.f_locals.get("triple")
            inventory = frame.f_locals.get("resident_inventory", {}).get(triple)
            try:
                inventory_metadata = _scalar_metadata(inventory)
            finally:
                del inventory
        finally:
            del frame
        result = original(**kwargs)
        resident_for = kwargs.get("resident_bytes_for")
        resident = (
            resident_for(result.compiled)
            if resident_for
            else kwargs.get("resident_bytes")
        )
        observer.emit(
            "workspace_selected",
            triple=list(triple) if triple else None,
            widths=dict(result.widths),
            compiler_id=id(result.compiled),
            peak_bytes=result.peak_bytes,
            resident_bytes=resident,
            budget_bytes=kwargs.get("budget_bytes"),
            inventory=inventory_metadata,
        )
        return result

    return plan


def _make_select_hook(*, original: Hook, observer: Observer) -> Hook:
    @functools.wraps(original)
    def select(**kwargs: Any) -> object:
        selected, donations = original(**kwargs)
        unit = kwargs["unit"]
        programs = kwargs["compiled_programs"]
        for name in unit.programs:
            triple = (unit.regime, unit.period, name)
            preferred = programs.executables[(unit.regime, unit.period)][name]
            fallback = programs.donation_fallbacks.get(triple)
            observer.emit(
                "runtime_selected",
                triple=list(triple),
                preferred_id=id(preferred.compiled),
                fallback_id=None if fallback is None else id(fallback.compiled),
                selected_id=id(selected[name].compiled),
                widths=dict(selected[name].tile_widths),
                donations=_scalar_metadata(donations.get(triple)),
            )
        return selected, donations

    return select


def _make_period_hook(*, original: Hook) -> Hook:
    @functools.wraps(original)
    def period(**kwargs: Any) -> object:
        # AgeGrid.values is host metadata; avoid device scalar transfers.
        token = _context.set(
            {"regime": kwargs["regime_name"], "period": kwargs["period"]}
        )
        try:
            return original(**kwargs)
        finally:
            _context.reset(token)

    return period


def _make_core_hook(
    *, original: Hook, observer: Observer, compiler_cache: dict[int, dict | None]
) -> Hook:
    from _lcm.execution.compiler_memory import compiler_memory_bytes

    @functools.wraps(original)
    def core(self: PlannedCore, *args: object, **kwargs: Any) -> object:
        parent = _context.get()
        if parent is None:
            return original(self, *args, **kwargs)
        compiled_id = id(self.compiled)
        if compiled_id not in compiler_cache:
            report = compiler_memory_bytes(compiled=self.compiled)
            compiler_cache[compiled_id] = (
                None if report is None else dataclasses.asdict(report)
            )
        tag = {
            **parent,
            "core": self.name,
            "compiler_id": compiled_id,
            "widths": dict(self.tile_widths),
            "donated_arguments": list(self.donated_arguments),
        }
        token = _context.set(tag)
        observer.emit(
            "core_enter",
            tag=tag,
            compiler=compiler_cache[compiled_id],
            transfers=_scalar_metadata(self.input_transfer_plan),
        )
        try:
            return original(self, *args, **kwargs)
        except Exception as error:
            _emit_error(observer=observer, event="core_error", error=error, tag=tag)
            raise
        finally:
            _context.reset(token)

    return core


def _make_compiled_hook(
    *,
    original: Hook,
    observer: Observer,
    compiler_cache: dict[int, dict | None],
    synchronize: bool,
) -> Hook:
    import jax

    @functools.wraps(original)
    def compiled(self: jax.stages.Compiled, *args: object, **kwargs: Any) -> object:
        tag = _context.get()
        if tag is None or tag.get("compiler_id") != id(self):
            return original(self, *args, **kwargs)
        if synchronize:
            observer.emit("inputs_wait_begin", tag=tag)
            try:
                jax.block_until_ready((args, kwargs))
            except Exception as error:
                _emit_error(
                    observer=observer, event="inputs_wait_error", error=error, tag=tag
                )
                raise
            observer.emit("inputs_wait_complete", tag=tag)
        return observer.run(
            tag=tag,
            metadata=lambda: {
                "compiler": compiler_cache[id(self)],
                "mode": "synchronize" if synchronize else "observe",
            },
            execute=lambda: original(self, *args, **kwargs),
            wait=jax.block_until_ready if synchronize else lambda _result: None,
            census=memory_census,
            completion_confirmed=synchronize,
        )

    return compiled


def _make_before_hook(*, original: Hook, observer: Observer) -> Hook:
    @functools.wraps(original)
    def before(self: PendingSolveWork, **kwargs: Any) -> None:
        tag = _context.get()
        observer.emit("pending_wait_begin", tag=tag)
        try:
            original(self, **kwargs)
        except Exception as error:
            _emit_error(
                observer=observer,
                event="pending_wait_error",
                error=error,
                tag=tag,
                attribution="earlier_work",
            )
            raise
        observer.emit("pending_wait_complete", tag=tag)

    return before


def _make_transfer_hook(
    *, original: Hook, observer: Observer, synchronize: bool
) -> Hook:
    import jax

    @functools.wraps(original)
    def transfer(**kwargs: Any) -> object:
        tag = _context.get()
        observer.emit("transfer_begin", tag=tag, census=memory_census())
        try:
            result = original(**kwargs)
            if synchronize:
                jax.block_until_ready(result)
        except Exception as error:
            _emit_error(observer=observer, event="transfer_error", error=error, tag=tag)
            raise

        else:
            observer.emit("transfer_complete", tag=tag, census=memory_census())
            return result

    return transfer


def _make_solve_hook(*, original: Hook) -> Hook:
    @functools.wraps(original)
    def solve(self: Model, *args: object, **kwargs: Any) -> object:
        original(self, *args, **kwargs)
        raise SolveComplete

    return solve


def memory_census() -> dict:
    """Observe transient array references and serialize metadata immediately."""
    import jax

    started = time.perf_counter()
    arrays = jax.live_arrays()
    physical: dict[tuple[int, int], dict] = {}
    deleted = 0
    errors: list[str] = []
    for array in arrays:
        if array.is_deleted():
            deleted += 1
            continue
        try:
            for shard in array.addressable_shards:
                buffer = shard.data
                key = (shard.device.id, buffer.unsafe_buffer_pointer())
                record = {
                    "device": shard.device.id,
                    "pointer": key[1],
                    "bytes": int(buffer.nbytes),
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                }
                if key not in physical or physical[key]["bytes"] < record["bytes"]:
                    physical[key] = record
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    wrapper_count = len(arrays)
    del arrays
    owners = _owner_census()
    totals = Counter()
    for record in physical.values():
        totals[record["device"]] += record["bytes"]
    result = {
        "device_stats": _device_stats(),
        "live_wrapper_count": wrapper_count,
        "deleted_wrappers": deleted,
        "physical_buffer_count": len(physical),
        "visible_buffer_bytes_by_device": dict(totals),
        "buffers": list(physical.values()),
        "owners": owners,
        "errors": errors,
        "limitations": (
            "Buffer-pointer deduplication is an observation; allocator cache, "
            "constants, workspace and partial buffer aliases are not a complete "
            "ownership accounting."
        ),
    }
    result["observer_seconds"] = time.perf_counter() - started
    return result


def _owner_census() -> dict:
    """Inspect named solve owners, releasing all frame references on return."""
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    try:
        while frame is not None:
            if (
                frame.f_code.co_name == "solve"
                and frame.f_globals.get("__name__")
                == "_lcm.solution.backward_induction"
            ):
                return {
                    name: _array_metadata(frame.f_locals[name])
                    for name in OWNER_NAMES
                    if name in frame.f_locals
                }
            frame = frame.f_back
        return {}
    finally:
        del frame


def _array_metadata(value: object) -> dict:
    seen: set[int] = set()
    leaves: list[dict] = []
    opaque: Counter[str] = Counter()
    _walk_owner(item=value, path="", seen=seen, leaves=leaves, opaque=opaque)
    return {"arrays": leaves, "opaque_types": dict(opaque)}


def _walk_owner(
    *, item: object, path: str, seen: set[int], leaves: list[dict], opaque: Counter[str]
) -> None:
    import jax

    from _lcm.execution.scheduler import BufferRegistry

    if isinstance(item, jax.Array):
        leaves.append(
            {
                "path": path,
                "array_id": id(item),
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "logical_bytes": int(item.nbytes),
                "deleted": item.is_deleted(),
                "shards": _shard_metadata(array=item),
            }
        )
        return
    if id(item) in seen:
        return
    seen.add(id(item))
    if type(item) is BufferRegistry:
        # Weak registry declarations are links, not retaining owners.
        leaves.append(
            {"path": path, "weak_registry": _registry_metadata(registry=item)}
        )
        return
    children = _owner_children(item=item)
    if children is not None:
        for name, child in children:
            _walk_owner(
                item=child,
                path=f"{path}/{name}",
                seen=seen,
                leaves=leaves,
                opaque=opaque,
            )
    elif item is not None and type(item) not in (bool, int, float, str, bytes):
        opaque[f"{type(item).__module__}.{type(item).__name__}"] += 1


def _owner_children(*, item: object) -> Iterator[tuple[str, object]] | None:
    from _lcm.execution.pending_work import PendingSolveWork
    from _lcm.execution.scheduler import PeriodTransferCache

    if type(item) in (dict, MappingProxyType):
        assert isinstance(item, Mapping)
        return ((str(key), child) for key, child in item.items())
    if type(item) in (list, tuple):
        assert isinstance(item, list | tuple)
        return ((str(index), child) for index, child in enumerate(item))
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return (
            (field.name, getattr(item, field.name))
            for field in dataclasses.fields(item)
        )
    if type(item) is PeriodTransferCache:
        # Read exact retaining fields; the diagnostic does not mutate cache ownership.
        return iter(
            (("_arrays", item._arrays), ("_pending_outputs", item._pending_outputs))  # noqa: SLF001
        )
    if type(item) is PendingSolveWork:
        return iter((("_records", item._records),))  # noqa: SLF001 - pending owner payloads
    return None


def _shard_metadata(*, array: jax.Array) -> list[dict]:
    if array.is_deleted():
        return []
    return [
        {
            "device": shard.device.id,
            "pointer": shard.data.unsafe_buffer_pointer(),
            "bytes": int(shard.data.nbytes),
        }
        for shard in array.addressable_shards
    ]


def _registry_metadata(*, registry: BufferRegistry) -> list[dict]:
    rows = []
    for shard, links in registry._keys_by_shard.items():  # noqa: SLF001 - inspect weak owner links
        for artifact, references in links.items():
            rows.append(
                {
                    "shard": list(shard),
                    "artifact": _scalar_metadata(artifact),
                    "references": _weak_reference_metadata(references),
                }
            )
    for shard, references in registry._unproduced_shards.items():  # noqa: SLF001 - inspect weak owner links
        rows.append(
            {
                "shard": list(shard),
                "unproduced": True,
                "references": _weak_reference_metadata(references),
            }
        )
    return rows


def _weak_reference_metadata(
    references: Iterable[Callable[[], jax.Array | None]],
) -> list[dict]:
    rows = []
    for reference in references:
        array = reference()
        rows.append(
            {
                "array_id": None if array is None else id(array),
                "dead": array is None,
                "deleted": None if array is None else array.is_deleted(),
            }
        )
        del array
    return rows


def _emit_error(
    *, observer: Observer, event: str, error: Exception, **metadata: object
) -> None:
    try:
        observer.emit(
            event, error_type=type(error).__name__, error=str(error), **metadata
        )
    except Exception as receipt_error:
        error.add_note(f"Error receipt {event} unavailable: {receipt_error}")


def _scalar_metadata(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): _scalar_metadata(v) for k, v in value.items()}
    if type(value) in (tuple, list, set, frozenset):
        assert isinstance(value, tuple | list | set | frozenset)
        return [_scalar_metadata(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _scalar_metadata(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    return {"type": f"{type(value).__module__}.{type(value).__name__}"}


def _device_stats() -> list[dict]:
    import jax

    return [
        {
            "id": d.id,
            "platform": d.platform,
            "kind": d.device_kind,
            "memory_stats": d.memory_stats(),
        }
        for d in jax.devices()
    ]


def _gpu_snapshot() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return {
        "gpu": result.stdout,
        "gpu_exit": result.returncode,
        "processes": processes.stdout,
        "processes_exit": processes.returncode,
    }


def _validate_runtime(
    *, modules: tuple[tuple[ModuleType, Path], ...], actual_x64: bool, precision: int
) -> None:
    for module, root in modules:
        origin = module.__file__
        if origin is None or not Path(origin).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(
                f"Imported source escaped its pinned root: {module.__name__}"
            )
    if actual_x64 != (precision == 64):
        raise RuntimeError("Runtime precision does not match requested precision.")


def _require_solve_sentinel() -> Never:
    raise RuntimeError("Automatic solve completion sentinel was not observed.")


def _configure(args: argparse.Namespace) -> None:
    os.environ["ACA_JAX_ENABLE_X64"] = str(int(args.precision == 64))
    os.environ["JAX_ENABLE_X64"] = str(int(args.precision == 64))
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.platform == "cuda" else "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    os.environ.pop("XLA_CLIENT_MEM_FRACTION", None)
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.output.resolve() / "jax-cache")
    sys.path[:0] = [
        str(args.pylcm_root / "src"),
        str(args.pylcm_root),
        str(args.aca_root / "src" if args.aca_root else args.aca_package_root.parent),
    ]


def _verify_sources(*, args: argparse.Namespace, observer: Observer) -> None:
    observer.emit(
        "source_verified",
        **verify_sources(
            pylcm_root=args.pylcm_root,
            aca_root=args.aca_root,
            aca_package_root=args.aca_package_root,
            manifest_path=args.manifest,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
