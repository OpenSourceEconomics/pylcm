"""Bind public capture requests and record the production dispatch boundary."""

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import jaxlib
import numpy as np
from jaxlib import (
    _hlo,  # ty: ignore[unresolved-import] - installed native API has no stub
)

from _lcm.egm.upper_envelope._exact_affine.ffi import _installed_native_directory
from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.output_layout import PlannedCore
from _lcm.execution.workspace_planning import compiler_memory_reservation
from _lcm.persistence.period import read_period_archive, write_period_archive
from _lcm.solution.period_capture import _period_layouts
from lcm.period_capture import PeriodCapture, PeriodCaptureRecord

_GRID_SEARCH_ROUTE = "_lcm.solution.grid_search._GridSearchPeriodKernel"


def validate_period_capture_cache() -> None:
    """Require the caller's explicit startup-disabled persistent cache contract."""
    if (
        os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "").lower() != "false"
        or jax.config.jax_enable_compilation_cache is not False
    ):
        raise ValueError(
            "Public period capture/replay requires JAX_ENABLE_COMPILATION_CACHE=false "
            "before process startup and JAX initialization. Enabled or undeclared "
            "persistent compilation cache modes are unsupported."
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CaptureContext:
    """Carry validated selection and identities through the ordinary solve chain."""

    request: PeriodCapture
    identity: dict[str, Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CapturedEntry:
    """Identify a durable entry without retaining its numerical inputs."""

    directory: Path
    digest: str


def prepare_period_capture(
    *,
    request: PeriodCapture,
    regimes: Mapping[str, Regime],
    execution: ResolvedExecution,
    enable_jit: bool,
    model_fingerprint: str,
    params_fingerprint: str,
) -> CaptureContext:
    """Refuse unsupported selections before solve planning or numerical dispatch."""
    validate_capture_route(
        periods=request.periods,
        regimes=regimes,
        execution=execution,
        enable_jit=enable_jit,
    )
    for name, period in request.periods:
        target = request.directory / f"{name}@{period}"
        if target.exists():
            raise FileExistsError(f"Period capture target already exists: {target}")
    return CaptureContext(
        request=request,
        identity=period_identity(
            model_fingerprint=model_fingerprint,
            params_fingerprint=params_fingerprint,
            source_identity=request.source_identity,
            execution=execution,
        ),
    )


def validate_capture_route(
    *,
    periods: tuple[tuple[str, int], ...],
    regimes: Mapping[str, Regime],
    execution: ResolvedExecution,
    enable_jit: bool,
) -> None:
    """Admit ordinary GridSearch values without opaque continuation payloads."""
    if not enable_jit or jax.config.jax_disable_jit or execution.invariant_block_widths:
        raise ValueError(
            "Period capture requires JIT and an unblocked GridSearch route."
        )
    if any(
        regime.solution.continuation_template is not None for regime in regimes.values()
    ):
        raise ValueError("Period capture does not support continuation payloads.")
    for name, period in periods:
        if name not in regimes or period not in regimes[name].active_periods:
            raise ValueError(f"Inactive period capture target: {name}@{period}")
        regime = regimes[name]
        kernel_type = type(regime.solution.period_kernels[period])
        route = f"{kernel_type.__module__}.{kernel_type.__qualname__}"
        if (
            route != _GRID_SEARCH_ROUTE
            or regime.stakeholders is not None
            or regime.gated_edges
        ):
            raise ValueError("Period capture supports ordinary GridSearch values only.")


def period_identity(
    *,
    model_fingerprint: str,
    params_fingerprint: str,
    source_identity: Mapping[str, str],
    execution: ResolvedExecution,
) -> dict[str, Any]:
    """Bind mathematical identity separately from strict execution provenance."""
    source_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for package in ("lcm", "_lcm"):
        for path in sorted((source_root / package).rglob("*.py")):
            digest.update(str(path.relative_to(source_root)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    devices = placed_devices_for_ids(
        submesh_device_ids=execution.device_ids, visible_device_ids=execution.device_ids
    )
    native_directory = _installed_native_directory()
    native_manifest = json.loads(
        (native_directory / "native-manifest.json").read_bytes()
    )
    return {
        "model": model_fingerprint,
        "parameters": params_fingerprint,
        "source": dict(source_identity),
        "installed_source_sha256": digest.hexdigest(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "native_build": native_manifest["fingerprint"],
        "native_libraries": {
            name: hashlib.sha256((native_directory / name).read_bytes()).hexdigest()
            for name in native_manifest["libraries"]
        },
        "execution": plain_metadata(execution),
        "devices": [
            {
                "id": int(device.id),
                "kind": device.device_kind,
                "platform": device.platform,
                "runtime": device.client.platform_version,
            }
            for device in devices
        ],
    }


def capture_public_entry(
    *,
    context: CaptureContext | None,
    regime: Regime,
    period: int,
    kernel_kwargs: dict[str, Any],
    compiled_cores: Mapping[str, PlannedCore],
    admission: Mapping[str, Mapping[str, int | None]],
) -> CapturedEntry | None:
    """Commit numerical entry inputs before the selected adapter is dispatched."""
    if (
        context is None
        or (kernel_kwargs["regime_name"], period) not in context.request.periods
    ):
        return None
    if (
        kernel_kwargs["next_regime_to_continuation"]
        or kernel_kwargs["next_edge_to_V_arr"]
    ):
        raise ValueError("Unsupported continuation or edge inputs in period capture.")
    if kernel_kwargs["selected_artifact_keys"] or any(
        core.donated_arguments for core in compiled_cores.values()
    ):
        raise ValueError(
            "Period capture supports values without donation or selected artifacts."
        )
    layouts = _period_layouts(
        regime=regime,
        period=period,
        kernel_kwargs=kernel_kwargs,
        compiled_cores=compiled_cores,
    )
    metadata = {
        "identity": context.identity,
        "regime": kernel_kwargs["regime_name"],
        "period": period,
        "age": float(kernel_kwargs["ages"].values[period]),
        "layouts": plain_metadata(layouts),
        "widths": {
            name: dict(core.tile_widths) for name, core in compiled_cores.items()
        },
        "admission": plain_metadata(admission),
        "optimized_hlo": optimized_hlo_records(compiled_cores=compiled_cores),
        "retain_replay": kernel_kwargs["retain_replay"],
    }
    arrays = {
        f"{channel}/{name}": np.asarray(array)
        for channel in ("next_regime_to_V_arr", "period_solution")
        for name, array in kernel_kwargs[channel].items()
    }
    directory = context.request.directory / f"{metadata['regime']}@{period}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    digest = write_period_archive(
        path=directory / "entry.h5", metadata=metadata, arrays=arrays
    )
    return CapturedEntry(directory=directory, digest=digest)


def complete_public_capture(
    *, entry: CapturedEntry, value: jax.Array | np.ndarray, seconds: float
) -> None:
    """Append a completed reference bound to the already durable entry."""
    array = np.asarray(value)
    write_period_archive(
        path=entry.directory / "completed.h5",
        metadata={"entry_sha256": entry.digest, "seconds": seconds},
        arrays={
            "value": array,
            "nan": np.isnan(array),
            "positive_inf": np.isposinf(array),
            "negative_inf": np.isneginf(array),
        },
    )


def load_period_capture(*, directory: Path) -> PeriodCaptureRecord:
    """Read a public capture, distinguishing entry-only from completed evidence."""
    metadata, _arrays, digest = read_period_archive(path=directory / "entry.h5")
    reference = None
    seconds = None
    if (directory / "completed.h5").exists():
        completed, arrays, _ = read_period_archive(path=directory / "completed.h5")
        if completed.get("entry_sha256") != digest or set(arrays) != {
            "value",
            "nan",
            "positive_inf",
            "negative_inf",
        }:
            raise ValueError("Completed reference belongs to a different entry.")
        reference = arrays["value"]
        for name, func in (
            ("nan", np.isnan),
            ("positive_inf", np.isposinf),
            ("negative_inf", np.isneginf),
        ):
            if not np.array_equal(arrays[name], func(reference)):
                raise ValueError("Completed reference nonfinite masks differ.")
        seconds = completed["seconds"]
        if type(seconds) is not float or not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid completed reference timing.")
        reference.flags.writeable = False
    return PeriodCaptureRecord(
        metadata=MappingProxyType(metadata),
        reference=reference,
        in_context_seconds=seconds,
    )


def optimized_hlo_records(
    *, compiled_cores: Mapping[str, PlannedCore]
) -> dict[str, Any]:
    """Require compiler evidence and identify complete optimized HLO modules."""
    options = _hlo.HloPrintOptions.canonical()
    options.canonicalize_computations = True
    options.print_ids = False
    options.print_large_constants = True
    options.print_backend_config = True
    records = {}
    for name, core in compiled_cores.items():
        if not isinstance(core.compiled, jax.stages.Compiled):
            raise TypeError("Period capture requires a compiled JAX executable.")
        executable = core.compiled.runtime_executable()
        if executable is None:
            raise ValueError(f"Core {name!r} exposes no runtime executable.")
        if any(device.platform == "gpu" for device in executable.local_devices()):
            if not any(
                isinstance(leaf, jax.ShapeDtypeStruct)
                and math.prod(leaf.shape) > 0
                and jax.dtypes.issubdtype(leaf.dtype, np.number)
                for leaf in jax.tree.leaves(core.compiled.out_info)
            ):
                raise ValueError(
                    f"Core {name!r}: GPU period capture/replay requires a nonempty "
                    "numeric output."
                )
            memory = compiler_memory_reservation(
                compiled=core.compiled, widths=core.tile_widths
            )
            if any(record.output_bytes == 0 for record in memory.records):
                raise ValueError(
                    f"Core {name!r}: unsupported GPU output layout; every compiler "
                    "memory record must report positive output allocation."
                )
            if any(record.peak_bytes == 0 for record in memory.records):
                raise ValueError(
                    f"Core {name!r}: compiler memory metadata is unavailable or "
                    "inconsistent for a nonempty output. Start a fresh process with "
                    "JAX_ENABLE_COMPILATION_CACHE=false and retain compiler "
                    "debug/HLO metadata."
                )
        modules = executable.hlo_modules()
        canonical = _canonicalize_optimized_hlo(
            "\n".join(module.to_string(options) for module in modules)
        )
        if not canonical:
            raise ValueError(f"Core {name!r} exposes no optimized HLO.")
        records[name] = {
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "text": canonical,
        }
    return records


def _canonicalize_optimized_hlo(text: str) -> str:
    """Normalize backend JSON ordering without discarding configuration values.

    XLA can reorder JSON members when deserializing the same executable. Skip
    quoted HLO strings when finding configuration fields, and retain everything
    outside their JSON values except insignificant trailing whitespace.
    """
    decoder = json.JSONDecoder()
    pieces = []
    cursor = 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|backend_config=', text):
        if match.group() != "backend_config=" or match.start() < cursor:
            continue
        value, end = decoder.raw_decode(text, match.end())
        pieces.append(text[cursor : match.end()])
        pieces.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
        cursor = end
    pieces.append(text[cursor:])
    return "\n".join(line.rstrip() for line in "".join(pieces).splitlines()).rstrip()


def plain_metadata(value: Any) -> Any:  # noqa: ANN401
    """Convert known metadata containers into non-executable JSON values."""
    if isinstance(value, enum.Enum):
        return plain_metadata(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: plain_metadata(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [plain_metadata(item) for item in items]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"Unsupported period metadata type: {type(value).__name__}")
