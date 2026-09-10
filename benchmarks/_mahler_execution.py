"""Capacity-admitted fp64 configuration for the Mahler-Yum ASV series."""

import csv
import ctypes
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    import jax

    from lcm import ExecutionConfig, Model


POLICY_LABEL = "capacity-half-a64-c4096-v1-fp64"
_FORMULA = "min(physical_free_bytes, allocator_limit_bytes - allocator_used_bytes) // 2"
_WORKLOAD_MODULE = "lcm_examples.mahler_yum_2024"
_RECEIPT_DIRECTORY = Path(__file__).resolve().parent.parent / ".asv" / "mahler-receipts"


def create_mahler_gpu_model() -> tuple[Model, Path]:
    """Build one fp64 model after recording its selected GPU capacity."""
    _establish_fp64_before_workload_import()
    import jax

    from lcm_examples.mahler_yum_2024 import create_model

    report_path = _unique_receipt_path(directory=_RECEIPT_DIRECTORY)
    config = create_mahler_execution_config(
        devices=tuple(jax.local_devices()[:1]), report_path=report_path
    )
    _assert_fp64()
    return create_model(execution_config=config), report_path


def create_mahler_execution_config(
    *, devices: tuple[jax.Device, ...], report_path: Path
) -> ExecutionConfig:
    """Reserve half the limiting physical or allocator headroom for one GPU."""
    import jax

    observations: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "status": "collecting",
        "policy": POLICY_LABEL,
        "source_hash": None,
        "started_at": _timestamp(),
        "formula": _FORMULA,
        "headroom_divisor": 2,
        "axis_widths": {"action_product": 64, "cell": 4096},
        "units": {
            "nvidia_smi_memory": "MiB",
            "jax_memory_stats": "bytes",
            "budget": "bytes",
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_CUDA_VISIBLE_DEVICES",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
        "precision": "64" if jax.config.x64_enabled else "32",
        "jax_version": jax.__version__,
        "devices": observations,
    }
    try:
        receipt["source_hash"] = _source_hash()
        _assert_fp64()
        config = _configure_observed_devices(
            devices=devices, observations=observations, receipt=receipt
        )
    except Exception as error:
        receipt.update(status="refused", error=f"{type(error).__name__}: {error}")
        raise ValueError(
            f"Cannot configure the Mahler GPU benchmark: {error}"
        ) from error
    else:
        receipt.update(status="accepted", budget_bytes=config.device_memory_bytes)
        return config
    finally:
        receipt["finished_at"] = _timestamp()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(report_path)


def _establish_fp64_before_workload_import() -> None:
    """Establish fp64 before importing the workload and reject precision drift."""
    import jax

    if _WORKLOAD_MODULE in sys.modules and not jax.config.x64_enabled:
        raise RuntimeError("Mahler-Yum workload was imported before fp64 was enabled.")
    if _WORKLOAD_MODULE not in sys.modules:
        jax.config.update("jax_enable_x64", val=True)
    _assert_fp64()


def _assert_fp64() -> None:
    import jax

    if not jax.config.x64_enabled:
        raise RuntimeError("Mahler-Yum fp64 benchmark could not enable x64 precision.")


def _configure_observed_devices(
    *,
    devices: tuple[jax.Device, ...],
    observations: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> ExecutionConfig:
    """Collect selected observations and apply the fixed workload policy."""
    from lcm import ExecutionConfig

    if not devices:
        raise ValueError("The workload must select at least one GPU.")
    rows = _read_nvidia_metadata(receipt=receipt)
    driver = _load_cuda_driver()
    remaining: list[int] = []
    selected_ids: list[int] = []
    selected_uuids: list[str] = []
    for device in devices:
        observation = {
            "id": device.id,
            "local_hardware_id": device.local_hardware_id,
            "cuda_visible_ordinal": device.local_hardware_id,
            "platform": device.platform,
            "device_kind": device.device_kind,
            "observed_at": _timestamp(),
            "memory_stats": device.memory_stats(),
        }
        observations.append(observation)
        device_id, uuid, available = _available_device_bytes(
            observation=observation, rows=rows, driver=driver
        )
        if device_id in selected_ids or uuid in selected_uuids:
            raise ValueError("Selected GPU ids and UUIDs must be distinct.")
        selected_ids.append(device_id)
        selected_uuids.append(uuid)
        remaining.append(available)
    budget = min(remaining) // 2
    if budget <= 0:
        raise ValueError("Measured headroom yields no positive workload budget.")
    return ExecutionConfig(
        devices=tuple(selected_ids),
        axis_widths={"action_product": 64, "cell": 4096},
        device_memory_bytes=budget,
    )


def _available_device_bytes(
    *, observation: dict[str, Any], rows: list[list[str]], driver: ctypes.CDLL
) -> tuple[int, str, int]:
    device_id = _nonnegative_integer(value=observation["id"], name="JAX id")
    ordinal = _nonnegative_integer(
        value=observation["local_hardware_id"], name="CUDA ordinal"
    )
    if observation["platform"] != "gpu":
        raise ValueError("The workload must select CUDA GPU devices.")
    uuid = _get_cuda_uuid(driver=driver, ordinal=ordinal)
    observation["uuid"] = uuid
    matched = [row for row in rows if row and row[0].strip() == uuid]
    if len(matched) != 1 or len(matched[0]) != 5:
        raise ValueError(f"Expected one complete NVIDIA record for {uuid}.")
    physical = matched[0]
    total = _mib_bytes(value=physical[3], name="physical total")
    free = _mib_bytes(value=physical[4], name="physical free")
    if free > total:
        raise ValueError(f"Physical free memory exceeds total for {uuid}.")
    stats = observation["memory_stats"]
    if not isinstance(stats, dict):
        raise TypeError(f"Missing JAX allocator statistics for {uuid}.")
    limit = _nonnegative_integer(value=stats.get("bytes_limit"), name="allocator limit")
    used = _nonnegative_integer(value=stats.get("bytes_in_use"), name="allocator use")
    if used > limit:
        raise ValueError(f"Allocator use exceeds its limit for {uuid}.")
    observation.update(
        physical_total_bytes=total,
        physical_free_bytes=free,
        allocator_limit_bytes=limit,
        allocator_used_bytes=used,
    )
    return device_id, uuid, min(free, limit - used)


def _read_nvidia_metadata(*, receipt: dict[str, Any]) -> list[list[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    observation: dict[str, Any] = {"command": command, "observed_at": _timestamp()}
    receipt["nvidia_smi"] = observation
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=10, check=False
    )
    observation.update(
        stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
    )
    if result.returncode:
        raise ValueError(f"NVIDIA memory query failed with status {result.returncode}.")
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def _load_cuda_driver() -> ctypes.CDLL:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    driver.cuDeviceGetUuid_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid_v2.restype = ctypes.c_int
    result = driver.cuInit(0)
    if result:
        raise ValueError(f"CUDA initialization for device identity failed: {result}.")
    return driver


def _get_cuda_uuid(*, driver: ctypes.CDLL, ordinal: int) -> str:
    handle = ctypes.c_int()
    result = driver.cuDeviceGet(ctypes.byref(handle), ordinal)
    if result:
        raise ValueError(f"CUDA ordinal {ordinal} has no device handle: {result}.")
    raw_uuid = (ctypes.c_ubyte * 16)()
    result = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), handle.value)
    if result:
        raise ValueError(f"CUDA UUID query failed for ordinal {ordinal}: {result}.")
    return f"GPU-{UUID(bytes=bytes(raw_uuid))}"


def _source_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    source_hash = result.stdout.strip()
    if result.returncode or len(source_hash) != 40:
        raise RuntimeError("Cannot record the benchmark source hash.")
    return source_hash


def _unique_receipt_path(*, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{POLICY_LABEL}-{uuid4().hex}.json"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _nonnegative_integer(*, value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
    return value


def _mib_bytes(*, value: str, name: str) -> int:
    return _nonnegative_integer(value=int(value), name=name) * 2**20
