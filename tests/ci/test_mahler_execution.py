"""Runner capacity is measured for precisely the workload's selected devices."""

import ctypes
import json
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import jax
import pytest

from tests.ci.mahler_execution import (
    create_mahler_execution_config,
    create_mahler_gpu_model,
)

_MIB = 2**20
_UUID_A = "GPU-11111111-1111-1111-1111-111111111111"
_UUID_B = "GPU-22222222-2222-2222-2222-222222222222"
_UUID_UNUSED = "GPU-33333333-3333-3333-3333-333333333333"
_FORMULA = "min(physical_free_bytes, allocator_limit_bytes - allocator_used_bytes) // 2"
_NVIDIA_STDOUT = (
    f"{_UUID_UNUSED}, unused, 580.1, N/A, N/A\n"
    f"{_UUID_B}, selected B, 580.1, 24, 6\n"
    f"{_UUID_A}, selected A, 580.1, 16, 10\n"
)


def test_workload_model_uses_the_device_whose_uuid_determined_its_budget(
    *, devices: tuple[_Device, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real model construction uses the selected addressable device's receipt."""
    _install_external_observers(monkeypatch=monkeypatch)
    selected = replace(devices[0], id=jax.devices()[0].id)
    monkeypatch.setattr(jax, "local_devices", lambda: (selected, devices[1]))
    report_path = tmp_path / "workload.json"

    model = create_mahler_gpu_model(report_path=report_path)

    receipt = _read_receipt(path=report_path, status="accepted")
    assert (
        model.execution_devices,
        model._execution.device_memory_bytes,
        dict(model._execution.axis_widths),
        [(record["id"], record["uuid"]) for record in receipt["devices"]],
    ) == (
        (selected.id,),
        (5 * _MIB + 3) // 2,
        {"action_product": 64, "cell": 4096},
        [(selected.id, _UUID_A)],
    )


@dataclass(frozen=True, kw_only=True)
class _Device:
    """External JAX metadata, with ids distinct from visible CUDA ordinals."""

    id: int
    """Public JAX device identity used in the resulting configuration."""
    local_hardware_id: int
    """CUDA ordinal after the process's visibility mapping."""
    stats: dict[str, object] | None
    """Raw allocator counts, including deliberately invalid refusal inputs."""
    device_kind: str = "fake selected NVIDIA device"
    """Device description retained in the observation receipt."""
    platform: str = "gpu"
    """JAX backend platform metadata."""

    def memory_stats(self) -> dict[str, object] | None:
        """Return the raw allocator observation supplied by the test."""
        return self.stats


@pytest.fixture
def devices() -> tuple[_Device, ...]:
    """A's allocator headroom and B's physical headroom are the two constraints."""
    return (
        _Device(
            id=42,
            local_hardware_id=1,
            stats={"bytes_limit": 8 * _MIB + 3, "bytes_in_use": 3 * _MIB},
        ),
        _Device(
            id=3,
            local_hardware_id=0,
            stats={"bytes_limit": 20 * _MIB, "bytes_in_use": _MIB},
        ),
    )


def _install_external_observers(
    *, monkeypatch: pytest.MonkeyPatch, stdout: str = _NVIDIA_STDOUT
) -> list[int]:
    """Emulate only nvidia-smi and CUDA's ordinal-to-UUID driver boundary."""
    ordinals: list[int] = []

    def cu_init(flags: int) -> int:
        """Accept the CUDA initialization flag used by the external API."""
        assert flags == 0
        return 0

    # keyword-only-exempt: library-callback=ctypes
    def cu_device_get(out: Any, ordinal: int) -> int:
        """Write a CUDA handle distinct from both the JAX id and ordinal."""
        ordinals.append(ordinal)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = {0: 17, 1: 71}[ordinal]
        return 0

    # keyword-only-exempt: library-callback=ctypes
    def cu_device_get_uuid(out: Any, device: int | ctypes.c_int) -> int:
        """Write a real 16-byte UUID through CUDA's output-pointer interface."""
        handle = device.value if isinstance(device, ctypes.c_int) else device
        uuid = {17: _UUID_B, 71: _UUID_A}[handle]
        ctypes.memmove(out, UUID(uuid.removeprefix("GPU-")).bytes, 16)
        return 0

    driver = SimpleNamespace(
        cuInit=cu_init,
        cuDeviceGet=cu_device_get,
        cuDeviceGetUuid_v2=cu_device_get_uuid,
    )

    def load_driver(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        """Return the driver boundary without loading a native library."""
        return driver

    def run_nvidia_smi(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return raw physical metadata in the requested field and unit format."""
        command = args[0] if args else kwargs["args"]
        assert Path(command[0]).name == "nvidia-smi"
        query = next(arg for arg in command if arg.startswith("--query-gpu="))
        assert query.removeprefix("--query-gpu=").split(",") == [
            "uuid",
            "name",
            "driver_version",
            "memory.total",
            "memory.free",
        ]
        format_arg = next(arg for arg in command if arg.startswith("--format="))
        assert set(format_arg.removeprefix("--format=").split(",")) == {
            "csv",
            "noheader",
            "nounits",
        }
        return subprocess.CompletedProcess(command, 0, stdout, "diagnostic stderr\n")

    monkeypatch.setattr(ctypes, "CDLL", load_driver)
    monkeypatch.setattr(subprocess, "run", run_nvidia_smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{_UUID_B},{_UUID_A}")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return ordinals


def _read_receipt(*, path: Path, status: str) -> dict[str, Any]:
    """Check durable status, the budget rule, and timezone-aware observation times."""
    receipt = json.loads(path.read_text())
    assert receipt["status"] == status
    assert receipt["formula"] == _FORMULA
    assert receipt["headroom_divisor"] == 2
    assert receipt["units"] == {
        "nvidia_smi_memory": "MiB",
        "jax_memory_stats": "bytes",
        "budget": "bytes",
    }
    start = datetime.fromisoformat(receipt["started_at"])
    finish = datetime.fromisoformat(receipt["finished_at"])
    assert start.utcoffset() == finish.utcoffset() == UTC.utcoffset(None)
    assert start <= finish
    for observation in (receipt.get("nvidia_smi"), *receipt.get("devices", ())):
        if observation and "observed_at" in observation:
            observed_at = datetime.fromisoformat(observation["observed_at"])
            assert observed_at.utcoffset() == UTC.utcoffset(None)
            assert start <= observed_at <= finish
    assert receipt["environment"]["CUDA_VISIBLE_DEVICES"] == f"{_UUID_B},{_UUID_A}"
    return receipt


def test_selected_devices_use_uuid_matched_physical_and_allocator_headroom(
    *, devices: tuple[_Device, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Remapping and an unrelated malformed row preserve the exact selected budget."""
    ordinals = _install_external_observers(monkeypatch=monkeypatch)
    report_path = tmp_path / "capacity.json"

    config = create_mahler_execution_config(
        devices=cast("tuple[jax.Device, ...]", devices), report_path=report_path
    )

    assert config.devices == (42, 3)
    assert dict(config.axis_widths) == {"action_product": 64, "cell": 4096}
    assert config.device_memory_bytes == (5 * _MIB + 3) // 2
    assert ordinals == [1, 0]
    receipt = _read_receipt(path=report_path, status="accepted")
    assert receipt["budget_bytes"] == config.device_memory_bytes
    assert receipt["nvidia_smi"]["stdout"] == _NVIDIA_STDOUT
    assert receipt["nvidia_smi"]["stderr"] == "diagnostic stderr\n"
    assert receipt["nvidia_smi"]["returncode"] == 0
    assert datetime.fromisoformat(receipt["nvidia_smi"]["observed_at"]).tzinfo
    assert [record["uuid"] for record in receipt["devices"]] == [_UUID_A, _UUID_B]
    for device, record in zip(devices, receipt["devices"], strict=True):
        assert record["id"] == device.id
        assert record["local_hardware_id"] == device.local_hardware_id
        assert record["device_kind"] == device.device_kind
        assert record["platform"] == device.platform
        assert record["memory_stats"] == device.stats
        assert datetime.fromisoformat(record["observed_at"]).tzinfo


def test_unselected_allocator_and_physical_capacity_do_not_constrain_budget(
    *, devices: tuple[_Device, ...], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selecting B alone uses B's six MiB physical headroom before halving."""
    ordinals = _install_external_observers(monkeypatch=monkeypatch)
    report_path = tmp_path / "capacity.json"
    config = create_mahler_execution_config(
        devices=cast("tuple[jax.Device, ...]", (devices[1],)), report_path=report_path
    )
    assert config.devices == (3,)
    assert config.device_memory_bytes == 3 * _MIB
    assert ordinals == [0]
    receipt = _read_receipt(path=report_path, status="accepted")
    assert [record["uuid"] for record in receipt["devices"]] == [_UUID_B]


@pytest.mark.parametrize(
    "stats",
    [
        None,
        {"bytes_in_use": 0},
        {"bytes_limit": _MIB},
        {"bytes_limit": True, "bytes_in_use": 0},
        {"bytes_limit": _MIB, "bytes_in_use": False},
        {"bytes_limit": _MIB, "bytes_in_use": 1.5},
        {"bytes_limit": "1048576", "bytes_in_use": 0},
        {"bytes_limit": -1, "bytes_in_use": 0},
        {"bytes_limit": _MIB, "bytes_in_use": -1},
        {"bytes_limit": _MIB, "bytes_in_use": _MIB + 1},
        {"bytes_limit": 0, "bytes_in_use": 0},
        {"bytes_limit": 1, "bytes_in_use": 0},
    ],
)
def test_invalid_selected_allocator_observation_is_recorded_and_refused(
    *,
    stats: dict[str, object] | None,
    devices: tuple[_Device, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An absent, inexact, inconsistent, or unusably small count cannot admit work."""
    _install_external_observers(monkeypatch=monkeypatch)
    report_path = tmp_path / "capacity.json"
    with pytest.raises(ValueError, match="Cannot configure the Mahler GPU workload"):
        create_mahler_execution_config(
            devices=cast("tuple[jax.Device, ...]", (replace(devices[0], stats=stats),)),
            report_path=report_path,
        )
    receipt = _read_receipt(path=report_path, status="refused")
    assert receipt["error"]
    assert "budget_bytes" not in receipt
    assert receipt["devices"][0]["memory_stats"] == stats


@pytest.mark.parametrize(
    "rows",
    [
        f"{_UUID_B}, selected B, 580.1, 24, 6\n",
        f"{_UUID_A}, A, 580.1, 16, 10\n{_UUID_A}, duplicate, 580.1, 16, 10\n",
        f"{_UUID_A}, A, 580.1, 16, N/A\n",
        f"{_UUID_A}, A, 580.1, 16, True\n",
        f"{_UUID_A}, A, 580.1, 16, 1.5\n",
        f"{_UUID_A}, A, 580.1, 16, -1\n",
        f"{_UUID_A}, A, 580.1, 16, 17\n",
        f"{_UUID_A}, A, 580.1, 0, 0\n",
    ],
)
def test_ambiguous_or_invalid_selected_physical_observation_is_recorded_and_refused(
    *,
    rows: str,
    devices: tuple[_Device, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """UUID matching and physical memory consistency are required for acceptance."""
    _install_external_observers(monkeypatch=monkeypatch, stdout=rows)
    report_path = tmp_path / "capacity.json"
    with pytest.raises(ValueError, match="Cannot configure the Mahler GPU workload"):
        create_mahler_execution_config(
            devices=cast("tuple[jax.Device, ...]", (devices[0],)),
            report_path=report_path,
        )
    receipt = _read_receipt(path=report_path, status="refused")
    assert receipt["error"]
    assert "budget_bytes" not in receipt
    assert receipt["nvidia_smi"]["stdout"] == rows
    assert datetime.fromisoformat(receipt["nvidia_smi"]["observed_at"]).tzinfo


def test_empty_device_selection_is_recorded_and_refused(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty workload device set has no admissible per-device memory budget."""
    _install_external_observers(monkeypatch=monkeypatch)
    report_path = tmp_path / "capacity.json"
    with pytest.raises(ValueError, match="Cannot configure the Mahler GPU workload"):
        create_mahler_execution_config(devices=(), report_path=report_path)
    receipt = _read_receipt(path=report_path, status="refused")
    assert receipt["error"]
    assert "budget_bytes" not in receipt
