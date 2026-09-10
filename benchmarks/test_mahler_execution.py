"""Controls for the Mahler ASV capacity admission policy."""

# ruff: noqa: SLF001

import ctypes
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import jax
import pytest

from benchmarks import _mahler_execution

_MIB = 2**20
_UUID_A = "GPU-11111111-1111-1111-1111-111111111111"
_UUID_B = "GPU-22222222-2222-2222-2222-222222222222"


@dataclass(frozen=True)
class _Device:
    id: int
    local_hardware_id: int
    stats: dict[str, int] | None
    platform: str = "gpu"
    device_kind: str = "fake GPU"

    def memory_stats(self) -> dict[str, int] | None:
        return self.stats


def _install_capacity_boundaries(
    *, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> list[int]:
    ordinals: list[int] = []

    # keyword-only-exempt: library-callback=ctypes
    def _device_get(out: Any, ordinal: int) -> int:
        ordinals.append(ordinal)
        ctypes.cast(out, ctypes.POINTER(ctypes.c_int))[0] = {0: 10, 1: 11}[ordinal]
        return 0

    # keyword-only-exempt: library-callback=ctypes
    def _device_uuid(out: Any, handle: int) -> int:
        uuid = {10: _UUID_B, 11: _UUID_A}[handle]
        ctypes.memmove(out, UUID(uuid.removeprefix("GPU-")).bytes, 16)
        return 0

    driver = SimpleNamespace(
        cuInit=lambda _flags: 0,
        cuDeviceGet=_device_get,
        cuDeviceGetUuid_v2=_device_uuid,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args: driver)
    monkeypatch.setattr(
        _mahler_execution.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout, ""),
    )
    monkeypatch.setattr(_mahler_execution, "_source_hash", lambda: "a" * 40)
    return ordinals


def test_capacity_policy_uses_uuid_matched_limiting_headroom(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Visible ordinal order cannot mix physical and allocator observations."""
    jax.config.update("jax_enable_x64", val=True)
    stdout = f"{_UUID_B}, B, 1, 24, 6\n{_UUID_A}, A, 1, 16, 10\n"
    ordinals = _install_capacity_boundaries(monkeypatch, stdout=stdout)
    report = tmp_path / "capacity.json"
    devices = (
        _Device(42, 1, {"bytes_limit": 8 * _MIB + 3, "bytes_in_use": 3 * _MIB}),
        _Device(3, 0, {"bytes_limit": 20 * _MIB, "bytes_in_use": _MIB}),
    )

    config = _mahler_execution.create_mahler_execution_config(
        devices=cast("tuple[jax.Device, ...]", devices), report_path=report
    )

    assert config.devices == (42, 3)
    assert dict(config.axis_widths) == {"action_product": 64, "cell": 4096}
    assert config.device_memory_bytes == (5 * _MIB + 3) // 2
    assert ordinals == [1, 0]
    receipt = json.loads(report.read_text())
    assert receipt["status"] == "accepted"
    assert receipt["policy"] == _mahler_execution.POLICY_LABEL
    assert receipt["precision"] == "64"
    assert [row["uuid"] for row in receipt["devices"]] == [_UUID_A, _UUID_B]


@pytest.mark.parametrize(
    ("stats", "stdout"),
    [
        (None, f"{_UUID_A}, A, 1, 16, 10\n"),
        (
            {"bytes_limit": _MIB, "bytes_in_use": _MIB + 1},
            f"{_UUID_A}, A, 1, 16, 10\n",
        ),
        ({"bytes_limit": _MIB, "bytes_in_use": 0}, f"{_UUID_A}, A, 1, 16, 0\n"),
        (
            {"bytes_limit": _MIB, "bytes_in_use": 0},
            f"{_UUID_A}, A, 1, 16, 0\n{_UUID_A}, A, 1, 16, 0\n",
        ),
    ],
)
def test_capacity_policy_refuses_bad_or_exhausted_observations(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stats: dict[str, int] | None,
    stdout: str,
) -> None:
    """A receipt is durable even when capacity admission rejects the workload."""
    jax.config.update("jax_enable_x64", val=True)
    _install_capacity_boundaries(monkeypatch, stdout=stdout)
    report = tmp_path / "capacity.json"

    with pytest.raises(ValueError, match="Cannot configure the Mahler GPU benchmark"):
        _mahler_execution.create_mahler_execution_config(
            devices=cast("tuple[jax.Device, ...]", (_Device(42, 1, stats),)),
            report_path=report,
        )

    receipt = json.loads(report.read_text())
    assert receipt["status"] == "refused"
    assert receipt["error"]
    assert receipt["source_hash"] == "a" * 40


def test_fp64_establishment_refuses_an_already_imported_fp32_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated setup cannot silently repair precision after model import."""
    monkeypatch.setitem(sys.modules, _mahler_execution._WORKLOAD_MODULE, object())
    jax.config.update("jax_enable_x64", val=False)

    with pytest.raises(RuntimeError, match="imported before fp64"):
        _mahler_execution._establish_fp64_before_workload_import()

    jax.config.update("jax_enable_x64", val=True)


def test_receipt_paths_are_unique_and_retained_under_asv_directory(
    tmp_path: Path,
) -> None:
    """Each construction leaves a separately collectable capacity receipt."""
    first = _mahler_execution._unique_receipt_path(directory=tmp_path)
    second = _mahler_execution._unique_receipt_path(directory=tmp_path)

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert ".asv" in str(_mahler_execution._RECEIPT_DIRECTORY)
