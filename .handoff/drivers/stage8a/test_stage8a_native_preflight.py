"""Strict native admission cannot turn tiny CPU plumbing into GPU acceptance."""

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import pytest
from test_stage8a_population import owner_driver as _imported_owner_driver

from _lcm.egm.upper_envelope._exact_affine import ffi
from tests.ci import probe_native

owner_driver = _imported_owner_driver


@pytest.mark.parametrize("payload", ["ready", "stale", "foreign_version"])
def test_native_preflight_refuses_cpu_or_stale_payload_before_aca_import(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    """Real CPU execution and a stale installed payload cannot admit production."""
    root = Path(os.environ["STAGE8A_RECEIPT_CORE_ROOT"])
    monkeypatch.setenv("PYLCM_DIR", str(root))
    if payload == "stale":
        monkeypatch.setattr(
            probe_native,
            "probe",
            lambda **_kwargs: probe_native.ProbeResult(
                exit_code=2,
                status="stale",
                detail="the installed manifest differs",
            ),
        )
    elif payload == "foreign_version":
        installed_version = importlib.metadata.version
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: (
                "foreign-pylcm" if name == "pylcm" else installed_version(name)
            ),
        )
    caught = None
    try:
        owner_driver._native_preflight(aca_slurm_src=Path("/no-owner-source/src"))
    except (AttributeError, RuntimeError) as error:
        caught = (type(error).__name__, str(error))

    assert (jax.default_backend(), caught) == (
        "cpu",
        ("RuntimeError", "Stage 8A requires an actual GPU backend")
        if payload == "ready"
        else (
            "RuntimeError",
            "Stage 8A native payload is not READY: stale",
        )
        if payload == "stale"
        else (
            "RuntimeError",
            "Stage 8A installed and source pylcm versions differ",
        ),
    )


@pytest.mark.parametrize(
    "change", ["count", "kind", "precision", "uuid", "prefix", "library"]
)
def test_native_preflight_refuses_inexact_gpu_allocation(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """A recording GPU boundary cannot admit missing devices or identity evidence.

    These controls exercise guard plumbing; the real runtime remains CPU.
    """

    root = Path(os.environ["STAGE8A_RECEIPT_CORE_ROOT"])
    if change == "library":
        monkeypatch.setattr(ffi, "_CUDA_LIBRARY", None)
    monkeypatch.setenv("PYLCM_DIR", str(root))
    if change != "prefix":
        monkeypatch.setattr(sys, "prefix", str(root / ".pixi/envs/benchmarks-cuda12"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(8))))
    monkeypatch.setattr(
        probe_native,
        "probe",
        lambda **_kwargs: probe_native.ProbeResult(
            exit_code=0,
            status="ready",
            detail="recording admission boundary",
        ),
    )
    devices = [
        SimpleNamespace(id=index, device_kind="NVIDIA A40") for index in range(8)
    ]
    if change == "count":
        devices.pop()
    elif change == "kind":
        devices[-1].device_kind = "NVIDIA A100"
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    monkeypatch.setattr(
        jax, "config", SimpleNamespace(jax_enable_x64=change == "precision")
    )
    output = "\n".join(f"GPU-{index}, NVIDIA A40" for index in range(8))
    if change == "uuid":
        output = "\n".join("GPU-0, NVIDIA A40" for _ in range(8))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=output,
            returncode=0,
        ),
    )
    # The allocated-GPU boundary is recorded; no CUDA payload exists locally.
    monkeypatch.setattr(owner_driver, "_file_sha256", lambda **_kwargs: "f" * 64)
    caught = None
    try:
        owner_driver._native_preflight(aca_slurm_src=Path("/no-owner-source/src"))
    except (AttributeError, RuntimeError) as error:
        caught = str(error)

    assert (
        caught
        == {
            "count": "Stage 8A requires eight distinct local A40 devices",
            "kind": "Stage 8A requires eight distinct local A40 devices",
            "precision": "Stage 8A requires fp32",
            "uuid": "Stage 8A requires eight distinct visible A40 GPU UUIDs",
            "prefix": "Stage 8A requires its preinstalled benchmarks-cuda12 prefix",
            "library": "Stage 8A CUDA native library is unavailable",
        }[change]
    )
