"""Collected component-job raw results land where the placement contract says.

A single-process block-major simulation cuts each invariant code's subjects into
chunks of the subject width. With several chunks in total, every raw leaf ends up
on the host assembly device: the first selected CPU on a CPU backend, CPU 0 on a
GPU backend. With one chunk, every raw leaf stays on the selected compute device.

Each test runs the campaign in a fresh process, checks the collected result's raw
bytes, analytical field values and per-leaf placement against that rule, saves the
collected solution and simulation, and reads both back in a second fresh process,
which must see the same bytes, placement, values and public panels.
"""

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_PRECISIONS = pytest.mark.parametrize("x64", [False, True], ids=["fp32", "fp64"])
_JIT = pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@_PRECISIONS
@_JIT
def test_collected_cpu_witness_sits_on_the_selected_host_device(
    *, tmp_path: Path, x64: bool, enable_jit: bool
) -> None:
    """Codes (2, 0, 2) at width 1 on CPU 1 form three chunks, all on CPU 1."""
    _assert_collected_result_meets_the_contract(
        directory=tmp_path,
        platform="cpu",
        codes=(2, 0, 2),
        width=1,
        device=1,
        expected_destination=("cpu", 1),
        x64=x64,
        enable_jit=enable_jit,
    )


@pytest.mark.requires(device="gpu")
@pytest.mark.coverage(backends=("gpu-small", "gpu-large"), precisions="64")
@pytest.mark.parametrize(
    ("codes", "expected_destination"),
    [((2, 0), ("cpu", 0)), ((2, 2), ("gpu", 0))],
    ids=["two-chunks", "one-chunk"],
)
@_PRECISIONS
@_JIT
def test_collected_gpu_result_sits_on_the_reference_device(
    *,
    tmp_path: Path,
    codes: tuple[int, ...],
    expected_destination: tuple[str, int],
    x64: bool,
    enable_jit: bool,
) -> None:
    """At width 8 on GPU 0, two codes give two chunks on CPU 0; one code stays on GPU 0.

    The two-code population fits one width in total but still forms one chunk
    per code, so its result is assembled on the host.
    """
    _assert_collected_result_meets_the_contract(
        directory=tmp_path,
        platform="gpu",
        codes=codes,
        width=8,
        device=0,
        expected_destination=expected_destination,
        x64=x64,
        enable_jit=enable_jit,
    )


def _assert_collected_result_meets_the_contract(
    *,
    directory: Path,
    platform: str,
    codes: tuple[int, ...],
    width: int,
    device: int,
    expected_destination: tuple[str, int],
    x64: bool,
    enable_jit: bool,
) -> None:
    assert (
        _oracle_destination(codes=codes, width=width, platform=platform, device=device)
        == expected_destination
    )
    arguments = ["--devices", str(device), "--codes", ",".join(map(str, codes))]
    arguments += ["--width", str(width), *([] if enable_jit else ["--eager"])]
    produced = _run_witness(
        mode="produce",
        directory=directory,
        arguments=arguments,
        platform=platform,
        x64=x64,
    )
    reloaded = _run_witness(
        mode="reload", directory=directory, arguments=[], platform=platform, x64=x64
    )

    assert produced["misplaced"] == []
    destination = f"{expected_destination[0]}:{expected_destination[1]}"
    assert {
        path: (leaf["devices"], leaf["memory_kind"])
        for path, leaf in produced["raw"].items()
    } == {
        path: (
            [list(expected_destination)],
            produced["default_memory_kinds"][destination],
        )
        for path in produced["raw"]
    }
    if expected_destination[0] == "cpu":
        assert {leaf["sharding"] for leaf in produced["raw"].values()} == {
            "SingleDeviceSharding"
        }
    assert {
        leaf["dtype"] for leaf in produced["raw"].values() if leaf["dtype"][1] == "f"
    } == {"<f8" if x64 else "<f4"}

    wealth = np.linspace(1.0, 2.0, len(codes))
    pref_type = np.asarray(codes, dtype=float)
    rtol = 1e-12 if x64 else 1e-6
    fields = produced["fields"]
    assert fields["live_choice"] == [1] * len(codes)
    assert fields["live_pref_type"] == list(codes)
    np.testing.assert_allclose(fields["live_wealth"], wealth, rtol=rtol)
    np.testing.assert_allclose(
        fields["live_value"], 1.9 * (wealth + pref_type) + 1.0, rtol=rtol
    )
    np.testing.assert_allclose(fields["dead_value"], wealth + pref_type, rtol=rtol)

    assert {key: reloaded[key] for key in ("raw", "values", "panels", "fields")} == {
        key: produced[key] for key in ("raw", "values", "panels", "fields")
    }
    assert reloaded["simulation_load_solution_values"] == produced["values"]
    assert reloaded["load_solution_values"] == produced["values"]


def _oracle_destination(
    *, codes: tuple[int, ...], width: int, platform: str, device: int
) -> tuple[str, int]:
    """Return the single-device destination the placement contract prescribes."""
    n_chunks = sum(-(-count // width) for count in Counter(codes).values())
    if n_chunks > 1:
        return ("cpu", device if platform == "cpu" else 0)
    return (platform, device)


def _run_witness(
    *,
    mode: str,
    directory: Path,
    arguments: list[str],
    platform: str,
    x64: bool,
) -> dict:
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("The fresh witness requires the active Pixi executable.")
    environment: dict[str, str] = {**os.environ, "JAX_ENABLE_X64": str(x64).lower()}
    if platform == "cpu":
        environment |= {
            "JAX_PLATFORMS": "cpu",
            "JAX_NUM_CPU_DEVICES": "8",
            "XLA_FLAGS": " ".join(
                (
                    os.environ.get("XLA_FLAGS", ""),
                    "--xla_force_host_platform_device_count=8",
                )
            ).strip(),
        }
    else:
        # The test process already holds its preallocated share of the GPU.
        environment.pop("JAX_PLATFORMS", None)
        environment.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    completed = subprocess.run(  # noqa: S603
        [
            pixi,
            "run",
            "--as-is",
            "--manifest-path",
            os.environ["PIXI_PROJECT_MANIFEST"],
            "-e",
            os.environ["PIXI_ENVIRONMENT_NAME"],
            "python",
            "-m",
            "tests.solution._component_acceptance_witness",
            mode,
            str(directory),
            *arguments,
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])
