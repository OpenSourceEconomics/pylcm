"""Collected component-job raw results sit where the single-process run puts them.

The witness runs in a fresh process with eight CPU host devices, so a selected
device other than the default one exists, and compares every case's collected
panel with the single-process block-major reference in values, raw bytes, public
panel and per-leaf placement.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import jax
import pytest

from tests.solution._component_placement_witness import CASES

pytestmark = pytest.mark.slow


def test_collected_raw_results_keep_the_reference_placement() -> None:
    """Every case's raw leaves match the reference bytes and placement exactly.

    A population cut into several chunks is assembled on the reference's host
    assembly device; a single chunk keeps the reference compute layout.
    """
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("The fresh witness requires the active Pixi executable.")
    environment = {
        **os.environ,
        "JAX_ENABLE_X64": str(jax.config.read("jax_enable_x64")).lower(),
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "8",
        "XLA_FLAGS": " ".join(
            (
                os.environ.get("XLA_FLAGS", ""),
                "--xla_force_host_platform_device_count=8",
            )
        ).strip(),
    }
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
            "tests.solution._component_placement_witness",
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=1200,
    )
    report = json.loads(completed.stdout.splitlines()[-1])

    assert report == {
        str(index): {"mismatches": [], "error": None} for index in range(len(CASES))
    }
