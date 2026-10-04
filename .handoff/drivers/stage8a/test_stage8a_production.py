"""Production component-job CLI checks without constructing an ACA model."""

import os
import shutil
import subprocess
from pathlib import Path


def test_run_refuses_unassigned_slurm_process_before_gpu_import(
    *, tmp_path: Path
) -> None:
    """Only Slurm processes zero, one and two own production type jobs."""
    driver = Path(__file__).with_name("stage8a_production.py")
    manifest = os.environ.get(
        "STAGE8A_TEST_MANIFEST", str(driver.parents[3] / "pyproject.toml")
    )
    pixi_path = shutil.which("pixi")
    if pixi_path is None:
        raise RuntimeError("pixi is required for the production driver CLI check")
    completed = subprocess.run(  # noqa: S603
        [
            pixi_path,
            "run",
            "--as-is",
            "--manifest-path",
            manifest,
            "-e",
            "tests-cpu",
            "python",
            str(driver),
            "run",
            "--job-from-slurm-procid",
            "--plan-directory",
            str(tmp_path / "plan"),
            "--out",
            str(tmp_path / "out"),
            "--aca-slurm-src",
            str(tmp_path / "aca-slurm" / "src"),
            "--planning-receipt",
            str(tmp_path / "planning-receipt.json"),
            "--planning-receipt-sha256",
            "a" * 64,
        ],
        env={**os.environ, "SLURM_PROCID": "3"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        completed.returncode,
        "SLURM_PROCID must be 0, 1, or 2" in completed.stderr,
        "Traceback" in completed.stderr,
    ) == (2, True, False)
