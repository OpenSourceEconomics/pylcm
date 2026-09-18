"""Proposed one-allocation B1 task; preparation is not a release."""

import os
import subprocess
from pathlib import Path
from typing import Annotated

import pytask
from pytask import Product

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "reports" / "b1"


@pytask.mark.slurm(
    partition="mlgpu_short",
    account="ag_iame_gaudecker",
    cpus_per_task=8,
    mem="64G",
    time="02:00:00",
    extra="--gres=gpu:a40:4 --nodes=1 --ntasks=1",
    python_unbuffered=True,
)
def task_discrete_baseline_head(
    output: Annotated[Path, Product] = OUTPUT / "B1.json",
) -> None:
    """Execute only following explicit release and clean normal installations."""
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("PYLCM_B1_RELEASE") == "approved"
    head = Path(os.environ["PYLCM_B1_HEAD_ROOT"]).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    subprocess.run(
        [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "6900s",
            "pixi",
            "run",
            "--frozen",
            "--no-install",
            "--manifest-path",
            str(head / "pyproject.toml"),
            "-e",
            "tests-cuda13",
            "python",
            str(HERE / "run.py"),
            "--output",
            str(output.parent),
        ],
        env=environment,
        check=True,
    )
