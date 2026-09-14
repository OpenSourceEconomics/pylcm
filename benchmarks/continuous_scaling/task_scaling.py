"""Submit one isolated, source-pinned scaling task through pytask-slurm."""

import os
import subprocess
from pathlib import Path
from typing import Annotated

import pytask
from pytask import Product

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "continuous-scaling-rerun"


@pytask.mark.slurm(
    partition="mlgpu_short",
    account="ag_iame_gaudecker",
    cpus_per_task=16,
    mem="128G",
    time="05:00:00",
    extra="--gres=gpu:a40:8 --nodes=1 --ntasks=1",
    python_unbuffered=True,
)
def task_continuous_gpu_scaling(
    output: Annotated[Path, Product] = OUTPUT / "scaling-summary.json",
) -> None:
    """Run the fixed workload on one, three, four, six and eight physical GPUs."""
    assert os.environ.get("SLURM_JOB_ID"), "Requires the allocated pytask-slurm worker"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "17700s",
        "pixi",
        "run",
        "--frozen",
        "--no-install",
        "--manifest-path",
        str(ROOT / "pyproject.toml"),
        "-e",
        "tests-cuda13",
        "python",
        str(Path(__file__).with_name("run.py")),
        "--output",
        str(output.parent),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
