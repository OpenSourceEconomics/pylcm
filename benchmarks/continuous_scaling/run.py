"""Run a bounded scaling experiment from an installed, pulled source tree."""

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run_command(
    *, arguments: list[str], log: Path, environment: dict[str, str], limit: int
) -> None:
    """Preserve the command and its own exit status, including timeouts."""
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=15s",
        f"{limit}s",
        "pixi",
        "run",
        "--frozen",
        "--no-install",
        "--manifest-path",
        str(ROOT / "pyproject.toml"),
        "-e",
        "tests-cuda13",
        *arguments,
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log.with_suffix(".command.json").write_text(
        json.dumps({"argv": command, "exit_code": result.returncode}, indent=2) + "\n"
    )
    result.check_returncode()


def prepare_harness(output: Path) -> Path:
    """Apply reviewed topology-only patches to exactly sealed test fixtures."""
    harness = output / "harness"
    harness.mkdir()
    seals = json.loads((HERE / "fixture-seals.json").read_text())
    for source_name, destination in (
        ("test_continuous_assets_sharding.py", "gpu_assets.py"),
        ("test_continuous_transfer_admission.py", "gpu_admission.py"),
    ):
        source = ROOT / "tests" / source_name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == seals[source_name]
        subprocess.run(
            [
                "patch",
                "--batch",
                "--fuzz=0",
                "--output",
                str(harness / destination),
                str(source),
                str(HERE / f"{destination}.patch"),
            ],
            check=True,
        )
    return harness


def check_xml(path: Path) -> None:
    """Require all six native cases to pass, without skips or empty selection."""
    cases = ET.parse(path).getroot().findall(".//testcase")  # noqa: S314 - own pytest XML
    assert len(cases) == 6
    assert all(
        case.find(tag) is None
        for case in cases
        for tag in ("failure", "error", "skipped")
    )


def run_scaling(output: Path) -> None:
    """Validate the installed backend before semantic checks and matched timings."""
    assert os.environ.get("SLURM_JOB_ID")
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), "Refusing to overwrite earlier evidence"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_ALLOCATOR="default",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
        XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=",
        OMP_NUM_THREADS="16",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    for name, command in (
        (
            "gpu-identity",
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv",
            ],
        ),
        ("topology", ["nvidia-smi", "topo", "-m"]),
        ("scheduler", ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
    ):
        with (output / f"{name}.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    run_command(
        arguments=[
            "python",
            str(HERE / "check_install.py"),
            "--output",
            str(output / "install.json"),
        ],
        log=output / "install.log",
        environment=environment,
        limit=180,
    )
    try:
        run_command(
            arguments=[
                "python",
                "-m",
                "tests.ci.probe_native",
                "--root",
                str(ROOT),
                "--report",
                str(output / "native.json"),
            ],
            log=output / "native.log",
            environment=environment,
            limit=180,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 2:
            raise
        # As in GPU CI, rebuild once only for proven absent/stale native payload.
        command = [
            "timeout",
            "900s",
            "pixi",
            "reinstall",
            "--manifest-path",
            str(ROOT / "pyproject.toml"),
            "--frozen",
            "-e",
            "tests-cuda13",
            "pylcm",
        ]
        with (output / "native-reinstall.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (output / "native-reinstall.command.json").write_text(
            json.dumps({"argv": command, "exit_code": result.returncode}, indent=2)
            + "\n"
        )
        result.check_returncode()
        run_command(
            arguments=[
                "python",
                "-m",
                "tests.ci.probe_native",
                "--root",
                str(ROOT),
                "--report",
                str(output / "native-reinstalled.json"),
            ],
            log=output / "native-reinstalled.log",
            environment=environment,
            limit=180,
        )
    run_command(
        arguments=[
            "python",
            str(HERE / "check_install.py"),
            "--output",
            str(output / "install-ready.json"),
        ],
        log=output / "install-ready.log",
        environment=environment,
        limit=180,
    )
    harness = prepare_harness(output)
    # Only test adapters enter PYTHONPATH; installed lcm resolves normally.
    environment["PYTHONPATH"] = os.pathsep.join((str(harness), str(ROOT)))
    for precision in (64, 32):
        for devices in (3, 4, 6, 8):
            directory = output / f"semantic-gpu{devices}-fp{precision}"
            directory.mkdir()
            environment.update(
                SCALING_DEVICES=str(devices),
                JAX_ENABLE_X64=str(int(precision == 64)),
                JAX_COMPILATION_CACHE_DIR=str(directory / "cache"),
            )
            run_command(
                arguments=[
                    "pytest",
                    "-p",
                    "tests.conftest",
                    "-c",
                    str(ROOT / "pyproject.toml"),
                    "--rootdir",
                    str(ROOT),
                    str(HERE / "test_gpu_acceptance.py"),
                    f"--precision={precision}",
                    "-n",
                    "0",
                    "-v",
                    "-x",
                    "--timeout=300",
                    "--timeout-method=thread",
                    f"--junitxml={directory / 'junit.xml'}",
                ],
                log=directory / "pytest.log",
                environment=environment,
                limit=900,
            )
            check_xml(directory / "junit.xml")
        for repeat in (1, 2):
            for devices in (1, 3, 4, 6, 8):
                directory = (
                    output / f"head-continuous-{devices}-fp{precision}-repeat{repeat}"
                )
                directory.mkdir()
                environment.update(
                    SCALING_DEVICES=str(devices),
                    JAX_ENABLE_X64=str(int(precision == 64)),
                    JAX_COMPILATION_CACHE_DIR=str(directory / "cache"),
                )
                run_command(
                    arguments=[
                        "python",
                        str(HERE / "measure.py"),
                        "--devices",
                        str(devices),
                        "--precision",
                        str(precision),
                        "--source",
                        str(ROOT),
                        "--output",
                        str(directory),
                    ],
                    log=directory / "measurement.log",
                    environment=environment,
                    limit=450,
                )
    run_command(
        arguments=[
            "python",
            str(HERE / "summarize_scaling.py"),
            "--results",
            str(output),
        ],
        log=output / "summary.log",
        environment=environment,
        limit=180,
    )
    run_command(
        arguments=[
            "python",
            str(HERE / "check_install.py"),
            "--output",
            str(output / "install-post.json"),
        ],
        log=output / "install-post.log",
        environment=environment,
        limit=180,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run_scaling(parser.parse_args().output)
