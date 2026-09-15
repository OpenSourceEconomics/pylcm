"""Proposed missing-only B1 runner with separate clean source environments."""

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE.parent / "SOURCE-MANIFEST.json").read_text())


def run(
    *,
    root: Path,
    arguments: list[str],
    output: Path,
    environment: dict[str, str],
    seconds: int,
) -> None:
    command = [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{seconds}s",
        "pixi",
        "run",
        "--frozen",
        "--no-install",
        "--manifest-path",
        str(root / "pyproject.toml"),
        "-e",
        "tests-cuda13",
        *arguments,
    ]
    with output.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output.with_suffix(".command.json").write_text(
        json.dumps({"argv": command, "exit_code": result.returncode}, indent=2) + "\n"
    )
    result.check_returncode()


def check_xml(path: Path) -> None:
    """Require the one own-generated semantic case without skips or failures."""
    cases = ET.parse(path).getroot().findall(".//testcase")  # noqa: S314 - own pytest XML
    assert len(cases) == 1
    assert all(
        case.find(tag) is None
        for case in cases
        for tag in ("failure", "error", "skipped")
    )


def prepare_source(
    *, arm: str, root: Path, output: Path, environment: dict[str, str]
) -> None:
    """Verify one normal source installation and rebuild only proven stale payloads."""
    for path, digest in MANIFEST["source_files"][MANIFEST[arm]].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    environment["PYLCM_SCALING_EXPECTED_HEAD"] = MANIFEST[arm]
    run(
        root=root,
        arguments=[
            "python",
            str(HERE / "check_install.py"),
            "--root",
            str(root),
            "--output",
            str(output / f"{arm}-install.json"),
        ],
        output=output / f"{arm}-install.log",
        environment=environment,
        seconds=60,
    )
    try:
        run(
            root=root,
            arguments=[
                "python",
                "-m",
                "tests.ci.probe_native",
                "--root",
                str(root),
                "--report",
                str(output / f"{arm}-native.json"),
            ],
            output=output / f"{arm}-native.log",
            environment=environment,
            seconds=60,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode != 2:
            raise
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=10s",
            "600s",
            "pixi",
            "reinstall",
            "--manifest-path",
            str(root / "pyproject.toml"),
            "--frozen",
            "-e",
            "tests-cuda13",
            "pylcm",
        ]
        with (output / f"{arm}-reinstall.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (output / f"{arm}-reinstall.command.json").write_text(
            json.dumps({"argv": command, "exit_code": result.returncode}, indent=2)
            + "\n"
        )
        result.check_returncode()
        run(
            root=root,
            arguments=[
                "python",
                "-m",
                "tests.ci.probe_native",
                "--root",
                str(root),
                "--report",
                str(output / f"{arm}-native-reinstalled.json"),
            ],
            output=output / f"{arm}-native-reinstalled.log",
            environment=environment,
            seconds=60,
        )
    run(
        root=root,
        arguments=[
            "python",
            str(HERE / "check_install.py"),
            "--root",
            str(root),
            "--output",
            str(output / f"{arm}-install-ready.json"),
        ],
        output=output / f"{arm}-install-ready.log",
        environment=environment,
        seconds=60,
    )


def main(output: Path) -> None:
    assert os.environ.get("SLURM_JOB_ID")
    assert os.environ.get("PYLCM_B1_RELEASE") == "approved"
    tooling_root = HERE.parents[2]
    tooling_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tooling_root, text=True
    ).strip()
    assert tooling_head == os.environ["PYLCM_B1_TOOLING_SHA"]
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=tooling_root, text=True
    )
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), "Refusing to overwrite prior observations"
    with (output / "tooling-integrity.log").open("w") as stream:
        subprocess.run(
            ["sha256sum", "--check", "SHA256SUMS"],
            cwd=HERE.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    roots = {
        arm: Path(os.environ[f"PYLCM_B1_{arm.upper()}_ROOT"]).resolve()
        for arm in ("base", "head")
    }
    assert roots["base"] != roots["head"]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        JAX_PLATFORMS="cuda,cpu",
        NVCCFLAGS="-arch=sm_86",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        XLA_PYTHON_CLIENT_ALLOCATOR="default",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
        XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=",
    )
    for filename, command in (
        ("topology.log", ["nvidia-smi", "topo", "-m"]),
        (
            "gpu-identity.log",
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv",
            ],
        ),
        ("scheduler.log", ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
    ):
        with (output / filename).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for arm, root in roots.items():
        prepare_source(arm=arm, root=root, output=output, environment=environment)
    # Correctness is separate from timing: its extra solves cannot warm timed caches.
    for precision in (64, 32):
        environment["JAX_ENABLE_X64"] = str(int(precision == 64))
        environment["PYLCM_B1_PRECISION"] = str(precision)
        for arm, root in roots.items():
            directory = output / f"{arm}-semantic-fp{precision}"
            directory.mkdir()
            environment["JAX_COMPILATION_CACHE_DIR"] = str(directory / "cache")
            run(
                root=root,
                arguments=[
                    "pytest",
                    "-c",
                    str(root / "pyproject.toml"),
                    "--rootdir",
                    str(root),
                    str(HERE / "test_discrete_control.py"),
                    "-n",
                    "0",
                    "-v",
                    "-x",
                    "--timeout=270",
                    f"--junitxml={directory / 'junit.xml'}",
                ],
                output=directory / "pytest.log",
                environment=environment,
                seconds=300,
            )
            check_xml(directory / "junit.xml")
        for repeat in (1, 2):
            # Reverse arm order on repeat2 to avoid a fixed-order comparison.
            for arm in ("base", "head") if repeat == 1 else ("head", "base"):
                directory = output / f"{arm}-fp{precision}-repeat{repeat}"
                directory.mkdir()
                environment["JAX_COMPILATION_CACHE_DIR"] = str(directory / "cache")
                run(
                    root=roots[arm],
                    arguments=[
                        "python",
                        str(HERE / "measure.py"),
                        "--source",
                        str(roots[arm]),
                        "--precision",
                        str(precision),
                        "--output",
                        str(directory),
                    ],
                    output=directory / "measure.log",
                    environment=environment,
                    seconds=450,
                )
    for arm, root in roots.items():
        environment["PYLCM_SCALING_EXPECTED_HEAD"] = MANIFEST[arm]
        run(
            root=root,
            arguments=[
                "python",
                str(HERE / "check_install.py"),
                "--root",
                str(root),
                "--output",
                str(output / f"{arm}-install-post.json"),
            ],
            output=output / f"{arm}-install-post.log",
            environment=environment,
            seconds=60,
        )

    with (output / "tooling-integrity-post.log").open("w") as stream:
        subprocess.run(
            ["sha256sum", "--check", "SHA256SUMS"],
            cwd=HERE.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    run(
        root=roots["head"],
        arguments=["python", str(HERE / "summarize.py"), "--output", str(output)],
        output=output / "summary.log",
        environment=environment,
        seconds=60,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
