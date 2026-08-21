"""Public launcher for bounded and full current-machine test policies."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

PrecisionChoice = Literal["auto", "32", "64"]


def default_environment(current: str | None) -> str:
    """Return the test environment a public launcher invocation should reuse."""
    if current in {"tests-cpu", "tests-cuda12", "tests-cuda13"}:
        return current
    return "tests-cpu"


def precision_legs(choice: PrecisionChoice) -> tuple[str, ...]:
    """Expand public precision selection into process-global pytest children."""
    if choice == "auto":
        return ("64", "32")
    return (choice,)


def isolation_legs() -> tuple[str, ...]:
    """Return process groups that must never share a pytest interpreter."""
    return ("shared", "fresh")


def child_command(
    *,
    environment: str,
    policy: str,
    profile: str,
    precision: str,
    isolation: str,
    report_dir: Path,
    pytest_args: tuple[str, ...],
) -> tuple[str, ...]:
    """Return one concrete, report-producing pytest child command."""
    suffix = f"{profile}-fp{precision}-{isolation}"
    return (
        "pixi",
        "run",
        "-e",
        environment,
        "pytest",
        *pytest_args,
        "--policy-child",
        f"--ci-policy={policy}",
        f"--hardware-profile={profile}",
        f"--precision={precision}",
        f"--isolation-mode={isolation}",
        f"--selection-report={report_dir / f'selection-{suffix}.json'}",
        "-n0",
        "-v",
        f"--junitxml={report_dir / f'junit-{suffix}.xml'}",
    )


def _empty_policy_child(report: Path) -> bool:
    """Return whether collection found nodes but this process group owns none."""
    try:
        records = json.loads(report.read_text())["tests"]
    except OSError, ValueError, TypeError, KeyError:
        return False
    selected = {"selected", "capability-skipped"}
    return bool(records) and all(
        record.get("disposition") not in selected for record in records
    )


def main(argv: list[str] | None = None) -> int:
    """Run every concrete child needed by one public policy invocation."""
    parser = argparse.ArgumentParser()
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--full-suite", action="store_true")
    policy.add_argument(
        "--ci-policy",
        choices=("pr", "relevant", "extended", "nightly"),
        default="pr",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=("auto", "cpu", "gpu-small", "gpu-large", "multi-gpu"),
        default="auto",
    )
    parser.add_argument("--precision", choices=("auto", "32", "64"), default="auto")
    parser.add_argument(
        "--environment",
        default=default_environment(os.environ.get("PIXI_ENVIRONMENT_NAME")),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/policy"))
    options, pytest_args = parser.parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["tests"]

    selected_policy = "full" if options.full_suite else options.ci_policy
    options.report_dir.mkdir(parents=True, exist_ok=True)
    for precision in precision_legs(options.precision):
        for isolation in isolation_legs():
            suffix = f"{options.hardware_profile}-fp{precision}-{isolation}"
            report = options.report_dir / f"selection-{suffix}.json"
            command = child_command(
                environment=options.environment,
                policy=selected_policy,
                profile=options.hardware_profile,
                precision=precision,
                isolation=isolation,
                report_dir=options.report_dir,
                pytest_args=tuple(pytest_args),
            )
            environment = os.environ.copy()
            cache_root = Path(
                environment.get("PYLCM_JAX_CACHE_ROOT", Path.home() / ".cache" / "jax")
            )
            environment["JAX_COMPILATION_CACHE_DIR"] = str(cache_root / suffix)
            completed = subprocess.run(  # noqa: S603
                command, env=environment, check=False
            )
            no_tests_in_group = completed.returncode == 5 and _empty_policy_child(
                report
            )
            if completed.returncode != 0 and not no_tests_in_group:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
