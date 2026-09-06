"""Contracts between the CPU workflow and supported platform capabilities."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[2]

# Each of these pins a four-CPU-device topology at import, a pin that only
# takes effect in a process where no other JAX-touching invocation shares it.
_FOUR_DEVICE_TEST_FILES = (
    "tests/test_distributed.py",
    "tests/execution/test_transfer_catalogue.py",
)


def test_windows_cpu_suite_has_no_missing_kernel_skip_policy():
    """Windows builds the native kernel, so its CPU suite needs no skip workaround."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    matrix_entries = workflow["jobs"]["tests"]["strategy"]["matrix"]["include"]
    windows = next(entry for entry in matrix_entries if entry["os"] == "windows-latest")

    arguments = shlex.split(windows.get("pytest_extra", ""))
    obsolete_options = {
        "--exact-kernel-skip-inventory",
        "--expected-exact-kernel-skip-inventory",
        "--max-total-skips",
    }

    assert not obsolete_options.intersection(
        argument.partition("=")[0] for argument in arguments
    )


def _step_run_block(*, job: str, step_name: str) -> str:
    """Return the `run:` script of one named step in one workflow job."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"][job]["steps"]
    step = next(entry for entry in steps if entry.get("name") == step_name)
    return step["run"]


def _pytest_invocation_argvs(*, job: str, step_name: str) -> list[list[str]]:
    """Return the argv of every `pixi run ... pytest ...` command in one step.

    A `run:` block chains commands with backslash line continuations and `&&`.
    Joining continuations collapses each chained command onto one line, so
    splitting on `&&` recovers the individual commands the step actually runs.
    """
    normalized = _step_run_block(job=job, step_name=step_name).replace("\\\n", " ")
    commands = [
        segment.strip()
        for line in normalized.splitlines()
        for segment in re.split(r"\s&&\s", line)
        if segment.strip()
    ]
    return [shlex.split(command) for command in commands if "pixi run" in command]


def test_four_device_test_files_run_alone_at_minus_n_zero():
    """Each four-CPU-device test file runs by itself, at `-n 0`, at both precisions.

    `tests/test_distributed.py` and `tests/execution/test_transfer_catalogue.py`
    pin a four-CPU-device topology at import; the pin only takes effect in a
    process that has not already touched a JAX backend. Folding either file
    back into a shared, `-n`-distributed invocation would silently skip every
    test in it, so each file must appear in exactly one invocation per
    precision leg, running no other test path alongside it, at `-n 0`.
    """
    for job, step_name in (
        ("tests", "Run pytest and collect coverage"),
        ("tests-fp32", "Run pytest at fp32"),
    ):
        invocations = _pytest_invocation_argvs(job=job, step_name=step_name)
        for four_device_file in _FOUR_DEVICE_TEST_FILES:
            matches = [argv for argv in invocations if four_device_file in argv]
            assert len(matches) == 1, (
                f"{job}/{step_name!r}: expected exactly one pytest invocation "
                f"naming {four_device_file}, found {len(matches)}"
            )
            argv = matches[0]

            other_targets = [
                argument
                for argument in argv
                if (argument == "tests" or argument.startswith("tests/"))
                and argument != four_device_file
            ]
            assert not other_targets, (
                f"{job}/{step_name!r}: {four_device_file} shares its "
                f"invocation with {other_targets}"
            )

            assert "-n" in argv, (
                f"{job}/{step_name!r}: {four_device_file}'s invocation is missing -n"
            )
            worker_count = argv[argv.index("-n") + 1]
            assert worker_count == "0", (
                f"{job}/{step_name!r}: {four_device_file} runs at "
                f"-n {worker_count!r} instead of its own process (-n 0)"
            )
