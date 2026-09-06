"""Contracts between the CPU workflow and supported platform capabilities."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
import yaml

from tests.ci.cpu_suite_invocations import (
    FOUR_DEVICE_TEST_FILES,
    carries_policy_activation_flags,
    cpu_suite_invocation_argvs,
)

_REPO_ROOT = Path(__file__).parents[2]

# The fp64 and fp32 legs each run every four-CPU-device file in one invocation
# of their own; every property below is checked once per (leg, file) pair.
_FOUR_DEVICE_INVOCATION_CASES = tuple(
    (job, step_name, four_device_file)
    for job, step_name in (
        ("tests", "Run pytest and collect coverage"),
        ("tests-fp32", "Run pytest at fp32"),
    )
    for four_device_file in FOUR_DEVICE_TEST_FILES
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
    """Return the argv of every CPU-suite pytest invocation in one step."""
    return cpu_suite_invocation_argvs(_step_run_block(job=job, step_name=step_name))


def _invocations_naming(
    *, job: str, step_name: str, four_device_file: str
) -> list[list[str]]:
    """Return every pytest invocation argv in one step that names `four_device_file`."""
    return [
        argv
        for argv in _pytest_invocation_argvs(job=job, step_name=step_name)
        if four_device_file in argv
    ]


def _sole_invocation(*, job: str, step_name: str, four_device_file: str) -> list[str]:
    """Return the one invocation argv naming `four_device_file` in one step.

    `test_four_device_file_appears_in_exactly_one_invocation` is the test that
    names and asserts this singleton precondition; every other property test
    below reuses this helper to reach the one invocation it inspects.
    """
    matches = _invocations_naming(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    return matches[0]


@pytest.mark.parametrize(
    ("job", "step_name", "four_device_file"), _FOUR_DEVICE_INVOCATION_CASES
)
def test_four_device_file_appears_in_exactly_one_invocation(
    *, job: str, step_name: str, four_device_file: str
):
    """Each four-CPU-device test file is named by exactly one pytest invocation.

    `tests/test_distributed.py` and `tests/execution/test_transfer_catalogue.py`
    pin a four-CPU-device topology at import, a pin that depends on running
    alone in its process; naming the file from zero or from more than one
    invocation means it either never runs or no longer runs alone.
    """
    matches = _invocations_naming(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    assert len(matches) == 1, (
        f"{job}/{step_name!r}: expected exactly one pytest invocation naming "
        f"{four_device_file}, found {len(matches)}"
    )


@pytest.mark.parametrize(
    ("job", "step_name", "four_device_file"), _FOUR_DEVICE_INVOCATION_CASES
)
def test_four_device_file_runs_without_other_test_paths(
    *, job: str, step_name: str, four_device_file: str
):
    """A four-CPU-device test file's invocation names no other test path.

    Sharing the invocation with `tests` or another `tests/...` target would
    fold the file back into a multi-file process, defeating the import-time
    device-count pin that assumes it runs alone.
    """
    argv = _sole_invocation(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    other_targets = [
        argument
        for argument in argv
        if (argument == "tests" or argument.startswith("tests/"))
        and argument != four_device_file
    ]
    assert not other_targets, (
        f"{job}/{step_name!r}: {four_device_file} shares its invocation with "
        f"{other_targets}"
    )


@pytest.mark.parametrize(
    ("job", "step_name", "four_device_file"), _FOUR_DEVICE_INVOCATION_CASES
)
def test_four_device_file_invocation_passes_the_worker_count_flag(
    *, job: str, step_name: str, four_device_file: str
):
    """A four-CPU-device test file's invocation states its worker count explicitly.

    An implicit worker count would leave the invocation's process-isolation
    guarantee undeclared; the sibling test then checks the count itself.
    """
    argv = _sole_invocation(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    assert "-n" in argv, (
        f"{job}/{step_name!r}: {four_device_file}'s invocation is missing -n"
    )


@pytest.mark.parametrize(
    ("job", "step_name", "four_device_file"), _FOUR_DEVICE_INVOCATION_CASES
)
def test_four_device_file_runs_at_worker_count_zero(
    *, job: str, step_name: str, four_device_file: str
):
    """A four-CPU-device test file's invocation runs at `-n 0`, its own process.

    Any other worker count would distribute the file's tests across xdist
    workers that fork before the file's own import-time device-count pin runs,
    so the pin would apply to at most one worker and the rest would skip.
    """
    argv = _sole_invocation(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    worker_count = argv[argv.index("-n") + 1]
    assert worker_count == "0", (
        f"{job}/{step_name!r}: {four_device_file} runs at -n {worker_count!r} "
        "instead of its own process (-n 0)"
    )


@pytest.mark.parametrize(
    ("job", "step_name", "four_device_file"), _FOUR_DEVICE_INVOCATION_CASES
)
def test_four_device_file_invocation_omits_policy_activation_flags(
    *, job: str, step_name: str, four_device_file: str
):
    """A four-CPU-device test file's invocation never activates the CI policy launcher.

    `--ci-policy` or `--full-suite` drives `pytest_policy.configure()`, which
    resolves an explicit `--hardware-profile` by querying `jax.default_backend()`
    during `pytest_configure` -- before pytest imports any test module. That
    query initialises the JAX backend ahead of this file's own import-time
    four-CPU-device pin, so the pin sees an already-initialised backend, never
    applies, and every test in the file silently skips.
    """
    argv = _sole_invocation(
        job=job, step_name=step_name, four_device_file=four_device_file
    )
    assert not carries_policy_activation_flags(argv), (
        f"{job}/{step_name!r}: {four_device_file}'s invocation carries a CI "
        "policy activation flag, which silently skips every test in the file"
    )
