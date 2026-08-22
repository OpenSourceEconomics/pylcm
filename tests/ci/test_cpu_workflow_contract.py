"""Contracts between the CPU workflow and supported platform capabilities."""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[2]


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
