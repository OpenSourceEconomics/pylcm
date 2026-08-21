"""Contracts between the CPU workflow and its checked-in inputs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[2]


def test_expected_skip_inventories_referenced_by_cpu_workflow_are_shipped():
    """Every expected skip inventory passed by CI exists and has the right schema."""
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    )
    matrix_entries = workflow["jobs"]["tests"]["strategy"]["matrix"]["include"]

    option = "--expected-exact-kernel-skip-inventory"
    referenced: list[str] = []
    for entry in matrix_entries:
        arguments = shlex.split(entry.get("pytest_extra", ""))
        for position, argument in enumerate(arguments):
            if argument == option:
                referenced.append(arguments[position + 1])
            elif argument.startswith(f"{option}="):
                referenced.append(argument.partition("=")[2])

    assert referenced
    for relative_path in referenced:
        inventory_path = _REPO_ROOT / relative_path
        assert inventory_path.is_file(), f"Missing CI inventory: {relative_path}"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        assert inventory["schema_version"] == 1
        assert isinstance(inventory["skipped"], list)
