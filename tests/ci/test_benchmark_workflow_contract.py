"""Contracts for the pull-request benchmark workflow."""

from pathlib import Path

import yaml


def test_benchmark_workflow_materializes_main_without_network_credentials() -> None:
    """ASV gets a local main ref after checkout credentials are removed."""
    workflow = yaml.safe_load(
        Path(".github/workflows/benchmark-pr.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["run-benchmarks"]["steps"]
    checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v7")
    ensure_main = next(
        step for step in steps if step.get("name") == "Ensure main ref exists"
    )

    assert checkout["with"]["persist-credentials"] is False
    assert ensure_main["run"] == "git branch --force main origin/main"
