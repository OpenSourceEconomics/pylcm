"""Actual main publishes complete phase outputs only after matched comparison.

Archive and numeric boundaries are recorders here. Genuine public persistence
is exercised separately by the tiny CPU archive witnesses; neither is native
ACA construction or GPU admission evidence.
"""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_stage8a_main_outputs import (
    production_runtime as _imported_production_runtime,
)
from test_stage8a_population import (
    fragment_helpers as _imported_fragment_helpers,
)
from test_stage8a_population import owner_driver as _imported_owner_driver

production_runtime = _imported_production_runtime
fragment_helpers = _imported_fragment_helpers
owner_driver = _imported_owner_driver


def test_reference_main_publishes_complete_returned_results_and_original_ids(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful reference phase publishes its complete solution before returning."""
    context = production_runtime
    solution = object()
    simulation = SimpleNamespace(solution=solution)
    original_ids = np.array([90, 10, 30], dtype=np.int64)
    calls = []
    observed = []

    def observe(**kwargs: Any) -> tuple[object, dict[str, object]]:
        observed.append(kwargs["label"])
        return kwargs["call"](), {"scope": "recorded-observations"}

    def reference(**kwargs: Any) -> tuple[object, np.ndarray]:
        calls.append(("reference", kwargs))
        return simulation, original_ids

    def publish(**kwargs: Any) -> Path:
        calls.append(("publish", kwargs))
        path = kwargs["out"] / "receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        context.helpers.write_json_atomically(
            path=path,
            payload={"phase": kwargs.get("phase", "reference"), "status": "completed"},
        )
        return path

    monkeypatch.setattr(context.driver, "_reference", reference)
    monkeypatch.setattr(context.driver, "_observe_call", observe)
    monkeypatch.setattr(context.driver, "_publish_reference", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "reference",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
        ],
    )

    context.driver.main()

    record = context.output / "receipt.json"
    published = next((kwargs for name, kwargs in calls if name == "publish"), {})
    plan = published.get("plan")
    assert (
        [name for name, _kwargs in calls],
        published.get("solution") is solution,
        published.get("simulation") is simulation,
        published.get("original_ids") is original_ids,
        published.get("out"),
        published.get("provenance"),
        [name for name, _kwargs in context.calls],
        None if plan is None else (plan.seed, plan.codes, plan.jobs, plan.n_subjects),
        json.loads(record.read_bytes()) if record.exists() else None,
        (context.output / "receipt.failed.json").exists(),
        observed,
    ) == (
        ["reference", "publish"],
        True,
        True,
        True,
        context.output,
        {"grid_config": "full_canonical_A100", "contract": {}, "runtime": {}},
        ["build", "plan"],
        (20_260_903, (0, 1, 2), ((0,), (1,), (2,)), 3),
        {"phase": "reference", "status": "completed"},
        False,
        ["reference_call"],
    )


@pytest.mark.parametrize("outcome", ["matching", "different"])
def test_collect_main_compares_immutable_reference_before_success_publication(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    """A complete matching result publishes a collect success receipt."""
    context = production_runtime
    context.directory.mkdir()
    context.helpers.write_json_atomically(
        path=context.directory / "plan.json",
        payload={"fixture": "complete-plan-boundary"},
    )
    plan = sys.modules["lcm.component_jobs"].load_component_job_plan(
        directory=context.directory,
    )
    original_ids = np.array([90, 10, 30], dtype=np.int64)
    planning = context.driver._publish_planned_receipt(
        out=tmp_path / "independent-planning-output",
        plan=plan,
        original_ids=original_ids,
        raw_input_count=4,
        canonical_input_count=3,
        provenance={"contract": {}, "scope": "binding-fixture"},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    reference = tmp_path / "separate-immutable-reference"
    reference.mkdir()
    reference_receipt = reference / "receipt.json"
    reference_receipt.write_bytes(b'{"scope":"reference-binding-fixture"}')
    reference_sha = context.driver._file_sha256(path=reference_receipt)
    solution = object()
    simulation = object()
    collected = SimpleNamespace(plan=plan, solution=solution, simulation=simulation)
    calls = []
    observed = []

    def observe(**kwargs: Any) -> tuple[object, dict[str, object]]:
        observed.append(kwargs["label"])
        return kwargs["call"](), {"scope": "recorded-observations"}

    def collect(**kwargs: Any) -> object:
        calls.append(("collect", kwargs))
        return collected

    def compare(**kwargs: Any) -> None:
        calls.append(("compare", kwargs))
        if outcome == "different":
            raise ValueError("Reference comparison differs: values")

    def publish(**kwargs: Any) -> Path:
        calls.append(("publish", kwargs))
        path = kwargs["out"] / "receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        context.helpers.write_json_atomically(
            path=path,
            payload={"phase": kwargs.get("phase"), "status": "completed"},
        )
        return path

    monkeypatch.setattr(context.driver, "_collect", collect)
    worker_proof = {
        "hosts": ["node-0", "node-1", "node-2"],
        "numeric_overlap_epoch": [102.0, 110.0],
    }
    monkeypatch.setattr(
        context.driver, "_validate_worker_receipts", lambda **_kwargs: worker_proof
    )
    monkeypatch.setattr(context.driver, "_observe_call", observe)
    monkeypatch.setattr(context.driver, "_compare_reference", compare, raising=False)
    monkeypatch.setattr(context.driver, "_publish_reference", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "collect",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
            "--planning-receipt",
            str(planning),
            "--planning-receipt-sha256",
            context.driver._file_sha256(path=planning),
            "--reference",
            str(reference),
            "--reference-receipt-sha256",
            reference_sha,
            "--worker-receipts",
            str(tmp_path / "workers"),
        ],
    )
    caught = None
    try:
        context.driver.main()
    except ValueError as error:
        caught = str(error)
    comparison = next((kwargs for name, kwargs in calls if name == "compare"), {})
    published = next((kwargs for name, kwargs in calls if name == "publish"), {})
    receipt = context.output / (
        "receipt.json" if outcome == "matching" else "receipt.failed.json"
    )
    record = json.loads(receipt.read_bytes()) if receipt.exists() else None
    fields = None if record is None else (record["phase"], record["status"])

    assert (
        [name for name, _kwargs in calls],
        comparison.get("collected") is collected,
        getattr(comparison.get("original_ids"), "tolist", lambda: None)(),
        comparison.get("reference"),
        comparison.get("reference_receipt_sha256"),
        (
            published.get("solution") is solution
            if outcome == "matching"
            else not published
        ),
        (
            published.get("simulation") is simulation
            if outcome == "matching"
            else not published
        ),
        fields,
        caught,
        (context.output / "receipt.json").exists(),
        observed,
        published.get("provenance", {}).get("worker_admission"),
    ) == (
        (
            ["collect", "compare", "publish"]
            if outcome == "matching"
            else ["collect", "compare"]
        ),
        True,
        [90, 10, 30],
        reference,
        reference_sha,
        True,
        True,
        ("collect", "completed") if outcome == "matching" else ("collect", "failed"),
        None if outcome == "matching" else "Reference comparison differs: values",
        outcome == "matching",
        ["collection_call", "comparison_call"],
        worker_proof if outcome == "matching" else None,
    )


def test_collect_cli_requires_immutable_reference_receipt_hash(
    *,
    owner_driver: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unbound reference directory cannot authorize collection success."""
    code = None
    try:
        owner_driver.parse_args(
            [
                "collect",
                "--plan-directory",
                str(tmp_path / "plan"),
                "--out",
                str(tmp_path / "out"),
                "--aca-slurm-src",
                str(tmp_path / "owner/src"),
                "--planning-receipt",
                str(tmp_path / "planning/receipt.json"),
                "--planning-receipt-sha256",
                "a" * 64,
                "--reference",
                str(tmp_path / "reference"),
                "--worker-receipts",
                str(tmp_path / "workers"),
            ]
        )
    except SystemExit as error:
        code = error.code

    assert (code, "--reference-receipt-sha256" in capsys.readouterr().err) == (2, True)


@pytest.mark.parametrize("phase", ["run", "collect"])
@pytest.mark.parametrize("invalid_sha", ["a" * 63, "A" * 64, "z" * 64])
def test_phase_cli_refuses_malformed_planning_receipt_hash(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    invalid_sha: str,
) -> None:
    """Receipt hashes are explicit lowercase SHA-256 digests before runtime imports."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    arguments = [
        phase,
        "--plan-directory",
        str(tmp_path / "plan"),
        "--out",
        str(tmp_path / "out"),
        "--aca-slurm-src",
        str(tmp_path / "owner/src"),
        "--planning-receipt",
        str(tmp_path / "planning/receipt.json"),
        "--planning-receipt-sha256",
        invalid_sha,
    ]
    arguments += (
        ["--job-from-slurm-procid"]
        if phase == "run"
        else [
            "--reference",
            str(tmp_path / "reference"),
            "--reference-receipt-sha256",
            "a" * 64,
            "--worker-receipts",
            str(tmp_path / "workers"),
        ]
    )
    code = None
    try:
        owner_driver.parse_args(arguments)
    except SystemExit as error:
        code = error.code

    assert (code, "64 lowercase hexadecimal characters" in capsys.readouterr().err) == (
        2,
        True,
    )


@pytest.mark.parametrize("invalid_sha", ["a" * 63, "A" * 64, "z" * 64])
def test_collect_cli_refuses_malformed_reference_receipt_hash(
    *,
    owner_driver: ModuleType,
    tmp_path: Path,
    invalid_sha: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reference digest obeys the same strict SHA-256 syntax as the plan binding."""
    code = None
    try:
        owner_driver.parse_args(
            [
                "collect",
                "--plan-directory",
                str(tmp_path / "plan"),
                "--out",
                str(tmp_path / "out"),
                "--aca-slurm-src",
                str(tmp_path / "owner/src"),
                "--planning-receipt",
                str(tmp_path / "planning/receipt.json"),
                "--planning-receipt-sha256",
                "a" * 64,
                "--reference",
                str(tmp_path / "reference"),
                "--reference-receipt-sha256",
                invalid_sha,
                "--worker-receipts",
                str(tmp_path / "workers"),
            ]
        )
    except SystemExit as error:
        code = error.code

    assert (code, "64 lowercase hexadecimal characters" in capsys.readouterr().err) == (
        2,
        True,
    )
