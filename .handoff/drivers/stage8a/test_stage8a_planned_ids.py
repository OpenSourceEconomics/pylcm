"""Worker and collector commands bind IDs to an explicit immutable plan receipt."""

import hashlib
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
from test_stage8a_population import owner_driver as _imported_owner_driver
from test_stage8a_receipts import (
    fragment_helpers as _imported_fragment_helpers,
)

production_runtime = _imported_production_runtime
owner_driver = _imported_owner_driver
fragment_helpers = _imported_fragment_helpers


@pytest.mark.parametrize("change", ["seed", "two_jobs", "permuted_jobs"])
def test_planned_id_binding_refuses_another_valid_campaign(
    *,
    production_runtime: SimpleNamespace,
    tmp_path: Path,
    change: str,
) -> None:
    """Self-consistent metadata cannot change production seed or rank assignments."""
    context = production_runtime
    context.directory.mkdir()
    context.helpers.write_json_atomically(
        path=context.directory / "plan.json",
        payload={"fixture": "complete-plan-boundary"},
    )
    plan = sys.modules["lcm.component_jobs"].load_component_job_plan(
        directory=context.directory,
    )
    if change == "seed":
        plan.seed = 7
    elif change == "two_jobs":
        plan.jobs = ((0, 1), (2,))
    else:
        plan.jobs = ((2,), (1,), (0,))
    original_ids = np.array([90, 10, 30], dtype=np.int64)
    receipt = context.driver._publish_planned_receipt(
        out=tmp_path / "another-campaign",
        plan=plan,
        original_ids=original_ids,
        raw_input_count=4,
        canonical_input_count=3,
        provenance={"contract": {}, "scope": "driver-binding-fixture"},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    caught = None
    try:
        context.driver._validate_planned_ids(
            plan=plan,
            receipt=receipt,
            receipt_sha256=context.driver._file_sha256(path=receipt),
            original_ids=original_ids,
        )
    except ValueError as error:
        caught = str(error)

    assert (plan.seed, plan.jobs, caught) == (
        7 if change == "seed" else 20_260_903,
        ((0, 1), (2,))
        if change == "two_jobs"
        else (((2,), (1,), (0,)) if change == "permuted_jobs" else ((0,), (1,), (2,))),
        "Component plan differs from the production campaign",
    )


@pytest.mark.parametrize("phase", ["run", "collect"])
def test_phase_cli_accepts_explicit_planning_receipt_and_hash(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
) -> None:
    """Each worker and collector names its exact planned-ID artifact."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    receipt = tmp_path / "separate-owner-workspace/planning/receipt.json"
    arguments = [
        phase,
        "--plan-directory",
        str(tmp_path / "component-plan"),
        "--out",
        str(tmp_path / "worker-output"),
        "--aca-slurm-src",
        str(tmp_path / "owner/src"),
        "--planning-receipt",
        str(receipt),
        "--planning-receipt-sha256",
        "a" * 64,
    ]
    arguments += (
        ["--job-from-slurm-procid"]
        if phase == "run"
        else [
            "--reference",
            str(tmp_path / "reference"),
            "--reference-receipt-sha256",
            "b" * 64,
            "--worker-receipts",
            str(tmp_path / "workers"),
        ]
    )
    observed = None
    try:
        args = owner_driver.parse_args(arguments)
        observed = (args.phase, args.planning_receipt, args.planning_receipt_sha256)
    except SystemExit:
        pass

    assert observed == (phase, receipt, "a" * 64)


@pytest.mark.parametrize("phase", ["run", "collect"])
def test_phase_cli_requires_explicit_planning_receipt_binding(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No worker or collector can silently omit the original-ID plan binding."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    arguments = [
        phase,
        "--plan-directory",
        str(tmp_path / "plan"),
        "--out",
        str(tmp_path / "out"),
        "--aca-slurm-src",
        str(tmp_path / "owner/src"),
    ]
    arguments += (
        ["--job-from-slurm-procid"]
        if phase == "run"
        else [
            "--reference",
            str(tmp_path / "reference"),
            "--reference-receipt-sha256",
            "b" * 64,
            "--worker-receipts",
            str(tmp_path / "workers"),
        ]
    )
    code = None
    try:
        owner_driver.parse_args(arguments)
    except SystemExit as error:
        code = error.code
    stderr = capsys.readouterr().err

    assert (
        code,
        "--planning-receipt" in stderr,
        "--planning-receipt-sha256" in stderr,
    ) == (2, True, True)


@pytest.mark.parametrize("phase", ["run", "collect"])
@pytest.mark.parametrize("change", ["ids", "contract"])
def test_main_refuses_external_ids_different_from_immutable_plan(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    change: str,
) -> None:
    """Dense-population equality cannot authorize another external-ID ordering."""
    context = production_runtime
    context.directory.mkdir()
    context.helpers.write_json_atomically(
        path=context.directory / "plan.json",
        payload={"fixture": "complete-plan-boundary"},
    )
    plan = SimpleNamespace(
        directory=context.directory,
        plan_id="literal-plan-id",
        digest=context.helpers.sha256_hex(
            (context.directory / "plan.json").read_bytes(),
        ),
        state_name="pref_type",
        codes=(0, 1, 2),
        jobs=((0,), (1,), (2,)),
        seed=20_260_903,
        n_subjects=3,
        initial_conditions_sha256="a" * 64,
    )
    # This separate immutable receipt authenticates a genuinely different ID order.
    receipt = context.driver._publish_planned_receipt(
        out=tmp_path / "separate-planning-output",
        plan=plan,
        original_ids=np.array(
            [10, 90, 30] if change == "ids" else [90, 10, 30], dtype=np.int64
        ),
        raw_input_count=4,
        canonical_input_count=3,
        provenance={
            "scope": "driver-binding-fixture",
            "contract": {"inputs": "other-frozen-inputs"},
        },
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    monkeypatch.setenv("SLURM_PROCID", "0")
    arguments = [
        "stage8a_production.py",
        phase,
        "--plan-directory",
        str(context.directory),
        "--out",
        str(context.output),
        "--aca-slurm-src",
        str(context.source),
        "--planning-receipt",
        str(receipt),
        "--planning-receipt-sha256",
        hashlib.sha256(receipt.read_bytes()).hexdigest(),
    ]
    arguments += (
        ["--job-from-slurm-procid"]
        if phase == "run"
        else [
            "--reference",
            str(tmp_path / "reference"),
            "--reference-receipt-sha256",
            "b" * 64,
            "--worker-receipts",
            str(tmp_path / "workers"),
        ]
    )

    # Recording the public collector prevents a missing fixture API from hiding
    # the required pre-component ID refusal.
    def collect_component_jobs(**kwargs: Any) -> object:
        context.calls.append(("collect", kwargs))
        return SimpleNamespace(plan=plan, solution=object(), simulation=object())

    monkeypatch.setattr(context.driver, "_collect", collect_component_jobs)
    monkeypatch.setattr(context.driver, "_compare_reference", lambda **_kwargs: None)
    monkeypatch.setattr(
        context.driver,
        "_publish_reference",
        lambda **_kwargs: context.output / "receipt.json",
    )

    monkeypatch.setattr(
        sys.modules["lcm.component_jobs"],
        "collect_component_jobs",
        collect_component_jobs,
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", arguments)
    caught = None
    try:
        context.driver.main()
    except (ValueError, SystemExit) as error:
        caught = (type(error).__name__, str(error))
    failed = context.output / (
        "job-0000/receipt.failed.json" if phase == "run" else "receipt.failed.json"
    )
    record = json.loads(failed.read_bytes()) if failed.exists() else None
    outcome = (
        None if record is None else (record["phase"], record["status"], record["error"])
    )

    message = (
        "Planned original subject IDs differ"
        if change == "ids"
        else "Planned production source or inputs differ"
    )
    assert (
        caught,
        outcome,
        [name for name, _kwargs in context.calls if name in {"run", "collect"}],
        (context.output / "receipt.json").exists(),
    ) == (
        ("ValueError", message),
        (
            phase,
            "failed",
            {"type": "ValueError", "message": message},
        ),
        [],
        False,
    )
