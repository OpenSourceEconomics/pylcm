"""Recorded receipts cannot substitute for distinct overlapping production workers."""

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


def test_numeric_intervals_use_stamped_engine_edges(
    *, owner_driver: ModuleType, tmp_path: Path
) -> None:
    """Numeric intervals exclude the surrounding worker's fragment-write bracket."""
    path = tmp_path / "worker_call.log"
    path.write_text(
        "100.000000000 INFO solve call abcdef phase backward_induction begin\n"
        "110.000000000 INFO solve call abcdef phase backward_induction "
        "end status=ok seconds=10.000\n"
        "111.000000000 INFO solve call abcdef phase simulation_chunk begin\n"
        "112.000000000 INFO solve call abcdef phase simulation_chunk "
        "end status=ok seconds=1.000\n"
    )
    caught = None
    result = None
    try:
        result = owner_driver._numerical_intervals(path=path)
    except AttributeError as error:
        caught = str(error)

    assert (caught, result) == (
        None,
        [
            {
                "call_id": "abcdef",
                "phase": "backward_induction",
                "begin_epoch": 100.0,
                "end_epoch": 110.0,
            },
            {
                "call_id": "abcdef",
                "phase": "simulation_chunk",
                "begin_epoch": 111.0,
                "end_epoch": 112.0,
            },
        ],
    )


@pytest.mark.parametrize(
    "change",
    ["none", "host", "step", "rank", "uuid", "overlap", "input", "fragment", "failure"],
)
# Keep the complete concrete protocol and its negative controls adjacent.
def test_worker_admission_requires_complete_matching_overlapping_receipts(  # noqa: C901
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    tmp_path: Path,
    change: str,
) -> None:
    """Only three distinct matching worker receipts can authorize collection.

    All topology/timestamps here are explicit recorded fixtures, not native evidence.
    """
    root = tmp_path / "workers"
    plan_directory = tmp_path / "plan"
    fragments = plan_directory / "fragments"
    fragments.mkdir(parents=True)
    contract = {"inputs": {f"input_{index}": "a" * 64 for index in range(11)}}
    for job in range(3):
        fragment = fragments / f"job-{job:04d}.h5"
        fragment.write_bytes(f"complete-fragment-{job}".encode())
        directory = root / f"job-{job:04d}"
        directory.mkdir(parents=True)
        payload = {
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": "run",
            "status": "completed",
            "job": job,
            "planning_receipt_sha256": "c" * 64,
            "fragment": str(fragment),
            "fragment_sha256": owner_driver._file_sha256(path=fragment),
            "provenance": {"contract": contract},
            "observations": {
                "host": f"node-{job}",
                "allocation": {
                    "SLURM_JOB_ID": "123",
                    "SLURM_STEP_ID": "2",
                    "SLURM_PROCID": str(job),
                    "SLURM_JOB_PARTITION": "mlgpu",
                },
                "gpu_uuids": [f"GPU-{job}-{index}" for index in range(8)],
                "gpu_exclusivity": {"exclusive": True},
                "sampler_failed": False,
                "numeric_intervals": [
                    {
                        "call_id": "abcdef",
                        "phase": "backward_induction",
                        "begin_epoch": 100.0 + job,
                        "end_epoch": 110.0 + job,
                    }
                ],
            },
        }
        if job == 2:
            if change == "host":
                payload["observations"]["host"] = "node-0"
            elif change == "step":
                payload["observations"]["allocation"]["SLURM_STEP_ID"] = "3"
            elif change == "rank":
                payload["observations"]["allocation"]["SLURM_PROCID"] = "0"
            elif change == "uuid":
                payload["observations"]["gpu_uuids"][0] = "GPU-0-0"
            elif change == "overlap":
                payload["observations"]["numeric_intervals"][0].update(
                    begin_epoch=120.0, end_epoch=130.0
                )
            elif change == "input":
                payload["provenance"]["contract"] = {"inputs": {"different": "b" * 64}}
            elif change == "fragment":
                payload["fragment_sha256"] = "0" * 64
            elif change == "failure":
                (directory / "receipt.failed.json").write_text('{"status":"failed"}')
        fragment_helpers.write_json_atomically(
            path=directory / "receipt.json", payload=payload
        )
    caught = None
    result = None
    try:
        result = owner_driver._validate_worker_receipts(
            directory=root,
            plan=SimpleNamespace(directory=plan_directory, jobs=((0,), (1,), (2,))),
            planning_receipt_sha256="c" * 64,
            production_contract=contract,
        )
    except (AttributeError, ValueError) as error:
        caught = (type(error).__name__, str(error))

    assert (caught, result is not None) == (
        None
        if change == "none"
        else (
            "ValueError",
            "Production worker receipts do not prove the matched three-node campaign",
        ),
        change == "none",
    )


def test_collector_requires_explicit_worker_receipt_directory(
    *,
    owner_driver: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Collection cannot infer a campaign's worker receipts from sibling paths."""
    caught = None
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
                "b" * 64,
            ]
        )
    except SystemExit as error:
        caught = error.code

    assert (caught, "--worker-receipts" in capsys.readouterr().err) == (2, True)


def test_main_refuses_unmatched_workers_before_collecting(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Worker admission happens before collection or archive publication."""

    context = production_runtime
    context.directory.mkdir()
    context.helpers.write_json_atomically(
        path=context.directory / "plan.json", payload={"fixture": "plan"}
    )
    plan = sys.modules["lcm.component_jobs"].load_component_job_plan(
        directory=context.directory
    )
    receipt = context.driver._publish_planned_receipt(
        out=tmp_path / "planning",
        plan=plan,
        original_ids=np.array([90, 10, 30]),
        raw_input_count=4,
        canonical_input_count=3,
        provenance={"contract": {}, "scope": "recorded-boundary"},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    workers = tmp_path / "workers"
    admissions = []
    refusal = ValueError("Production workers refused")

    def admit(**kwargs: Any) -> dict[str, object]:
        admissions.append(kwargs)
        raise refusal

    def collect(**kwargs: Any) -> SimpleNamespace:
        context.calls.append(("collect", kwargs))
        return SimpleNamespace(plan=plan, solution=object(), simulation=object())

    monkeypatch.setattr(context.driver, "_validate_worker_receipts", admit)
    monkeypatch.setattr(context.driver, "_collect", collect)
    monkeypatch.setattr(context.driver, "_compare_reference", lambda **_kwargs: None)
    monkeypatch.setattr(
        context.driver,
        "_publish_reference",
        lambda **_kwargs: context.output / "receipt.json",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "collect",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
            "--planning-receipt",
            str(receipt),
            "--planning-receipt-sha256",
            context.driver._file_sha256(path=receipt),
            "--reference",
            str(tmp_path / "reference"),
            "--reference-receipt-sha256",
            "b" * 64,
            "--worker-receipts",
            str(workers),
        ],
    )
    caught = None
    try:
        context.driver.main()
    except ValueError as error:
        caught = error
    failed = context.output / "receipt.failed.json"
    record = json.loads(failed.read_bytes()) if failed.exists() else None

    assert (
        caught is refusal,
        len(admissions),
        [name for name, _ in context.calls],
        None if not admissions else admissions[0]["directory"],
        None if record is None else record["error"],
        (context.output / "receipt.json").exists(),
    ) == (
        True,
        1,
        ["build"],
        workers,
        {"type": "ValueError", "message": "Production workers refused"},
        False,
    )
