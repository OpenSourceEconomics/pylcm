"""Plan receipts bind exact original IDs and distinguish unsuccessful attempts."""

import importlib
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_stage8a_population import (
    fragment_helpers as _imported_fragment_helpers,
)
from test_stage8a_population import (
    owner_driver as _imported_owner_driver,
)
from test_stage8a_routing import _ExecutionPolicy as _imported__ExecutionPolicy

fragment_helpers = _imported_fragment_helpers
owner_driver = _imported_owner_driver
_ExecutionPolicy = _imported__ExecutionPolicy


def test_planned_receipt_publishes_bound_original_ids_and_exact_phase_record(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    tmp_path: Path,
) -> None:
    """A planned receipt binds its complete population IDs and actual plan artifact."""
    directory = tmp_path / "plan"
    directory.mkdir()
    plan_file = directory / "plan.json"
    fragment_helpers.write_json_atomically(
        path=plan_file, payload={"fixture": "complete-plan-boundary"}
    )
    plan = SimpleNamespace(
        directory=directory,
        plan_id="literal-plan-id",
        digest=fragment_helpers.sha256_hex(plan_file.read_bytes()),
        state_name="pref_type",
        codes=(0, 1, 2),
        jobs=((0,), (1,), (2,)),
        seed=20_260_903,
        n_subjects=3,
        initial_conditions_sha256="a" * 64,
    )
    original_ids = np.array([90, 10, 30], dtype=np.int64)
    output = tmp_path / "planned"
    provenance = {
        "scope": "driver-binding-fixture",
        "aca_model_commit": "ad38653696ec366e318ac61b9a81b597a4ecb700",
        "driver_sha256": "b" * 64,
        "helper_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
    }

    receipt = owner_driver._publish_planned_receipt(
        out=output,
        plan=plan,
        original_ids=original_ids,
        raw_input_count=4,
        canonical_input_count=3,
        provenance=provenance,
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    ids_file = output / "original_ids.npy"
    saved_ids = np.load(ids_file, allow_pickle=False)
    observed = json.loads(receipt.read_bytes())
    expected = {
        "format": "aca-stage8a-phase-receipt",
        "format_version": 1,
        "phase": "plan",
        "status": "planned",
        "started_at": "2026-10-03T00:00:00+00:00",
        "finished_at": "2026-10-03T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "plan_id": "literal-plan-id",
        "plan_sha256": plan.digest,
        "state_name": "pref_type",
        "codes": [0, 1, 2],
        "jobs": [[0], [1], [2]],
        "seed": 20_260_903,
        "raw_input_count": 4,
        "canonical_input_count": 3,
        "raw_input_sha256": None,
        "initial_conditions_sha256": "a" * 64,
        "provenance": provenance,
        "artifacts": {
            "original_ids.npy": {
                "sha256": fragment_helpers.sha256_hex(ids_file.read_bytes()),
                "array_checksum": fragment_helpers.array_checksum(
                    identity={"field": "original_subject_ids"},
                    array=original_ids,
                ),
                "dtype": "<i8",
                "shape": [3],
            },
        },
    }

    assert (
        receipt.name,
        observed,
        receipt.read_bytes(),
        saved_ids.tolist(),
        saved_ids.dtype.str,
        saved_ids.shape,
        sorted(path.name for path in output.iterdir()),
    ) == (
        "receipt.json",
        expected,
        fragment_helpers.canonical_json(expected),
        [90, 10, 30],
        "<i8",
        (3,),
        ["original_ids.npy", "receipt.json"],
    )


def test_plan_cli_failure_publishes_only_failed_receipt(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real routed phase error propagates and cannot publish a successful receipt."""
    del fragment_helpers  # The fixture installs the recorded public boundary.
    initial = pd.DataFrame(
        {
            "age": [50, 60, 51, 50],
            "regime_name": ["work"] * 4,
            "pref_type": [2, 0, 0, 1],
            "assets": [7, 8, 9, 11],
        },
        index=[90, 80, 10, 30],
    )
    model = SimpleNamespace(initial_nodes=((50, "work"), (51, "work")))
    params = {"literal_parameter": 17}
    error = RuntimeError("component failed")
    output = tmp_path / "failed-plan"
    policy = _ExecutionPolicy(
        axis_widths=MappingProxyType({"subject": 2048}),
        invariant_block_widths=MappingProxyType({}),
    )

    def build(**_kwargs: Any) -> tuple[object, object, pd.DataFrame, dict[str, str]]:
        return model, params, initial, {"grid_config": "full_canonical_A100"}

    def plan_component_jobs(**_kwargs: Any) -> object:
        raise error

    lcm = ModuleType("lcm")
    lcm.__dict__["__path__"] = []
    lcm.__dict__["InvariantBlockSchedule"] = SimpleNamespace(BLOCK_MAJOR="block_major")
    api = ModuleType("lcm.component_jobs")
    api.__dict__["_initial_conditions_sha256"] = importlib.import_module(
        "lcm.component_jobs"
    )._initial_conditions_sha256
    api.__dict__["plan_component_jobs"] = plan_component_jobs
    slurm = ModuleType("aca_slurm")
    slurm.__dict__["__path__"] = []
    config = ModuleType("aca_slurm.config")
    config.__dict__["SIMULATION_SEED"] = 20_260_903
    config.__dict__["make_execution_config"] = lambda **_kwargs: policy
    harness = ModuleType("stage3_arms")
    harness.__dict__["_build"] = build
    for name, module in (
        ("lcm", lcm),
        ("lcm.component_jobs", api),
        ("aca_slurm", slurm),
        ("aca_slurm.config", config),
        ("stage3_arms", harness),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # This witness records admission and tests the actual phase exception boundary.
    monkeypatch.setattr(owner_driver, "_native_preflight", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_production_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "plan",
            "--plan-directory",
            str(tmp_path / "plan"),
            "--out",
            str(output),
            "--aca-slurm-src",
            str(tmp_path / "owner/src"),
        ],
    )
    caught = None
    try:
        owner_driver.main()
    except RuntimeError as failure:
        caught = failure
    failed = output / "receipt.failed.json"
    payload = json.loads(failed.read_bytes()) if failed.exists() else None
    observed = (
        None
        if payload is None
        else (
            payload["phase"],
            payload["status"],
            payload["error"],
        )
    )

    assert (
        caught is error,
        observed,
        sorted(path.name for path in output.iterdir()) if output.exists() else [],
        (tmp_path / "plan/fragments").exists(),
    ) == (
        True,
        ("plan", "failed", {"type": "RuntimeError", "message": "component failed"}),
        ["receipt.failed.json"],
        False,
    )
