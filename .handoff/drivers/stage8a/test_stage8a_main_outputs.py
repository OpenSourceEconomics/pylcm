"""Production main publishes bound phase outputs without mixing worker ranks."""

import importlib
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_stage8a_population import owner_driver as _imported_owner_driver
from test_stage8a_receipts import (
    fragment_helpers as _imported_fragment_helpers,
)
from test_stage8a_routing import _ExecutionPolicy as _imported__ExecutionPolicy

owner_driver = _imported_owner_driver
fragment_helpers = _imported_fragment_helpers
_ExecutionPolicy = _imported__ExecutionPolicy


@pytest.fixture
def production_runtime(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    """Supply an authenticated owner selector and recording production boundaries."""
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
    directory = tmp_path / "plan"
    output = tmp_path / "output"
    source = tmp_path / "owner/src"
    context = SimpleNamespace(
        driver=owner_driver,
        helpers=fragment_helpers,
        initial=initial,
        model=model,
        params=params,
        directory=directory,
        output=output,
        source=source,
        calls=[],
        fail_build=False,
        fail_worker=False,
        error=RuntimeError("construction refused"),
        validate_workers=owner_driver._validate_worker_receipts,
    )
    policy = _ExecutionPolicy(
        axis_widths=MappingProxyType({"subject": 2048}),
        invariant_block_widths=MappingProxyType({}),
    )

    def build(**kwargs: Any) -> tuple[object, object, pd.DataFrame, dict[str, str]]:
        context.calls.append(("build", kwargs))
        if context.fail_build:
            raise context.error
        return model, params, initial, {"grid_config": "full_canonical_A100"}

    def load_component_job_plan(*, directory: Path) -> object:
        plan_file = directory / "plan.json"
        return SimpleNamespace(
            directory=directory,
            plan_id="literal-plan-id",
            digest=fragment_helpers.sha256_hex(plan_file.read_bytes()),
            state_name="pref_type",
            codes=(0, 1, 2),
            jobs=((0,), (1,), (2,)),
            seed=20_260_903,
            n_subjects=3,
            initial_conditions_sha256="a" * 64,
            identity={"model": "full-model-fixture", "params": "full-params-fixture"},
            job_rows_sha256=("b" * 64, "c" * 64, "d" * 64),
        )

    def plan_component_jobs(**kwargs: Any) -> object:
        context.calls.append(("plan", kwargs))
        directory.mkdir()
        fragment_helpers.write_json_atomically(
            path=directory / "plan.json",
            payload={"fixture": "complete-plan-boundary"},
        )
        return load_component_job_plan(directory=directory)

    def run_component_job(**kwargs: Any) -> Path:
        context.calls.append(("run", kwargs))
        if context.fail_worker:
            raise RuntimeError(f"worker {kwargs['job']} failed")
        path = directory / f"fragments/job-{kwargs['job']:04d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"binding-fragment-{kwargs['job']}".encode())
        return path

    lcm = ModuleType("lcm")
    lcm.__dict__["__path__"] = []
    lcm.__dict__["InvariantBlockSchedule"] = SimpleNamespace(BLOCK_MAJOR="block_major")
    api = ModuleType("lcm.component_jobs")
    api.__dict__["_initial_conditions_sha256"] = importlib.import_module(
        "lcm.component_jobs"
    )._initial_conditions_sha256
    api.__dict__["plan_component_jobs"] = plan_component_jobs
    api.__dict__["load_component_job_plan"] = load_component_job_plan
    api.__dict__["run_component_job"] = run_component_job
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
    # A recorded admission boundary is not a native GPU setup observation.
    monkeypatch.setattr(owner_driver, "_native_preflight", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_production_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_validate_worker_receipts", lambda **_kwargs: {})
    monkeypatch.setattr(
        owner_driver,
        "_observe_call",
        lambda **kwargs: (kwargs["call"](), {"scope": "recorded-observations"}),
    )
    return context


def test_plan_main_publishes_full_original_id_binding(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning records every admitted external ID independently of population bytes."""
    context = production_runtime
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "plan",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
        ],
    )

    context.driver.main()

    path = context.output / "receipt.json"
    observed = None
    if path.exists():
        record = json.loads(path.read_bytes())
        artifact = context.output / "original_ids.npy"
        ids = np.load(artifact, allow_pickle=False)
        observed = (
            record["phase"],
            record["status"],
            record["codes"],
            record["jobs"],
            record["seed"],
            record["raw_input_count"],
            record["canonical_input_count"],
            record["initial_conditions_sha256"],
            ids.tolist(),
            ids.dtype.str,
            record.get("raw_input_sha256"),
            ids.shape,
            record["artifacts"]["original_ids.npy"],
        )
    expected_ids = np.array([90, 10, 30], dtype=np.int64)
    expected_artifact = None
    artifact = context.output / "original_ids.npy"
    if artifact.exists():
        expected_artifact = {
            "sha256": context.helpers.sha256_hex(artifact.read_bytes()),
            "array_checksum": context.helpers.array_checksum(
                identity={"field": "original_subject_ids"},
                array=expected_ids,
            ),
            "dtype": "<i8",
            "shape": [3],
        }

    assert (observed, context.initial.index.tolist()) == (
        (
            "plan",
            "planned",
            [0, 1, 2],
            [[0], [1], [2]],
            20_260_903,
            4,
            3,
            "a" * 64,
            [90, 10, 30],
            "<i8",
            sys.modules["lcm.component_jobs"]._initial_conditions_sha256(
                initial_conditions=context.initial
            ),
            (3,),
            expected_artifact,
        ),
        [90, 80, 10, 30],
    )


def test_native_preflight_refusal_precedes_model_construction_and_is_recorded(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native admission refusal prevents construction and records a failed phase.

    The admission boundary is a refusing recorder. This verifies routing and
    failure publication, without claiming any source or GPU setup passed.
    """
    context = production_runtime
    admission_calls = []
    refusal = RuntimeError("installed native payload is stale")

    def preflight(*, aca_slurm_src: Path) -> dict[str, object]:
        admission_calls.append(aca_slurm_src)
        raise refusal

    monkeypatch.setattr(context.driver, "_native_preflight", preflight, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "plan",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
        ],
    )
    caught = None
    try:
        context.driver.main()
    except RuntimeError as error:
        caught = error
    path = context.output / "receipt.failed.json"
    record = json.loads(path.read_bytes()) if path.exists() else None

    assert (
        caught is refusal,
        admission_calls,
        context.calls,
        record,
        (context.output / "receipt.json").exists(),
        context.directory.exists(),
    ) == (
        True,
        [context.source],
        [],
        {
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": "plan",
            "status": "failed",
            "error": {
                "type": "RuntimeError",
                "message": "installed native payload is stale",
            },
        },
        False,
        False,
    )


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_worker_ranks_publish_distinct_receipts(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    "Two worker ranks retain their own outcome instead of replacing a shared record."
    context = production_runtime
    context.directory.mkdir()
    context.helpers.write_json_atomically(
        path=context.directory / "plan.json",
        payload={"fixture": "complete-plan-boundary"},
    )
    context.fail_worker = outcome == "failure"
    receipt = context.driver._publish_planned_receipt(
        out=context.directory.parent / "planned-output",
        plan=sys.modules["lcm.component_jobs"].load_component_job_plan(
            directory=context.directory,
        ),
        original_ids=np.array([90, 10, 30], dtype=np.int64),
        raw_input_count=4,
        canonical_input_count=3,
        provenance={"contract": {}, "scope": "driver-binding-fixture"},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    errors = []
    for job in (0, 2):
        monkeypatch.setenv("SLURM_PROCID", str(job))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stage8a_production.py",
                "run",
                "--job-from-slurm-procid",
                "--plan-directory",
                str(context.directory),
                "--out",
                str(context.output),
                "--aca-slurm-src",
                str(context.source),
                "--planning-receipt",
                str(receipt),
                "--planning-receipt-sha256",
                context.helpers.sha256_hex(receipt.read_bytes()),
            ],
        )
        try:
            context.driver.main()
        except RuntimeError as error:
            errors.append(str(error))
    filename = "receipt.json" if outcome == "success" else "receipt.failed.json"
    records = []
    for job in (0, 2):
        path = context.output / f"job-{job:04d}" / filename
        if path.exists():
            payload = json.loads(path.read_bytes())
            records.append((job, payload["phase"], payload["status"], payload["job"]))
    paths = (
        sorted(
            path.relative_to(context.output).as_posix()
            for path in context.output.rglob("*.json")
        )
        if context.output.exists()
        else []
    )
    expected = {
        "success": (
            [],
            [(0, "run", "completed", 0), (2, "run", "completed", 2)],
            ["job-0000/receipt.json", "job-0002/receipt.json"],
        ),
        "failure": (
            ["worker 0 failed", "worker 2 failed"],
            [(0, "run", "failed", 0), (2, "run", "failed", 2)],
            ["job-0000/receipt.failed.json", "job-0002/receipt.failed.json"],
        ),
    }

    assert (errors, records, paths) == expected[outcome]


def test_model_construction_failure_is_a_failed_phase_attempt(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder refusal propagates and records failure before component work starts."""
    context = production_runtime
    context.fail_build = True
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "plan",
            "--plan-directory",
            str(context.directory),
            "--out",
            str(context.output),
            "--aca-slurm-src",
            str(context.source),
        ],
    )
    caught = None
    try:
        context.driver.main()
    except RuntimeError as error:
        caught = error
    path = context.output / "receipt.failed.json"
    record = json.loads(path.read_bytes()) if path.exists() else None

    assert (caught is context.error, record, context.directory.exists()) == (
        True,
        {
            "format": "aca-stage8a-phase-receipt",
            "format_version": 1,
            "phase": "plan",
            "status": "failed",
            "error": {"type": "RuntimeError", "message": "construction refused"},
        },
        False,
    )


def test_worker_main_receipts_are_accepted_by_the_campaign_validator(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual rank publishers bind the planning receipt and complete fragment bytes.

    Numeric calls and topology observations are recording stand-ins, not native proof.
    """
    context = production_runtime
    context.directory.mkdir()
    (context.directory / "plan.json").write_text("{}")
    plan = sys.modules["lcm.component_jobs"].load_component_job_plan(
        directory=context.directory
    )
    receipt = context.driver._publish_planned_receipt(
        out=context.directory.parent / "planning",
        plan=plan,
        original_ids=np.array([90, 10, 30], dtype=np.int64),
        raw_input_count=4,
        canonical_input_count=3,
        provenance={"contract": {}, "scope": "recorded-boundary"},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )
    planning_sha = context.driver._file_sha256(path=receipt)
    for job in range(3):
        observed = {
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
                    "phase": "backward_induction",
                    "call_id": "abcdef",
                    "begin_epoch": 100.0 + job,
                    "end_epoch": 110.0 + job,
                }
            ],
        }
        monkeypatch.setattr(
            context.driver,
            "_observe_call",
            lambda observed=observed, **kwargs: (kwargs["call"](), observed),
        )
        monkeypatch.setenv("SLURM_PROCID", str(job))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "driver",
                "run",
                "--job-from-slurm-procid",
                "--plan-directory",
                str(context.directory),
                "--out",
                str(context.output),
                "--aca-slurm-src",
                str(context.source),
                "--planning-receipt",
                str(receipt),
                "--planning-receipt-sha256",
                planning_sha,
            ],
        )
        context.driver.main()
    result = None
    caught = None
    try:
        result = context.validate_workers(
            directory=context.output,
            plan=plan,
            planning_receipt_sha256=planning_sha,
            production_contract={},
        )
    except ValueError as error:
        caught = str(error)

    assert (caught, result) == (
        None,
        {
            "hosts": ["node-0", "node-1", "node-2"],
            "allocation": ("123", "2"),
            "gpu_uuids": [
                f"GPU-{job}-{index}" for job in range(3) for index in range(8)
            ],
            "numeric_overlap_epoch": [102.0, 110.0],
            "worker_receipt_sha256": [
                context.driver._file_sha256(
                    path=context.output / f"job-{job:04d}/receipt.json"
                )
                for job in range(3)
            ],
        },
    )
