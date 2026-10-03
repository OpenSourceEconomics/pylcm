"""Production CLI phases retain the owner model and whole population bindings."""

import hashlib
import importlib
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


@pytest.mark.parametrize("phase", ["run", "reference", "collect"])
# Keep the complete concrete protocol and its negative controls adjacent.
def test_production_cli_routes_complete_owner_phase(  # noqa: C901, PLR0915
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
) -> None:
    """Each CLI phase uses one owner construction and its complete phase inputs."""
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
    calls: dict[str, list[Any]] = {"config": [], "build": [], "phase": []}
    model = SimpleNamespace(initial_nodes=((50, "work"), (51, "work")))
    params = {"literal_parameter": 17}
    directory = tmp_path / "plan"
    source = tmp_path / "owner/src"
    output = tmp_path / phase
    reference_path = tmp_path / "immutable-reference"
    receipt_plan = None
    planning_receipt = None
    solution = object()
    simulation = SimpleNamespace(solution=solution)
    collected = SimpleNamespace(solution=solution, simulation=simulation)
    policy = _ExecutionPolicy(
        axis_widths=MappingProxyType({"subject": 2048}),
        invariant_block_widths=MappingProxyType({}),
    )

    def make_execution_config(**kwargs: Any) -> _ExecutionPolicy:
        calls["config"].append(kwargs)
        return policy

    def build(**kwargs: Any) -> tuple[object, object, pd.DataFrame, dict[str, str]]:
        calls["build"].append(kwargs)
        return model, params, initial, {"grid_config": "full_canonical_A100"}

    def load_component_job_plan(*, directory: Path) -> object:
        calls["phase"].append(("load", directory))
        return receipt_plan

    def run_component_job(**kwargs: Any) -> Path:
        calls["phase"].append(("run", kwargs))
        fragment = directory / "fragments/job-0002.h5"
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_bytes(b"recorded-full-fragment")
        return fragment

    def simulate(**kwargs: Any) -> object:
        calls["phase"].append(("reference", kwargs))
        return simulation

    def collect_component_jobs(**kwargs: Any) -> object:
        calls["phase"].append(("collect", kwargs))
        return collected

    model.simulate = simulate
    lcm = ModuleType("lcm")
    lcm.__dict__["__path__"] = []
    lcm.__dict__["InvariantBlockSchedule"] = SimpleNamespace(BLOCK_MAJOR="block_major")
    api = ModuleType("lcm.component_jobs")
    api.__dict__["_initial_conditions_sha256"] = importlib.import_module(
        "lcm.component_jobs"
    )._initial_conditions_sha256
    api.__dict__["load_component_job_plan"] = load_component_job_plan
    api.__dict__["run_component_job"] = run_component_job
    api.__dict__["collect_component_jobs"] = collect_component_jobs
    slurm = ModuleType("aca_slurm")
    slurm.__dict__["__path__"] = []
    config = ModuleType("aca_slurm.config")
    config.__dict__["SIMULATION_SEED"] = 20_260_903
    config.__dict__["make_execution_config"] = make_execution_config
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
    # Numeric forwarding is checked with explicitly recorded admission/archives.
    monkeypatch.setattr(owner_driver, "_native_preflight", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_production_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_validate_worker_receipts", lambda **_kwargs: {})
    monkeypatch.setattr(
        owner_driver,
        "_observe_call",
        lambda **kwargs: (kwargs["call"](), {"scope": "recorded-observations"}),
    )
    monkeypatch.setattr(
        owner_driver, "_publish_reference", lambda **_kwargs: output / "receipt.json"
    )
    monkeypatch.setattr(owner_driver, "_compare_reference", lambda **_kwargs: None)
    monkeypatch.setattr(owner_driver, "_plan", lambda **_kwargs: object())
    if phase in {"run", "collect"}:
        directory.mkdir()
        plan_file = directory / "plan.json"
        plan_file.write_text("{}")
        receipt_plan = SimpleNamespace(
            directory=directory,
            plan_id="binding-plan",
            digest=hashlib.sha256(plan_file.read_bytes()).hexdigest(),
            state_name="pref_type",
            codes=(0, 1, 2),
            jobs=((0,), (1,), (2,)),
            seed=20_260_903,
            n_subjects=3,
            initial_conditions_sha256="a" * 64,
        )
        planning_receipt = owner_driver._publish_planned_receipt(
            out=tmp_path / "separate-planning",
            plan=receipt_plan,
            original_ids=np.array([90, 10, 30], dtype=np.int64),
            raw_input_count=4,
            canonical_input_count=3,
            provenance={"contract": {}, "scope": "driver-binding-fixture"},
            started_at="2026-10-03T00:00:00+00:00",
            finished_at="2026-10-03T00:00:01+00:00",
            elapsed_seconds=1.0,
        )
        collected.plan = receipt_plan
    argv = [
        "stage8a_production.py",
        phase,
        "--plan-directory",
        str(directory),
        "--out",
        str(output),
        "--aca-slurm-src",
        str(source),
    ]
    if phase == "run":
        argv.append("--job-from-slurm-procid")
        monkeypatch.setenv("SLURM_PROCID", "2")
    if phase == "collect":
        argv.extend(
            [
                "--reference",
                str(reference_path),
                "--reference-receipt-sha256",
                "b" * 64,
                "--worker-receipts",
                str(tmp_path / "workers"),
            ]
        )
    if planning_receipt is not None:
        argv.extend(
            [
                "--planning-receipt",
                str(planning_receipt),
                "--planning-receipt-sha256",
                hashlib.sha256(planning_receipt.read_bytes()).hexdigest(),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    refusal = None
    try:
        owner_driver.main()
    except SystemExit as error:
        refusal = str(error)
    built = None
    if calls["build"]:
        forwarded = calls["build"][0]
        execution = forwarded["production_execution_config"]
        built = (
            len(calls["build"]),
            {
                name: value
                for name, value in forwarded.items()
                if name != "production_execution_config"
            },
            execution.devices,
            execution.sharded_states,
            dict(execution.axis_widths),
            execution.simulation_sharding,
            execution.device_memory_headroom_fraction,
            dict(execution.invariant_block_widths),
            execution.invariant_block_schedule,
            execution.device_memory_bytes,
        )
    observed = []
    for name, payload in calls["phase"]:
        if name == "load":
            observed.append((name, payload))
            continue
        population = payload.get("initial_conditions")
        observed.append(
            (
                name,
                set(payload),
                payload.get("model", model) is model,
                payload["params"] is params,
                payload.get("directory"),
                payload.get("job"),
                payload.get("seed"),
                payload["log_level"],
                None if population is None else population.index.tolist(),
                None if population is None else population.to_dict("list"),
            )
        )
    dense = {
        "age": [50, 51, 50],
        "regime_name": ["work", "work", "work"],
        "pref_type": [2, 0, 1],
        "assets": [7, 9, 11],
    }
    expected = {
        "run": [
            ("load", directory),
            ("load", directory),
            (
                "run",
                {
                    "model",
                    "params",
                    "directory",
                    "job",
                    "initial_conditions",
                    "log_level",
                },
                True,
                True,
                directory,
                2,
                None,
                "progress",
                [0, 1, 2],
                dense,
            ),
        ],
        "reference": [
            (
                "reference",
                {"params", "initial_conditions", "seed", "log_level"},
                True,
                True,
                None,
                None,
                20_260_903,
                "progress",
                [0, 1, 2],
                dense,
            )
        ],
        "collect": [
            ("load", directory),
            (
                "collect",
                {"model", "params", "directory", "log_level"},
                True,
                True,
                directory,
                None,
                None,
                "progress",
                None,
                None,
            ),
        ],
    }

    assert (refusal, calls["config"], built, observed, initial.index.tolist()) == (
        None,
        [{"solver": "brute_force", "continuous_sharding": True}],
        (
            1,
            {"workload": "production", "aca_slurm_src": source, "n_subjects": 1},
            tuple(range(8)),
            ("assets",),
            {"subject": 2048},
            "subjects",
            0.15,
            {"pref_type": 1},
            "block_major",
            None,
        ),
        expected[phase],
        [90, 80, 10, 30],
    )
