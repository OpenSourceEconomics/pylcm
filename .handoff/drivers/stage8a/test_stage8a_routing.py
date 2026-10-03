"""The production CLI routes through the owner builder and complete population."""

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from test_stage8a_population import (
    fragment_helpers as _imported_fragment_helpers,
)
from test_stage8a_population import (
    owner_driver as _imported_owner_driver,
)

fragment_helpers = _imported_fragment_helpers
owner_driver = _imported_owner_driver


@dataclass(frozen=True)
class _ExecutionPolicy:
    """Record the owner policy fields preserved by CLI forwarding."""

    devices: tuple[int, ...] = tuple(range(8))
    sharded_states: tuple[str, ...] = ("assets",)
    axis_widths: object = None
    simulation_sharding: str = "subjects"
    device_memory_headroom_fraction: float = 0.15
    invariant_block_widths: object = None
    invariant_block_schedule: object = "period_major"
    device_memory_bytes: object = "device"


def test_plan_cli_routes_full_owner_population_and_configuration(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Planning builds one full owner model and forwards every admitted type row."""
    del fragment_helpers  # The fixture installs the recorded public boundary.
    initial = pd.DataFrame(
        {
            "age": [50, 60, 51, 50],
            "regime_name": ["work", "work", "work", "work"],
            "pref_type": [2, 0, 0, 1],
            "assets": [7, 8, 9, 11],
        },
        index=[90, 80, 10, 30],
    )
    model = SimpleNamespace(
        initial_nodes=((50, "work"), (51, "work")),
        user_regimes={
            "work": SimpleNamespace(
                states={"age": None, "pref_type": None, "assets": None}
            )
        },
    )
    params = {"literal_parameter": 17}
    directory = tmp_path / "plan"
    source = tmp_path / "owner/src"
    output = tmp_path / "planning"
    calls: dict[str, list[Any]] = {"config": [], "build": [], "plan": []}
    owner_policy = _ExecutionPolicy(
        axis_widths=MappingProxyType({"subject": 2048}),
        invariant_block_widths=MappingProxyType({}),
    )

    def make_execution_config(**kwargs: Any) -> _ExecutionPolicy:
        calls["config"].append(kwargs)
        return owner_policy

    def build(**kwargs: Any) -> tuple[object, object, pd.DataFrame, dict[str, str]]:
        calls["build"].append(kwargs)
        return model, params, initial, {"grid_config": "full_canonical_A100"}

    def plan_component_jobs(**kwargs: Any) -> object:
        calls["plan"].append(kwargs)
        directory.mkdir()
        plan_file = directory / "plan.json"
        plan_file.write_text("{}")
        return SimpleNamespace(
            directory=directory,
            plan_id="binding-plan",
            digest=hashlib.sha256(plan_file.read_bytes()).hexdigest(),
            state_name="pref_type",
            codes=(0, 1, 2),
            jobs=((0,), (1,), (2,)),
            seed=20_260_903,
            n_subjects=3,
            initial_conditions_sha256="a" * 64,
            identity={"model": "recorded-model"},
            job_rows_sha256=("b" * 64,) * 3,
        )

    lcm = ModuleType("lcm")
    lcm.__dict__["__path__"] = []
    lcm.__dict__["ExecutionConfig"] = _ExecutionPolicy
    lcm.__dict__["InvariantBlockSchedule"] = SimpleNamespace(
        PERIOD_MAJOR="period_major", BLOCK_MAJOR="block_major"
    )
    api = ModuleType("lcm.component_jobs")
    api.__dict__["_initial_conditions_sha256"] = importlib.import_module(
        "lcm.component_jobs"
    )._initial_conditions_sha256
    api.__dict__["plan_component_jobs"] = plan_component_jobs
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
    # Native/input admission is recorded here; complete numeric forwarding is tested.
    monkeypatch.setattr(owner_driver, "_native_preflight", lambda **_kwargs: {})
    monkeypatch.setattr(owner_driver, "_production_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage8a_production.py",
            "plan",
            "--plan-directory",
            str(directory),
            "--out",
            str(output),
            "--aca-slurm-src",
            str(source),
        ],
    )
    refusal = None
    try:
        owner_driver.main()
    except SystemExit as error:
        refusal = str(error)
    observed = None
    if calls["build"] and calls["plan"]:
        forwarded = calls["build"][0]
        execution = forwarded["production_execution_config"]
        planned = calls["plan"][0]
        observed = (
            calls["config"],
            len(calls["build"]),
            {
                key: value
                for key, value in forwarded.items()
                if key != "production_execution_config"
            },
            execution.devices,
            execution.sharded_states,
            dict(execution.axis_widths),
            execution.simulation_sharding,
            execution.device_memory_headroom_fraction,
            dict(execution.invariant_block_widths),
            execution.invariant_block_schedule,
            execution.device_memory_bytes,
            len(calls["plan"]),
            planned["model"] is model,
            planned["params"] is params,
            planned["directory"],
            planned["assignment"],
            planned["seed"],
            planned["initial_conditions"].index.tolist(),
            planned["initial_conditions"].to_dict("list"),
            initial.index.tolist(),
        )

    assert (refusal, observed) == (
        None,
        (
            [{"solver": "brute_force", "continuous_sharding": True}],
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
            1,
            True,
            True,
            directory,
            ((0,), (1,), (2,)),
            20_260_903,
            [0, 1, 2],
            {
                "age": [50, 51, 50],
                "regime_name": ["work", "work", "work"],
                "pref_type": [2, 0, 1],
                "assets": [7, 9, 11],
            },
            [90, 80, 10, 30],
        ),
    )
