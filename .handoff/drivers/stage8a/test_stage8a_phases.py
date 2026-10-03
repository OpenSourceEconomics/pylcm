"""Worker, reference and collector phases preserve the production API bindings."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from test_stage8a_population import (
    owner_driver as _imported_owner_driver,
)

owner_driver = _imported_owner_driver


@pytest.fixture
def initial() -> pd.DataFrame:
    """Provide all three types plus one row outside the owner's entry ages."""
    return pd.DataFrame(
        {
            "age": [50, 60, 51, 50],
            "regime_name": ["work", "work", "work", "work"],
            "pref_type": [2, 0, 0, 1],
            "assets": [7, 8, 9, 11],
            "claimed_ss": [False, True, True, False],
        },
        index=[90, 80, 10, 30],
    )


@pytest.fixture
def component_api(*, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Supply a recording API boundary and the canonical owner seed."""
    api = ModuleType("lcm.component_jobs")
    lcm = ModuleType("lcm")
    lcm.__dict__["__path__"] = []
    monkeypatch.setitem(sys.modules, "lcm", lcm)
    monkeypatch.setitem(sys.modules, "lcm.component_jobs", api)
    config = ModuleType("aca_slurm.config")
    config.__dict__["SIMULATION_SEED"] = 20_260_903
    slurm = ModuleType("aca_slurm")
    slurm.__dict__["__path__"] = []
    monkeypatch.setitem(sys.modules, "aca_slurm", slurm)
    monkeypatch.setitem(sys.modules, "aca_slurm.config", config)
    return api


def test_worker_passes_full_canonical_population_to_its_original_component_job(
    *,
    owner_driver: ModuleType,
    component_api: ModuleType,
    initial: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """A worker forwards the whole dense population; the engine selects its code."""
    directory = tmp_path / "plan"
    published = directory / "fragments/job-0002.h5"
    calls: list[tuple[str, Any]] = []
    model = SimpleNamespace(
        initial_nodes=((50, "work"), (51, "work")),
        user_regimes={
            "work": SimpleNamespace(
                states={"age": None, "pref_type": None, "assets": None}
            )
        },
    )
    params = {"literal_parameter": 17}

    def load_component_job_plan(*, directory: Path) -> object:
        calls.append(("load", directory))
        return SimpleNamespace(
            codes=(0, 1, 2), jobs=((0,), (1,), (2,)), seed=20_260_903, n_subjects=3
        )

    def run_component_job(**kwargs: Any) -> Path:
        calls.append(("run", kwargs))
        return published

    component_api.__dict__["load_component_job_plan"] = load_component_job_plan
    component_api.__dict__["run_component_job"] = run_component_job

    result = owner_driver._run(
        model=model,
        params=params,
        initial_conditions=initial,
        directory=directory,
        job=2,
    )

    assert (
        result,
        len(calls),
        calls[0],
        calls[1][0],
        set(calls[1][1]),
        calls[1][1]["model"] is model,
        calls[1][1]["params"] is params,
        calls[1][1]["directory"],
        calls[1][1]["job"],
        calls[1][1]["log_level"],
        calls[1][1]["initial_conditions"].index.tolist(),
        calls[1][1]["initial_conditions"].to_dict("list"),
    ) == (
        published,
        2,
        ("load", directory),
        "run",
        {"model", "params", "directory", "job", "initial_conditions", "log_level"},
        True,
        True,
        directory,
        2,
        "progress",
        [0, 1, 2],
        {
            "age": [50, 51, 50],
            "regime_name": ["work", "work", "work"],
            "pref_type": [2, 0, 1],
            "assets": [7, 9, 11],
        },
    )


def test_reference_uses_owner_adapter_seed_and_original_id_order(
    *, owner_driver: ModuleType, component_api: ModuleType, initial: pd.DataFrame
) -> None:
    """The matched reference uses the same admitted rows, seed and external IDs."""
    del component_api  # The fixture installs the recorded public boundary.
    calls: list[dict[str, Any]] = []
    simulation = object()
    params = {"literal_parameter": 17}

    def simulate(**kwargs: Any) -> object:
        calls.append(kwargs)
        return simulation

    model = SimpleNamespace(
        initial_nodes=((50, "work"), (51, "work")),
        user_regimes={
            "work": SimpleNamespace(
                states={"age": None, "pref_type": None, "assets": None}
            )
        },
        simulate=simulate,
    )

    result, original_ids = owner_driver._reference(
        model=model, params=params, initial_conditions=initial
    )

    assert (
        result is simulation,
        original_ids.tolist(),
        len(calls),
        set(calls[0]),
        calls[0]["params"] is params,
        calls[0]["seed"],
        calls[0]["log_level"],
        calls[0]["initial_conditions"].index.tolist(),
        calls[0]["initial_conditions"].to_dict("list"),
    ) == (
        True,
        [90, 10, 30],
        1,
        {"initial_conditions", "params", "seed", "log_level"},
        True,
        20_260_903,
        "progress",
        [0, 1, 2],
        {
            "age": [50, 51, 50],
            "regime_name": ["work", "work", "work"],
            "pref_type": [2, 0, 1],
            "assets": [7, 9, 11],
        },
    )


def test_collector_uses_complete_engine_result_without_an_initial_argument(
    *, owner_driver: ModuleType, component_api: ModuleType, tmp_path: Path
) -> None:
    """Collection preserves the engine's complete solution and simulation result."""
    directory = tmp_path / "plan"
    model = object()
    params = {"literal_parameter": 17}
    solution = object()
    simulation = object()
    collected = SimpleNamespace(solution=solution, simulation=simulation)
    calls: list[dict[str, Any]] = []

    def collect_component_jobs(**kwargs: Any) -> object:
        calls.append(kwargs)
        return collected

    component_api.__dict__["collect_component_jobs"] = collect_component_jobs

    result = owner_driver._collect(model=model, params=params, directory=directory)

    assert (
        result is collected,
        result.solution is solution,
        result.simulation is simulation,
        calls,
    ) == (
        True,
        True,
        True,
        [
            {
                "model": model,
                "params": params,
                "directory": directory,
                "log_level": "progress",
            }
        ],
    )
