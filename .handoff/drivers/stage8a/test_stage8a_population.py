"""Production population binding uses the frozen owner's admission rule."""

import hashlib
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def fragment_helpers(*, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Use maintained receipt helpers from the explicit component source root."""
    root = Path(os.environ["STAGE8A_RECEIPT_CORE_ROOT"]).resolve()
    monkeypatch.syspath_prepend(str(root / "src"))
    for package in ("_lcm", "lcm"):
        module = importlib.import_module(package)
        if (
            Path(str(module.__file__)).resolve()
            != root / "src" / package / "__init__.py"
        ):
            raise RuntimeError(
                "Receipt runtime imported outside the explicit core root"
            )
    helpers = importlib.import_module("_lcm.solution.component_fragments")
    if Path(helpers.__file__).resolve() != (
        root / "src/_lcm/solution/component_fragments.py"
    ):
        raise RuntimeError("Receipt helpers imported outside the explicit core root")
    return helpers


@pytest.fixture
def owner_driver(*, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the candidate with the authenticated frozen owner's selector."""
    repository = Path(
        os.environ.get(
            "STAGE8A_ACA_MODEL_GIT",
            str(Path(__file__).resolve().parents[5] / "aca-model"),
        )
    )
    source = subprocess.run(  # noqa: S603
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "show",
            "ad38653696ec366e318ac61b9a81b597a4ecb700:src/aca_model/simulation.py",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if hashlib.sha256(source).hexdigest() != (
        "3bf6abfdfa74721a60b399eedff0e2f4a69168201c790ce2aedc5b1426e4a8d3"
    ):
        raise RuntimeError(
            "The frozen owner simulation module does not match its source SHA"
        )
    owner = ModuleType("aca_model.simulation")
    exec(compile(source, "frozen-owner-simulation.py", "exec"), owner.__dict__)  # noqa: S102
    package = ModuleType("aca_model")
    package.__dict__["__path__"] = []
    monkeypatch.setitem(sys.modules, "aca_model", package)
    monkeypatch.setitem(sys.modules, "aca_model.simulation", owner)
    path = Path(__file__).with_name("stage8a_production.py")
    spec = importlib.util.spec_from_file_location("stage8a_population_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The candidate production driver cannot be loaded")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def test_canonical_population_retains_every_owner_admitted_row_and_original_id(
    *, owner_driver: ModuleType
) -> None:
    """All owner-admitted types retain their input order and external IDs."""
    initial = pd.DataFrame(
        {
            "age": [50, 60, 51, 50],
            "regime_name": ["work", "work", "work", "work"],
            "pref_type": [2, 0, 0, 1],
            "assets": [7, 8, 9, 11],
        },
        index=[90, 80, 10, 30],
    )
    model = SimpleNamespace(initial_nodes=((50, "work"), (51, "work")))

    dense, original_ids = owner_driver._canonical_population(
        model=model, initial_conditions=initial
    )

    assert (
        len(initial),
        initial.index.tolist(),
        len(dense),
        dense.index.tolist(),
        dense.to_dict("list"),
        original_ids.tolist(),
    ) == (
        4,
        [90, 80, 10, 30],
        3,
        [0, 1, 2],
        {
            "age": [50, 51, 50],
            "regime_name": ["work", "work", "work"],
            "pref_type": [2, 0, 1],
            "assets": [7, 9, 11],
        },
        [90, 10, 30],
    )


def test_plan_binds_full_canonical_population_original_codes_and_owner_seed(
    *, owner_driver: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Planning forwards every canonical row with original codes and owner seed."""
    calls = []

    def plan_component_jobs(**kwargs: Any) -> object:
        calls.append(kwargs)
        return SimpleNamespace(jobs=((0,), (1,), (2,)))

    api = ModuleType("lcm.component_jobs")
    api.__dict__["plan_component_jobs"] = plan_component_jobs
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
    model = SimpleNamespace(initial_nodes=((50, "work"), (51, "work")))
    params = {"literal_parameter": 17}
    initial = pd.DataFrame(
        {
            "age": [50, 60, 51, 50],
            "regime_name": ["work", "work", "work", "work"],
            "pref_type": [2, 0, 0, 1],
            "assets": [7, 8, 9, 11],
        },
        index=[90, 80, 10, 30],
    )
    directory = tmp_path / "plan"

    owner_driver._plan(
        model=model, params=params, initial_conditions=initial, directory=directory
    )

    assert [
        (
            call["model"] is model,
            call["params"] is params,
            call["directory"],
            call["assignment"],
            call["seed"],
            call["initial_conditions"].index.tolist(),
            call["initial_conditions"].to_dict("list"),
        )
        for call in calls
    ] == [
        (
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
        )
    ]
