"""Production receipts bind maintained inputs without changing their contents."""

import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


@pytest.mark.parametrize("changed", [False, True])
def test_main_refuses_input_changes_before_success_publication(
    *,
    production_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    changed: bool,
) -> None:
    """A changed production input prevents a completed planning receipt."""
    context = production_runtime
    observations = []
    initial = {"inputs": {f"input-{index}": "a" * 64 for index in range(11)}}
    changed_record = {"inputs": {**initial["inputs"], "input-0": "b" * 64}}

    def production_inputs(*, aca_slurm_src: Path) -> dict[str, dict[str, str]]:
        observations.append(aca_slurm_src)
        return changed_record if changed and len(observations) > 1 else initial

    monkeypatch.setattr(
        context.driver, "_production_inputs", production_inputs, raising=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
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
        caught = str(error)
    success = context.output / "receipt.json"
    failure = context.output / "receipt.failed.json"
    record = json.loads(success.read_bytes()) if success.exists() else None
    failed = json.loads(failure.read_bytes()) if failure.exists() else None

    assert (
        observations,
        caught,
        success.exists(),
        None if record is None else record["provenance"].get("contract"),
        None if failed is None else failed["error"]["message"],
    ) == (
        [context.source, context.source],
        "Production source or input files changed during the phase"
        if changed
        else None,
        not changed,
        None if changed else initial,
        "Production source or input files changed during the phase"
        if changed
        else None,
    )


def test_production_inputs_authenticates_owner_modules_and_all_eleven_files(
    *,
    owner_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The maintained input mapping is stream-hashed with its actual owner sources."""
    installed_core = Path(os.environ["STAGE8A_RECEIPT_CORE_ROOT"])
    # A clean authenticated HEAD snapshot keeps pending CI-only edits separate.
    core = tmp_path / "committed-source"
    subprocess.run(  # noqa: S603 - fixed Git arguments clone the authenticated fixture
        ["/usr/bin/git", "clone", "--shared", str(installed_core), str(core)],
        check=True,
        capture_output=True,
    )
    core_commit = subprocess.run(  # noqa: S603 - read the authenticated clone's exact source
        ["/usr/bin/git", "-C", str(core), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    driver_path = core / ".handoff/drivers/stage8a/stage8a_production.py"
    spec = importlib.util.spec_from_file_location("stage8a_input_driver", driver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The committed input-guard driver cannot be loaded")
    committed_driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(committed_driver)
    model = Path(os.environ["STAGE8A_ACA_MODEL_GIT"])
    slurm = model.parent / "aca-slurm"
    source = slurm / "src"
    paths = {f"input_{index}": tmp_path / f"{index}.pkl" for index in range(11)}
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"full-input-{index}".encode())
    module = ModuleType("aca_slurm._simulate")
    module.__dict__["__file__"] = str(source / "aca_slurm/_simulate.py")
    module.__dict__["_production_input_paths"] = lambda: paths
    package = ModuleType("aca_slurm")
    package.__dict__["__path__"] = []
    package.__dict__["_simulate"] = module
    monkeypatch.setitem(sys.modules, "aca_slurm", package)
    monkeypatch.setitem(sys.modules, "aca_slurm._simulate", module)
    monkeypatch.setattr(
        sys.modules["aca_model.simulation"],
        "__file__",
        str(model / "src/aca_model/simulation.py"),
        raising=False,
    )
    monkeypatch.setattr(
        sys.modules["aca_model"],
        "simulation",
        sys.modules["aca_model.simulation"],
        raising=False,
    )
    helper = driver_path.parents[1] / "stage5b/stage3_arms.py"
    for name, value in {
        "PYLCM_DIR": str(core),
        "PYLCM_COMMIT": core_commit,
        "ACA_MODEL_DIR": str(model),
        "ACA_MODEL_COMMIT": "ad38653696ec366e318ac61b9a81b597a4ecb700",
        "ACA_SLURM_DIR": str(slurm),
        "ACA_SLURM_COMMIT": "b650981593437799ae3235d1d912ea2cd9d5feda",
        "PIXI_LOCK_SHA256": hashlib.sha256(
            (core / "pixi.lock").read_bytes()
        ).hexdigest(),
        "DRIVER_SHA256": hashlib.sha256(driver_path.read_bytes()).hexdigest(),
        "BENCHMARK_HELPER_SHA256": hashlib.sha256(helper.read_bytes()).hexdigest(),
    }.items():
        monkeypatch.setenv(name, value)
    record = None
    caught = None
    try:
        record = committed_driver._production_inputs(aca_slurm_src=source)
    except AttributeError as error:
        caught = str(error)

    assert (
        inspect.getsource(committed_driver._production_inputs)
        == inspect.getsource(owner_driver._production_inputs),
        caught,
        None if record is None else record["inputs"],
    ) == (
        True,
        None,
        {
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in paths.items()
        },
    )
