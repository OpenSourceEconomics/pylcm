"""Node-local component jobs publish the complete block-major result."""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal, Never

import h5py
import jax
import numpy as np
import pytest

import lcm
from _lcm import version
from _lcm.egm.upper_envelope._exact_affine import ffi
from _lcm.persistence.solution import _array_checksum
from lcm import InvariantBlockSchedule, Model
from lcm.exceptions import (
    ExecutionPlanningError,
    ModelIdentityError,
    SolutionIntegrityError,
)
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import UserParams
from tests.simulation import test_type_grouped_simulation as life_cycle
from tests.solution import test_block_major_lifetime as lifetime

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")

type CompleteCampaign = tuple[Path, Model, UserParams, SimulationResult]


def _model(*, enable_jit: bool = True, durable_identity: bool = True) -> Model:
    """Return the typed life cycle with block-major execution."""
    model = life_cycle._model(typed_dead=True)
    return Model(
        edges=model.edges,
        regimes=model.user_regimes,
        ages=model.ages,
        regime_id_class=life_cycle._RegimeId,
        initial_nodes={0: "work"},
        execution_config=lifetime._config(
            schedule=InvariantBlockSchedule.BLOCK_MAJOR, subject_width=3
        ),
        enable_jit=enable_jit,
        durable_identity=durable_identity,
    )


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("existing_directory", [False, True])
def test_planning_ephemeral_jobs_is_refused_before_publication(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulate: bool,
    existing_directory: bool,
) -> None:
    """Component plans require a durable identity before writing any files."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model(durable_identity=False)
    directory = tmp_path / "jobs"
    if existing_directory:
        directory.mkdir()
    before = _campaign_files(directory=directory)
    monkeypatch.setattr(jax.stages.Lowered, "compile", _forbid_numerical_execution)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", _forbid_numerical_execution)

    with pytest.raises(
        ExecutionPlanningError,
        match=r"plan_component_jobs.*ephemeral.*durable_identity=True",
    ):
        component_jobs.plan_component_jobs(
            model=model,
            params=life_cycle._params(typed_dead=True),
            directory=directory,
            n_jobs=3,
            initial_conditions=life_cycle._initial() if simulate else None,
            seed=7 if simulate else None,
        )

    assert _campaign_files(directory=directory) == before


def _forbid_numerical_execution(*_args: object, **_kwargs: object) -> Never:
    """Expose compilation or dispatch past an unsupported-operation guard."""
    msg = "Ephemeral operations must reject before compilation or dispatch."
    raise AssertionError(msg)


@pytest.mark.parametrize("simulate", [False, True])
@pytest.mark.parametrize("existing_directory", [False, True])
def test_planning_requires_an_intact_durable_identity(
    *, tmp_path: Path, simulate: bool, existing_directory: bool
) -> None:
    """A durable model needs its binding seal before publishing a component plan."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model()
    model._sealed_bindings = None
    directory = tmp_path / "jobs"
    if existing_directory:
        directory.mkdir()
    before = _campaign_files(directory=directory)

    with pytest.raises(ModelIdentityError, match="binding seal"):
        component_jobs.plan_component_jobs(
            model=model,
            params=life_cycle._params(typed_dead=True),
            directory=directory,
            n_jobs=3,
            initial_conditions=life_cycle._initial() if simulate else None,
            seed=7 if simulate else None,
        )

    assert _campaign_files(directory=directory) == before


def _complete_result_bytes(
    *, solution: object, simulation: SimulationResult | None
) -> tuple[object, ...]:
    """Encode every value, raw leaf and published panel cell for exact comparison."""
    if not isinstance(solution, SolutionResult):
        msg = "A complete component result includes a SolutionResult."
        raise TypeError(msg)
    values = tuple(
        (period, regime, lifetime._leaf_bytes(value))
        for period, regimes in solution.values.items()
        for regime, value in regimes.items()
    )
    if simulation is None:
        return values, None
    leaves, structure = jax.tree.flatten(simulation.raw_results)
    raw_order = tuple(
        (regime, tuple(periods)) for regime, periods in simulation.raw_results.items()
    )
    raw = raw_order, structure, tuple(lifetime._leaf_bytes(leaf) for leaf in leaves)
    frame = simulation.to_dataframe()
    columns = []
    for name in frame.columns:
        array = np.asarray(frame[name])
        # Object columns contain categorical labels; their pointers are not values.
        content = repr(array.tolist()) if array.dtype.hasobject else array.tobytes()
        columns.append(
            (name, repr(frame[name].dtype), array.dtype.str, array.shape, content)
        )
    index_dtypes = tuple(
        repr(frame.index.get_level_values(level).dtype)
        for level in range(frame.index.nlevels)
    )
    panel = (
        tuple(frame.index.names),
        index_dtypes,
        repr(frame.index.tolist()),
        tuple(columns),
    )
    return values, raw, panel


@pytest.fixture(scope="module", name="complete_campaign")
def _create_complete_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompleteCampaign:
    """Publish one complete campaign and its independent single-process reference."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model()
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    directory = tmp_path_factory.mktemp("complete-component-campaign") / "jobs"
    plan = component_jobs.plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        n_jobs=3,
        initial_conditions=initial,
        seed=7,
    )
    for job in range(len(plan.jobs)):
        component_jobs.run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level="off",
        )
    reference = _model().simulate(
        params=params,
        initial_conditions=initial,
        seed=7,
        log_level="off",
    )
    return directory, model, params, reference


def _reseal_manifest(*, file: h5py.File, manifest: dict[str, object]) -> None:
    """Publish the checksum of a deliberately edited fragment manifest."""
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    del file["manifest"]
    dataset = file.create_dataset(
        "manifest", data=np.frombuffer(encoded, dtype=np.uint8)
    )
    dataset.attrs["sha256"] = hashlib.sha256(encoded).hexdigest()


def test_collected_jobs_equal_complete_single_process_result_bitwise(
    complete_campaign: CompleteCampaign,
) -> None:
    """Three component jobs preserve every value, raw leaf and subject's panel."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    directory, model, params, reference = complete_campaign
    collected = component_jobs.collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )

    assert _complete_result_bytes(
        solution=collected.solution, simulation=collected.simulation
    ) == _complete_result_bytes(solution=reference.solution, simulation=reference)


@pytest.mark.parametrize("operation", ["run_component_job", "collect_component_jobs"])
@pytest.mark.parametrize("existing_campaign", [False, True])
def test_ephemeral_jobs_are_refused_before_reading_or_altering_campaigns(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete_campaign: CompleteCampaign,
    operation: str,
    existing_campaign: bool,
) -> None:
    """Workers and collectors require durable identity before campaign access."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, _, params, _ = complete_campaign
    model = _model(durable_identity=False)
    directory = tmp_path / "jobs"
    if existing_campaign:
        shutil.copytree(campaign, directory)
    before = _campaign_files(directory=directory)
    monkeypatch.setattr(jax.stages.Lowered, "compile", _forbid_numerical_execution)
    monkeypatch.setattr(jax.stages.Compiled, "__call__", _forbid_numerical_execution)

    if operation == "run_component_job":
        invoke = partial(
            component_jobs.run_component_job,
            model=model,
            params=params,
            directory=directory,
            job=0,
            initial_conditions=life_cycle._initial(),
            log_level="off",
        )
    else:
        invoke = partial(
            component_jobs.collect_component_jobs,
            model=model,
            params=params,
            directory=directory,
            log_level="off",
        )
    with pytest.raises(
        ExecutionPlanningError,
        match=rf"{operation}.*ephemeral.*durable_identity=True",
    ):
        invoke()

    assert _campaign_files(directory=directory) == before


def test_simulating_job_copies_its_raw_results_to_the_host_in_one_transfer(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A simulating job gathers all of its raw-result leaves with one `device_get`."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model()
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    directory = tmp_path / "jobs"
    component_jobs.plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        n_jobs=3,
        initial_conditions=initial,
        seed=7,
    )
    device_get = jax.device_get
    callers: list[str] = []

    def _recording_device_get(tree: object) -> object:
        callers.append(sys._getframe(1).f_code.co_name)
        return device_get(tree)

    monkeypatch.setattr(jax, "device_get", _recording_device_get)
    component_jobs.run_component_job(
        model=model,
        params=params,
        directory=directory,
        job=0,
        initial_conditions=initial,
        log_level="off",
    )

    assert callers.count("_simulate_job") == 1


def test_solve_only_jobs_preserve_every_value_without_a_simulation(
    tmp_path: Path,
) -> None:
    """A solve-only campaign collects every value and has no simulation result."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model()
    params = life_cycle._params(typed_dead=True)
    directory = tmp_path / "jobs"
    plan = component_jobs.plan_component_jobs(
        model=model, params=params, directory=directory, n_jobs=3
    )
    for job in range(len(plan.jobs)):
        component_jobs.run_component_job(
            model=model, params=params, directory=directory, job=job, log_level="off"
        )
    collected = component_jobs.collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )
    reference = _model().solve(params=params, log_level="off")

    assert (
        collected.simulation,
        _complete_result_bytes(solution=collected.solution, simulation=None),
    ) == (None, _complete_result_bytes(solution=reference, simulation=None))


@pytest.mark.parametrize(
    ("enable_jit", "log_level"),
    [
        (True, "warning"),
        (True, "progress"),
        (True, "debug"),
        (False, "off"),
        (False, "warning"),
        (False, "progress"),
        (False, "debug"),
    ],
)
def test_jobs_preserve_complete_results_in_each_execution_mode(
    *,
    tmp_path: Path,
    enable_jit: bool,
    log_level: Literal["off", "warning", "progress", "debug"],
) -> None:
    """Matched eager or JIT jobs preserve complete results at each logging level."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    model = _model(enable_jit=enable_jit)
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    directory = tmp_path / "jobs"
    plan = component_jobs.plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        n_jobs=3,
        initial_conditions=initial,
        seed=7,
    )
    for job in range(len(plan.jobs)):
        component_jobs.run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level=log_level,
        )
    collected = component_jobs.collect_component_jobs(
        model=model, params=params, directory=directory, log_level=log_level
    )
    reference = _model(enable_jit=enable_jit).simulate(
        params=params, initial_conditions=initial, seed=7, log_level=log_level
    )

    assert _complete_result_bytes(
        solution=collected.solution, simulation=collected.simulation
    ) == _complete_result_bytes(solution=reference.solution, simulation=reference)


def test_duplicate_value_address_is_refused(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """A fragment cannot publish the same logical value address twice."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    path = directory / "fragments/job-0000.h5"
    with h5py.File(path, "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        duplicate = dict(manifest["values"][0])
        array = np.asarray(file[duplicate["dataset"]][()])
        duplicate["dataset"] = "values/duplicate"
        file.create_dataset(duplicate["dataset"], data=array)
        manifest["values"].append(duplicate)
        _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match=r"duplicate|twice"):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def test_duplicate_panel_address_is_refused(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """A fragment cannot publish the same logical raw-result address twice."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    with h5py.File(directory / "fragments/job-0000.h5", "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        duplicate = dict(manifest["panel"]["leaves"][0])
        array = np.asarray(file[duplicate["dataset"]][()])
        duplicate["dataset"] = "panel/duplicate"
        file.create_dataset(duplicate["dataset"], data=array)
        manifest["panel"]["leaves"].append(duplicate)
        _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(SolutionIntegrityError, match=r"duplicate|twice"):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def test_one_dataset_cannot_supply_distinct_panel_addresses(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """Every logical raw-result leaf owns a distinct physical dataset."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    # Recompute transport checksums so the witness exercises address ownership.

    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    with h5py.File(directory / "fragments/job-0000.h5", "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        leaves = manifest["panel"]["leaves"]
        source = next(
            leaf for leaf in leaves if leaf["identity"]["field"] == "in_regime"
        )
        target = next(
            leaf
            for leaf in leaves
            if leaf["identity"]["field"] == "nested_policy_fallback"
            and leaf["identity"]["regime"] == source["identity"]["regime"]
            and leaf["identity"]["period"] == source["identity"]["period"]
        )
        array = np.asarray(file[source["dataset"]][()])
        target_array = np.asarray(file[target["dataset"]][()])
        if array.dtype != target_array.dtype or array.shape != target_array.shape:
            msg = "The dataset-alias witness requires matching leaf representations."
            raise ValueError(msg)
        del file[target["dataset"]]
        target["dataset"] = source["dataset"]
        target["sha256"] = _array_checksum(identity=target["identity"], array=array)
        _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(
        SolutionIntegrityError,
        match=r"dataset.*(duplicate|twice)|duplicate.*dataset|referenced.*twice",
    ):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "code-bool",
        "code-fraction",
        "period-fraction",
        "kind",
        "shape-fraction",
        "format-version-bool",
    ],
)
def test_fragment_metadata_preserves_exact_address_types(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """Checksummed metadata cannot coerce a malformed address into a valid one."""
    component_jobs = importlib.import_module("lcm.component_jobs")

    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    with h5py.File(directory / "fragments/job-0000.h5", "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        leaf = manifest["values"][0]
        identity = leaf["identity"]
        array = np.asarray(file[leaf["dataset"]][()])
        if corruption == "code-bool":
            if identity["code"] != 0:
                msg = "The Boolean-code witness requires original code zero."
                raise ValueError(msg)
            identity["code"] = False
        elif corruption == "code-fraction":
            identity["code"] += 0.5
        elif corruption == "period-fraction":
            identity["period"] += 0.5
        elif corruption == "kind":
            identity["kind"] = "panel"
        elif corruption == "shape-fraction":
            leaf["shape"][0] += 0.5
        else:
            manifest["format_version"] = True
        leaf["sha256"] = _array_checksum(identity=identity, array=array)
        _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def _replace_campaign_plan(*, directory: Path, plan: dict[str, object]) -> None:
    """Replace one test plan and bind its fragments to the replacement bytes."""
    encoded = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    (directory / "plan.json").write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    for path in sorted((directory / "fragments").glob("*.h5")):
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            manifest["plan_sha256"] = digest
            _reseal_manifest(file=file, manifest=manifest)


@pytest.mark.parametrize(
    "corruption", ["code-bool", "code-fraction", "duplicate-assignment", "version-bool"]
)
def test_loaded_plan_refuses_malformed_codes_and_assignments(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """A loaded plan preserves exact original codes and their unique assignment."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, _, _, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    if corruption == "code-bool":
        plan["codes"][0] = plan["jobs"][0][0] = False
    elif corruption == "code-fraction":
        plan["codes"][0] += 0.5
        plan["jobs"][0][0] += 0.5
    elif corruption == "duplicate-assignment":
        plan["jobs"][1] = list(plan["jobs"][0])
    else:
        plan["format_version"] = True
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.load_component_job_plan(directory=directory)


@pytest.mark.parametrize("corruption", ["state-name", "reversed-codes"])
def test_collector_binds_plan_coordinates_to_the_model(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """The actual model owns its blocked-state header and original code order."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    if corruption == "state-name":
        plan["state_name"] = "wealth"
    else:
        plan["codes"].reverse()
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises((ExecutionPlanningError, SolutionIntegrityError)):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def test_collector_requires_values_of_an_empty_subject_type(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """Complete subject coverage does not authorize omitting an original value code."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    plan["codes"] = [0, 2]
    plan["jobs"] = [[0], [2]]
    (directory / "fragments/job-0001.h5").unlink()
    remaining = directory / "fragments/job-0002.h5"
    with h5py.File(remaining, "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        manifest["job"] = 1
        _reseal_manifest(file=file, manifest=manifest)
    remaining.rename(directory / "fragments/job-0001.h5")
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises((ExecutionPlanningError, SolutionIntegrityError), match="code"):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def test_collector_requires_model_owned_raw_state_fields(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """Matching fragment schemas cannot omit the model's required wealth state."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    removed = 0
    for path in sorted((directory / "fragments").glob("*.h5")):
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            leaves = manifest["panel"]["leaves"]
            required = [
                leaf
                for leaf in leaves
                if leaf["identity"]["field"] == "states"
                and leaf["identity"]["key"] == "wealth"
            ]
            for leaf in required:
                del file[leaf["dataset"]]
                leaves.remove(leaf)
            removed += len(required)
            _reseal_manifest(file=file, manifest=manifest)
    if removed == 0:
        msg = "The raw-schema witness must remove a required wealth-state leaf."
        raise ValueError(msg)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


@pytest.mark.parametrize(
    "corruption",
    ["mask-shape", "role-dtype", "mask-dtype", "value-shape", "value-dtype"],
)
def test_collector_validates_model_owned_fixed_raw_fields(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """Fragment agreement cannot authorize incompatible raw shapes or dtypes."""
    component_jobs = importlib.import_module("lcm.component_jobs")

    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    changed = 0
    field_name = (
        "V_arr"
        if corruption.startswith("value")
        else "own_stakeholder"
        if corruption == "role-dtype"
        else "in_regime"
    )
    for path in sorted((directory / "fragments").glob("*.h5")):
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            for leaf in manifest["panel"]["leaves"]:
                if leaf["identity"]["field"] != field_name:
                    continue
                array = np.asarray(file[leaf["dataset"]][()])
                if corruption.endswith("shape"):
                    if array.ndim != 1:
                        msg = "The shape witness requires a singleton raw array."
                        raise ValueError(msg)
                    array = array[:, None]
                elif corruption == "role-dtype":
                    array = array.astype(np.float64)
                elif corruption == "mask-dtype":
                    array = array.astype(np.int32)
                else:
                    array = array.astype(
                        np.float32 if array.dtype == np.float64 else np.float64
                    )
                del file[leaf["dataset"]]
                file.create_dataset(leaf["dataset"], data=array)
                leaf["shape"] = list(array.shape)
                leaf["dtype"] = array.dtype.str
                leaf["sha256"] = _array_checksum(identity=leaf["identity"], array=array)
                changed += 1
            _reseal_manifest(file=file, manifest=manifest)
    if not changed:
        msg = "The fixed-field witness must modify populated raw leaves."
        raise ValueError(msg)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def _campaign_files(*, directory: Path) -> tuple[object, ...]:
    """Snapshot caller-owned files and directories, including campaign existence."""
    return directory.exists(), tuple(
        (
            str(path.relative_to(directory)),
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in sorted(directory.rglob("*"))
    )


@pytest.mark.parametrize("argument", ["n-jobs", "assignment-code", "job-index", "seed"])
def test_public_job_integers_refuse_boolean_before_publication(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, argument: str
) -> None:
    """Boolean counts, codes, job indices and seeds cannot publish or dispatch."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = (
        shutil.copytree(campaign, tmp_path / "jobs")
        if argument == "job-index"
        else tmp_path / "jobs"
    )
    before = _campaign_files(directory=directory)
    refused = False
    try:
        if argument == "job-index":
            component_jobs.run_component_job(
                model=model,
                params=params,
                directory=directory,
                job=False,
                initial_conditions=life_cycle._initial(),
                log_level="off",
            )
        else:
            component_jobs.plan_component_jobs(
                model=model,
                params=params,
                directory=directory,
                n_jobs=True
                if argument == "n-jobs"
                else 3
                if argument == "seed"
                else None,
                assignment=((False,), (1,), (2,))
                if argument == "assignment-code"
                else None,
                initial_conditions=life_cycle._initial(),
                seed=True if argument == "seed" else 7,
            )
    except ExecutionPlanningError:
        refused = True

    assert (refused, _campaign_files(directory=directory)) == (True, before)


def test_planner_refuses_empty_population_before_publication(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """An empty overall population cannot publish a simulating component plan."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    _, model, params, _ = complete_campaign
    directory = tmp_path / "jobs"
    before = _campaign_files(directory=directory)
    initial = {name: array[:0] for name, array in life_cycle._initial().items()}
    refused = False
    try:
        component_jobs.plan_component_jobs(
            model=model,
            params=params,
            directory=directory,
            n_jobs=3,
            initial_conditions=initial,
            seed=7,
        )
    except ExecutionPlanningError:
        refused = True

    assert (refused, _campaign_files(directory=directory)) == (True, before)


@pytest.mark.parametrize(
    "corruption",
    [
        "header-code",
        "header-job",
        "header-period",
        "panel-period",
        "panel-kind",
        "population-count",
        "chunk-width",
    ],
)
def test_fragment_headers_and_panels_preserve_exact_metadata(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """Transport metadata preserves exact coordinates, kinds and population counts."""
    component_jobs = importlib.import_module("lcm.component_jobs")

    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    with h5py.File(directory / "fragments/job-0000.h5", "r+") as file:
        manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
        if corruption == "header-code":
            if manifest["codes"][0] != 0:
                raise ValueError(
                    "The Boolean header-code witness requires original zero."
                )
            manifest["codes"][0] = False
        elif corruption == "header-job":
            manifest["job"] = False
        elif corruption == "header-period":
            manifest["coordinates"][0][0] += 0.5
        elif corruption in {"panel-period", "panel-kind"}:
            leaf = manifest["panel"]["leaves"][0]
            if corruption == "panel-period":
                leaf["identity"]["period"] += 0.5
            else:
                leaf["identity"]["kind"] = "value"
            leaf["sha256"] = _array_checksum(
                identity=leaf["identity"], array=np.asarray(file[leaf["dataset"]][()])
            )
        elif corruption == "population-count":
            manifest["panel"]["n_subjects"] += 0.5
        else:
            manifest["panel"]["subject_batch_size"] += 0.5
        _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


@pytest.mark.parametrize("field_name", ["seed", "n_subjects"])
def test_loaded_plan_preserves_exact_simulation_integers(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, field_name: str
) -> None:
    """A loaded simulation plan cannot round its seed or population count."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, _, _, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    plan["simulation"][field_name] += 0.5
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.load_component_job_plan(directory=directory)


@pytest.mark.parametrize(
    ("field_name", "foreign"),
    [("seed", None), ("n_subjects", 0), ("initial_conditions_sha256", 123)],
)
def test_loaded_simulation_plan_preserves_a_complete_admitted_bundle(
    *,
    tmp_path: Path,
    complete_campaign: CompleteCampaign,
    field_name: str,
    foreign: object,
) -> None:
    """A simulation plan needs an integer seed, subjects and a population digest."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, _, _, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    plan["simulation"][field_name] = foreign
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.load_component_job_plan(directory=directory)


@pytest.mark.parametrize(
    "corruption", ["missing", "short", "wrong-sum", "negative", "boolean"]
)
def test_loaded_simulation_plan_requires_one_count_per_code_summing_to_population(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """A plan's per-code counts are exact, one per code, and sum to the population."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, _, _, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    plan = json.loads((directory / "plan.json").read_bytes())
    counts = list(plan["simulation"]["code_counts"])
    plan["simulation"]["code_counts"] = {
        "missing": None,
        "short": counts[:-1],
        "wrong-sum": [counts[0] + 1, *counts[1:]],
        "negative": [-1, counts[1] + counts[0] + 1, *counts[2:]],
        "boolean": [True, *counts[1:]],
    }[corruption]
    _replace_campaign_plan(directory=directory, plan=plan)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.load_component_job_plan(directory=directory)


@pytest.mark.parametrize(
    ("field_name", "foreign"),
    [
        ("jax_version", "foreign-jax"),
        ("jaxlib_version", "foreign-jaxlib"),
        ("xla_flags", "foreign-compiler-options"),
        ("log_level", "warning"),
        ("enable_jit", False),
        ("action_partitions", {"work": 2}),
        ("native_build_fingerprint", "0" * 64),
    ],
)
def test_collector_binds_execution_to_its_own_runtime(
    *,
    tmp_path: Path,
    complete_campaign: CompleteCampaign,
    field_name: str,
    foreign: object,
) -> None:
    """Fragment agreement cannot authorize another compiler or execution mode."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    for path in sorted((directory / "fragments").glob("*.h5")):
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            if manifest["execution"].get(field_name) == foreign:
                raise ValueError(
                    "The runtime witness must change the admitted execution."
                )
            manifest["execution"][field_name] = foreign
            _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(ExecutionPlanningError, match="execution"):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicated", "failed", "stale", "identity", "checksum", "partial"],
)
def test_collector_refuses_incomplete_or_untrusted_campaigns(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """A missing, duplicated, failed, stale or damaged job cannot yield a result."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    path = directory / "fragments/job-0001.h5"
    if corruption == "missing":
        path.unlink()
    elif corruption == "duplicated":
        shutil.copyfile(directory / "fragments/job-0000.h5", path)
    elif corruption == "failed":
        path.unlink()
        path.with_suffix(".failed.json").write_text(json.dumps({"job": 1}))
    elif corruption == "partial":
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    else:
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            if corruption == "stale":
                manifest["plan_id"] = "another-plan"
            elif corruption == "identity":
                manifest["identity"]["params_fingerprint"] = "another-model-parameters"
            else:
                dataset = file[manifest["values"][0]["dataset"]]
                array = np.asarray(dataset[()])
                array.reshape(-1).view(np.uint8)[0] ^= 1
                dataset[...] = array
            _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises((ExecutionPlanningError, SolutionIntegrityError)):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


_FRESH_WORKER_SCRIPT = """
import hashlib
import json
import os
import sys
from pathlib import Path

import jax
import jaxlib
import lcm
from _lcm import version
from _lcm.egm.upper_envelope._exact_affine import ffi
from lcm.component_jobs import run_component_job
from tests.ci.probe_native import probe
from tests.solution.test_component_jobs import _model
from tests.simulation import test_type_grouped_simulation as life_cycle

native_probe = probe(root=Path.cwd())
if native_probe.exit_code:
    msg = f"The fresh worker's native payload is not ready: {native_probe}"
    raise RuntimeError(msg)
run_component_job(
    model=_model(), params=life_cycle._params(typed_dead=True),
    directory=Path(sys.argv[1]), job=0,
    initial_conditions=life_cycle._initial(), log_level="off",
)
print(json.dumps({
    "pid": os.getpid(), "lcm_file": lcm.__file__,
    "source_version": version.__version__, "jax": jax.__version__,
    "jaxlib": jaxlib.__version__, "x64": jax.config.read("jax_enable_x64"),
    "backend": jax.default_backend(), "xla_flags": os.environ.get("XLA_FLAGS", ""),
    "native_probe": native_probe.status,
    "native_cpu_payload": str(ffi._CPU_LIBRARY),
    "native_cpu_sha256": hashlib.sha256(ffi._CPU_LIBRARY.read_bytes()).hexdigest(),
}, sort_keys=True))
"""


def test_fresh_pixi_process_preserves_complete_component_result(
    *,
    tmp_path: Path,
    complete_campaign: CompleteCampaign,
    record_property: Callable[[str, object], None],
) -> None:
    """A fresh frozen Pixi worker preserves every value, raw leaf and panel cell."""

    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, reference = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    (directory / "fragments/job-0000.h5").unlink()
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = str(jax.config.read("jax_enable_x64")).lower()
    pixi = shutil.which("pixi")
    if pixi is None:
        raise RuntimeError("The fresh worker requires the active Pixi executable.")
    completed = subprocess.run(  # noqa: S603
        [
            pixi,
            "run",
            "--as-is",
            "--manifest-path",
            os.environ["PIXI_PROJECT_MANIFEST"],
            "-e",
            os.environ["PIXI_ENVIRONMENT_NAME"],
            "python",
            "-c",
            _FRESH_WORKER_SCRIPT,
            str(directory),
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    receipt = json.loads(completed.stdout.splitlines()[-1])
    record_property("child_stdout", completed.stdout)
    record_property("child_stderr", completed.stderr)
    collected = component_jobs.collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )

    assert (
        receipt["pid"] != os.getpid(),
        receipt["lcm_file"],
        receipt["source_version"],
        receipt["x64"],
        receipt["native_probe"],
        receipt["native_cpu_sha256"],
        _complete_result_bytes(
            solution=collected.solution, simulation=collected.simulation
        ),
    ) == (
        True,
        lcm.__file__,
        version.__version__,
        jax.config.read("jax_enable_x64"),
        "ready",
        hashlib.sha256(ffi._CPU_LIBRARY.read_bytes()).hexdigest(),
        _complete_result_bytes(solution=reference.solution, simulation=reference),
    )


@pytest.mark.parametrize("corruption", ["job-membership", "row-shape", "row-dtype"])
def test_collector_binds_original_rows_to_the_planned_jobs(
    *, tmp_path: Path, complete_campaign: CompleteCampaign, corruption: str
) -> None:
    """Global coverage cannot authorize another job's rows or malformed row arrays."""
    component_jobs = importlib.import_module("lcm.component_jobs")

    campaign, model, params, _ = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    if corruption == "job-membership":
        replacements = {}
        for job in (0, 2):
            with h5py.File(directory / f"fragments/job-{job:04d}.h5", "r") as file:
                manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
                replacements[job] = np.asarray(
                    file[manifest["panel"]["rows"]["dataset"]][()]
                )
        if not len(replacements[0]) or not len(replacements[2]):
            raise ValueError("The row-membership witness requires two populated jobs.")
        replacements[0][0], replacements[2][0] = replacements[2][0], replacements[0][0]
        replacements = {job: np.sort(rows) for job, rows in replacements.items()}
    else:
        replacements = {}
    for path in sorted((directory / "fragments").glob("*.h5")):
        with h5py.File(path, "r+") as file:
            manifest = json.loads(np.asarray(file["manifest"][()]).tobytes())
            leaf = manifest["panel"]["rows"]
            array = np.asarray(file[leaf["dataset"]][()])
            if corruption == "job-membership":
                array = replacements.get(manifest["job"], array)
            elif corruption == "row-shape":
                array = array[:, None]
            else:
                array = array.astype(np.float64)
            del file[leaf["dataset"]]
            file.create_dataset(leaf["dataset"], data=array)
            leaf["shape"] = list(array.shape)
            leaf["dtype"] = array.dtype.str
            leaf["sha256"] = _array_checksum(identity=leaf["identity"], array=array)
            _reseal_manifest(file=file, manifest=manifest)

    with pytest.raises(SolutionIntegrityError):
        component_jobs.collect_component_jobs(
            model=model, params=params, directory=directory, log_level="off"
        )


def test_actual_worker_publication_failure_can_be_retried(
    *, tmp_path: Path, complete_campaign: CompleteCampaign
) -> None:
    """A successful worker retry clears its recorded failure and restores the result."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    campaign, model, params, reference = complete_campaign
    directory = shutil.copytree(campaign, tmp_path / "jobs")
    path = directory / "fragments/job-0000.h5"
    path.unlink()
    path.mkdir()
    error_type = ""
    try:
        component_jobs.run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=0,
            initial_conditions=life_cycle._initial(),
            log_level="off",
        )
    except OSError as error:
        error_type = type(error).__name__
    failed_path = directory / "fragments/job-0000.failed.json"
    failure = json.loads(failed_path.read_bytes())
    recorded = (failure["job"], failure["codes"], failure["error_type"])
    path.rmdir()
    component_jobs.run_component_job(
        model=model,
        params=params,
        directory=directory,
        job=0,
        initial_conditions=life_cycle._initial(),
        log_level="off",
    )
    collected = component_jobs.collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )

    assert (
        bool(error_type),
        recorded,
        failed_path.exists(),
        _complete_result_bytes(
            solution=collected.solution, simulation=collected.simulation
        ),
    ) == (
        True,
        (0, [0], error_type),
        False,
        _complete_result_bytes(solution=reference.solution, simulation=reference),
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "jax_disable_jit",
        "jax_default_prng_impl",
        "jax_random_seed_offset",
        "jax_threefry_partitionable",
    ],
)
def test_collector_refuses_another_ambient_jax_execution(
    *, complete_campaign: CompleteCampaign, config_name: str
) -> None:
    """Collection requires the workers' effective JIT and seeded-random settings."""
    component_jobs = importlib.import_module("lcm.component_jobs")
    directory, model, params, _ = complete_campaign
    original = getattr(jax.config, config_name)
    foreign = (
        ("rbg" if original != "rbg" else "threefry2x32")
        if config_name == "jax_default_prng_impl"
        else original + 1
        if config_name == "jax_random_seed_offset"
        else not original
    )
    jax.config.update(config_name, foreign)
    try:
        with pytest.raises(ExecutionPlanningError, match="execution"):
            component_jobs.collect_component_jobs(
                model=model, params=params, directory=directory, log_level="off"
            )
    finally:
        jax.config.update(config_name, original)
