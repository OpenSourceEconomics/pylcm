"""Tiny local CPU archives preserve complete results and exact reference equality.

The numeric campaign uses production seed20260903 with the existing tiny model.
It proves persistence/comparison behavior, not full ACA or native GPU acceptance.
"""

import copy
import dataclasses
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest
from test_stage8a_population import owner_driver as _imported_owner_driver
from test_stage8a_receipts import (
    fragment_helpers as _imported_fragment_helpers,
)

from lcm._solver_api.contract import ArtifactRef, OmissionReason
from lcm._solver_api.identity import ArtifactKey
from lcm.component_jobs import CollectedComponentJobs
from lcm.exceptions import IncompatibleSolutionError

owner_driver = _imported_owner_driver
fragment_helpers = _imported_fragment_helpers


@pytest.fixture(scope="module")
def tiny_campaign(*, tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Generate a new eleven-row CPU campaign with the owner production seed."""
    root = Path(os.environ["STAGE8A_RECEIPT_CORE_ROOT"]).resolve()
    modules = {}
    for name, relative in (
        ("lcm", "src/lcm/__init__.py"),
        ("lcm.component_jobs", "src/lcm/component_jobs.py"),
        ("tests.solution.test_component_jobs", "tests/solution/test_component_jobs.py"),
        (
            "tests.simulation.test_type_grouped_simulation",
            "tests/simulation/test_type_grouped_simulation.py",
        ),
    ):
        module = importlib.import_module(name)
        if Path(str(module.__file__)).resolve() != root / relative:
            raise RuntimeError(f"Tiny archive campaign imported another source: {name}")
        modules[name] = module
    jax = importlib.import_module("jax")
    if jax.default_backend() != "cpu":
        raise RuntimeError("The tiny archive witness requires a CPU backend")
    fixture = modules["tests.solution.test_component_jobs"]
    life_cycle = modules["tests.simulation.test_type_grouped_simulation"]
    api = modules["lcm.component_jobs"]
    model = fixture._model()
    params = life_cycle._params(typed_dead=True)
    initial = life_cycle._initial()
    directory = tmp_path_factory.mktemp("tiny-seed20260903") / "plan"
    plan = api.plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        assignment=((0,), (1,), (2,)),
        initial_conditions=initial,
        seed=20_260_903,
    )
    for job in (0, 1, 2):
        api.run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level="off",
        )
    reference = fixture._model().simulate(
        params=params,
        initial_conditions=initial,
        seed=20_260_903,
        log_level="off",
    )
    return SimpleNamespace(
        model=model,
        params=params,
        api=api,
        fixture=fixture,
        directory=directory,
        initial=initial,
        plan=plan,
        reference=reference,
        expected=fixture._complete_result_bytes(
            solution=reference.solution,
            simulation=reference,
        ),
        original_ids=np.array([90, 10, 30, 70, 50, 20, 80, 40, 60, 100, 110]),
    )


@pytest.fixture
def complete_result(*, tiny_campaign: SimpleNamespace) -> CollectedComponentJobs:
    """Collect a fresh complete result without consuming the independent reference."""
    return tiny_campaign.api.collect_component_jobs(
        model=tiny_campaign.model,
        params=tiny_campaign.params,
        directory=tiny_campaign.directory,
        log_level="off",
    )


def _publish(
    *,
    driver: ModuleType,
    campaign: SimpleNamespace,
    result: CollectedComponentJobs,
    out: Path,
) -> Path:
    """Invoke the proposed driver publication seam with genuine complete results."""
    return driver._publish_reference(
        plan=result.plan,
        solution=result.solution,
        simulation=result.simulation,
        original_ids=campaign.original_ids,
        out=out,
        provenance={"scope": "tiny-local-cpu", "seed": 20_260_903},
        started_at="2026-10-03T00:00:00+00:00",
        finished_at="2026-10-03T00:00:01+00:00",
        elapsed_seconds=1.0,
    )


def test_collected_publication_records_its_actual_phase(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """A complete collector archive identifies itself as collection."""
    caught = None
    receipt = None
    try:
        receipt = owner_driver._publish_reference(
            plan=complete_result.plan,
            solution=complete_result.solution,
            simulation=complete_result.simulation,
            original_ids=tiny_campaign.original_ids,
            out=tmp_path / "collected",
            phase="collect",
            provenance={"scope": "tiny-local-cpu"},
            started_at="2026-10-03T00:00:00+00:00",
            finished_at="2026-10-03T00:00:01+00:00",
            elapsed_seconds=1.0,
        )
    except TypeError as error:
        caught = str(error)
    record = None if receipt is None else json.loads(receipt.read_bytes())

    assert (caught, None if record is None else record["phase"]) == (None, "collect")


def test_reference_publication_keeps_complete_solution_before_simulation_save(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """Complete values, schemas, raw leaves, panel and IDs survive publication."""
    campaign = tiny_campaign
    solution = complete_result.solution
    before = campaign.fixture._complete_result_bytes(
        solution=solution,
        simulation=complete_result.simulation,
    )
    expected_catalog = [
        {
            "period": period,
            "regime": regime,
            "axis_names": list(
                solution.metadata.value_schemas[(period, regime)].axis_names,
            ),
            "dtype": np.asarray(value).dtype.str,
            "shape": list(np.asarray(value).shape),
            "array_checksum": fragment_helpers.array_checksum(
                identity={"period": period, "regime": regime},
                array=np.asarray(value),
            ),
        }
        for period, regimes in solution.values.items()
        for regime, value in regimes.items()
    ]
    expected_raw_order = [
        [regime, list(periods)]
        for regime, periods in complete_result.simulation.raw_results.items()
    ]
    expected_raw_schemas = [
        {
            "regime": regime,
            "period": period,
            "actions": list(data.actions),
            "states": list(data.states),
        }
        for regime, periods in complete_result.simulation.raw_results.items()
        for period, data in periods.items()
    ]
    original_leaves = [
        (address, np.asarray(leaf))
        for regime, periods in complete_result.simulation.raw_results.items()
        for period, data in periods.items()
        for address, leaf in campaign.api._data_leaves(
            regime=regime,
            period=period,
            data=data,
        )
    ]
    expected_raw_catalog = [
        {
            "address": list(address),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "array_checksum": fragment_helpers.array_checksum(
                identity={
                    "regime": address[0],
                    "period": address[1],
                    "field": address[2],
                    "key": address[3],
                },
                array=array,
            ),
        }
        for address, array in original_leaves
    ]
    expected_artifacts = (
        tuple(solution.retained_continuations),
        tuple(solution.replay_artifacts),
        tuple(solution.auxiliary_artifacts),
        tuple(solution.diagnostics),
        dict(solution.omissions),
    )
    output = tmp_path / "reference"

    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=complete_result,
        out=output,
    )

    persistence = importlib.import_module("lcm.persistence")
    results = importlib.import_module("lcm.result")
    loaded = persistence.load_solution(
        path=output / "solution.h5",
        verify_checksums=True,
    )
    simulation = results.SimulationResult.load(directory=output / "simulation")
    record = json.loads(receipt.read_bytes())
    ids_file = output / "original_ids.npy"
    ids = np.load(ids_file, allow_pickle=False)
    # Archive storage sorts addresses; the receipt preserves publication order.
    ordered_values = {
        period: {regime: loaded.values[period][regime] for regime in regimes}
        for period, regimes in solution.values.items()
    }
    observed = campaign.fixture._complete_result_bytes(
        solution=dataclasses.replace(loaded, values=ordered_values),
        simulation=simulation,
    )
    loaded_leaves = {
        address: np.asarray(leaf)
        for regime, periods in simulation.raw_results.items()
        for period, data in periods.items()
        for address, leaf in campaign.api._data_leaves(
            regime=regime,
            period=period,
            data=data,
        )
    }
    raw_domain_exact = set(loaded_leaves) == {address for address, _ in original_leaves}
    loaded_raw_bytes = (
        tuple(
            (
                address,
                loaded_leaves[address].dtype.str,
                loaded_leaves[address].shape,
                loaded_leaves[address].tobytes(),
            )
            for address, _ in original_leaves
        )
        if raw_domain_exact
        else None
    )
    original_raw_bytes = tuple(
        (address, array.dtype.str, array.shape, array.tobytes())
        for address, array in original_leaves
    )
    artifact_sha256 = {
        name: entry["sha256"] for name, entry in record["artifacts"].items()
    }
    expected_sha256 = {}
    for payload in output.rglob("*"):
        if payload.is_file() and payload != receipt:
            with payload.open("rb") as handle:
                expected_sha256[payload.relative_to(output).as_posix()] = (
                    hashlib.file_digest(handle, "sha256").hexdigest()
                )
    loaded_artifacts = (
        tuple(loaded.retained_continuations),
        tuple(loaded.replay_artifacts),
        tuple(loaded.auxiliary_artifacts),
        tuple(loaded.diagnostics),
        dict(loaded.omissions),
    )

    assert (
        campaign.plan.seed,
        campaign.plan.codes,
        campaign.plan.jobs,
        campaign.plan.n_subjects,
        before,
        observed[0],
        raw_domain_exact,
        loaded_raw_bytes,
        observed[2],
        record["phase"],
        record["status"],
        record["value_catalog"],
        record["raw_coordinates"],
        record["raw_schemas"],
        record["raw_catalog"],
        record["provenance"],
        dataclasses.replace(loaded.metadata, source=solution.metadata.source),
        loaded.metadata.source.value,
        loaded_artifacts,
        ids.tolist(),
        ids.dtype.str,
        ids.shape,
        record["artifacts"]["original_ids.npy"]["array_checksum"],
        record["artifacts"]["original_ids.npy"]["sha256"],
        complete_result.simulation.solution,
        artifact_sha256,
    ) == (
        20_260_903,
        (0, 1, 2),
        ((0,), (1,), (2,)),
        11,
        campaign.expected,
        campaign.expected[0],
        True,
        original_raw_bytes,
        campaign.expected[2],
        "reference",
        "completed",
        expected_catalog,
        expected_raw_order,
        expected_raw_schemas,
        expected_raw_catalog,
        {"scope": "tiny-local-cpu", "seed": 20_260_903},
        solution.metadata,
        "persisted",
        expected_artifacts,
        [90, 10, 30, 70, 50, 20, 80, 40, 60, 100, 110],
        campaign.original_ids.dtype.str,
        (11,),
        fragment_helpers.array_checksum(
            identity={"field": "original_subject_ids"},
            array=campaign.original_ids,
        ),
        hashlib.sha256(ids_file.read_bytes()).hexdigest(),
        None,
        expected_sha256,
    )


def test_complete_collection_matches_immutable_reference_bitwise(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """Ordered values, raw fields, panel cells and IDs equal the reference."""
    campaign = tiny_campaign
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )

    result = owner_driver._compare_reference(
        collected=complete_result,
        original_ids=campaign.original_ids,
        reference=output,
        reference_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )

    assert (
        result,
        campaign.fixture._complete_result_bytes(
            solution=complete_result.solution,
            simulation=complete_result.simulation,
        ),
    ) == (None, campaign.expected)


@pytest.mark.parametrize(
    "change",
    ["one_ulp", "signed_zero", "dtype", "shape", "omission", "order"],
)
# Keep the complete concrete protocol and its negative controls adjacent.
def test_reference_comparison_refuses_changed_values(  # noqa: C901
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
    change: str,
) -> None:
    """One representable neighbor, a zero sign or an absent cell breaks equality."""
    campaign = tiny_campaign
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    values = {
        period: {regime: np.asarray(value).copy() for regime, value in regimes.items()}
        for period, regimes in complete_result.solution.values.items()
    }
    period = next(iter(values))
    regime = next(iter(values[period]))
    array = values[period][regime]
    reference_values = {
        p: {r: np.asarray(value).copy() for r, value in regimes.items()}
        for p, regimes in reference_result.solution.values.items()
    }
    if change == "one_ulp":
        array.flat[0] = np.nextafter(array.flat[0], np.array(np.inf, dtype=array.dtype))
    elif change == "signed_zero":
        reference_values[period][regime].flat[0] = 0.0
        array.flat[0] = -0.0
    elif change == "dtype":
        # Reinterpret the same payload bytes: a bare SHA cannot distinguish this.
        values[period][regime] = array.view(
            np.uint32 if array.dtype.itemsize == 4 else np.uint64,
        )
    elif change == "shape":
        values[period][regime] = array.reshape(-1)
    elif change == "omission":
        del values[period][regime]
    elif change == "order":
        values = dict(reversed(tuple(values.items())))
    reference_array = reference_values[period][regime]
    if change == "one_ulp":
        integer_dtype = np.uint32 if array.dtype.itemsize == 4 else np.uint64
        original = np.asarray(reference_array.flat[0])
        changed = np.asarray(array.flat[0])
        mutation_is_concrete = bool(
            np.isfinite(original)
            and np.isfinite(changed)
            and abs(
                int(original.view(integer_dtype)) - int(changed.view(integer_dtype)),
            )
            == 1
            and original.tobytes() != changed.tobytes()
        )
    elif change == "signed_zero":
        mutation_is_concrete = bool(
            reference_array.flat[0] == array.flat[0] == 0
            and not np.signbit(reference_array.flat[0])
            and np.signbit(array.flat[0])
            and reference_array.tobytes() != array.tobytes()
        )
    elif change in {"dtype", "shape"}:
        changed = values[period][regime]
        mutation_is_concrete = changed.tobytes() == reference_array.tobytes() and (
            changed.dtype != reference_array.dtype
            if change == "dtype"
            else changed.shape != reference_array.shape
        )
    else:
        mutation_is_concrete = [
            (p, r) for p, regimes in values.items() for r in regimes
        ] != [(p, r) for p, regimes in reference_values.items() for r in regimes]
    candidate = dataclasses.replace(
        complete_result,
        solution=dataclasses.replace(complete_result.solution, values=values),
    )
    reference_result = dataclasses.replace(
        reference_result,
        solution=dataclasses.replace(
            reference_result.solution,
            values=reference_values,
        ),
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )
    caught = None
    try:
        owner_driver._compare_reference(
            collected=candidate,
            original_ids=campaign.original_ids,
            reference=output,
            reference_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )
    except ValueError as error:
        caught = str(error)

    assert (mutation_is_concrete, caught) == (
        True,
        "Reference comparison differs: values",
    )


@pytest.mark.parametrize(
    "change",
    ["raw_leaf", "raw_field", "panel_order", "external_ids", "receipt_sha"],
)
def test_reference_comparison_refuses_changed_population_outputs(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every raw field, panel row, external ID and immutable receipt is required."""
    campaign = tiny_campaign
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )
    expected_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    candidate = complete_result
    original_ids = campaign.original_ids.copy()
    mutation_is_concrete = False
    if change in {"raw_leaf", "raw_field"}:
        simulation = copy.copy(complete_result.simulation)
        raw = {
            regime: dict(periods) for regime, periods in simulation.raw_results.items()
        }
        regime = "work"
        period = 0
        data = raw[regime][period]
        if change == "raw_leaf":
            original_leaf = np.asarray(data.V_arr)
            leaf = np.asarray(data.V_arr).copy()
            leaf.flat[0] = np.nextafter(
                leaf.flat[0],
                np.array(np.inf, dtype=leaf.dtype),
            )
            integer_dtype = np.uint32 if leaf.dtype.itemsize == 4 else np.uint64
            mutation_is_concrete = bool(
                np.isfinite(original_leaf.flat[0])
                and np.isfinite(leaf.flat[0])
                and abs(
                    int(np.asarray(original_leaf.flat[0]).view(integer_dtype))
                    - int(np.asarray(leaf.flat[0]).view(integer_dtype))
                )
                == 1
                and original_leaf.tobytes() != leaf.tobytes()
            )
            data = dataclasses.replace(data, V_arr=jax.device_put(leaf))
        else:
            states = dict(data.states)
            del states[next(iter(states))]
            mutation_is_concrete = tuple(states) != tuple(data.states)
            data = dataclasses.replace(data, states=MappingProxyType(states))
        raw[regime][period] = data
        # Deliberately alter only this copied result's raw boundary.
        simulation._raw_results = MappingProxyType(
            {name: MappingProxyType(periods) for name, periods in raw.items()}
        )
        candidate = dataclasses.replace(complete_result, simulation=simulation)
    elif change == "panel_order":
        original_frame = complete_result.simulation.to_dataframe()
        frame = original_frame.iloc[::-1].copy()
        mutation_is_concrete = not frame.index.equals(original_frame.index)
        monkeypatch.setattr(
            complete_result.simulation,
            "to_dataframe",
            lambda **_kwargs: frame,
        )
    elif change == "external_ids":
        original_ids[[0, 1]] = original_ids[[1, 0]]
        mutation_is_concrete = original_ids.tobytes() != campaign.original_ids.tobytes()
    elif change == "receipt_sha":
        expected_sha = "0" * 64
        mutation_is_concrete = (
            expected_sha != hashlib.sha256(receipt.read_bytes()).hexdigest()
        )
    caught = None
    try:
        owner_driver._compare_reference(
            collected=candidate,
            original_ids=original_ids,
            reference=output,
            reference_receipt_sha256=expected_sha,
        )
    except ValueError as error:
        caught = str(error)
    expected = {
        "raw_leaf": "Reference comparison differs: raw_results",
        "raw_field": "Reference comparison differs: raw_results",
        "panel_order": "Reference comparison differs: panel",
        "external_ids": "Reference comparison differs: original_ids",
        "receipt_sha": "Reference receipt SHA-256 differs",
    }

    assert (mutation_is_concrete, caught) == (True, expected[change])


def test_simulation_persistence_failure_cannot_complete_reference_publication(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """A failed simulation archive propagates after saving the complete solution."""
    output = tmp_path / "reference"
    output.mkdir()
    obstruction = output / "simulation"
    obstruction.write_bytes(b"A regular file cannot be a simulation directory.")
    caught = None
    try:
        _publish(
            driver=owner_driver,
            campaign=tiny_campaign,
            result=complete_result,
            out=output,
        )
    except OSError as error:
        caught = error
    solution_path = output / "solution.h5"
    values = None
    if solution_path.exists():
        loaded = importlib.import_module("lcm.persistence").load_solution(
            path=solution_path,
            verify_checksums=True,
        )
        values = tuple(
            (
                period,
                regime,
                tiny_campaign.fixture.lifetime._leaf_bytes(
                    loaded.values[period][regime],
                ),
            )
            for period, regimes in complete_result.solution.values.items()
            for regime in regimes
        )

    assert (
        type(caught).__name__,
        getattr(caught, "filename", None),
        (output / "receipt.json").exists(),
        values,
        obstruction.read_bytes(),
    ) == (
        "FileExistsError",
        str(obstruction),
        False,
        tiny_campaign.expected[0],
        b"A regular file cannot be a simulation directory.",
    )


@pytest.mark.parametrize("change", ["changed", "missing", "extra"])
def test_reference_comparison_authenticates_every_artifact_before_deserialization(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An immutable receipt requires its entire unchanged archive file population."""
    campaign = tiny_campaign
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if change == "changed":
        payload = output / "simulation/metadata.pkl"
        payload.write_bytes(b"Changed serialized metadata must never be deserialized.")
    elif change == "missing":
        (output / "original_ids.npy").unlink()
    else:
        (output / "unlisted-payload.bin").write_bytes(b"not in the immutable receipt")
    loads = []

    def load(**kwargs: Any) -> object:
        loads.append(kwargs)
        raise RuntimeError("Unverified reference deserialization was attempted")

    results = importlib.import_module("lcm.result")
    persistence = importlib.import_module("lcm.persistence")
    monkeypatch.setattr(results.SimulationResult, "load", load)
    monkeypatch.setattr(persistence, "load_solution", load)
    caught = None
    try:
        owner_driver._compare_reference(
            collected=complete_result,
            original_ids=campaign.original_ids,
            reference=output,
            reference_receipt_sha256=receipt_sha,
        )
    except ValueError as error:
        caught = str(error)

    assert (caught, loads) == (
        "Reference artifact differs: simulation/metadata.pkl"
        if change == "changed"
        else "Reference artifact inventory differs",
        [],
    )


@pytest.mark.parametrize("boundary", ["raw", "panel"])
def test_reference_comparison_refuses_numeric_signed_zero_at_each_population_boundary(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal numeric zeros with different sign bits fail raw and panel equality."""
    campaign = tiny_campaign
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    simulations = []
    zero_leaves = []
    for result, zero in (
        (reference_result, 0.0),
        (complete_result, -0.0 if boundary == "raw" else 0.0),
    ):
        simulation = copy.copy(result.simulation)
        raw = {
            regime: dict(periods) for regime, periods in simulation.raw_results.items()
        }
        data = raw["work"][0]
        leaf = np.asarray(data.V_arr).copy()
        leaf[0] = zero
        raw["work"][0] = dataclasses.replace(data, V_arr=jax.device_put(leaf))
        simulation._raw_results = MappingProxyType(
            {regime: MappingProxyType(periods) for regime, periods in raw.items()}
        )
        simulations.append(simulation)
        zero_leaves.append(leaf)
    reference_result = dataclasses.replace(reference_result, simulation=simulations[0])
    candidate = dataclasses.replace(complete_result, simulation=simulations[1])
    if boundary == "panel":
        frame = candidate.simulation.to_dataframe().copy()
        frame.loc[0, "value"] = -0.0
        reference_zero = simulations[0].to_dataframe()["value"].to_numpy()[0]
        candidate_zero = frame["value"].to_numpy()[0]
        monkeypatch.setattr(
            candidate.simulation, "to_dataframe", lambda **_kwargs: frame
        )
    else:
        reference_zero, candidate_zero = zero_leaves[0][0], zero_leaves[1][0]
    mutation_is_concrete = bool(
        reference_zero == candidate_zero == 0
        and not np.signbit(reference_zero)
        and np.signbit(candidate_zero)
        and np.asarray(reference_zero).dtype == np.asarray(candidate_zero).dtype
        and np.asarray(reference_zero).tobytes() != np.asarray(candidate_zero).tobytes()
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )
    caught = None
    try:
        owner_driver._compare_reference(
            collected=candidate,
            original_ids=campaign.original_ids,
            reference=output,
            reference_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )
    except ValueError as error:
        caught = str(error)

    assert (mutation_is_concrete, caught) == (
        True,
        (
            "Reference comparison differs: "
            f"{'raw_results' if boundary == 'raw' else 'panel'}"
        ),
    )


def test_reference_publication_refuses_output_with_a_previous_failure_receipt(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """A new reference archive cannot relabel a failed output directory as completed."""
    output = tmp_path / "reference"
    output.mkdir()
    failed = output / "receipt.failed.json"
    failed.write_bytes(b'{"phase":"reference","status":"failed"}')
    caught = None
    try:
        _publish(
            driver=owner_driver,
            campaign=tiny_campaign,
            result=complete_result,
            out=output,
        )
    except FileExistsError as error:
        caught = str(error)

    assert (
        caught,
        (output / "receipt.json").exists(),
        (output / "solution.h5").exists(),
        failed.read_bytes(),
        complete_result.simulation.solution is complete_result.solution,
    ) == (
        "Reference output has a previous phase receipt",
        False,
        False,
        b'{"phase":"reference","status":"failed"}',
        True,
    )


def test_reference_comparison_refuses_foreign_seed_with_identical_result_bytes(
    *,
    owner_driver: ModuleType,
    fragment_helpers: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """Identical numeric output cannot authenticate a different planned campaign."""
    campaign = tiny_campaign
    foreign = campaign.api.plan_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=tmp_path / "foreign-plan",
        assignment=((0,), (1,), (2,)),
        initial_conditions=campaign.initial,
        seed=7,
    )
    reference_result = campaign.api.collect_component_jobs(
        model=campaign.model,
        params=campaign.params,
        directory=campaign.directory,
        log_level="off",
    )
    bytes_unchanged = campaign.fixture._complete_result_bytes(
        solution=reference_result.solution,
        simulation=reference_result.simulation,
    ) == campaign.fixture._complete_result_bytes(
        solution=complete_result.solution,
        simulation=complete_result.simulation,
    )
    output = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=reference_result,
        out=output,
    )
    record = json.loads(receipt.read_bytes())
    record["campaign"] = {
        "identity": dict(foreign.identity),
        "state_name": foreign.state_name,
        "codes": list(foreign.codes),
        "jobs": [list(job) for job in foreign.jobs],
        "seed": foreign.seed,
        "n_subjects": foreign.n_subjects,
        "initial_conditions_sha256": foreign.initial_conditions_sha256,
        "job_rows_sha256": list(foreign.job_rows_sha256),
    }
    fragment_helpers.write_json_atomically(path=receipt, payload=record)
    caught = None
    try:
        owner_driver._compare_reference(
            collected=complete_result,
            original_ids=campaign.original_ids,
            reference=output,
            reference_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )
    except ValueError as error:
        caught = str(error)

    assert (bytes_unchanged, foreign.seed, complete_result.plan.seed, caught) == (
        True,
        7,
        20_260_903,
        "Reference campaign differs from the component plan",
    )


def test_independently_constructed_reference_matches_complete_collection(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    "Separate model instances with the same durable campaign preserve all result bytes."
    campaign = tiny_campaign
    reference = campaign.fixture._model().simulate(
        params=campaign.params,
        initial_conditions=campaign.initial,
        seed=20_260_903,
        log_level="off",
    )
    different_uuid = (
        reference.solution.metadata.model_instance_id
        != complete_result.solution.metadata.model_instance_id
    )
    same_bytes = campaign.fixture._complete_result_bytes(
        solution=reference.solution,
        simulation=reference,
    ) == campaign.fixture._complete_result_bytes(
        solution=complete_result.solution,
        simulation=complete_result.simulation,
    )
    receipt = _publish(
        driver=owner_driver,
        campaign=campaign,
        result=CollectedComponentJobs(
            plan=campaign.plan,
            solution=reference.solution,
            simulation=reference,
        ),
        out=tmp_path / "reference",
    )
    caught = None
    try:
        owner_driver._compare_reference(
            collected=complete_result,
            original_ids=campaign.original_ids,
            reference=receipt.parent,
            reference_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )
    except ValueError as error:
        caught = str(error)

    assert (different_uuid, same_bytes, caught) == (True, True, None)


@pytest.mark.parametrize("changed", [False, True])
def test_reference_comparison_binds_full_production_input_contract(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    fragment_helpers: ModuleType,
    tmp_path: Path,
    changed: bool,
) -> None:
    """Equal numerical bytes cannot authorize another immutable input contract."""
    contract = {"inputs": {f"input_{index}": "a" * 64 for index in range(11)}}
    receipt = _publish(
        driver=owner_driver,
        campaign=tiny_campaign,
        result=complete_result,
        out=tmp_path / "reference",
    )
    record = json.loads(receipt.read_bytes())
    record["provenance"]["contract"] = copy.deepcopy(contract)
    if changed:
        record["provenance"]["contract"]["inputs"]["input_0"] = "b" * 64
    fragment_helpers.write_json_atomically(path=receipt, payload=record)
    candidate = tiny_campaign.api.collect_component_jobs(
        model=tiny_campaign.model,
        params=tiny_campaign.params,
        directory=tiny_campaign.directory,
        log_level="off",
    )
    caught = None
    try:
        owner_driver._compare_reference(
            collected=candidate,
            original_ids=tiny_campaign.original_ids,
            reference=receipt.parent,
            reference_receipt_sha256=owner_driver._file_sha256(path=receipt),
            production_contract=contract,
        )
    except (TypeError, ValueError) as error:
        caught = (type(error).__name__, str(error))

    assert caught == (
        ("ValueError", "Reference production input/source contract differs")
        if changed
        else None
    )


def test_publication_records_full_retention_before_consumption_and_archive_wall(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """The complete values-only retention policy and archive wall remain explicit."""
    receipt = _publish(
        driver=owner_driver,
        campaign=tiny_campaign,
        result=complete_result,
        out=tmp_path / "reference",
    )
    record = json.loads(receipt.read_bytes())

    assert (
        record.get("retention"),
        record.get("archive_wall_seconds", 0) > 0,
        record["elapsed_seconds"] > 1.0,
        complete_result.simulation.solution,
    ) == (
        {
            "retained_continuations": [],
            "replay_artifacts": [],
            "auxiliary_artifacts": [],
            "diagnostics": [],
            "omissions": {},
        },
        True,
        True,
        None,
    )


def test_publication_refuses_unpersisted_target_channels_before_consuming_simulation(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
) -> None:
    """The values-only target refuses nonempty retention channels before saving."""

    changed = dataclasses.replace(
        complete_result.solution,
        omissions={
            ArtifactRef(
                period=0,
                regime="work",
                key=ArtifactKey(type_id="fixture.omitted", schema_version=1),
            ): OmissionReason.NOT_PERSISTED,
        },
    )
    simulation = complete_result.simulation
    out = tmp_path / "unsupported-target"
    caught = None
    try:
        owner_driver._publish_reference(
            plan=complete_result.plan,
            solution=changed,
            simulation=simulation,
            original_ids=tiny_campaign.original_ids,
            out=out,
            provenance={"scope": "tiny-cpu"},
            started_at="start",
            finished_at="end",
            elapsed_seconds=0.0,
        )
    except (ValueError, IncompatibleSolutionError) as error:
        caught = str(error)

    assert (caught, simulation.solution is complete_result.solution, out.exists()) == (
        "Production archive requires the complete values-only GridSearch target",
        True,
        False,
    )


@pytest.mark.parametrize("boundary", ["publication", "comparison"])
def test_full_archive_refuses_missing_solution_or_simulation_before_io(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    tmp_path: Path,
    boundary: str,
) -> None:
    """Full-production archives require both complete public result containers."""
    caught = None
    out = tmp_path / "full-results"
    if boundary == "comparison":
        receipt = _publish(
            driver=owner_driver, campaign=tiny_campaign, result=complete_result, out=out
        )
        missing = dataclasses.replace(complete_result, simulation=None)

        def invoke() -> object:
            return owner_driver._compare_reference(
                collected=missing,
                original_ids=tiny_campaign.original_ids,
                reference=out,
                reference_receipt_sha256=owner_driver._file_sha256(path=receipt),
            )
    else:

        def invoke() -> object:
            return owner_driver._publish_reference(
                plan=complete_result.plan,
                solution=None,
                simulation=complete_result.simulation,
                original_ids=tiny_campaign.original_ids,
                out=out,
                provenance={},
                started_at="start",
                finished_at="end",
                elapsed_seconds=0.0,
            )

    try:
        invoke()
    except (ValueError, AttributeError) as error:
        caught = (type(error).__name__, str(error))

    assert (caught, out.exists()) == (
        ("ValueError", "Full production results require a solution and simulation"),
        boundary == "comparison",
    )


def test_reference_comparison_restores_both_external_id_sequences_with_owner_helper(
    *,
    owner_driver: ModuleType,
    tiny_campaign: SimpleNamespace,
    complete_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both complete positional panels restore the same original external IDs."""

    owner = sys.modules["aca_model.simulation"]
    restore = owner.restore_subject_ids
    expected_ids = [90, 10, 30, 70, 50, 20, 80, 40, 60, 100, 110]
    observed = []

    def restore_and_record(*, panel: Any, original_ids: Any) -> Any:
        restored = restore(panel=panel, original_ids=original_ids)
        observed.append(
            (
                restored["id"].tolist(),
                restored["id"].dtype.str,
                [expected_ids[int(position)] for position in panel["subject_id"]],
            )
        )
        return restored

    monkeypatch.setattr(owner, "restore_subject_ids", restore_and_record)
    out = tmp_path / "reference"
    receipt = _publish(
        driver=owner_driver, campaign=tiny_campaign, result=complete_result, out=out
    )
    candidate = tiny_campaign.api.collect_component_jobs(
        model=tiny_campaign.model,
        params=tiny_campaign.params,
        directory=tiny_campaign.directory,
        log_level="off",
    )
    owner_driver._compare_reference(
        collected=candidate,
        original_ids=tiny_campaign.original_ids,
        reference=out,
        reference_receipt_sha256=owner_driver._file_sha256(path=receipt),
    )

    assert (
        len(observed),
        [ids == literal and dtype == "<i8" for ids, dtype, literal in observed],
    ) == (2, [True, True])
