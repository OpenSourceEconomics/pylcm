"""Qualification replay uses the real evaluator without granting an admission."""

import json
from importlib.util import find_spec
from importlib import import_module
from pathlib import Path
from typing import cast

import pytest

import _lcm_regime_finder
from _lcm_regime_finder.fibre_source import _mapping
from _lcm_regime_finder.fibre_source import Wire
from _lcm_regime_finder.implicit.check_projection import _project
from _lcm_regime_finder.implicit.implicit_native import _produce
from _lcm_regime_finder.implicit.source_versions import (
    SourceVersion,
    assert_runtime_source,
    resolve_source_candidate,
    sha,
)
from _lcm_regime_finder.implicit.verify_native import _replay
from tests.regime_finder.test_fibre_source import _original


def installed_candidate() -> SourceVersion:
    """Select a known descriptor by exact bytes, without a production admission."""
    root = Path(_lcm_regime_finder.__file__).resolve().parent
    actual = {
        path.name: sha(path)
        for path in root.iterdir()
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    matches = [
        resolve_source_candidate(source_version=name)
        for name in (
            "source587",
            "source645",
            "source639-builtin-int-v1",
            "source645-builtin-int-v1",
        )
        if resolve_source_candidate(source_version=name).source_files == actual
    ]
    assert len(matches) == 1, "The installed evaluator needs an explicit descriptor"
    version = matches[0]
    assert_runtime_source(package=root, version=version).check()
    return version


def candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path, SourceVersion]:
    """Use an original tiny fixture without editing its declaration dialect."""
    version = installed_candidate()
    snapshot, axes, artifact = [
        tmp_path / name for name in ("source.json", "axes.json", "facts.json")
    ]
    snapshot.write_text(json.dumps(_original()))
    axes.write_text(json.dumps({"original": ["x", "history"]}))
    artifact.write_text(
        json.dumps(_produce(snapshot=snapshot, axes_file=axes, version=version))
    )
    return snapshot, axes, artifact, version


@pytest.mark.parametrize("removed", [frozenset(), frozenset({("original", "history")})])
def test_candidate_transaction_has_only_neutral_evidence(
    tmp_path: Path,
    removed: frozenset[tuple[str, str]],
) -> None:
    """Candidate replay and finite checks cannot issue production receipts."""
    snapshot, axes, artifact, version = candidate_inputs(tmp_path)
    packet = json.loads(artifact.read_text())
    replay = _replay(
        snapshot=snapshot, axes_file=axes, artifact=artifact, version=version
    )
    projection = _project(
        snapshot=snapshot,
        axes_file=axes,
        artifact=artifact,
        removed=removed,
        version=version,
    )
    assert (packet["status"], replay["status"], projection["status"]) == (
        "PRODUCED_UNVERIFIED",
        "FACT_REPLAY_COMPLETE",
        "PROJECTION_REPLAY_COMPLETE",
    )
    assert _mapping(projection["screen"])["conditions_pass"] is True


@pytest.mark.parametrize(
    "field",
    [
        "source_version",
        "reference_program_id",
        "source_map_digest",
        "semantic_descriptor_sha256",
    ],
)
def test_candidate_replay_rejects_mismatched_version_identity(
    tmp_path: Path,
    field: str,
) -> None:
    """An untrusted packet cannot choose its own evaluator identity."""
    snapshot, axes, artifact, version = candidate_inputs(tmp_path)
    record = json.loads(artifact.read_text())
    record[field] = "foreign"
    artifact.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="caller-selected evaluator version differ"):
        _replay(snapshot=snapshot, axes_file=axes, artifact=artifact, version=version)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_versioned_bridge_checks_complete_root_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A version label cannot replace the original complete fact inventory."""
    snapshot, axes, artifact, version = candidate_inputs(tmp_path)
    record = json.loads(artifact.read_text())
    roots = record["cells"][0]["diagram"]["roots"]
    if mutation == "missing":
        roots.pop("direct")
    else:
        roots["invented"] = roots["direct"]
    artifact.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Missing or invented fact roots"):
        _project(
            snapshot=snapshot,
            axes_file=axes,
            artifact=artifact,
            removed=frozenset(),
            version=version,
        )


def test_qualification_controls_are_fixed_by_the_implementation() -> None:
    """Selecting metadata inputs cannot remove mandatory qualification controls."""
    name = "tests.regime_finder.implicit_version_qualification"
    assert find_spec(name) is not None, "Missing guarded qualification orchestrator"
    helper = import_module(name)
    assert set(helper.CONTROL_TARGETS) == set(
        import_module("_lcm_regime_finder.implicit.source_versions").MANDATORY_CONTROLS
    )


def test_cases_cannot_replace_the_control_manifest(tmp_path: Path) -> None:
    """A caller can select metadata cases but cannot omit required tests."""
    helper = import_module("tests.regime_finder.implicit_version_qualification")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            {"schema": "implicit-qualification-cases/v1", "cases": [], "controls": []}
        )
    )
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="not qualification controls"):
        helper.qualify_source_version(
            source_version=installed_candidate().version_id,
            cases_file=cases,
            evidence_directory=output,
        )
    assert not output.exists()


def native_capture() -> Wire:
    """Capture the tested conditional-process model using its actual native types."""
    from lcm import DiscreteGrid, InitialNodes, Model, Periods, Regime
    from _lcm_regime_finder.model_snapshot import read_model
    from tests.test_graphs import (
        GraphRegimes,
        ProcessConditionHistory,
        _carry_process_history,
        _conditional_persistent_process,
        _conditional_process_utility,
    )

    process = _conditional_persistent_process(conditioned=True)
    model = Model(
        n_periods=2,
        regime_id_class=GraphRegimes,
        regimes={
            "alive": Regime(
                states={"z": process, "history": DiscreteGrid(ProcessConditionHistory)},
                functions={"utility": _conditional_process_utility},
                state_transitions={"history": _carry_process_history},
            ),
            "final": Regime(
                states={"z": process, "history": DiscreteGrid(ProcessConditionHistory)},
                functions={"utility": _conditional_process_utility},
            ),
        },
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"final": Periods(values=(0,))}},
        fixed_params={"discount_factor": 1.0},
        enable_jit=False,
    )
    return cast(Wire, read_model(model))


def test_genuine_native_capture_replays_without_relabelling(tmp_path: Path) -> None:
    """A new capture uses the actual native types and keeps every qualification."""
    version = installed_candidate()
    snapshot, axes, artifact = [
        tmp_path / name for name in ("native.json", "axes.json", "facts.json")
    ]
    original = native_capture()
    snapshot.write_text(json.dumps(original))
    axes.write_text(json.dumps({"alive": ["history"], "final": ["history"]}))
    packet = _produce(snapshot=snapshot, axes_file=axes, version=version)
    artifact.write_text(json.dumps(packet))
    result = _project(
        snapshot=snapshot,
        axes_file=axes,
        artifact=artifact,
        removed=frozenset(),
        version=version,
    )
    assert result["status"] == "PROJECTION_REPLAY_COMPLETE"
    assert result["source_obligations"] == packet["source_obligations"]
    assert snapshot.read_text() == json.dumps(original)
