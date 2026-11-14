"""Only qualified, byte-matched evaluator versions can issue fact receipts."""

from dataclasses import fields
from importlib import import_module
from importlib.util import find_spec
import json
from pathlib import Path
from types import ModuleType
from dataclasses import replace

import pytest

from _lcm_regime_finder.implicit.implicit_native import compile_table

NEW_SOURCE_IDS = ("source639-builtin-int-v1",)


def registry() -> ModuleType:
    """Load the installed source-version boundary."""
    name = "_lcm_regime_finder.implicit.source_versions"
    assert find_spec(name) is not None, "Missing explicit source-version boundary"
    return import_module(name)


def test_semantic_descriptor_is_separate_from_admission() -> None:
    """Qualification metadata cannot enter its own semantic identity."""
    version = registry().resolve_source_candidate(source_version="source587")
    assert tuple(field.name for field in fields(version)) == (
        "version_id",
        "source_commit",
        "point_table_protocol_id",
        "reference_program_id",
        "source_files",
        "source_map_digest",
        "capture_dialect_id",
        "binding_purity_contract_id",
    )


@pytest.mark.parametrize("source_version", ["", "../source587", "unregistered"])
def test_unknown_version_is_rejected_before_input_loading(
    tmp_path: Path,
    source_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input bytes cannot select an evaluator or disable authentication."""

    def forbidden_load(path: Path) -> None:
        del path
        pytest.fail("An unknown evaluator must fail before loading inputs")

    monkeypatch.setattr("_lcm_regime_finder.cli.load_problem", forbidden_load)
    with pytest.raises(ValueError, match="(?:Unknown|unknown) source version"):
        compile_table(
            snapshot=tmp_path / "not-loaded.json",
            axes_file=tmp_path / "not-loaded-axes.json",
            source_version=source_version,
        )


def test_candidate_lookup_does_not_admit_ordinary_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known descriptor without a qualification receipt stays unqualified."""
    module = registry()
    admissions = tmp_path / "admissions.json"
    admissions.write_text(
        json.dumps({"schema": "implicit-source-admissions/v1", "admissions": []})
    )
    monkeypatch.setattr(module, "_ADMISSIONS", admissions)
    candidate = module.resolve_source_candidate(source_version="source645")
    assert candidate.version_id == "source645"
    with pytest.raises(ValueError, match="[Uu]nqualified source version"):
        module.resolve_source_version(source_version="source645")


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_complete_source_inventory_is_checked(mutation: str) -> None:
    """An admission cannot replace the exact installed evaluator inventory."""
    module = registry()
    version = module.resolve_source_candidate(source_version="source587")
    files = dict(version.source_files)
    if mutation == "changed":
        files["fibre_source.py"] = "0" * 64
    elif mutation == "missing":
        files.pop("fibre_source.py")
    else:
        files["invented.py"] = "0" * 64
    filename = module.__file__
    assert filename is not None
    with pytest.raises(ValueError, match="[Aa]uthoritative source bytes"):
        module.assert_runtime_source(
            package=Path(filename).resolve().parents[1],
            version=replace(version, source_files=files),
        )


def test_historical_packets_are_not_relabelled() -> None:
    """A matching fact leaf does not change a packet's evaluator identity."""
    module = registry()
    version = module.resolve_source_candidate(source_version="source645")
    with pytest.raises(ValueError, match="sealed matching installation"):
        module.check_packet_identity(
            record={"schema": "exact-implicit-fibre-table/v1"}, version=version
        )


@pytest.mark.parametrize(
    "source_version, incompatible",
    [
        ("source587", "lcm.processes.UniformIIDProcess"),
        ("source645", "_lcm.processes.iid.UniformIIDProcess"),
        ("source645", "_lcm.regime_building.schedules._Constant"),
    ],
)
def test_foreign_native_capture_spellings_are_not_rewritten(
    source_version: str,
    incompatible: str,
) -> None:
    """Native compatibility requires its own unchanged declaration dialect."""
    module = registry()
    version = module.resolve_source_candidate(source_version=source_version)
    record = {
        "schema": "constructed-pylcm-model-graph-1",
        "native": {"type": incompatible},
    }
    before = json.dumps(record)
    with pytest.raises(ValueError, match="Capture dialect is incompatible"):
        module.check_capture_admission(original=record, version=version)
    assert json.dumps(record) == before


def test_module_origins_cannot_be_selected_by_hashing_another_directory(
    tmp_path: Path,
) -> None:
    """An evaluator descriptor cannot swap the functions already imported."""
    module = registry()
    version = module.resolve_source_candidate(source_version="source587")
    with pytest.raises(ValueError, match="Mixed evaluator module origins"):
        module.assert_runtime_source(package=tmp_path, version=version)


@pytest.mark.parametrize(
    "changed",
    ["source", "removed", "extra", "descriptor", "admissions", "receipt", "registry"],
)
def test_final_guard_rechecks_inventory_and_every_authority_file(
    tmp_path: Path,
    changed: str,
) -> None:
    """No changed code, descriptor or admission can survive a final transaction guard."""
    module = registry()
    package = tmp_path / "evaluator"
    package.mkdir()
    source = package / "policy.py"
    source.write_text("VALUE = 1\n")
    authorities = {
        name: tmp_path / f"{name}.json"
        for name in ("descriptor", "admissions", "receipt", "registry")
    }
    for path in authorities.values():
        path.write_text("{}\n")
    guards = {path: module.sha(path) for path in (source, *authorities.values())}
    identity = module.SourceIdentity(
        package=package,
        source_files={"policy.py": module.sha(source)},
        algorithm_sources=module.algorithm_sources(),
        guarded_sha256=guards,
    )
    identity.check()
    if changed == "source":
        source.write_text("VALUE = 2\n")
    elif changed == "removed":
        source.unlink()
    elif changed == "extra":
        (package / "invented.py").write_text("VALUE = 1\n")
    else:
        authorities[changed].write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="Source/version transaction changed"):
        identity.check()


def test_claimed_admission_requires_its_exact_reviewed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admission flag without the separately pinned receipt is insufficient."""
    module = registry()
    version = module.resolve_source_candidate(source_version="source587")
    table = tmp_path / "admissions.json"
    table.write_text(
        json.dumps(
            {
                "schema": "implicit-source-admissions/v1",
                "admissions": [
                    {
                        "semantic_descriptor_sha256": module.descriptor_digest(version),
                        "algorithm_map_sha256": module.canonical_digest(
                            module.algorithm_sources()
                        ),
                        "qualification_status": "QUALIFIED",
                        "qualification_receipt_sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(module, "_ADMISSIONS", table)
    with pytest.raises(ValueError, match="Qualification receipt changed or is missing"):
        module.resolve_source_version(source_version="source587")


@pytest.mark.parametrize("source_version", NEW_SOURCE_IDS)
def test_new_semantic_candidates_need_their_own_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_version: str
) -> None:
    """Bundling exact new semantics does not qualify them for ordinary use."""
    module = registry()
    table = tmp_path / "unqualified-admissions.json"
    table.write_text(
        json.dumps({"schema": "implicit-source-admissions/v1", "admissions": []})
    )
    monkeypatch.setattr(module, "_ADMISSIONS", table)
    version = module.resolve_source_candidate(source_version=source_version)
    assert version.version_id == source_version
    assert len(version.source_commit) == 40
    assert set(version.source_commit) <= set("0123456789abcdef")
    assert (
        version.reference_program_id
        == f"FibreSource-point-table@{version.source_commit}"
    )
    assert version.source_map_digest == module.canonical_digest(
        dict(version.source_files)
    )
    with pytest.raises(ValueError, match="[Uu]nqualified source version"):
        module.resolve_source_version(source_version=source_version)


@pytest.mark.parametrize("source_version", NEW_SOURCE_IDS)
@pytest.mark.parametrize(
    "field",
    [
        "source_version",
        "point_table_protocol_id",
        "reference_program_id",
        "semantic_descriptor_sha256",
        "source_map_digest",
        "capture_dialect_id",
    ],
)
def test_new_candidates_reject_foreign_packet_identity(
    source_version: str, field: str
) -> None:
    module = registry()
    version = module.resolve_source_candidate(source_version=source_version)
    record = {
        "schema": "exact-implicit-fibre-table/v2",
        **module.packet_identity(version),
    }
    record[field] = "foreign"
    with pytest.raises(ValueError, match="caller-selected evaluator version differ"):
        module.check_packet_identity(record=record, version=version)


@pytest.mark.parametrize("source_version", NEW_SOURCE_IDS)
@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_new_candidates_still_require_the_complete_source_map(
    source_version: str, mutation: str
) -> None:
    module = registry()
    filename = module.__file__
    assert filename is not None
    version = module.resolve_source_candidate(source_version=source_version)
    inventory = dict(version.source_files)
    if mutation == "changed":
        inventory["fibre_source.py"] = "0" * 64
    elif mutation == "missing":
        inventory.pop("fibre_source.py")
    else:
        inventory["invented.py"] = "0" * 64
    with pytest.raises(ValueError, match="[Aa]uthoritative source bytes"):
        module.assert_runtime_source(
            package=Path(filename).resolve().parents[1],
            version=replace(version, source_files=inventory),
        )


@pytest.mark.parametrize(
    "dialect, rejected, accepted",
    [
        (
            "constructed-graph/native-private-processes-v1",
            "lcm.processes.UniformIIDProcess",
            "_lcm.processes.iid.UniformIIDProcess",
        ),
        (
            "constructed-graph/native-public-processes-v1",
            "_lcm.processes.iid.UniformIIDProcess",
            "lcm.processes.UniformIIDProcess",
        ),
    ],
)
def test_capture_policy_uses_the_trusted_dialect_not_the_version_name(
    dialect: str, rejected: str, accepted: str
) -> None:
    module = registry()
    version = replace(
        module.resolve_source_candidate(source_version="source587"),
        version_id="renamed-candidate",
        capture_dialect_id=dialect,
    )
    original = {
        "schema": "constructed-pylcm-model-graph-1",
        "native": {"type": accepted},
    }
    before = json.dumps(original)
    assert accepted in module.check_capture_admission(
        original=original, version=version
    )
    assert json.dumps(original) == before
    foreign = {
        "schema": "constructed-pylcm-model-graph-1",
        "native": {"type": rejected},
    }
    with pytest.raises(ValueError, match="Capture dialect is incompatible"):
        module.check_capture_admission(original=foreign, version=version)


def test_unknown_capture_dialect_is_rejected() -> None:
    module = registry()
    version = replace(
        module.resolve_source_candidate(source_version="source587"),
        capture_dialect_id="unsupported-dialect",
    )
    with pytest.raises(ValueError, match="Unknown capture dialect"):
        module.check_capture_admission(
            original={"schema": "constructed-pylcm-model-graph-1"}, version=version
        )


@pytest.mark.parametrize("source_version", NEW_SOURCE_IDS)
@pytest.mark.parametrize("authority", ["descriptor", "admissions", "registry"])
def test_new_candidate_final_guard_retains_its_authority_files(
    source_version: str, authority: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model a changed authority without modifying any installed frozen file."""
    module = registry()
    filename = module.__file__
    assert filename is not None
    version = module.resolve_source_candidate(source_version=source_version)
    package = Path(filename).resolve().parents[1]
    actual = {
        path.name: module.sha(path)
        for path in package.iterdir()
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    if actual != dict(version.source_files):
        # Both variants can be bundled, but only the actual installed root can
        # pass its positive final guard. A foreign candidate must fail closed.
        with pytest.raises(ValueError, match="[Aa]uthoritative source bytes"):
            module.assert_runtime_source(package=package, version=version)
        return
    identity = module.assert_runtime_source(package=package, version=version)
    selected = {
        "descriptor": module._VERSIONS / f"{source_version}.json",
        "admissions": module._ADMISSIONS,
        "registry": Path(filename).resolve(),
    }[authority]
    assert selected in identity.guarded_sha256
    original_sha = module.sha
    monkeypatch.setattr(
        module, "sha", lambda path: "0" * 64 if path == selected else original_sha(path)
    )
    with pytest.raises(ValueError, match="Source/version transaction changed"):
        identity.check()


def test_qualification_includes_the_predicate_capture_regressions() -> None:
    helper = import_module("tests.regime_finder.implicit_version_qualification")
    assert (
        "tests/regime_finder/test_round5_bindings.py"
        in helper.CONTROL_TARGETS["native-namespace-and-immutable-capture"]
    )
