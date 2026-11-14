"""Trusted evaluator descriptors and independent qualification admissions.

A descriptor identifies semantics. An admission binds those semantics to the
installed algorithms and a separately reviewed receipt. Neither record hashes
itself, and a candidate descriptor cannot authorize an ordinary fact receipt.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
from types import MappingProxyType

from ..fibre_source import Wire, _mapping, _rows, _text
from ..records import JSONValue

_HERE = Path(__file__).resolve().parent
_VERSIONS = _HERE / "source_versions"
_ADMISSIONS = _VERSIONS / "admissions.json"
_IDS = frozenset(
    {"source587", "source645", "source639-builtin-int-v1", "source645-builtin-int-v1"}
)
_PROTOCOL = "original-cell-point-query-order/v1"
type SourceFile = str
MANDATORY_CONTROLS = (
    "trusted-version-boundary",
    "typed-terminals-and-differential-facts",
    "support-closure-and-finite-premises",
    "cache-order-and-binding-depth",
    "native-namespace-and-immutable-capture",
    "guarded-candidate-transaction",
)


def sha(path: Path) -> str:
    """Hash exact file bytes, including their line endings."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: JSONValue) -> str:
    """Identify a semantic JSON value independently of document formatting."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SourceVersion:
    """A complete immutable evaluator descriptor, without admission metadata."""

    version_id: str
    """Bundled identifier selected explicitly by the caller."""
    source_commit: str
    """Revision whose evaluator source map is authenticated."""
    point_table_protocol_id: str
    """Original cell, point and query-order protocol."""
    reference_program_id: str
    """Selected evaluator program, distinct from the query protocol."""
    source_files: Mapping[SourceFile, str]
    """Exact complete root source and normative JSON inventory."""
    source_map_digest: str
    """Canonical digest of the selected source inventory."""
    capture_dialect_id: str
    """Supported native declaration spelling contract."""
    binding_purity_contract_id: str
    """Purity proof or original shared-cache ordering contract."""


@dataclass(frozen=True)
class SourceIdentity:
    """Installed evaluator, algorithm and transaction dependencies."""

    package: Path
    """Actual imported evaluator package."""
    source_files: Mapping[SourceFile, str]
    """Authenticated evaluator inventory."""
    algorithm_sources: Mapping[str, str]
    """Current complete implicit-fact implementation inventory."""
    guarded_sha256: Mapping[Path, str]
    """Descriptor, admission and installed source bytes checked at completion."""

    def check(self) -> None:
        """Reject any changed or removed dependency."""
        inventory = {
            path.name: sha(path)
            for path in self.package.iterdir()
            if path.is_file() and path.suffix in {".py", ".json"}
        }
        if (
            inventory != self.source_files
            or algorithm_sources() != self.algorithm_sources
            or any(
                not path.is_file() or sha(path) != digest
                for path, digest in self.guarded_sha256.items()
            )
        ):
            raise ValueError("Source/version transaction changed")


def semantic_record(version: SourceVersion) -> Wire:
    """Serialize only the evaluator's semantic fields."""
    return {
        "version_id": version.version_id,
        "source_commit": version.source_commit,
        "point_table_protocol_id": version.point_table_protocol_id,
        "reference_program_id": version.reference_program_id,
        "source_files": dict(version.source_files),
        "source_map_digest": version.source_map_digest,
        "capture_dialect_id": version.capture_dialect_id,
        "binding_purity_contract_id": version.binding_purity_contract_id,
    }


def descriptor_digest(version: SourceVersion) -> str:
    """Identify semantics without a receipt or admission-table dependency."""
    return canonical_digest(semantic_record(version))


def algorithm_sources() -> dict[SourceFile, str]:
    """Bind compiler, independent verifier, bridge, screen and registry code."""
    return {
        str(path.relative_to(_HERE)): sha(path) for path in sorted(_HERE.rglob("*.py"))
    }


def resolve_source_candidate(*, source_version: str) -> SourceVersion:
    """Resolve only a bundled descriptor; this does not grant an admission."""
    if source_version not in _IDS:
        raise ValueError(f"Unknown source version: {source_version!r}")
    record = _mapping(json.loads((_VERSIONS / f"{source_version}.json").read_text()))
    names = tuple(SourceVersion.__dataclass_fields__)
    if set(record) != set(names) or record["version_id"] != source_version:
        raise ValueError("Malformed source-version descriptor")
    files = {
        name: _text(value) for name, value in _mapping(record["source_files"]).items()
    }
    if not files or any(
        Path(name).name != name or Path(name).suffix not in {".py", ".json"}
        for name in files
    ):
        raise ValueError("Source inventory must contain named root source files")
    if canonical_digest(files) != record["source_map_digest"]:
        raise ValueError("Source-map digest differs from its complete inventory")
    if record["point_table_protocol_id"] != _PROTOCOL:
        raise ValueError("Unknown point-table protocol")
    return SourceVersion(
        version_id=source_version,
        source_commit=_text(record["source_commit"]),
        point_table_protocol_id=_text(record["point_table_protocol_id"]),
        reference_program_id=_text(record["reference_program_id"]),
        source_files=MappingProxyType(files),
        source_map_digest=_text(record["source_map_digest"]),
        capture_dialect_id=_text(record["capture_dialect_id"]),
        binding_purity_contract_id=_text(record["binding_purity_contract_id"]),
    )


def _receipt_path(version: SourceVersion) -> Path:
    table = _mapping(json.loads(_ADMISSIONS.read_text()))
    if table.get("schema") != "implicit-source-admissions/v1":
        raise ValueError("Unknown source-admission table")
    descriptor = descriptor_digest(version)
    algorithms = algorithm_sources()
    algorithm_digest = canonical_digest(algorithms)
    matches = [
        row
        for row in _rows(table["admissions"])
        if row.get("semantic_descriptor_sha256") == descriptor
        and row.get("algorithm_map_sha256") == algorithm_digest
    ]
    if len(matches) != 1 or matches[0].get("qualification_status") != "QUALIFIED":
        raise ValueError(f"Unqualified source version: {version.version_id}")
    admission = matches[0]
    if set(admission) != {
        "semantic_descriptor_sha256",
        "algorithm_map_sha256",
        "qualification_status",
        "qualification_receipt_sha256",
    }:
        raise ValueError("Malformed source admission")
    digest = _text(admission["qualification_receipt_sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid qualification receipt digest")
    path = _VERSIONS / "receipts" / f"{digest}.json"
    if not path.is_file() or sha(path) != digest:
        raise ValueError("Qualification receipt changed or is missing")
    receipt = _mapping(json.loads(path.read_text()))
    if (
        receipt.get("schema") != "implicit-source-qualification/v1"
        or receipt.get("status") != "QUALIFICATION_COMPLETE"
        or receipt.get("semantic_descriptor_sha256") != descriptor
        or receipt.get("source_files") != dict(version.source_files)
        or receipt.get("algorithm_sources") != algorithms
    ):
        raise ValueError("Qualification receipt does not bind this implementation")
    controls = _mapping(receipt.get("control_results", {}))
    if set(controls) != set(MANDATORY_CONTROLS) or any(
        _mapping(value).get("status") != "PASS" for value in controls.values()
    ):
        raise ValueError("Mandatory source qualification controls are incomplete")
    inputs = _rows(receipt.get("input_results", []))
    if not inputs or any(
        row.get("status") != "QUALIFICATION_REPLAY_COMPLETE" for row in inputs
    ):
        raise ValueError("Source qualification has no completed input replay")
    return path


def resolve_source_version(*, source_version: str) -> SourceVersion:
    """Require a reviewed admission for the currently installed algorithm bytes."""
    version = resolve_source_candidate(source_version=source_version)
    _receipt_path(version)
    return version


def assert_runtime_source(*, package: Path, version: SourceVersion) -> SourceIdentity:
    """Bind the exact complete evaluator map and actual imported module origins."""
    package = package.resolve()
    if package != _HERE.parent:
        raise ValueError("Mixed evaluator module origins")
    files = {
        path.name: sha(path)
        for path in sorted(package.iterdir())
        if path.is_file() and path.suffix in {".py", ".json"}
    }
    if files != version.source_files:
        raise ValueError("Authoritative source bytes or complete inventory changed")
    for name, callable_name in (
        ("fibre_source", "FibreSource"),
        ("semantic_check", "SourceProof"),
        ("cli", "load_problem"),
        ("structural_store", "final_input_check"),
    ):
        module = import_module(f"_lcm_regime_finder.{name}")
        expected = package / f"{name}.py"
        filename = module.__file__
        if (
            filename is None
            or Path(filename).resolve() != expected
            or Path(inspect.getfile(getattr(module, callable_name))).resolve()
            != expected
        ):
            raise ValueError("Mixed evaluator module origins")
    algorithms = algorithm_sources()
    guarded = {package / name: digest for name, digest in files.items()}
    guarded.update({_HERE / name: digest for name, digest in algorithms.items()})
    descriptor = _VERSIONS / f"{version.version_id}.json"
    guarded[descriptor] = sha(descriptor)
    guarded[_ADMISSIONS] = sha(_ADMISSIONS)
    try:
        receipt = _receipt_path(version)
    except ValueError as error:
        if not str(error).startswith("Unqualified source version:"):
            raise
    else:
        guarded[receipt] = sha(receipt)
    return SourceIdentity(
        package,
        MappingProxyType(files),
        MappingProxyType(algorithms),
        MappingProxyType(guarded),
    )


def check_capture_admission(
    *, original: Wire, version: SourceVersion
) -> tuple[str, ...]:
    """Keep captured declarations unchanged and reject incompatible native dialects."""
    if original.get("schema") != "constructed-pylcm-model-graph-1":
        raise ValueError("Expected an original constructed Model capture")
    types: set[str] = set()

    def visit(value: JSONValue) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("type"), str):
                types.add(_text(value["type"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(original)
    old = {"_lcm.regime_building.schedules._Constant"}
    new = {
        "_lcm.regime_building.schedules._RegimeCodeConstant",
        "_lcm.regime_building.schedules._ProbabilityConstant",
        "lcm.state_conditioned.StateConditioned",
    }
    if version.capture_dialect_id == "constructed-graph/native-public-processes-v1":
        rejected = {
            name
            for name in types
            if name in old
            or name.startswith(("_lcm.processes.iid.", "_lcm.processes.ar1."))
        }
    elif version.capture_dialect_id == "constructed-graph/native-private-processes-v1":
        rejected = {
            name
            for name in types
            if name in new
            or name.startswith(("lcm.processes.", "_lcm.processes.ar1_on_nodes."))
        }
    else:
        raise ValueError(f"Unknown capture dialect: {version.capture_dialect_id!r}")
    if rejected:
        raise ValueError(
            f"Capture dialect is incompatible with {version.version_id}: {sorted(rejected)}"
        )
    return tuple(sorted(types))


def packet_identity(version: SourceVersion) -> Wire:
    """Versioned packet identity independently reconstructible by every consumer."""
    return {
        "source_version": version.version_id,
        "point_table_protocol_id": version.point_table_protocol_id,
        "reference_program_id": version.reference_program_id,
        "semantic_descriptor_sha256": descriptor_digest(version),
        "source_map_digest": version.source_map_digest,
        "capture_dialect_id": version.capture_dialect_id,
    }


def check_packet_identity(*, record: Wire, version: SourceVersion) -> None:
    """Never infer an evaluator from producer-supplied labels or source maps."""
    if record.get("schema") != "exact-implicit-fibre-table/v2":
        raise ValueError(
            "Use the sealed matching installation for historical v1 packets"
        )
    if any(
        record.get(name) != value for name, value in packet_identity(version).items()
    ):
        raise ValueError("Fact packet and caller-selected evaluator version differ")
