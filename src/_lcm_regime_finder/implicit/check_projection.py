"""Check one supplied joint projection, with explicit optional box expansion.

Run after implicit_native has produced a canonical fact packet. Full original
fact replay runs before expansion. Box refinement can be exponential; this is
not the default fact representation or a replacement for witness-order search.
"""

import argparse
import json
from pathlib import Path
from typing import cast

from ..cli import load_problem
from ..fibre_source import FibreSource, Wire, _mapping, _strings
from ..records import JSONValue
from ..structural_store import final_input_check
from .compact_screen import check_screen, least_liveness, possible_closure
from .factored_bridge import as_box_package, build_factored_table
from .verify_native import _replay, sha
from .source_versions import (
    SourceVersion,
    assert_runtime_source,
    check_capture_admission,
    packet_identity,
    resolve_source_version,
)


def _project(
    *,
    snapshot: Path,
    axes_file: Path,
    artifact: Path,
    removed: frozenset[tuple[str, str]],
    version: SourceVersion,
) -> Wire:
    """Verify original facts, then check a proposal using the explicit box adapter.

    Source qualifications remain binding. The result is a finite supplied-fact
    check, not native adoption, economic equivalence or a best-graph certificate.
    No result is returned after an interrupted replay or a changed input.
    """
    import _lcm_regime_finder

    here = Path(__file__).parent
    identity = assert_runtime_source(
        package=Path(_lcm_regime_finder.__file__).parent, version=version
    )
    guarded = {path: sha(path) for path in (snapshot, axes_file, artifact)}
    guarded.update({path: sha(path) for path in here.glob("*.py")})
    guarded[here / "reference_sources.json"] = sha(here / "reference_sources.json")
    fact_receipt = _replay(
        snapshot=snapshot, axes_file=axes_file, artifact=artifact, version=version
    )
    original = load_problem(snapshot)
    check_capture_admission(original=original, version=version)
    source = FibreSource(original)
    record = _mapping(json.loads(artifact.read_text()))
    axes = {
        name: _strings(value)
        for name, value in _mapping(json.loads(axes_file.read_text())).items()
    }
    table = build_factored_table(
        source=source, record=record, status_axes=axes, version=version
    )
    package = as_box_package(table)
    kept = possible_closure(package)
    live = least_liveness(package)
    screen = check_screen(package, removed)
    final_input_check(original)
    if any(sha(path) != digest for path, digest in guarded.items()):
        raise ValueError("Source/certificate transaction changed")
    # The original analyzer remains pinned independently of the new package.
    import _lcm_regime_finder

    root = Path(_lcm_regime_finder.__file__).parent
    if any(
        sha(root / name) != digest
        for name, digest in _mapping(fact_receipt["reference_sources"]).items()
    ):
        raise ValueError("Authoritative source changed after fact replay")
    identity.check()
    return {
        "schema": "source-bound-implicit-projection-evidence/v2",
        **packet_identity(version),
        "status": "PROJECTION_REPLAY_COMPLETE",
        "removed": [list(axis) for axis in sorted(removed)],
        "forced_actions": [],
        "forced_action_savings": "not measured",
        "fact_verification": fact_receipt,
        "screen": cast(JSONValue, screen),
        "kept_counts": {cell: region.cardinality for cell, region in kept.items()},
        "live_counts": [
            {"cell": cell, "coordinate": coordinate, "count": region.cardinality}
            for (cell, coordinate), region in sorted(live.items())
        ],
        "source_obligations": list(source.source_obligations),
        "guarded_sha256": {str(path): digest for path, digest in guarded.items()},
        "qualification": "All original facts independently replayed; explicit box reconstruction and finite checks. Original source/proof qualifications remain. No native adoption, maximum-removal, cost or optimality claim.",
    }


def check_projection(
    *,
    snapshot: Path,
    axes_file: Path,
    artifact: Path,
    removed: frozenset[tuple[str, str]],
    source_version: str = "source587",
) -> Wire:
    """Replay original facts and supplied finite premises for a qualified version."""
    version = resolve_source_version(source_version=source_version)
    result = _project(
        snapshot=snapshot,
        axes_file=axes_file,
        artifact=artifact,
        removed=removed,
        version=version,
    )
    resolve_source_version(source_version=source_version)
    return {
        **result,
        "schema": "source-bound-implicit-projection/v2",
        "status": "FINITE_PREMISES_PROVED"
        if _mapping(result["screen"])["conditions_pass"]
        else "FINITE_PREMISES_REFUTED",
        "fact_verification": {
            **_mapping(result["fact_verification"]),
            "schema": "exact-implicit-fibre-verification/v2",
            "status": "COMPLETE_FACT_EQUALITY",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("axes", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "removed",
        type=Path,
        help="JSON array of [original regime, status coordinate] pairs",
    )
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-version", default="source587")
    parser.add_argument(
        "--expand-boxes",
        action="store_true",
        required=True,
        help="Acknowledge that exact common refinement can be exponential",
    )
    args = parser.parse_args()
    pairs = tuple(_strings(row) for row in json.loads(args.removed.read_text()))
    if any(len(pair) != 2 for pair in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("Removal proposal must contain distinct regime/status pairs")
    removed = frozenset((pair[0], pair[1]) for pair in pairs)
    proposal_sha = sha(args.removed)
    result = check_projection(
        snapshot=args.snapshot,
        axes_file=args.axes,
        artifact=args.artifact,
        removed=removed,
        source_version=args.source_version,
    )
    if sha(args.removed) != proposal_sha:
        raise ValueError("Removal proposal changed during verification")
    result["proposal_sha256"] = proposal_sha
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
