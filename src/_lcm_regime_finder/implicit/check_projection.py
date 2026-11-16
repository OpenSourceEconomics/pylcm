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


def _project(
    *,
    snapshot: Path,
    axes_file: Path,
    artifact: Path,
    removed: frozenset[tuple[str, str]],
) -> Wire:
    """Verify original facts, then check a proposal using the explicit box adapter.

    Source qualifications remain binding. The result is a finite supplied-fact
    check, not native adoption, economic equivalence or a best-graph certificate.
    No result is returned after an interrupted replay or a changed input.
    """
    here = Path(__file__).parent
    guarded = {path: sha(path) for path in (snapshot, axes_file, artifact)}
    guarded.update({path: sha(path) for path in here.glob("*.py")})
    fact_receipt = _replay(snapshot=snapshot, axes_file=axes_file, artifact=artifact)
    original = load_problem(snapshot)
    if original.get("schema") != "constructed-pylcm-model-graph-1":
        raise ValueError("Expected an original constructed Model capture")
    source = FibreSource(original)
    record = _mapping(json.loads(artifact.read_text()))
    axes = {
        name: _strings(value)
        for name, value in _mapping(json.loads(axes_file.read_text())).items()
    }
    table = build_factored_table(source=source, record=record, status_axes=axes)
    package = as_box_package(table)
    kept = possible_closure(package)
    live = least_liveness(package)
    screen = check_screen(package, removed)
    final_input_check(original)
    if any(sha(path) != digest for path, digest in guarded.items()):
        raise ValueError("Source/certificate transaction changed")
    return {
        "schema": "source-bound-implicit-projection-evidence/v2",
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
) -> Wire:
    """Replay original facts, then check the supplied finite projection premises."""
    result = _project(
        snapshot=snapshot, axes_file=axes_file, artifact=artifact, removed=removed
    )
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
