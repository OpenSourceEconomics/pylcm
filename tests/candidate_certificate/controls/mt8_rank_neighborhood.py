#!/usr/bin/env python3
"""Check that the candidate ordering is swept, not fixed.

Under one fixed ordering only a single candidate is ever the maximizer of a fully
feasible neighborhood, so an omission of any other candidate changes nothing that the
certificate observes. This control shows the discrimination the swept ordering buys and
verifies that the swept matrix is structurally required rather than merely present.

Exit 1 means the ordering neighborhood is fail-open. Exit 0 means it is closed.
Exit 2 means the control itself failed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CERTIFICATE = Path("tests/test_grid_search_candidate_certificate.py")
VERIFIER = Path("tests/candidate_certificate/verify.py")
ROUTES = (
    "test_singleton_solve_matches_reference_over_every_nonempty_feasibility_mask",
    "test_singleton_simulate_matches_reference_over_every_nonempty_feasibility_mask",
    "test_collective_solve_matches_reference_over_every_nonempty_feasibility_mask",
    "test_collective_simulate_matches_reference_over_every_nonempty_feasibility_mask",
)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_verifier(root: Path) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, str(root / VERIFIER), "--repo-root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
    return completed.returncode, payload


def _oracle_discrimination(root: Path) -> dict[str, Any]:
    """Compare a fixed ordering against the swept one on the same omission."""
    sys.path.insert(0, str(root / "tests" / "candidate_certificate"))
    try:
        import verify as verifier  # noqa: PLC0415 -- loaded from the copied tree
    finally:
        sys.path.pop(0)
    return verifier.run_rank_neighborhood_controls()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    source_root = args.repo_root.resolve()
    try:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            shutil.copytree(
                source_root,
                root,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".git", ".pixi", "__pycache__", "*.pyc", ".venv"
                ),
            )
            clean_exit, clean_payload = _run_verifier(root)
            oracle = _oracle_discrimination(root)

            certificate = root / CERTIFICATE
            original = certificate.read_text(encoding="utf-8")
            cases: dict[str, dict[str, Any]] = {}

            # Collapsing the swept matrix back to one fixed ordering must be rejected.
            collapsed = original.replace(
                "_RANK_VECTORS = rank_vectors(len(_CANDIDATES))",
                "_RANK_VECTORS = ("
                "tuple(float(i + 1) for i in range(len(_CANDIDATES))),)",
                1,
            )
            certificate.write_text(collapsed, encoding="utf-8")
            exit_code, payload = _run_verifier(root)
            cases["ordering_matrix_collapsed_to_one_fixed_ordering"] = {
                "exit": exit_code,
                "errors": payload.get("errors", []),
            }
            certificate.write_text(original, encoding="utf-8")

            # Dropping the ordering loop from any single route must be rejected.
            for route in ROUTES:
                marker = f"def {route}("
                start = original.index(marker)
                body = original[start:]
                mutated_body = body.replace(
                    "    for ranks in _RANK_VECTORS:\n"
                    "        for mask in _NONEMPTY_FEASIBILITY_MASKS:",
                    "    ranks = _RANK_VECTORS[0]\n"
                    "    if True:\n"
                    "        for mask in _NONEMPTY_FEASIBILITY_MASKS:",
                    1,
                )
                certificate.write_text(
                    original[:start] + mutated_body, encoding="utf-8"
                )
                exit_code, payload = _run_verifier(root)
                cases[f"ordering_loop_removed:{route}"] = {
                    "exit": exit_code,
                    "errors": payload.get("errors", []),
                }
                certificate.write_text(original, encoding="utf-8")

            failures = [name for name, case in cases.items() if case["exit"] == 0]
            fixed = oracle["fixed_ascending_ordering"]
            swept = oracle["ordering_won_by_candidate_zero"]
            coverage = oracle["every_candidate_wins_exactly_one_ordering"]
            discriminates = (
                not fixed["detected"]
                and swept["detected"]
                and bool(coverage["covers_every_candidate"])
            )
            closed = clean_exit == 0 and not failures and discriminates
            _emit(
                {
                    "schema_version": "1",
                    "result": "gap_not_reproduced" if closed else "gap_reproduced",
                    "clean_exit": clean_exit,
                    "clean_errors": clean_payload.get("errors", []),
                    "oracle_discrimination": oracle,
                    "swept_ordering_detects_what_fixed_ordering_misses": discriminates,
                    "mutations": cases,
                    "admitted_mutations": failures,
                }
            )
            return 0 if closed else 1
    except Exception as error:
        _emit(
            {
                "schema_version": "1",
                "result": "instrument_error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
