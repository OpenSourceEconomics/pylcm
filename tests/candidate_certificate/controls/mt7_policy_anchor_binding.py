#!/usr/bin/env python3
"""Check that the compiled profile policy is bound to the whole source inventory.

A compiled policy can express one path and one digest per profile, so the contract
anchors on the generated inventory file rather than on one certified source. This
control asserts the anchor actually binds: changing any certified source's bytes must
move the inventory file, and the stale anchor must then be rejected.

Exit 1 means the binding is fail-open. Exit 0 means every perturbation is rejected.
Exit 2 means the control itself failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CONTRACT = Path(".pro-audit/profile-contract.yaml")
GENERATOR = Path("tests/candidate_certificate/generate_sources.py")
INVENTORY = Path("tests/candidate_certificate/sources.json")
VERIFIER = Path("tests/candidate_certificate/verify.py")
PROFILES = ("fast", "certified")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _compile_policy(contract: Path) -> dict[str, Any]:
    """Project the contract as a bundler does: one source and digest per profile."""
    profiles: dict[str, Any] = {}
    current: str | None = None
    candidate_indent: int | None = None
    source: str | None = None
    digest: str | None = None

    def flush() -> None:
        nonlocal source, digest
        if current is not None and (source is not None or digest is not None):
            profiles[current] = {
                "candidate_source": source,
                "candidate_source_digest": digest,
            }
        source = None
        digest = None

    for line in [*contract.read_text(encoding="utf-8").splitlines(), ""]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 2 and stripped.endswith(":"):
            flush()
            current = stripped[:-1]
            candidate_indent = None
            continue
        if current is None:
            continue
        if stripped == "candidate_set_error:" and indent > 2:
            flush()
            candidate_indent = indent
            continue
        if candidate_indent is None:
            continue
        if stripped and indent <= candidate_indent:
            flush()
            candidate_indent = None
            continue
        if indent == candidate_indent + 2 and stripped.startswith("source:"):
            source = stripped.split(":", 1)[1].strip().strip('"')
        elif indent == candidate_indent + 2 and stripped.startswith("source_digest:"):
            digest = stripped.split(":", 1)[1].strip().strip('"')
    flush()
    return {"contract_version": "1", "target": "", "profiles": profiles}


def _run_join(root: Path, policy_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    policy_path = root / "compiled-policy.json"
    policy_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(root / VERIFIER),
            "--repo-root",
            str(root),
            "--contract",
            str(root / CONTRACT),
            "--policy",
            str(policy_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
    return completed.returncode, payload


def _regenerate(root: Path) -> None:
    subprocess.run(
        [sys.executable, str(root / GENERATOR), "--repo-root", str(root), "--write"],
        capture_output=True,
        text=True,
        check=False,
    )


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
            certified = [
                item["path"]
                for item in json.loads((root / INVENTORY).read_text(encoding="utf-8"))[
                    "sources"
                ]
            ]
            clean_policy = _compile_policy(root / CONTRACT)
            clean_exit, clean_payload = _run_join(root, clean_policy)

            cases: dict[str, dict[str, Any]] = {}

            # Every certified source must move the anchor when its bytes change.
            for path in certified:
                target = root / path
                original = target.read_bytes()
                target.write_bytes(original + b"\n# certified-source perturbation\n")
                _regenerate(root)
                exit_code, payload = _run_join(root, clean_policy)
                cases[f"byte_change:{path}"] = {
                    "exit": exit_code,
                    "errors": payload.get("errors", []),
                }
                target.write_bytes(original)
                _regenerate(root)

            # A policy naming one certified source instead of the inventory is stale.
            for path in certified:
                stale = {
                    "contract_version": "1",
                    "target": "",
                    "profiles": {
                        profile: {
                            "candidate_source": path,
                            "candidate_source_digest": hashlib.sha256(
                                (root / path).read_bytes()
                            ).hexdigest(),
                        }
                        for profile in PROFILES
                    },
                }
                exit_code, payload = _run_join(root, stale)
                cases[f"anchor_replaced_by:{path}"] = {
                    "exit": exit_code,
                    "errors": payload.get("errors", []),
                }

            # A policy that declares no anchor at all must not pass.
            empty = {
                "contract_version": "1",
                "target": "",
                "profiles": {profile: {} for profile in PROFILES},
            }
            exit_code, payload = _run_join(root, empty)
            cases["anchor_absent"] = {
                "exit": exit_code,
                "errors": payload.get("errors", []),
            }

            failures = [name for name, case in cases.items() if case["exit"] == 0]
            closed = clean_exit == 0 and not failures
            _emit(
                {
                    "schema_version": "1",
                    "result": "gap_not_reproduced" if closed else "gap_reproduced",
                    "clean_exit": clean_exit,
                    "clean_errors": clean_payload.get("errors", []),
                    "certified_sources": certified,
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
