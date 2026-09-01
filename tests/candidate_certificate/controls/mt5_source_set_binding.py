#!/usr/bin/env python3
"""Port MT5 to a repository-root candidate-certificate control.

Exit 1 means the source-set gap is reproduced. Exit 0 means the unified
inventory rejects deletion, addition, duplication, rename, and byte change.
Exit 2 means the control itself failed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CERTIFICATE = Path("tests/test_grid_search_candidate_certificate.py")
INVENTORY = Path("tests/candidate_certificate/sources.json")
VERIFIER = Path("tests/candidate_certificate/verify.py")
PROFILES = ("fast", "certified")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _baseline_sets(path: Path) -> tuple[set[str], set[str], bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constant: set[str] = set()
    parsed: set[str] = set()
    test_reads_external_inventory = False
    for node in tree.body:
        value = None
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "CERTIFIED_SOURCES"
            and isinstance(value, ast.Tuple)
        ):
            constant = {
                str(item.value)
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
        if isinstance(node, ast.FunctionDef) and node.name in {
            "test_the_obligations_read_exactly_the_declared_certified_sources",
            "test_no_obligation_rests_on_an_undeclared_source",
        }:
            body = ast.unparse(node)
            test_reads_external_inventory = any(
                marker in body
                for marker in ("sources.json", "generate_sources", "verify_inventory")
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _callee_name(node) == "_parse"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            parsed.add(node.args[0].value)
    return constant, parsed, test_reads_external_inventory


def _load_inventory(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload["sources"]]


def _contract(
    *,
    entries: list[dict[str, str]],
    anchor: dict[str, str],
    duplicate: bool = False,
) -> str:
    """Render the contract grammar: an inventory anchor plus the certified set.

    The primary `source`/`source_digest` pair names the generated inventory, not one
    of the certified sources, so every certified source is declared under
    `additional_sources` and the mutations below perturb that set alone.
    """
    lines = ["profiles:"]
    for profile in PROFILES:
        lines.extend(
            [
                f"  {profile}:",
                "    numerical:",
                "      candidate_set_error:",
                f'        source: "{anchor["path"]}"',
                f'        source_digest: "sha256:{anchor["sha256"]}"',
                "        additional_sources:",
            ]
        )
        lines.extend(
            f'          "{item["path"]}": "sha256:{item["sha256"]}"' for item in entries
        )
        if duplicate:
            first = entries[0]
            lines.append(f'          "{first["path"]}": "sha256:{first["sha256"]}"')
    return "\n".join(lines) + "\n"


def _policy(entries: list[dict[str, str]]) -> str:
    return (
        json.dumps(
            {
                "schema_version": "candidate-source-policy-1",
                "profiles": {
                    profile: {"candidate_sources": entries} for profile in PROFILES
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _run_verifier(
    *, verifier: Path, repo_root: Path, contract: Path
) -> tuple[int, dict[str, Any]]:
    process = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--repo-root",
            str(repo_root),
            "--contract",
            str(contract),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        payload = {"raw_stdout": process.stdout, "raw_stderr": process.stderr}
    return process.returncode, payload


def _copy_minimal_repo(
    *, source: Path, destination: Path, entries: list[dict[str, str]]
) -> None:
    paths = [CERTIFICATE, INVENTORY, *[Path(item["path"]) for item in entries]]
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    certificate = repo / CERTIFICATE
    inventory = repo / INVENTORY
    verifier = repo / VERIFIER
    try:
        if not inventory.exists() or not verifier.exists():
            constant, parsed, reads_external = _baseline_sets(certificate)
            removed = min(parsed) if parsed else None
            mutated_declared = set(parsed)
            if removed is not None:
                mutated_declared.remove(removed)
            gap = bool(
                parsed
                and constant == parsed
                and parsed != mutated_declared
                and not reads_external
            )
            _emit(
                {
                    "schema_version": "1",
                    "result": "gap_reproduced" if gap else "gap_not_reproduced",
                    "mutation": "delete one certified source from the declared set",
                    "baseline_contract_source_set": sorted(parsed),
                    "mutated_contract_source_set": sorted(mutated_declared),
                    "certificate_internal_source_set": sorted(constant),
                    "certificate_parse_literal_source_set": sorted(parsed),
                    "certificate_source_set_test_reads_generated_inventory": (
                        reads_external
                    ),
                    "offending_paths": [removed] if removed else [],
                }
            )
            return 1 if gap else 0

        entries = _load_inventory(inventory)
        expected_paths = [item["path"] for item in entries]
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            clean_contract = tmp / "clean-contract.yaml"
            anchor = {
                "path": INVENTORY.as_posix(),
                "sha256": hashlib.sha256((repo / INVENTORY).read_bytes()).hexdigest(),
            }
            clean_contract.write_text(
                _contract(entries=entries, anchor=anchor), encoding="utf-8"
            )
            clean_exit, _clean_payload = _run_verifier(
                verifier=verifier,
                repo_root=repo,
                contract=clean_contract,
            )

            cases: dict[str, dict[str, Any]] = {}

            deleted = entries[:-1]
            path = tmp / "delete.yaml"
            path.write_text(_contract(entries=deleted, anchor=anchor), encoding="utf-8")
            code, payload = _run_verifier(
                verifier=verifier, repo_root=repo, contract=path
            )
            cases["delete"] = {
                "exit": code,
                "payload": payload,
                "path": entries[-1]["path"],
            }

            extra_path = "src/_lcm/solution/contract.py"
            extra = {
                "path": extra_path,
                "sha256": hashlib.sha256((repo / extra_path).read_bytes()).hexdigest(),
            }
            added = [*entries, extra]
            path = tmp / "add.yaml"
            path.write_text(_contract(entries=added, anchor=anchor), encoding="utf-8")
            code, payload = _run_verifier(
                verifier=verifier, repo_root=repo, contract=path
            )
            cases["add"] = {"exit": code, "payload": payload, "path": extra_path}

            path = tmp / "duplicate.yaml"
            path.write_text(
                _contract(entries=entries, anchor=anchor, duplicate=True),
                encoding="utf-8",
            )
            code, payload = _run_verifier(
                verifier=verifier, repo_root=repo, contract=path
            )
            cases["duplicate"] = {
                "exit": code,
                "payload": payload,
                "path": entries[0]["path"],
            }

            renamed_path = entries[-1]["path"] + ".renamed"
            renamed = [*entries[:-1], {**entries[-1], "path": renamed_path}]
            path = tmp / "rename.yaml"
            path.write_text(_contract(entries=renamed, anchor=anchor), encoding="utf-8")
            code, payload = _run_verifier(
                verifier=verifier, repo_root=repo, contract=path
            )
            cases["rename"] = {
                "exit": code,
                "payload": payload,
                "path": renamed_path,
            }

            mutated_repo = tmp / "mutated-repo"
            _copy_minimal_repo(source=repo, destination=mutated_repo, entries=entries)
            changed_path = entries[-1]["path"]
            with (mutated_repo / changed_path).open("ab") as stream:
                stream.write(b" ")
            code, payload = _run_verifier(
                verifier=verifier,
                repo_root=mutated_repo,
                contract=clean_contract,
            )
            cases["byte_change"] = {
                "exit": code,
                "payload": payload,
                "path": changed_path,
            }

        failures = []
        for name, case in cases.items():
            rendered = json.dumps(case["payload"], sort_keys=True)
            if case["exit"] == 0 or case["path"] not in rendered:
                failures.append(name)
        closed = clean_exit == 0 and not failures
        _emit(
            {
                "schema_version": "1",
                "result": "gap_not_reproduced" if closed else "gap_reproduced",
                "clean_exit": clean_exit,
                "clean_source_set": expected_paths,
                "mutations": cases,
                "failed_mutations": failures,
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
