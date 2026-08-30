#!/usr/bin/env python3
"""Synchronize every anchor, then attack the route-local direct-flow proof.

Each mutation changes a certified production source, regenerates the AST-derived
source inventory, rewrites the profile contract's inventory anchor and complete
additional-source set, recompiles the policy, refreshes the relevant manifest
records, and only then runs the unified verifier.  A digest mismatch therefore
cannot make the control red: the direct-flow proof itself must reject the
candidate-changing transformation.

Exit 0 means every mutation is rejected after synchronized re-anchoring.
Exit 1 means at least one mutation remains fail-open.  Exit 2 is an instrument
error.
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
from types import ModuleType
from typing import Any, cast

REQUIRED_PROFILES = ("fast", "certified")
INVENTORY = Path("tests/candidate_certificate/sources.json")
GENERATOR = Path("tests/candidate_certificate/generate_sources.py")
VERIFIER = Path("tests/candidate_certificate/verify.py")
MAX_Q_SOURCE = Path("src/_lcm/regime_building/max_Q_over_a.py")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_direct_flow(repo_root: Path) -> ModuleType:
    sys.path.insert(0, str(repo_root / "tests" / "candidate_certificate"))
    try:
        import direct_flow  # noqa: PLC0415 -- loaded from the copied tree
    finally:
        sys.path.pop(0)
    return direct_flow


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _regenerate_inventory(repo_root: Path) -> None:
    result = _run(
        [
            sys.executable,
            str(repo_root / GENERATOR),
            "--repo-root",
            str(repo_root),
            "--write",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"inventory generation failed ({result.returncode}): {result.stderr}"
        )


def _sync_contract(contract: Path, *, repo_root: Path) -> None:
    """Rewrite both candidate blocks from the generated inventory."""
    inventory_path = repo_root / INVENTORY
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    source_digest = _sha256(inventory_path)
    sources = payload["sources"]
    lines = contract.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    candidate_indent: int | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "candidate_set_error:":
            candidate_indent = indent
            output.append(line)
            index += 1
            continue
        if candidate_indent is not None and stripped and indent <= candidate_indent:
            candidate_indent = None
        if candidate_indent is not None and stripped.startswith("source_digest:"):
            output.append(" " * indent + f'source_digest: "sha256:{source_digest}"')
            index += 1
            continue
        if candidate_indent is not None and stripped == "additional_sources:":
            output.append(line)
            child_indent = indent + 2
            index += 1
            while index < len(lines):
                child = lines[index]
                child_stripped = child.strip()
                child_level = len(child) - len(child.lstrip(" "))
                if child_stripped and child_level <= indent:
                    break
                if child_level == child_indent and child_stripped.startswith('"src/'):
                    index += 1
                    continue
                break
            output.extend(
                " " * child_indent + f'"{item["path"]}": "sha256:{item["sha256"]}"'
                for item in sources
            )
            continue
        output.append(line)
        index += 1
    contract.write_text("\n".join(output) + "\n", encoding="utf-8")


def _sync_policy(policy: Path, *, contract: Path, repo_root: Path) -> None:
    inventory_path = repo_root / INVENTORY
    anchor = f"sha256:{_sha256(inventory_path)}"
    try:
        payload = cast("dict[str, Any]", json.loads(policy.read_text(encoding="utf-8")))
    except OSError, json.JSONDecodeError:
        payload = {"contract_version": "1", "target": "", "profiles": {}}
    profiles = cast("dict[str, Any]", payload.setdefault("profiles", {}))
    for profile in REQUIRED_PROFILES:
        entry = cast("dict[str, Any]", profiles.setdefault(profile, {}))
        entry["candidate_source"] = INVENTORY.as_posix()
        entry["candidate_source_digest"] = anchor
    payload["digest"] = _sha256(contract)
    policy.write_text(_canonical_json(payload), encoding="utf-8")


def _sync_manifest(
    manifest: Path | None,
    *,
    bundle_root: Path | None,
    repo_root: Path,
    contract: Path | None,
    policy: Path | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    payload = cast("dict[str, Any]", json.loads(manifest.read_text(encoding="utf-8")))
    manifest_files = cast("list[dict[str, Any]]", payload["files"])
    path_to_entry: dict[str, dict[str, Any]] = {
        str(item["path"]): item for item in manifest_files
    }
    inventory_payload = json.loads((repo_root / INVENTORY).read_text(encoding="utf-8"))
    updates: list[tuple[str, Path, str]] = [
        (
            f"project/{item['path']}",
            repo_root / item["path"],
            item["path"],
        )
        for item in inventory_payload["sources"]
    ]
    updates.append(
        (
            f"project/{INVENTORY.as_posix()}",
            repo_root / INVENTORY,
            INVENTORY.as_posix(),
        )
    )
    if contract is not None:
        updates.append(
            (
                "protocol/profile-contract.yaml",
                contract,
                ".pro-audit/profile-contract.yaml",
            )
        )
    if policy is not None:
        updates.append(
            (
                "protocol/profile-contract.policy.json",
                policy,
                "(compiled from the shipped profile contract)",
            )
        )
    for manifest_path, disk_path, source_path in updates:
        entry = path_to_entry.get(manifest_path)
        if entry is None:
            role = (
                "profile_contract"
                if manifest_path.endswith("profile-contract.yaml")
                else "profile_contract_policy"
                if manifest_path.endswith("policy.json")
                else "code"
            )
            entry: dict[str, Any] = {
                "path": manifest_path,
                "source_path": source_path,
                "role": role,
            }
            payload["files"].append(entry)
            path_to_entry[manifest_path] = entry
        entry["bytes"] = disk_path.stat().st_size
        entry["sha256"] = _sha256(disk_path)
    if contract is not None:
        payload["profile_contract_digest"] = _sha256(contract)
    if policy is not None:
        payload["profile_contract_policy"] = json.loads(
            policy.read_text(encoding="utf-8")
        )
    payload.pop("manifest_digest", None)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["manifest_digest"] = digest
    manifest.write_text(_canonical_json(payload), encoding="utf-8")
    return {
        "manifest_digest": digest,
        "relevant_entries_match": all(
            path_to_entry[path]["sha256"] == _sha256(disk)
            for path, disk, _source in updates
        ),
        "bundle_root": str(bundle_root) if bundle_root is not None else None,
    }


def _run_verifier(
    repo_root: Path, *, contract: Path | None, policy: Path | None
) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        "-S",
        str(repo_root / VERIFIER),
        "--repo-root",
        str(repo_root),
    ]
    if contract is not None:
        command += ["--contract", str(contract)]
    if policy is not None:
        command += ["--policy", str(policy)]
    completed = _run(command)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": completed.stdout, "stderr": completed.stderr}
    return completed.returncode, payload


def _synchronize(
    *,
    repo_root: Path,
    contract: Path | None,
    policy: Path | None,
    manifest: Path | None,
    bundle_root: Path | None,
) -> dict[str, Any] | None:
    _regenerate_inventory(repo_root)
    if contract is not None:
        _sync_contract(contract, repo_root=repo_root)
    if policy is not None:
        if contract is None:
            raise ValueError("--policy requires --contract")
        _sync_policy(policy, contract=contract, repo_root=repo_root)
    return _sync_manifest(
        manifest,
        bundle_root=bundle_root,
        repo_root=repo_root,
        contract=contract,
        policy=policy,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-root", type=Path)
    source.add_argument("--bundle-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    try:
        source_bundle = args.bundle_root.resolve() if args.bundle_root else None
        if source_bundle is None and args.repo_root is None:
            raise ValueError("one of --repo-root or --bundle-root is required")
        if source_bundle is not None:
            source_repo = source_bundle / "project"
        else:
            assert args.repo_root is not None
            source_repo = args.repo_root.resolve()
        if source_bundle is not None:
            source_contract = source_bundle / "protocol/profile-contract.yaml"
            source_policy = source_bundle / "protocol/profile-contract.policy.json"
            source_manifest = source_bundle / "BUNDLE-MANIFEST.json"
        else:
            source_contract = args.contract.resolve() if args.contract else None
            source_policy = args.policy.resolve() if args.policy else None
            source_manifest = args.manifest.resolve() if args.manifest else None

        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            if source_bundle is not None:
                bundle = temp / "bundle"
                shutil.copytree(source_bundle, bundle, symlinks=True)
                repo = bundle / "project"
                contract = bundle / "protocol/profile-contract.yaml"
                policy = bundle / "protocol/profile-contract.policy.json"
                manifest = bundle / "BUNDLE-MANIFEST.json"
            else:
                bundle = None
                repo = temp / "repo"
                shutil.copytree(
                    source_repo,
                    repo,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(
                        ".git", ".pixi", "__pycache__", "*.pyc", ".venv"
                    ),
                )
                contract = None
                policy = None
                manifest = None
                if source_contract is not None:
                    contract = temp / "profile-contract.yaml"
                    shutil.copy2(source_contract, contract)
                if source_policy is not None:
                    policy = temp / "profile-contract.policy.json"
                    shutil.copy2(source_policy, policy)
                if source_manifest is not None:
                    manifest = temp / "BUNDLE-MANIFEST.json"
                    shutil.copy2(source_manifest, manifest)

            direct_flow = _load_direct_flow(repo)
            mutation_specs = direct_flow.direct_flow_mutation_specs(repo_root=repo)
            mutated_paths = sorted({spec["path"] for spec in mutation_specs.values()})
            originals = {
                relative: (repo / relative).read_text(encoding="utf-8")
                for relative in mutated_paths
            }
            baseline_manifest = _synchronize(
                repo_root=repo,
                contract=contract,
                policy=policy,
                manifest=manifest,
                bundle_root=bundle,
            )
            clean_exit, clean_payload = _run_verifier(
                repo, contract=contract, policy=policy
            )
            clean_inventory = (repo / INVENTORY).read_text(encoding="utf-8")
            clean_contract = (
                contract.read_text(encoding="utf-8") if contract is not None else None
            )
            clean_policy = (
                policy.read_text(encoding="utf-8") if policy is not None else None
            )
            clean_manifest = (
                manifest.read_text(encoding="utf-8") if manifest is not None else None
            )

            cases: dict[str, dict[str, Any]] = {}
            for name, spec in mutation_specs.items():
                relative = spec["path"]
                source_path = repo / relative
                source_path.write_text(spec["source"], encoding="utf-8")
                manifest_state = _synchronize(
                    repo_root=repo,
                    contract=contract,
                    policy=policy,
                    manifest=manifest,
                    bundle_root=bundle,
                )
                exit_code, payload = _run_verifier(
                    repo, contract=contract, policy=policy
                )
                errors = payload.get("errors", [])
                semantic_errors = [
                    error for error in errors if str(error).startswith("direct flow:")
                ]
                anchor_errors = [
                    error
                    for error in errors
                    if any(
                        marker in str(error)
                        for marker in (
                            "source bytes",
                            "sources.json",
                            "profile contract",
                            "derived policy",
                            "inventory anchor mismatch",
                        )
                    )
                    and not str(error).startswith("direct flow:")
                ]
                cases[name] = {
                    "path": relative,
                    "exit": exit_code,
                    "rejected_by_direct_flow": exit_code == 1 and bool(semantic_errors),
                    "semantic_errors": semantic_errors,
                    "anchor_errors": anchor_errors,
                    "manifest": manifest_state,
                }
                source_path.write_text(originals[relative], encoding="utf-8")
                (repo / INVENTORY).write_text(clean_inventory, encoding="utf-8")
                if contract is not None and clean_contract is not None:
                    contract.write_text(clean_contract, encoding="utf-8")
                if policy is not None and clean_policy is not None:
                    policy.write_text(clean_policy, encoding="utf-8")
                if manifest is not None and clean_manifest is not None:
                    manifest.write_text(clean_manifest, encoding="utf-8")

            admitted = sorted(
                name
                for name, result in cases.items()
                if not result["rejected_by_direct_flow"] or result["anchor_errors"]
            )
            closed = clean_exit == 0 and not admitted
            print(
                _canonical_json(
                    {
                        "schema_version": "1",
                        "result": "gap_not_reproduced" if closed else "gap_reproduced",
                        "clean_exit": clean_exit,
                        "clean_errors": clean_payload.get("errors", []),
                        "baseline_manifest": baseline_manifest,
                        "mutation_count": len(cases),
                        "mutations": cases,
                        "admitted_mutations": admitted,
                        "all_anchors_synchronized_before_each_verifier_run": True,
                    }
                ),
                end="",
            )
            return 0 if closed else 1
    except Exception as error:
        print(
            _canonical_json(
                {
                    "schema_version": "1",
                    "result": "instrument_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            end="",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
