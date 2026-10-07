#!/usr/bin/env python3
"""Generate the candidate certificate's exact source inventory from its AST.

The inventory lists each certified source once, under `sources`. Each profile in
`derived_policy.profiles` references that list by name and carries one explicit
override slot, `exclude_sources`; `profile_sources` resolves the effective set of
every profile, and `upgrade_inventory` carries an older-schema inventory forward.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

CERTIFICATE_PATH = "tests/test_grid_search_candidate_certificate.py"
INVENTORY_PATH = "tests/candidate_certificate/sources.json"
REQUIRED_PROFILES = ("fast", "certified")
SCHEMA_VERSION = "2"
# The top-level key holding the canonical source list every profile references.
INVENTORY_KEY = "sources"


def sha256_file(path: Path) -> str:
    """Hash UTF-8 text with checkout-independent LF newlines."""
    canonical = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def derive_source_paths(certificate: Path) -> tuple[str, ...]:
    """Derive unique repo-relative sources from literal ``_parse`` obligations."""
    tree = ast.parse(certificate.read_text(encoding="utf-8"), filename=str(certificate))
    paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "_parse":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        value = node.args[0].value
        if isinstance(value, str):
            paths.append(Path(value).as_posix())
    unique = tuple(sorted(set(paths)))
    if not unique:
        raise ValueError("certificate contains no literal _parse source obligations")
    return unique


def inventory_digest(sources: list[dict[str, str]]) -> str:
    """Hash the canonical ordered source-record representation."""
    payload = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def upgrade_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    """Return `payload` in the current schema, carrying every profile override.

    Schema 1 repeated the whole source list in each profile. A profile that listed
    every inventory source becomes a plain reference; one that listed a subset
    becomes a reference whose `exclude_sources` names the omitted paths. A schema-1
    profile entry absent from the inventory has no exclusion form and is refused.
    """
    version = payload.get("schema_version")
    if version == SCHEMA_VERSION:
        return payload
    if version != "1":
        raise ValueError(f"unknown inventory schema_version {version!r}")
    paths = [item["path"] for item in payload.get("sources", [])]
    profiles: dict[str, Any] = {}
    for profile, entry in payload.get("derived_policy", {}).get("profiles", {}).items():
        listed = [item["path"] for item in entry.get("candidate_sources", [])]
        stray = sorted(set(listed) - set(paths))
        if stray:
            raise ValueError(
                f"schema-1 profile {profile!r} lists sources outside the inventory: "
                f"{stray}"
            )
        profiles[profile] = {
            "inventory": INVENTORY_KEY,
            "exclude_sources": [path for path in paths if path not in listed],
        }
    return {
        **payload,
        "schema_version": SCHEMA_VERSION,
        "derived_policy": {"profiles": profiles},
    }


def profile_sources(payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Resolve each required profile's candidate sources from the one inventory.

    A profile entry is `{"inventory": "sources", "exclude_sources": [...]}`: it
    references the canonical list by name and drops the paths its override slot
    names. An override naming a path absent from the inventory, or naming one
    twice, is refused rather than ignored.
    """
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"inventory schema_version {version!r} is not {SCHEMA_VERSION!r}; "
            "regenerate it with generate_sources.py --write"
        )
    sources = payload.get(INVENTORY_KEY)
    if not isinstance(sources, list):
        raise ValueError(f"inventory has no {INVENTORY_KEY!r} list")
    policy = payload.get("derived_policy")
    profiles = policy.get("profiles") if isinstance(policy, dict) else None
    if not isinstance(profiles, dict):
        raise ValueError("inventory has no derived_policy profiles object")
    paths = [item.get("path") for item in sources]
    resolved: dict[str, list[dict[str, str]]] = {}
    for profile in REQUIRED_PROFILES:
        entry = profiles.get(profile)
        if not isinstance(entry, dict):
            raise ValueError(f"inventory has no derived policy profile {profile!r}")
        reference = entry.get("inventory")
        if reference != INVENTORY_KEY:
            raise ValueError(
                f"profile {profile!r} references {reference!r}, not {INVENTORY_KEY!r}"
            )
        excluded = entry.get("exclude_sources")
        if not isinstance(excluded, list) or not all(
            isinstance(path, str) for path in excluded
        ):
            raise ValueError(f"profile {profile!r} exclude_sources is not a path list")
        repeated = sorted({path for path in excluded if excluded.count(path) > 1})
        unknown = sorted(set(excluded) - set(paths))
        if repeated or unknown:
            raise ValueError(
                f"profile {profile!r} exclude_sources names paths twice {repeated} "
                f"or outside the inventory {unknown}"
            )
        resolved[profile] = [item for item in sources if item["path"] not in excluded]
    return resolved


def _committed_overrides(root: Path) -> dict[str, list[str]]:
    """Read each profile's hand-set exclusions from the committed inventory.

    The override slot is the one hand-maintained part of the inventory, so
    regeneration carries it forward, upgrading an older schema on the way. A
    missing or unreadable inventory has no overrides.
    """
    try:
        committed = json.loads((root / INVENTORY_PATH).read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    if not isinstance(committed, dict):
        return {}
    profiles = upgrade_inventory(committed)["derived_policy"]["profiles"]
    return {
        profile: list(entry.get("exclude_sources", []))
        for profile, entry in profiles.items()
    }


def build_inventory(repo_root: Path) -> dict[str, Any]:
    """Build the canonical inventory from certificate obligations and source bytes."""
    root = repo_root.resolve()
    certificate = root / CERTIFICATE_PATH
    sources = [
        {"path": relative, "sha256": sha256_file(root / relative)}
        for relative in derive_source_paths(certificate)
    ]
    overrides = _committed_overrides(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "certificate": CERTIFICATE_PATH,
        "generation_rule": (
            "unique sorted literal _parse(<repo-relative path>) call arguments "
            "in the certificate AST"
        ),
        INVENTORY_KEY: sources,
        "source_inventory_sha256": inventory_digest(sources),
        "derived_policy": {
            "profiles": {
                profile: {
                    "inventory": INVENTORY_KEY,
                    "exclude_sources": overrides.get(profile, []),
                }
                for profile in REQUIRED_PROFILES
            }
        },
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    generated = build_inventory(root)
    inventory_path = root / INVENTORY_PATH

    if args.write:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(canonical_json(generated), encoding="utf-8")

    if args.check:
        try:
            committed = json.loads(inventory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result = {
                "schema_version": "1",
                "result": "fail",
                "inventory": INVENTORY_PATH,
                "error": str(error),
            }
            print(canonical_json(result), end="")
            return 1
        matches = committed == generated
        result = {
            "schema_version": "1",
            "result": "pass" if matches else "fail",
            "inventory": INVENTORY_PATH,
            "certificate": CERTIFICATE_PATH,
            "derived_source_paths": [item["path"] for item in generated["sources"]],
            "source_inventory_sha256": generated["source_inventory_sha256"],
            "matches_generated_inventory": matches,
        }
        print(canonical_json(result), end="")
        return 0 if matches else 1

    rendered = canonical_json(generated)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
