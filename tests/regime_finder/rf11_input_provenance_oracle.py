"""RF11 independent byte/path oracle for the supplied structural fixture.

This module imports no _lcm_regime_finder code. Its read-set rule is deliberately
literal and narrow: the single supplied generated structural capture resolves
its store relative to the resolved snapshot file, while every supplied input
name itself must remain readable and unchanged. It is not a replacement loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


def identity(path):
    path = Path(path)
    entry, target = path.lstat(), path.stat()
    if not stat.S_ISREG(target.st_mode):
        raise AssertionError(f"Not an ordinary readable file: {path}")
    payload = path.read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "entry_type": stat.S_IFMT(entry.st_mode),
        "entry_permissions": stat.S_IMODE(entry.st_mode),
        "target_permissions": stat.S_IMODE(target.st_mode),
        "link_target": str(path.readlink()) if path.is_symlink() else None,
    }


def fixture_dependencies(supplied_snapshot):
    """The independently specified read set for this one known fixture shape."""
    supplied = Path(supplied_snapshot).absolute()
    actual = supplied.resolve(strict=True)
    document = json.loads(actual.read_text(encoding="utf-8"))
    if document.get("schema") != "regime-declarations/v1":
        raise AssertionError("This oracle accepts only the unchanged declaration fixture")
    header = document["structural_store"]
    relative = Path(header["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise AssertionError("Fixture locator must be a simple relative store path")
    canonical_store = actual.parent / relative
    lexical_store = supplied.parent / relative
    paths = [supplied, actual, canonical_store / "manifest.json"]
    for key in sorted(header["manifest"]["blobs"]):
        if not key.startswith("p") or not key[1:].isdigit():
            raise AssertionError("Unexpected fixture chunk key")
        paths.append(canonical_store / (key + ".bin"))
    return {
        "paths": list(dict.fromkeys(paths)),
        "actual_manifest": canonical_store / "manifest.json",
        "lexical_manifest": lexical_store / "manifest.json",
        "locator": str(relative),
        "basis": "Known structural fixture: declared store relative to resolved snapshot parent",
    }


def snapshot(paths):
    return {str(Path(path)): identity(path) for path in dict.fromkeys(paths)}


def assess(*, protected, exit_code, result_path=None, stdout=None, positive=False,
           preexisting_temporary=None, result_is_original=False):
    """Safe rejection or coherent success is allowed for deliberately unsafe layouts."""
    failures = []
    for path, expected in protected.items():
        try:
            observed = identity(path)
        except (OSError, AssertionError) as error:
            failures.append({"invariant": "original_input_available", "path": path,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        for field, value in expected.items():
            if observed[field] != value:
                failures.append({"invariant": "original_input_unchanged", "path": path,
                                 "field": field, "expected": value,
                                 "observed": observed[field]})
    result = None
    if result_path is not None:
        result_path = Path(result_path)
        temporary = result_path.with_name(result_path.name + ".writing")
        if temporary.exists() or temporary.is_symlink():
            unchanged_rejected_alias = False
            if exit_code in (1, 2) and preexisting_temporary is not None:
                try:
                    unchanged_rejected_alias = identity(temporary) == preexisting_temporary
                except (OSError, AssertionError):
                    pass
            if not unchanged_rejected_alias:
                failures.append({"invariant": "no_new_or_changed_result_temporary", "path": str(temporary)})
        # Safe pre-mutation rejection may leave a requested destination that is
        # itself the protected original. Its old input schema is not a result.
        if result_path.exists() and not (result_is_original and exit_code in (1, 2)):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                failures.append({"invariant": "readable_public_result", "error": str(error)})
    elif stdout is not None:
        try:
            result = json.loads(stdout)
        except ValueError as error:
            failures.append({"invariant": "readable_stdout_result", "error": str(error)})
    status = result.get("status") if isinstance(result, dict) else None
    if positive:
        if exit_code != 0 or status != "PASS":
            failures.append({"invariant": "positive_supported_fixture", "exit_code": exit_code,
                             "status": status})
    elif exit_code == 0:
        if status != "PASS":
            failures.append({"invariant": "coherent_success", "exit_code": exit_code,
                             "status": status})
    else:
        if exit_code not in (1, 2):
            failures.append({"invariant": "coherent_rejection", "exit_code": exit_code})
        if result is not None:
            if status not in {"FAIL", "INVALID", "INCOMPLETE", "FAILED"}:
                failures.append({"invariant": "no_success_after_rejection", "status": status})
            elif any(result.get(key) for key in (
                    "transformation_authorized", "recommendation", "decisions", "proposal_decisions")):
                failures.append({"invariant": "no_authority_after_rejection"})
    return failures
