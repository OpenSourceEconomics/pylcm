"""RF10-OR1: external byte/permission oracle for completed CLI invocations.

This oracle imports no _lcm_regime_finder module, destination resolver, cleanup list,
resource runner, or status mapper. Every protected pathname and the public
destination are literal test inputs. It is intentionally separate from the
implementation's proposed ownership/write-set inventory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat


def identity(path):
    """Record both the supplied input pathname and the file it currently reads."""
    path = Path(path)
    entry = path.lstat()
    target = path.stat()
    if not stat.S_ISREG(target.st_mode):
        raise AssertionError(f"Expected an ordinary readable input: {path}")
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "entry_type": stat.S_IFMT(entry.st_mode),
        "entry_permissions": stat.S_IMODE(entry.st_mode),
        "target_permissions": stat.S_IMODE(target.st_mode),
        "link_target": str(path.readlink()) if path.is_symlink() else None,
    }


def snapshot(paths):
    return {str(Path(path)): identity(path) for path in paths}


def assess(*, exit_code, result_path, protected, success):
    """Return every violated invariant, without fixing one alias policy.

    Positive controls require normal PASS. For mutation cases, safe rejection
    and coherent success are both valid: a writer may safely replace an output
    alias and continue, provided the protected original inputs remain unchanged.
    """
    problems = []
    for path, expected in protected.items():
        try:
            observed = identity(path)
        except (OSError, AssertionError) as exc:
            problems.append({"invariant": "original_input_available", "path": path,
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        for field, value in expected.items():
            if observed[field] != value:
                problems.append({"invariant": "original_input_unchanged", "path": path,
                                 "field": field, "expected": value,
                                 "observed": observed[field]})

    result_path = Path(result_path)
    temporary = result_path.with_name(result_path.name + ".writing")
    if temporary.exists() or temporary.is_symlink():
        problems.append({"invariant": "no_result_temporary", "path": str(temporary)})
    result = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append({"invariant": "valid_public_result", "error": str(exc)})
    if success:
        if exit_code != 0 or not isinstance(result, dict) or result.get("status") != "PASS":
            problems.append({"invariant": "normal_positive_control", "exit_code": exit_code,
                             "status": result.get("status") if isinstance(result, dict) else None})
    elif exit_code == 0:
        if not isinstance(result, dict) or result.get("status") != "PASS":
            problems.append({"invariant": "coherent_success", "exit_code": exit_code,
                             "status": result.get("status") if isinstance(result, dict) else None})
    else:
        if exit_code == 3:
            problems.append({"invariant": "coherent_alias_disposition", "exit_code": exit_code,
                             "reason": "Expected a supported fixture PASS or an explicit failed boundary"})
        if result is not None:
            if not isinstance(result, dict) or result.get("status") not in {
                    "FAIL", "INVALID", "INCOMPLETE", "FAILED"}:
                problems.append({"invariant": "no_success_after_rejection",
                                 "status": result.get("status") if isinstance(result, dict) else None})
            elif any(result.get(key) for key in (
                    "transformation_authorized", "recommendation", "decisions", "proposal_decisions")):
                problems.append({"invariant": "no_authority_after_rejection"})
    return problems
