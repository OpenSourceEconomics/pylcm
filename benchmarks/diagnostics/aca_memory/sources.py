"""Validate the source and input bytes used by the bounded ACA diagnostic."""

import hashlib
import json
from pathlib import Path

PYLCM_SHA = "c710f9b522ec6041830211d9eaf2536f1c842826"
ACA_SHA = "b941507c14932b85b482d77ed8d7f3034c5dbfee"


def verify_sources(
    *,
    pylcm_root: Path,
    manifest_path: Path,
    aca_root: Path | None = None,
    aca_package_root: Path | None = None,
) -> dict:
    """Check source files without importing numerical packages."""
    if (aca_root is None) == (aca_package_root is None):
        raise ValueError(
            "Specify exactly one ACA source root or installed package root."
        )
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest)
    checked = 0
    ignored = []
    aca_boundary = aca_root if aca_root is not None else aca_package_root
    assert aca_boundary is not None
    for namespace, root in (("pylcm", pylcm_root), ("aca", aca_boundary)):
        for relative, expected in manifest[namespace].items():
            target = _source_path(
                namespace=namespace,
                relative=relative,
                root=root,
                installed_aca=aca_package_root is not None,
            )
            if target is None:
                ignored.append(relative)
                continue
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {namespace}/{relative}")
            checked += 1
    return {
        "pylcm_sha": PYLCM_SHA,
        "aca_sha": ACA_SHA,
        "checked_files": checked,
        "ignored_packaging_files": ignored,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _validate_manifest(manifest: dict) -> None:
    """Require the exact source revisions and a populated numerical source manifest."""
    if manifest["pylcm_sha"] != PYLCM_SHA or manifest["aca_sha"] != ACA_SHA:
        raise ValueError("Unexpected pinned source revisions.")
    for namespace, prefix in (("pylcm", "src/"), ("aca", "src/aca_model/")):
        if not any(
            name.startswith(prefix) and name.endswith(".py")
            for name in manifest[namespace]
        ):
            raise ValueError(f"Missing numerical source population: {namespace}")


def _source_path(
    *, namespace: str, relative: str, root: Path, installed_aca: bool
) -> Path | None:
    """Resolve a contained source file and identify packaging-only exclusions."""
    if namespace == "aca" and installed_aca:
        if relative == "pyproject.toml":
            return None
        if not relative.startswith("src/aca_model/"):
            raise ValueError(f"Unsupported installed package path: {relative}")
        target = root / relative.removeprefix("src/aca_model/")
    else:
        target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes root: {relative}")
    return target
