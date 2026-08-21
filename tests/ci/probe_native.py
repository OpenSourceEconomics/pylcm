"""Strict setup probe for pylcm's installed exact-affine payload."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import hatch_build
from _lcm.egm.upper_envelope._exact_affine import ffi

BROKEN_PAYLOAD = 1
REINSTALL_REQUIRED = 2


@dataclass(frozen=True)
class ProbeResult:
    """Machine-readable native-payload setup verdict."""

    exit_code: int
    status: str
    detail: str


def payload_present() -> bool:
    """Return whether the selected backend has every required library file."""
    return ffi.kernel_built_for_current_backend()


def manifest_matches(root: Path) -> bool:
    """Return whether the installed payload matches this source and toolchain."""
    path = ffi._DIRECTORY / hatch_build.NATIVE_MANIFEST
    try:
        installed = json.loads(path.read_text())
    except OSError, ValueError, TypeError:
        return False
    return installed.get("inputs") == hatch_build.native_build_inputs(root=root)


def kernel_available() -> bool:
    """Return whether the present payload loads and registers successfully."""
    return ffi.kernel_available_for_current_backend()


def probe(*, root: Path | None = None) -> ProbeResult:
    """Distinguish a cache miss from a broken installed native payload."""
    root = root or Path.cwd()
    if not payload_present():
        return ProbeResult(
            exit_code=REINSTALL_REQUIRED,
            status="absent",
            detail="the selected backend's installed native library is absent",
        )
    if not manifest_matches(root):
        return ProbeResult(
            exit_code=REINSTALL_REQUIRED,
            status="stale",
            detail="the installed native manifest does not match this build",
        )
    if not kernel_available():
        return ProbeResult(
            exit_code=BROKEN_PAYLOAD,
            status="broken",
            detail="the installed native library is present but cannot register",
        )
    return ProbeResult(exit_code=0, status="ready", detail="native payload is ready")


def main(argv: list[str] | None = None) -> int:
    """Print and optionally persist the probe verdict for a CI setup step."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=None)
    options = parser.parse_args(argv)
    result = probe(root=options.root)
    payload = asdict(result)
    if options.report is not None:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))  # noqa: T201
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
