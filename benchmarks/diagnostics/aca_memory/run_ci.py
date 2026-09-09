"""Run one source-pinned ACA diagnostic through the benchmark runner."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from observer import Observer


def main() -> int:
    """Launch one bounded diagnostic and return its exit status."""
    scripts = Path(__file__).resolve().parent
    checkout = scripts.parents[2]
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("GitHub run and attempt identifiers must be decimal integers.")
    output = checkout / "reports" / "aca-memory-diagnostic" / f"{run_id}-{attempt}"
    output.mkdir(parents=True, exist_ok=False)
    observer = Observer(output / "launch.jsonl")
    observer.emit("ci_start", run_id=run_id, attempt=attempt)
    try:
        actual_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        manifest = scripts / "sources.json"
        source_base = json.loads(manifest.read_text())["pylcm_sha"]
        aca_root = _installed_aca_root()
        version = checkout / "src" / "_lcm" / "version.py"
        command = [
            sys.executable,
            str(scripts / "run_bounded.py"),
            "--output",
            str(output / "supervisor"),
            "--seconds",
            "900",
            "--rss-gib",
            "32",
            "--",
            sys.executable,
            str(scripts / "diagnose.py"),
            "--pylcm-root",
            str(checkout),
            "--aca-package-root",
            str(aca_root),
            "--manifest",
            str(manifest),
            "--output",
            str(output / "worker"),
            "--precision",
            "64",
            "--platform",
            "cuda",
            "--mode",
            "synchronize",
        ]
        observer.emit(
            "ci_launch",
            checkout_sha=actual_sha,
            algorithmic_source_base=source_base,
            generated_version=version.read_text() if version.exists() else None,
            aca_package_root=str(aca_root),
            command=command,
        )
        result = subprocess.run(command, cwd=checkout, check=False)
    except Exception as error:
        try:
            observer.emit("ci_error", error_type=type(error).__name__, error=str(error))
        except Exception as receipt_error:
            error.add_note(f"CI error receipt unavailable: {receipt_error}")
        raise
    else:
        observer.emit("ci_complete", exit_code=result.returncode)
        return result.returncode


def _installed_aca_root() -> Path:
    """Locate the installed caller package without executing its module."""
    package = importlib.util.find_spec("aca_model")
    if package is None or package.origin is None:
        raise RuntimeError("The benchmark environment must contain aca_model.")
    return Path(package.origin).resolve().parent


if __name__ == "__main__":
    raise SystemExit(main())
