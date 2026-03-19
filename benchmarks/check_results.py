"""Check whether benchmark results exist for the current PR commits.

Usage:
    python benchmarks/check_results.py [--org-site-dir DIR] [--pr-commits SHA,SHA,...]

Exit codes:
    0 — results are current, stale, or check is skipped
    1 — results are missing for all PR commits on at least one required machine
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ORG_REPO = "https://github.com/OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_SUBDIR = "pylcm-benchmarks"


def _get_pr_commits() -> list[str]:
    """Return commit SHAs from origin/main..HEAD, newest first."""
    result = subprocess.run(
        ["git", "log", "--format=%H", "origin/main..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.strip().splitlines() if line]


def _clone_org_site(target: Path) -> None:
    """Shallow-clone the org site repo into *target*."""
    subprocess.run(
        ["git", "clone", "--depth", "1", _ORG_REPO, str(target)],
        check=True,
        capture_output=True,
    )


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _get_required_machines(org_site_dir: Path) -> list[str] | None:
    """Return required machine hashes, or None to skip the check."""
    benchmarks_root = org_site_dir / _SUBDIR

    if not benchmarks_root.is_dir():
        print("Benchmark directory does not exist on org site.")
        return None

    config_path = benchmarks_root / "benchmark-config.json"
    if not config_path.exists():
        print("No benchmark-config.json found — skipping check.")
        return None

    config = _load_json(config_path)
    machines = config.get("required_machines", [])
    if not machines:
        print("No required machines configured — skipping check.")
        return None

    return machines


def _check_machine(
    benchmarks_root: Path,
    machine: str,
    pr_commits: list[str],
    head_sha: str,
) -> str:
    """Check one machine's results. Return "current", "stale", or "missing"."""
    manifest_path = benchmarks_root / "results" / machine / "manifest.json"

    if not manifest_path.exists():
        print(f"Machine `{machine}`: no results.")
        return "missing"

    manifest = _load_json(manifest_path)
    manifest_shas = {entry["commit_sha"] for entry in manifest}

    found_sha = next((sha for sha in pr_commits if sha in manifest_shas), None)

    if found_sha is None:
        print(f"Machine `{machine}`: no results found for any PR commit.")
        return "missing"

    if found_sha == head_sha:
        print(f"Machine `{machine}`: results are current (HEAD).")
        return "current"

    short = found_sha[:12]
    print(f"Machine `{machine}`: results for `{short}`, not HEAD.")
    result = subprocess.run(
        ["git", "log", "--format=- %h %s", f"{found_sha}..HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        print(f"  Commits since:\n{result.stdout.rstrip()}")
    return "stale"


def check(org_site_dir: Path, pr_commits: list[str]) -> str:
    """Check benchmark status and print details.

    Returns:
        One of "current", "stale", "missing", or "skip".

    """
    machines = _get_required_machines(org_site_dir)
    if machines is None or not pr_commits:
        if not pr_commits:
            print("No PR commits found — skipping check.")
        return "skip"

    head_sha = pr_commits[0]
    benchmarks_root = org_site_dir / _SUBDIR
    statuses = [
        _check_machine(benchmarks_root, m, pr_commits, head_sha) for m in machines
    ]

    if "missing" in statuses:
        return "missing"
    if "stale" in statuses:
        return "stale"
    return "current"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--org-site-dir",
        type=Path,
        default=None,
        help="Path to org site clone (default: temp clone).",
    )
    parser.add_argument(
        "--pr-commits",
        default=None,
        help="Comma-separated SHAs (default: read from git via "
        "`git log origin/main..HEAD`).",
    )
    args = parser.parse_args()

    pr_commits = args.pr_commits.split(",") if args.pr_commits else _get_pr_commits()

    if args.org_site_dir is not None:
        status = check(args.org_site_dir, pr_commits)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            org_site_dir = Path(tmpdir) / "org-site"
            print(f"Cloning org site to {org_site_dir} ...")
            _clone_org_site(org_site_dir)
            status = check(org_site_dir, pr_commits)

    print(f"\nStatus: {status}")
    sys.exit(1 if status == "missing" else 0)


if __name__ == "__main__":
    main()
