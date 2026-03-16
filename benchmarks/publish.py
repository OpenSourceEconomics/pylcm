"""Publish benchmark results to the gh-pages branch.

Usage: pixi run benchmarks-publish

Reads the latest saved benchmark JSON from .benchmarks/{machine_hash}/,
converts it to the github-action-benchmark format, and pushes it to the
gh-pages branch along with the raw results.
"""

import contextlib
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def _machine_hash() -> str:
    """Stable hash from CPU model + JAX backend + device."""
    import jax

    parts = [platform.processor(), jax.default_backend()]
    with contextlib.suppress(RuntimeError):
        parts.append(str(jax.devices()[0]))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


_REPO_URL = "git@github.com:OpenSourceEconomics/pylcm.git"
_GH_PAGES_BRANCH = "gh-pages"
_SUITE_NAME = "PyLCM Benchmarks"
_MAX_DASHBOARD_ENTRIES = 200

_INITIAL_INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PyLCM Benchmarks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
  <div id="main"></div>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="dev/bench/data.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/benchmark-action/github-action-benchmark@v1/dev/bench/index.js"></script>
</body>
</html>
"""

_INITIAL_DATA_JS = 'window.BENCHMARK_DATA = {"entries": {}};\n'

_INITIAL_CONFIG = {"required_machines": []}


def publish() -> None:
    """Publish the latest benchmark results to gh-pages."""
    machine = _machine_hash()
    benchmark_dir = Path(f".benchmarks/{machine}")

    json_files = sorted(benchmark_dir.rglob("*.json"), key=lambda p: p.name)
    if not json_files:
        msg = (
            f"No benchmark results in {benchmark_dir}. "
            "Run `pixi run benchmarks-save` first."
        )
        raise FileNotFoundError(msg)

    latest = json_files[-1]
    data = json.loads(latest.read_text())

    commit_sha = data["commit_info"]["id"]
    commit_sha_short = commit_sha[:12]
    branch = data["commit_info"]["branch"]
    timestamp = data["commit_info"]["time"]

    print(
        f"Publishing benchmarks for {commit_sha_short} "
        f"(branch: {branch}, machine: {machine})"
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _clone_or_init_gh_pages(tmp_path)
        _write_result(tmp_path, machine, commit_sha_short, latest)
        _update_manifest(tmp_path, machine, commit_sha, branch, timestamp)
        _update_dashboard_data(tmp_path, data, commit_sha, timestamp)
        _commit_and_push(tmp_path, commit_sha_short, machine)

    print("Done.")


def _clone_or_init_gh_pages(tmp_path: Path) -> None:
    """Clone the gh-pages branch, or create an orphan branch if it doesn't exist."""
    result = subprocess.run(
        ["git", "ls-remote", "--heads", _REPO_URL, _GH_PAGES_BRANCH],
        capture_output=True,
        text=True,
        check=True,
    )

    if result.stdout.strip():
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                _GH_PAGES_BRANCH,
                "--depth",
                "1",
                _REPO_URL,
                str(tmp_path),
            ],
            check=True,
        )
    else:
        print("gh-pages branch does not exist, creating it...")
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, str(tmp_path)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--orphan", _GH_PAGES_BRANCH],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            ["git", "rm", "-rf", "."],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        _init_gh_pages_content(tmp_path)


def _init_gh_pages_content(tmp_path: Path) -> None:
    """Create initial gh-pages content."""
    (tmp_path / "index.html").write_text(_INITIAL_INDEX_HTML)

    dev_bench = tmp_path / "dev" / "bench"
    dev_bench.mkdir(parents=True)
    (dev_bench / "data.js").write_text(_INITIAL_DATA_JS)

    (tmp_path / "benchmark-config.json").write_text(
        json.dumps(_INITIAL_CONFIG, indent=2) + "\n"
    )

    # Prevent Jekyll from ignoring underscore-prefixed files
    (tmp_path / ".nojekyll").write_text("")


def _write_result(
    tmp_path: Path,
    machine_hash: str,
    commit_sha_short: str,
    source_json: Path,
) -> None:
    """Copy the benchmark JSON to results/{machine_hash}/{sha}.json."""
    results_dir = tmp_path / "results" / machine_hash
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, results_dir / f"{commit_sha_short}.json")


def _update_manifest(
    tmp_path: Path,
    machine_hash: str,
    commit_sha: str,
    branch: str,
    timestamp: str,
) -> None:
    """Append an entry to the manifest for this machine."""
    manifest_path = tmp_path / "results" / machine_hash / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = []

    # Don't duplicate entries for the same commit
    if not any(entry["commit_sha"] == commit_sha for entry in manifest):
        manifest.append(
            {
                "commit_sha": commit_sha,
                "branch": branch,
                "timestamp": timestamp,
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _update_dashboard_data(
    tmp_path: Path,
    benchmark_data: dict,
    commit_sha: str,
    timestamp: str,
) -> None:
    """Convert pytest-benchmark results to github-action-benchmark format."""
    data_js_path = tmp_path / "dev" / "bench" / "data.js"
    data_js_path.parent.mkdir(parents=True, exist_ok=True)

    if data_js_path.exists():
        raw = data_js_path.read_text()
        # Strip "window.BENCHMARK_DATA = " prefix and ";\n" suffix
        json_str = (
            raw.removeprefix("window.BENCHMARK_DATA = ").removesuffix(";\n").rstrip(";")
        )
        dashboard = json.loads(json_str)
    else:
        dashboard = {"entries": {}}

    # Build the new entry
    benches = []
    for bench in benchmark_data["benchmarks"]:
        stats = bench["stats"]
        benches.append(
            {
                "name": bench["name"],
                "unit": "s",
                "value": stats["mean"],
                "range": f"\u00b1 {stats['stddev']:.6f}",
                "extra": (
                    f"rounds: {stats['rounds']}, "
                    f"min: {stats['min']:.6f}, "
                    f"max: {stats['max']:.6f}"
                ),
            }
        )

    # Parse timestamp to epoch
    dt = datetime.fromisoformat(timestamp)
    epoch_ms = int(dt.timestamp() * 1000)

    commit_info = benchmark_data.get("commit_info", {})
    entry = {
        "commit": {
            "id": commit_sha,
            "message": commit_info.get("message", ""),
            "timestamp": timestamp,
            "author": {
                "name": commit_info.get("author", ""),
                "email": commit_info.get("author_email", ""),
            },
        },
        "date": epoch_ms,
        "benches": benches,
    }

    entries = dashboard.setdefault("entries", {})
    suite = entries.setdefault(_SUITE_NAME, [])
    suite.append(entry)

    # Trim to keep only the latest entries
    if len(suite) > _MAX_DASHBOARD_ENTRIES:
        entries[_SUITE_NAME] = suite[-_MAX_DASHBOARD_ENTRIES:]

    data_js_path.write_text(
        "window.BENCHMARK_DATA = " + json.dumps(dashboard, indent=2) + ";\n"
    )


def _commit_and_push(tmp_path: Path, commit_sha_short: str, machine_hash: str) -> None:
    """Commit and push changes to gh-pages."""
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=tmp_path,
        capture_output=True,
    )
    if result.returncode == 0:
        print("No new changes to publish.")
        return

    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            f"Publish benchmarks for {commit_sha_short} (machine: {machine_hash})",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "push", "origin", _GH_PAGES_BRANCH],
        cwd=tmp_path,
        check=True,
    )
    print(f"Pushed to {_GH_PAGES_BRANCH}.")


if __name__ == "__main__":
    publish()
