"""Publish benchmark results to the OpenSourceEconomics.github.io repo.

Usage: pixi run benchmarks-publish

Reads the latest saved benchmark JSON from .benchmarks/{machine_hash}/,
converts it to a Chart.js dashboard, and pushes it to the org site repo
under pylcm-benchmarks/. The dashboard is at open-econ.org/pylcm-benchmarks/.
"""

import contextlib
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _machine_hash() -> str:
    """Stable hash from CPU model + JAX backend + device."""
    import jax

    parts = [platform.processor(), jax.default_backend()]
    with contextlib.suppress(RuntimeError):
        parts.append(str(jax.devices()[0]))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


_ORG_REPO = "git@github.com:OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_BRANCH = "main"
_SITE_DIR = Path(".benchmark-site")
_SUBDIR = "pylcm-benchmarks"
_SUITE_NAME = "PyLCM Benchmarks"
_MAX_DASHBOARD_ENTRIES = 200

_INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PyLCM Benchmarks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    .chart-box { margin: 1.5rem 0; max-width: 900px; }
    canvas { max-height: 350px; }
    details { margin: 0.5rem 0; }
    summary { cursor: pointer; font-weight: 600; }
    p.meta { color: #666; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>PyLCM Benchmarks</h1>
  <p class="meta">Performance tracking across commits.</p>
  <div id="charts"></div>
  <script src="dev/bench/data.js"></script>
  <script>
  (function() {
    var data = window.BENCHMARK_DATA;
    if (!data || !data.entries) {
      document.getElementById('charts').innerHTML =
        '<p>No benchmark data available yet.</p>';
      return;
    }
    var box = document.getElementById('charts');
    Object.keys(data.entries).forEach(function(suite) {
      var entries = data.entries[suite];
      if (!entries.length) return;
      var nameSet = {};
      entries.forEach(function(e) {
        e.benches.forEach(function(b) { nameSet[b.name] = 1; });
      });
      var names = Object.keys(nameSet);
      var groups = {};
      names.forEach(function(n) {
        var g = n.replace(/\\[.*/, '');
        if (!groups[g]) groups[g] = [];
        groups[g].push(n);
      });
      Object.keys(groups).forEach(function(group) {
        var gNames = groups[group];
        var det = document.createElement('details');
        det.open = true;
        var sum = document.createElement('summary');
        sum.textContent = group;
        det.appendChild(sum);
        var div = document.createElement('div');
        div.className = 'chart-box';
        var cv = document.createElement('canvas');
        div.appendChild(cv);
        det.appendChild(div);
        box.appendChild(det);
        var labels = entries.map(function(e) {
          return e.commit.id ? e.commit.id.slice(0, 8) : '?';
        });
        var ds = gNames.map(function(name, i) {
          var hue = (i * 360 / gNames.length) % 360;
          return {
            label: name,
            data: entries.map(function(e) {
              var b = e.benches.find(
                function(x) { return x.name === name; }
              );
              return b ? b.value * 1000 : null;
            }),
            borderColor: 'hsl('+hue+',70%,50%)',
            tension: 0.2,
            spanGaps: true
          };
        });
        new Chart(cv, {
          type: 'line',
          data: { labels: labels, datasets: ds },
          options: {
            responsive: true,
            plugins: {
              legend: {
                position: 'bottom',
                labels: { boxWidth: 12, font: { size: 11 } }
              }
            },
            scales: {
              y: {
                title: { display: true, text: 'Time (ms)' },
                beginAtZero: false
              },
              x: {
                title: { display: true, text: 'Commit' }
              }
            }
          }
        });
      });
    });
  })();
  </script>
</body>
</html>
"""

_INITIAL_DATA_JS = 'window.BENCHMARK_DATA = {"entries": {}};\n'

_INITIAL_CONFIG = {"required_machines": []}


def publish() -> None:
    """Publish the latest benchmark results to the org site."""
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

    _ensure_site_clone()
    root = _SITE_DIR / _SUBDIR
    _init_benchmark_dir(root)
    (root / "index.html").write_text(_INDEX_HTML)
    _write_result(root, machine, commit_sha_short, latest)
    _update_manifest(root, machine, commit_sha, branch, timestamp)
    _update_dashboard_data(root, data, commit_sha, timestamp)
    _commit_and_push(commit_sha_short, machine)

    print("Done.")


def _ensure_site_clone() -> None:
    """Clone the org site repo or pull latest if already cloned."""
    if (_SITE_DIR / ".git").exists():
        subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=_SITE_DIR,
            check=True,
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                _BRANCH,
                _ORG_REPO,
                str(_SITE_DIR),
            ],
            check=True,
        )


def _init_benchmark_dir(root: Path) -> None:
    """Create the benchmark subdirectory with initial content if needed."""
    if root.exists():
        return

    root.mkdir(parents=True)

    dev_bench = root / "dev" / "bench"
    dev_bench.mkdir(parents=True)
    (dev_bench / "data.js").write_text(_INITIAL_DATA_JS)

    (root / "index.html").write_text(_INDEX_HTML)

    (root / "benchmark-config.json").write_text(
        json.dumps(_INITIAL_CONFIG, indent=2) + "\n"
    )


def _write_result(
    root: Path,
    machine_hash: str,
    commit_sha_short: str,
    source_json: Path,
) -> None:
    """Copy the benchmark JSON to results/{machine_hash}/{sha}.json."""
    results_dir = root / "results" / machine_hash
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, results_dir / f"{commit_sha_short}.json")


def _update_manifest(
    root: Path,
    machine_hash: str,
    commit_sha: str,
    branch: str,
    timestamp: str,
) -> None:
    """Append an entry to the manifest for this machine."""
    manifest_path = root / "results" / machine_hash / "manifest.json"

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
    root: Path,
    benchmark_data: dict,
    commit_sha: str,
    timestamp: str,
) -> None:
    """Convert pytest-benchmark results to Chart.js dashboard data."""
    data_js_path = root / "dev" / "bench" / "data.js"
    data_js_path.parent.mkdir(parents=True, exist_ok=True)

    if data_js_path.exists():
        raw = data_js_path.read_text()
        json_str = (
            raw.removeprefix("window.BENCHMARK_DATA = ").removesuffix(";\n").rstrip(";")
        )
        dashboard = json.loads(json_str)
    else:
        dashboard = {"entries": {}}

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

    if len(suite) > _MAX_DASHBOARD_ENTRIES:
        entries[_SUITE_NAME] = suite[-_MAX_DASHBOARD_ENTRIES:]

    data_js_path.write_text(
        "window.BENCHMARK_DATA = " + json.dumps(dashboard, indent=2) + ";\n"
    )


def _commit_and_push(commit_sha_short: str, machine_hash: str) -> None:
    """Commit and push changes to the org site repo."""
    subprocess.run(
        ["git", "add", _SUBDIR],
        cwd=_SITE_DIR,
        check=True,
    )

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=_SITE_DIR,
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
            f"pylcm: publish benchmarks for "
            f"{commit_sha_short} (machine: {machine_hash})",
        ],
        cwd=_SITE_DIR,
        check=True,
    )
    subprocess.run(
        ["git", "push", "origin", _BRANCH],
        cwd=_SITE_DIR,
        check=True,
    )
    print("Pushed to org site.")


if __name__ == "__main__":
    publish()
