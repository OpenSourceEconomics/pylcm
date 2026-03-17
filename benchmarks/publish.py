"""Publish benchmark results to the OpenSourceEconomics.github.io repo.

Usage: pixi run benchmarks-publish

Reads the latest saved benchmark JSON from .benchmarks/{machine_hash}/,
converts it to a Plotly dashboard, and pushes it to the org site repo
under pylcm-benchmarks/. The dashboard is at open-econ.org/pylcm-benchmarks/.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

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
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@2"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    .chart-box { margin: 1rem 0; max-width: 900px; }
    details { margin: 0.5rem 0; }
    summary { cursor: pointer; font-weight: 600; }
    p.meta { color: #666; font-size: 0.9rem; }
    p.subtitle { color: #888; font-size: 0.85rem; margin: 0.2rem 0 0.5rem; }
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
    var GH = 'https://github.com/OpenSourceEconomics/pylcm/commit/';
    var DESCRIPTIONS = {
      'test_solve': 'Precautionary savings model \\u2014 solve only',
      'test_grid_lookup': 'Precautionary savings \\u2014 lin vs irreg grids',
      'test_simulate': 'Precautionary savings model \\u2014 simulation only',
      'test_mortality': 'Mortality model',
      'test_mahler_yum_2024': 'Mahler & Yum (2024) replication'
    };
    function formatTitle(group) {
      var s = group.replace(/^test_/, '');
      return s.replace(/(^|_)([a-z0-9])/g, function(m, sep, ch) {
        return (sep ? ' ' : '') + ch.toUpperCase();
      });
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
        sum.textContent = formatTitle(group);
        det.appendChild(sum);
        if (DESCRIPTIONS[group]) {
          var sub = document.createElement('p');
          sub.className = 'subtitle';
          sub.textContent = DESCRIPTIONS[group];
          det.appendChild(sub);
        }
        var plotDiv = document.createElement('div');
        plotDiv.className = 'chart-box';
        det.appendChild(plotDiv);
        box.appendChild(det);
        var commitIds = entries.map(function(e) {
          return e.commit.id || '';
        });
        var shortShas = commitIds.map(function(id) {
          return id ? id.slice(0, 8) : '?';
        });
        var dateLabels = entries.map(function(e) {
          var ts = e.commit.timestamp || '';
          if (!ts) return '?';
          var d = new Date(ts);
          var yyyy = d.getFullYear();
          var mm = String(d.getMonth() + 1).padStart(2, '0');
          var dd = String(d.getDate()).padStart(2, '0');
          return yyyy + '-' + mm + '-' + dd;
        });
        var xIndices = entries.map(function(_, i) { return i; });
        var traces = gNames.map(function(name) {
          var yVals = [];
          var hoverTexts = [];
          entries.forEach(function(e, i) {
            var b = e.benches.find(function(x) { return x.name === name; });
            yVals.push(b ? b.value * 1000 : null);
            var sha = shortShas[i];
            var url = commitIds[i] ? GH + commitIds[i] : '';
            var shaLink = url
              ? '<a href="' + url + '" target="_blank">' + sha + '</a>'
              : sha;
            if (b) {
              hoverTexts.push(
                '<b>' + name + '</b><br>' +
                (b.value * 1000).toFixed(2) + ' ms ' + b.range + '<br>' +
                b.extra + '<br>' +
                shaLink
              );
            } else {
              hoverTexts.push('');
            }
          });
          return {
            x: xIndices,
            y: yVals,
            name: name,
            mode: 'lines+markers',
            connectgaps: true,
            hoverinfo: 'text',
            hovertext: hoverTexts,
            marker: { size: 6 }
          };
        });
        var layout = {
          template: 'plotly_white',
          height: 350,
          margin: { l: 60, r: 20, t: 10, b: 80 },
          xaxis: {
            title: 'Date',
            tickangle: -45,
            tickfont: { size: 11 },
            tickvals: xIndices,
            ticktext: dateLabels
          },
          yaxis: {
            title: 'Time (ms)'
          },
          legend: {
            orientation: 'h',
            yanchor: 'top',
            y: -0.25,
            xanchor: 'center',
            x: 0.5
          },
          hovermode: 'closest'
        };
        var config = {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          displaylogo: false
        };
        Plotly.newPlot(plotDiv, traces, layout, config);
        plotDiv.on('plotly_click', function(clickData) {
          var idx = clickData.points[0].pointIndex;
          var sha = commitIds[idx];
          if (sha) window.open(GH + sha, '_blank');
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
    benchmark_root = Path(".benchmarks")

    json_files = sorted(benchmark_root.rglob("*.json"), key=lambda p: p.name)
    if not json_files:
        msg = (
            f"No benchmark results in {benchmark_root}. "
            "Run `pixi run benchmarks-save` first."
        )
        raise FileNotFoundError(msg)

    latest = json_files[-1]
    data = json.loads(latest.read_text())

    # Derive machine hash from directory structure: .benchmarks/{machine}/...
    machine = latest.relative_to(benchmark_root).parts[0]

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
    """Convert pytest-benchmark results to Plotly dashboard data."""
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

    # Deduplicate by commit SHA (keep last occurrence), then add new entry
    deduped: dict[str, dict] = {}
    for existing in suite:
        cid = existing.get("commit", {}).get("id", id(existing))
        deduped[cid] = existing
    deduped[commit_sha] = entry
    suite[:] = list(deduped.values())

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
