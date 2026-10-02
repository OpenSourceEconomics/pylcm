"""Publish ASV benchmark results to the OpenSourceEconomics.github.io repo.

Usage: pixi run asv-run-and-publish-main

Downloads previous results from the org site, merges them with the new run,
normalises the merged history so each machine draws one continuous series,
generates the HTML dashboard via ``asv publish``, then pushes everything back.
This is intended for the main branch only — PR branches should use
``asv-run-and-pr-comment`` instead.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from benchmarks.asv_machine import stable_ram
from benchmarks.pr_comment import display_names, display_sort_key

logger = logging.getLogger(__name__)

_ORG_REPO = "git@github.com:OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_BRANCH = "main"
_SITE_DIR = Path(".benchmark-site")
_SUBDIR = "pylcm-benchmarks"

# Separates the benchmark label from the statistic label in a dashboard title.
_TITLE_SEPARATOR = " \u2014 "

# Benchmarks renamed from an ASV-native `time_*` method to `track_execution_time`.
# Both report seconds over the same params, but `time_*` is ASV's own repeat
# statistic and `track_execution_time` the median of a few warm calls, so the
# carried-over history shows a step at the rename.
_RENAMED_BENCHMARKS = {
    f"bench_collective_household.{cls}.time_execution": (
        f"bench_collective_household.{cls}.track_execution_time"
    )
    for cls in ("CollectiveHouseholdSimulate", "ReferenceChainSolve")
}


def publish() -> None:
    """Publish benchmark results and dashboard to the org site."""
    results_dir = Path(".asv/results")
    html_dir = Path(".asv/html")

    commit_sha_short = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    print(f"Publishing benchmarks for {commit_sha_short}")

    _ensure_site_clone()
    _download_previous_results(results_dir)
    _normalise_results(results_dir)

    subprocess.run(["asv", "publish"], check=True)
    _patch_html_title(html_dir / "index.html")
    _default_x_axis_to_date(html_dir / "graphdisplay.js")
    _default_y_axis_to_log(html_dir / "graphdisplay.js")
    _log_scale_summary_thumbnails(html_dir / "summarygrid.js")
    _group_summary_grid_by_title(html_dir / "summarygrid.js")
    _title_and_order_benchmarks(html_dir / "index.json")
    _pad_sparse_graphs(html_dir / "graphs")

    _generate_comparison(results_dir)

    root = _SITE_DIR / _SUBDIR

    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(html_dir, root)

    results_dest = root / "results"
    if results_dir.exists():
        shutil.copytree(results_dir, results_dest)

    _commit_and_push(commit_sha_short)
    print("Done.")


def _download_previous_results(results_dir: Path) -> None:
    """Copy previous benchmark results from the org site into .asv/results/.

    This ensures ``asv publish`` sees the full history, not just the current run.
    Existing local results (from the current run) take precedence over downloaded
    ones.
    """
    site_results = _SITE_DIR / _SUBDIR / "results"
    if not site_results.is_dir():
        print("No previous results on org site.")
        return

    for machine_dir in site_results.iterdir():
        if not machine_dir.is_dir():
            continue

        local_machine_dir = results_dir / machine_dir.name
        local_machine_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for result_file in machine_dir.iterdir():
            dest = local_machine_dir / result_file.name
            if not dest.exists():
                shutil.copy2(result_file, dest)
                count += 1

        if count:
            print(f"Downloaded {count} previous result(s) for {machine_dir.name}")


def _normalise_results(results_dir: Path) -> None:
    """Rewrite every stored result so a machine's history is one continuous series.

    ASV starts a new graph series whenever a machine param changes, and hides a
    stored result whose version stamp differs from the current benchmark's. Both
    happen without anything about the measurement changing: the RAM the kernel
    reports drifts by kilobytes, the kernel itself is updated, and a version bump
    marks a deliberate change of workload. This rewrites, in place:

    - the machine params of every result (all but `python`) to the machine's current
      `machine.json`, with the RAM in stable whole gigabytes;
    - the version column of every result to null, which ASV treats as matching any
      version;
    - the names in `_RENAMED_BENCHMARKS` to their current names.

    Nulling versions deliberately trades ASV's guard against mixing measurement
    semantics on one line for continuity: a version bump now shows as a step on the
    same line instead of hiding all earlier history. Steps at version bumps are
    expected. Files are only rewritten when their content changes, so the pass is
    idempotent and the merged, normalised results are what gets pushed back.
    """
    for machine_json in results_dir.glob("*/machine.json"):
        machine = json.loads(machine_json.read_text(encoding="utf-8"))
        machine["ram"] = stable_ram(machine["ram"])
        _write_json_if_changed(path=machine_json, data=machine)
        current = {key: val for key, val in machine.items() if key != "version"}

        for result_file in machine_json.parent.glob("*.json"):
            if result_file.name == "machine.json" or result_file.name.endswith(
                "-compare.json"
            ):
                continue
            data = json.loads(result_file.read_text(encoding="utf-8"))
            data["params"].update(current)
            results = data["results"]
            for old, new in _RENAMED_BENCHMARKS.items():
                if old in results:
                    results.setdefault(new, results.pop(old))
            version_column = data["result_columns"].index("version")
            for entry in results.values():
                if len(entry) > version_column:
                    entry[version_column] = None
            _write_json_if_changed(path=result_file, data=data)


def _write_json_if_changed(*, path: Path, data: dict) -> None:
    """Write `data` to `path` as ASV formats JSON, unless the content is unchanged."""
    if json.loads(path.read_text(encoding="utf-8")) != data:
        path.write_text(json.dumps(data, indent=4, sort_keys=True), encoding="utf-8")


def _generate_comparison(results_dir: Path) -> None:
    """Generate a comparison JSON file against the main merge-base.

    Find the merge-base commit between main and HEAD, check if local ASV
    results exist for it, and if so run ``asv compare`` and save the output.
    This is best-effort — failures are logged but do not stop publishing.
    """
    try:
        head_sha = _get_short_hash("HEAD")
        base_sha_full = subprocess.run(
            ["git", "merge-base", "main", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        base_sha = base_sha_full[:8]

        machine_dir = _find_machine_dir(results_dir)
        if machine_dir is None:
            logger.warning("No machine directory found in %s", results_dir)
            return

        if not list(machine_dir.glob(f"{base_sha}*.json")):
            logger.warning(
                "No results for merge-base %s — skipping comparison", base_sha
            )
            return

        comparison_text = subprocess.run(
            ["asv", "compare", base_sha_full, "HEAD", "--split", "--factor", "1.05"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

        compare_data = {
            "base_commit": base_sha,
            "head_commit": head_sha,
            "base_branch": "main",
            "machine": machine_dir.name,
            "comparison": comparison_text,
        }
        out_path = machine_dir / f"{head_sha}-compare.json"
        out_path.write_text(json.dumps(compare_data, indent=2), encoding="utf-8")
        print(f"Comparison saved to {out_path.name}")

    except subprocess.CalledProcessError, OSError:
        logger.warning("Could not generate comparison — skipping", exc_info=True)


def _get_short_hash(ref: str) -> str:
    """Return the short (8-char) hash for a git ref."""
    return subprocess.run(
        ["git", "rev-parse", "--short=8", ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _find_machine_dir(results_dir: Path) -> Path | None:
    """Return the first machine directory under results_dir, or None."""
    if not results_dir.is_dir():
        return None
    for path in results_dir.iterdir():
        if path.is_dir():
            return path
    return None


def _pad_graphs_in_folder(folder: Path) -> int:
    """Pad every `bench_*.json` in `folder` to the folder-wide x-range."""
    target_revs: set[int] = set()
    for f in folder.glob("bench_*.json"):
        for entry in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(entry, list) and entry:
                target_revs.add(entry[0])
    if not target_revs:
        return 0
    endpoints = {min(target_revs), max(target_revs)}
    padded = 0
    for f in folder.glob("bench_*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        have = {e[0] for e in data if isinstance(e, list) and e}
        missing = endpoints - have
        if not missing:
            continue
        data.extend([rev, None] for rev in missing)
        data.sort(key=lambda e: e[0])
        f.write_text(json.dumps(data), encoding="utf-8")
        padded += len(missing)
    return padded


def _pad_sparse_graphs(graphs_dir: Path) -> None:
    """Pad benchmark series with `[rev, null]` entries to the full x-range.

    asv writes per-benchmark graph JSONs containing only revisions where the
    benchmark actually ran. flot's auto-fit then sizes each chart's x-axis
    to the benchmark's own range, so two recent measurements span the full
    chart width even though most of the project history has no data for
    that benchmark.

    Inject `[rev, null]` markers at the first and last revision of the folder
    when this series does not cover them. flot fits the x-axis to them and draws
    no point there, so the x-axis matches the rest of the grid. Interior
    revisions are not padded: flot breaks a line at every null, so padding them
    would cut a series wherever a sibling ran and it did not.

    Runs over the summary directory (`graphs/summary/`) and every
    per-environment leaf directory (`graphs/arch-*/.../`).
    """
    if not graphs_dir.is_dir():
        return

    total = _pad_graphs_in_folder(graphs_dir / "summary")
    for env_root in graphs_dir.iterdir():
        if env_root.name == "summary" or not env_root.is_dir():
            continue
        for leaf in {p.parent for p in env_root.rglob("bench_*.json")}:
            total += _pad_graphs_in_folder(leaf)
    if total:
        print(f"Padded sparse benchmark graphs with {total} null entries.")


def _default_x_axis_to_date(graphdisplay_js: Path) -> None:
    """Default the per-benchmark graph x-axis to the date scale.

    asv's detail-view defaults the x-axis to the revision index and switches to a
    real date axis only when the `x-axis-scale=date` URL param is present. Add an
    `else` branch to the param parser so the date scale is the default when no
    param is given. Best-effort — a parser change upstream just leaves the asv
    default in place.
    """
    if not graphdisplay_js.is_file():
        logger.warning("graphdisplay.js not found — skipping date-axis default")
        return
    anchor = "            delete params['x-axis-scale'];\n        }\n"
    replacement = (
        "            delete params['x-axis-scale'];\n"
        "        } else {\n"
        "            $('#date-scale').addClass('active');\n"
        "            date_scale = true;\n"
        "        }\n"
    )
    text = graphdisplay_js.read_text(encoding="utf-8")
    if anchor not in text:
        logger.warning(
            "x-axis-scale parser not found in graphdisplay.js — skipping date default"
        )
        return
    graphdisplay_js.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


def _default_y_axis_to_log(graphdisplay_js: Path) -> None:
    """Default the per-benchmark graph y-axis to the log scale.

    asv's detail view switches to a log y-axis only when the `y-axis-scale=log` URL
    param is present. Add an `else` branch to the param parser so log is the default
    when no param is given, and make the log toggle write `linear` when switched off,
    so that switching it off does not fall back to the new default. Best-effort — a
    parser change upstream just leaves the asv default in place.
    """
    if not graphdisplay_js.is_file():
        logger.warning("graphdisplay.js not found — skipping log-axis default")
        return
    edits = {
        "            delete params['y-axis-scale'];\n        }\n": (
            "            delete params['y-axis-scale'];\n"
            "        } else {\n"
            "            $('#log-scale').addClass('active');\n"
            "            log_scale = true;\n"
            "        }\n"
        ),
        "log_scale ? ['log']: []": "log_scale ? ['log'] : ['linear']",
    }
    text = graphdisplay_js.read_text(encoding="utf-8")
    if not all(anchor in text for anchor in edits):
        logger.warning(
            "y-axis-scale handling not found in graphdisplay.js — skipping log default"
        )
        return
    for anchor, replacement in edits.items():
        text = text.replace(anchor, replacement, 1)
    graphdisplay_js.write_text(text, encoding="utf-8")


def _log_scale_summary_thumbnails(summarygrid_js: Path) -> None:
    """Draw the front-page thumbnail graphs on a log y-axis.

    Non-positive values have no logarithm and are left out of the thumbnail.
    Best-effort — a layout change upstream just leaves the linear thumbnails.
    """
    if not summarygrid_js.is_file():
        logger.warning("summarygrid.js not found — skipping log thumbnails")
        return
    anchor = "                        ticks: [],\n                        min: 0\n"
    replacement = (
        "                        ticks: [],\n"
        "                        transform: function (v) {\n"
        "                            return v > 0 ? Math.log(v) : null;\n"
        "                        },\n"
        "                        inverseTransform: function (v) {\n"
        "                            return Math.exp(v);\n"
        "                        }\n"
    )
    text = summarygrid_js.read_text(encoding="utf-8")
    if anchor not in text:
        logger.warning("thumbnail y-axis not found in summarygrid.js — skipping")
        return
    summarygrid_js.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


def _title_and_order_benchmarks(index_json: Path) -> None:
    """Title and order the dashboard's benchmarks as the PR comparison table does.

    Each benchmark's `pretty_name`, which asv shows in the grid, the navigation and
    the detail view, becomes "<benchmark label> — <statistic label>" from the labels
    in `pr_comment`, so the dashboard and the PR table cannot drift apart. The
    benchmarks are reordered by the table's order, which puts each family's execution
    time first; the grid lays thumbnails out in this order.
    """
    data = json.loads(index_json.read_text(encoding="utf-8"))
    for name, benchmark in data["benchmarks"].items():
        benchmark["pretty_name"] = _TITLE_SEPARATOR.join(display_names(name))
    data["benchmarks"] = {
        name: data["benchmarks"][name]
        for name in sorted(data["benchmarks"], key=display_sort_key)
    }
    index_json.write_text(json.dumps(data), encoding="utf-8")


def _group_summary_grid_by_title(summarygrid_js: Path) -> None:
    """Group the front-page grid by benchmark label instead of by module.

    asv heads each group with the benchmark's module and each thumbnail with its
    `pretty_name`. Split the `pretty_name` written by `_title_and_order_benchmarks`
    instead: the benchmark label heads the group and the statistic label the
    thumbnail, mirroring the two columns of the PR comparison table. Best-effort --
    a layout change upstream just leaves asv's grouping.
    """
    if not summarygrid_js.is_file():
        logger.warning("summarygrid.js not found — skipping title grouping")
        return
    separator = json.dumps(_TITLE_SEPARATOR)
    edits = {
        "            var group = bm_name.slice(0, i);\n": (
            "            var group = bm.pretty_name ? "
            f"bm.pretty_name.split({separator})[0] : bm_name.slice(0, i);\n"
        ),
        "        var display_name = bm.pretty_name || "
        "bm.name.slice(bm.name.indexOf('.') + 1);\n": (
            "        var display_name = bm.pretty_name ? "
            f"bm.pretty_name.split({separator}).pop() : "
            "bm.name.slice(bm.name.indexOf('.') + 1);\n"
        ),
    }
    text = summarygrid_js.read_text(encoding="utf-8")
    if not all(anchor in text for anchor in edits):
        logger.warning("grid grouping not found in summarygrid.js — skipping")
        return
    for anchor, replacement in edits.items():
        text = text.replace(anchor, replacement, 1)
    summarygrid_js.write_text(text, encoding="utf-8")


def _patch_html_title(index_html: Path) -> None:
    """Replace ASV's default page title with a project-specific one."""
    text = index_html.read_text(encoding="utf-8")
    text = text.replace(
        "<title>airspeed velocity</title>",
        "<title>pylcm benchmarks</title>",
    )
    index_html.write_text(text, encoding="utf-8")


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


def _commit_and_push(commit_sha_short: str) -> None:
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
            f"pylcm: publish benchmarks for {commit_sha_short}",
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
