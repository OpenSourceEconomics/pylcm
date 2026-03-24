"""Post benchmark results as a PR comment via ``gh``.

Usage: pixi run asv-pr-comment

Compares HEAD against the merge-base with main (if local results exist for it)
and posts a formatted markdown comment on the current pull request.  If no
merge-base results are available, posts raw benchmark numbers instead.

The comment uses the ``<!-- benchmark-check -->`` marker so the CI workflow
can verify that benchmarks have been run.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

_REPO_URL = "https://github.com/OpenSourceEconomics/pylcm"

_MARKER = "<!-- benchmark-check -->"
_RESULTS_DIR = Path(".asv/results")

_CLASS_DISPLAY = {
    "MahlerYum": "Mahler-Yum",
    "Mortality": "Mortality",
    "PrecautionarySavingsSolve": "Precautionary Savings - Solve",
    "PrecautionarySavingsSimulate": ("Precautionary Savings - Simulate"),
    "PrecautionarySavingsSimulateWithSolve": (
        "Precautionary Savings - Solve & Simulate"
    ),
    "PrecautionarySavingsSimulateWithSolveIrreg": (
        "Precautionary Savings - Solve & Simulate (irreg)"
    ),
}

_METHOD_DISPLAY = {
    "time_execution": "execution time",
    "peakmem_execution": "peak mem usage",
    "track_compilation_time": "compilation time",
}

_METHOD_SORT = {
    name: i
    for i, name in enumerate(
        ("time_execution", "track_compilation_time", "peakmem_execution")
    )
}

_CLASS_SORT = {name: i for i, name in enumerate(_CLASS_DISPLAY)}

_ALERT_RATIO = 1.10

_DATA_ROW_RE = re.compile(
    r"^\|\s*[-+~x]?\s*"  # | change_indicator
    r"\|\s*(\S+)\s*"  # | before_value
    r"\|\s*(\S+)\s*"  # | after_value
    r"\|\s*~?([\d.]+)\s*"  # | ratio (strip ~ prefix)
    r"\|\s*(.+?)\s*\|$"  # | benchmark_name |
)
_BENCH_NAME_RE = re.compile(r"(?:\w+\.)*(\w+)\.(\w+)(?:\(([^)]*)\))?$")
_HASH_RE = re.compile(r"\[(\w+)\]")


class _BenchmarkRow(NamedTuple):
    class_name: str
    method_name: str
    params: str
    before_value: str
    after_value: str
    ratio: float


def post_pr_comment() -> None:
    """Post benchmark comparison (or raw results) as a PR comment."""
    head_sha_full = _get_full_hash("HEAD")
    head_sha = head_sha_full[:8]

    machine_dir = _find_machine_dir(_RESULTS_DIR)
    if machine_dir is None:
        print("No machine directory found in .asv/results — nothing to post.")
        sys.exit(1)

    head_result_file = _ensure_head_result(machine_dir, head_sha, head_sha_full)

    comparison_md = _try_comparison(machine_dir, head_sha_full)
    processed = (
        _postprocess_comparison(comparison_md) if comparison_md is not None else None
    )

    if processed is not None:
        body = _format_comparison_comment(head_sha, processed)
    else:
        raw_md = _format_raw_results(head_result_file, head_sha)
        body = _format_raw_comment(head_sha, raw_md)

    _upsert_pr_comment(body)
    print(f"Benchmark comment posted for {head_sha}.")


def _try_comparison(
    machine_dir: Path,
    head_sha_full: str,
) -> str | None:
    """Run ``asv compare`` against the merge-base, returning markdown or None."""
    try:
        base_sha_full = subprocess.run(
            ["git", "merge-base", "main", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None

    base_sha = base_sha_full[:8]
    base_file = _find_result_file(machine_dir, base_sha)
    if base_file is None:
        print(
            f"No results for merge-base {base_sha} — "
            "will post raw numbers instead of comparison."
        )
        return None

    head_file = _find_result_file(machine_dir, head_sha_full[:8])
    if head_file is not None:
        _backfill_base_results(base_file, head_file)

    try:
        result = subprocess.run(
            [
                "asv",
                "compare",
                base_sha_full,
                head_sha_full,
                "--split",
                "--factor",
                "1.05",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _backfill_base_results(
    base_file: Path,
    head_file: Path,
) -> None:
    """Sync the base result file so ``asv compare`` finds matching benchmarks.

    Copies HEAD entries into the base when the benchmark is missing or its
    parameter set has changed (e.g. after adding/removing param values).
    Since both runs use the same ``existing:python`` environment, the
    values are comparable.

    """
    base_data: dict[str, Any] = json.loads(base_file.read_text(encoding="utf-8"))
    head_data: dict[str, Any] = json.loads(head_file.read_text(encoding="utf-8"))

    base_results = base_data.setdefault("results", {})
    head_results = head_data.get("results", {})

    updates = {
        key: head_val
        for key, head_val in head_results.items()
        if _needs_backfill(base_results.get(key), head_val)
    }

    if not updates:
        return

    base_results.update(updates)
    base_file.write_text(json.dumps(base_data, indent=4), encoding="utf-8")


def _needs_backfill(
    base_val: list[Any] | None,
    head_val: Any,
) -> bool:
    """Return True when a base entry is missing or has different params."""
    if base_val is None:
        return True
    return (
        isinstance(head_val, list)
        and len(head_val) > 1
        and isinstance(base_val, list)
        and len(base_val) > 1
        and head_val[1] != base_val[1]
    )


def _format_comparison_comment(
    head_sha: str,
    processed_md: str,
) -> str:
    """Format the full PR comment body for a comparison."""
    return "\n".join(
        [
            _MARKER,
            f"<!-- head-sha:{head_sha} -->",
            "",
            "### Benchmark comparison (main \u2192 HEAD)",
            "",
            processed_md,
        ]
    )


def _postprocess_comparison(raw: str) -> str | None:
    """Parse ASV compare output and reformat as a grouped benchmark table.

    Return ``None`` when no rows with numeric ratios are found (e.g. all
    benchmarks were renamed between base and HEAD, producing only ``n/a``
    ratios).

    """
    rows, hashes = _parse_comparison_rows(raw)

    if not rows:
        return None

    parts: list[str] = []

    if len(hashes) >= 2:
        base_link = f"[`{hashes[0]}`]({_REPO_URL}/commit/{hashes[0]})"
        head_link = f"[`{hashes[1]}`]({_REPO_URL}/commit/{hashes[1]})"
        parts.append(f"Comparing {base_link} (main) \u2192 {head_link} (HEAD)")
        parts.append("")

    parts.append(_build_grouped_table(rows))

    return "\n".join(parts)


def _parse_comparison_rows(
    raw: str,
) -> tuple[list[_BenchmarkRow], list[str]]:
    """Parse ASV compare text into structured rows and commit hashes."""
    rows: list[_BenchmarkRow] = []
    hashes: list[str] = []

    for line in raw.splitlines():
        hash_matches = _HASH_RE.findall(line)
        if hash_matches and not hashes:
            hashes = hash_matches
            continue

        row_match = _DATA_ROW_RE.match(line)
        if not row_match:
            continue

        before_val, after_val, ratio_str, bench_name = row_match.groups()

        name_match = _BENCH_NAME_RE.match(bench_name)
        if not name_match:
            continue

        class_name, method_name, params = name_match.groups()

        if class_name not in _CLASS_DISPLAY and method_name not in _METHOD_DISPLAY:
            continue

        rows.append(
            _BenchmarkRow(
                class_name=class_name,
                method_name=method_name,
                params=params or "",
                before_value=before_val,
                after_value=after_val,
                ratio=float(ratio_str),
            )
        )

    return rows, hashes


def _build_grouped_table(rows: list[_BenchmarkRow]) -> str:
    """Build a grouped markdown table from parsed benchmark rows."""
    groups: dict[tuple[str, str], list[_BenchmarkRow]] = {}
    for row in rows:
        key = (row.class_name, row.params)
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(
            key=lambda r: _METHOD_SORT.get(r.method_name, len(_METHOD_SORT))
        )

    sorted_keys = sorted(
        groups,
        key=lambda k: (
            _CLASS_SORT.get(k[0], len(_CLASS_SORT)),
            k[1],
        ),
    )

    lines = [
        "| Benchmark | Statistic | before | after | Ratio | Alert |",
        "|---|---|---|---|---|---|",
    ]

    for class_name, params in sorted_keys:
        display_name = _CLASS_DISPLAY.get(class_name, class_name)
        if params:
            display_name = f"{display_name} ({params})"

        for i, row in enumerate(groups[(class_name, params)]):
            bench_col = display_name if i == 0 else ""
            stat_col = _METHOD_DISPLAY.get(row.method_name, row.method_name)
            alert = "\u274c" if row.ratio > _ALERT_RATIO else ""
            lines.append(
                f"| {bench_col} | {stat_col} "
                f"| {row.before_value} | {row.after_value} "
                f"| {row.ratio:.2f} | {alert} |"
            )

    return "\n".join(lines)


def _format_raw_results(result_file: Path, head_sha: str) -> str:
    """Extract a grouped summary from an ASV results JSON file."""
    entries = _parse_raw_entries(result_file)

    if not entries:
        return "No benchmark results found."

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for class_name, method_name, params, value in entries:
        groups.setdefault((class_name, params), []).append((method_name, value))

    for group in groups.values():
        group.sort(key=lambda x: _METHOD_SORT.get(x[0], len(_METHOD_SORT)))

    sorted_keys = sorted(
        groups,
        key=lambda k: (
            _CLASS_SORT.get(k[0], len(_CLASS_SORT)),
            k[1],
        ),
    )

    lines = [
        f"| Benchmark ({head_sha}) | Statistic | Value |",
        "|---|---|---|",
    ]

    for class_name, params in sorted_keys:
        display_name = _CLASS_DISPLAY.get(class_name, class_name)
        if params:
            display_name = f"{display_name} ({params})"

        for i, (method_name, value) in enumerate(groups[(class_name, params)]):
            bench_col = display_name if i == 0 else ""
            stat_col = _METHOD_DISPLAY.get(method_name, method_name)
            lines.append(f"| {bench_col} | {stat_col} | {value} |")

    return "\n".join(lines)


def _parse_raw_entries(
    result_file: Path,
) -> list[tuple[str, str, str, str]]:
    """Parse an ASV result JSON into (class, method, params, value) tuples."""
    data: dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
    results: dict[str, list[Any]] = data.get("results", {})

    entries: list[tuple[str, str, str, str]] = []

    for bench_name, values in results.items():
        if not values or values[0] is None:
            continue

        name_match = _BENCH_NAME_RE.match(bench_name)
        if not name_match:
            continue

        class_name, method_name, _ = name_match.groups()
        raw_values = values[0]
        params_list = values[1] if len(values) > 1 else []

        if params_list:
            for idx, combo_str in enumerate(_expand_params(params_list)):
                if idx < len(raw_values) and raw_values[idx] is not None:
                    entries.append(
                        (
                            class_name,
                            method_name,
                            combo_str,
                            _format_value(method_name, raw_values[idx]),
                        )
                    )
        elif isinstance(raw_values, list) and len(raw_values) == 1:
            entries.append(
                (
                    class_name,
                    method_name,
                    "",
                    _format_value(method_name, raw_values[0]),
                )
            )

    return entries


def _expand_params(params: list[list[str]]) -> list[str]:
    """Expand parameter lists into comma-separated combination strings."""
    if not params:
        return []
    if len(params) == 1:
        return [str(v) for v in params[0]]
    combos = [str(v) for v in params[0]]
    for dim in params[1:]:
        combos = [f"{c}, {v}" for c in combos for v in dim]
    return combos


def _format_value(bench_name: str, value: float) -> str:
    """Format a benchmark value with appropriate units."""
    if "peakmem" in bench_name:
        if value >= 1e9:
            return f"{value / 1e9:.2f} GB"
        return f"{value / 1e6:.0f} MB"
    if "compilation_time" in bench_name or "track" in bench_name:
        return f"{value:.2f} s"
    if value >= 1.0:
        return f"{value:.3f} s"
    if value >= 0.001:
        return f"{value * 1000:.1f} ms"
    return f"{value * 1e6:.1f} \u00b5s"


def _format_raw_comment(head_sha: str, raw_md: str) -> str:
    """Format the full PR comment body for raw results (no baseline)."""
    return "\n".join(
        [
            _MARKER,
            f"<!-- head-sha:{head_sha} -->",
            "",
            "### Benchmark results (HEAD only \u2014 no baseline comparison available)",
            "",
            raw_md,
            "",
            "*No merge-base results found locally. "
            "Run benchmarks on main first for a comparison.*",
        ]
    )


def _upsert_pr_comment(body: str) -> None:
    """Create or update the benchmark PR comment via ``gh``."""
    pr_number = _get_current_pr_number()
    if pr_number is None:
        print("No open PR found for the current branch. Skipping comment.")
        sys.exit(0)

    existing_id = _find_marker_comment(pr_number)

    if existing_id is not None:
        _run_gh(
            [
                "gh",
                "api",
                "--method",
                "PATCH",
                f"repos/{{owner}}/{{repo}}/issues/comments/{existing_id}",
                "-f",
                f"body={body}",
            ],
            check=True,
        )
    else:
        _run_gh(
            [
                "gh",
                "pr",
                "comment",
                str(pr_number),
                "--body",
                body,
            ],
            check=True,
        )


def _run_gh(
    cmd: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command, raising a clear error if gh is missing."""
    try:
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        print(
            "The GitHub CLI (gh) is not installed. "
            "Install it from https://cli.github.com/ and run `gh auth login`."
        )
        sys.exit(1)


def _get_current_pr_number() -> int | None:
    """Return the PR number for the current branch, or None.

    Checks the ``GITHUB_PR_NUMBER`` environment variable first (set by the
    benchmark-pr workflow), falling back to ``gh pr view`` for local use.
    """
    env_number = os.environ.get("GITHUB_PR_NUMBER")
    if env_number:
        try:
            return int(env_number)
        except ValueError:
            pass

    result = _run_gh(
        [
            "gh",
            "pr",
            "view",
            "--json",
            "number",
            "--jq",
            ".number",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _find_marker_comment(pr_number: int) -> int | None:
    """Find the comment ID with our marker on the given PR, or None."""
    result = _run_gh(
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments",
            "--paginate",
            "--jq",
            f'[.[] | select(.body | contains("{_MARKER}"))][0].id',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _get_full_hash(ref: str) -> str:
    """Return the full hash for a git ref."""
    return subprocess.run(
        ["git", "rev-parse", ref],
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


def _find_result_file(
    machine_dir: Path,
    short_hash: str,
) -> Path | None:
    """Return the first result file matching the short hash, or None."""
    matches = [
        p for p in machine_dir.glob(f"{short_hash}*.json") if "-compare" not in p.name
    ]
    return matches[0] if matches else None


def _find_latest_result_file(machine_dir: Path) -> Path | None:
    """Return the most recently modified result file, or None.

    Fallback when ``--set-commit-hash`` did not produce a file matching
    HEAD.  ASV may store results under a different commit hash depending
    on how it resolves the configured branches.

    """
    candidates = [
        p
        for p in machine_dir.glob("*.json")
        if p.name != "machine.json" and "-compare" not in p.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ensure_head_result(
    machine_dir: Path,
    head_sha: str,
    head_sha_full: str,
) -> Path:
    """Ensure a result file exists under HEAD's hash.

    ASV ignores ``--set-commit-hash`` when iterating over configured
    branches, storing results under the main-branch commit instead.
    When no file matches HEAD, copy the most recent result file and
    retag it so ``asv compare`` can find two distinct commits.

    """
    existing = _find_result_file(machine_dir, head_sha)
    if existing is not None:
        return existing

    latest = _find_latest_result_file(machine_dir)
    if latest is None:
        print("No ASV results found. Run `asv-run` first.")
        sys.exit(1)

    old_prefix = latest.name.split("-", 1)[0]
    new_path = latest.parent / latest.name.replace(old_prefix, head_sha, 1)

    data: dict[str, Any] = json.loads(latest.read_text(encoding="utf-8"))
    data["commit_hash"] = head_sha_full
    new_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    return new_path


if __name__ == "__main__":
    post_pr_comment()
