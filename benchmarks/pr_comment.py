"""Post benchmark results as a PR comment via ``gh``.

Usage: pixi run asv-pr-comment

Compares HEAD against the merge-base with main (if local results exist for it)
and posts a formatted markdown comment on the current pull request.  If no
merge-base results are available, posts raw benchmark numbers instead.

The comment uses the ``<!-- benchmark-check -->`` marker so the CI workflow
can verify that benchmarks have been run.
"""

import json
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
        "Precautionary Savings - Simulate with Solve"
    ),
    "PrecautionarySavingsGridLookup": ("Precautionary Savings - Grid Lookup"),
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

_DATA_ROW_RE = re.compile(r"^\s*(?:[-+x]\s+)?(\S+)\s+(\S+)\s+([\d.]+)\s+(\S+)\s*$")
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

    head_result_file = _find_result_file(machine_dir, head_sha)
    if head_result_file is None:
        print(f"No ASV results for HEAD ({head_sha}). Run `asv-run` first.")
        sys.exit(1)

    comparison_md = _try_comparison(machine_dir, head_sha_full)
    if comparison_md is not None:
        body = _format_comparison_comment(head_sha, comparison_md)
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
    if not list(machine_dir.glob(f"{base_sha}*.json")):
        print(
            f"No results for merge-base {base_sha} — "
            "will post raw numbers instead of comparison."
        )
        return None

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


def _format_comparison_comment(
    head_sha: str,
    comparison_md: str,
) -> str:
    """Format the full PR comment body for a comparison."""
    return "\n".join(
        [
            _MARKER,
            f"<!-- head-sha:{head_sha} -->",
            "",
            "### Benchmark comparison (main \u2192 HEAD)",
            "",
            _postprocess_comparison(comparison_md),
        ]
    )


def _postprocess_comparison(raw: str) -> str:
    """Parse ASV compare output and reformat as a grouped benchmark table."""
    rows, hashes = _parse_comparison_rows(raw)

    if not rows:
        return re.sub(
            r"\[(\w+)\]\s*<[^>]+>",
            lambda m: f"[`{m.group(1)}`]({_REPO_URL}/commit/{m.group(1)})",
            raw,
        )

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
    """Extract a readable summary from an ASV results JSON file."""
    data: dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
    results: dict[str, list[Any]] = data.get("results", {})

    lines: list[str] = []
    for bench_name, values in sorted(results.items()):
        if not values or values[0] is None:
            continue
        raw_values = values[0]
        params = values[1] if len(values) > 1 else []

        if bench_name.startswith("bench_") or "." in bench_name:
            short_name = (
                bench_name.rsplit(".", 1)[-1] if "." in bench_name else bench_name
            )
        else:
            short_name = bench_name

        if params:
            param_combos = _expand_params(params)
            for idx, combo_str in enumerate(param_combos):
                if idx < len(raw_values) and raw_values[idx] is not None:
                    lines.append(
                        f"| {short_name}({combo_str}) | "
                        f"{_format_value(short_name, raw_values[idx])}"
                        " |"
                    )
        elif isinstance(raw_values, list) and len(raw_values) == 1:
            lines.append(
                f"| {short_name} | {_format_value(short_name, raw_values[0])} |"
            )

    header = f"| Benchmark ({head_sha}) | Value |\n|---|---|"
    return header + "\n" + "\n".join(lines) if lines else "No benchmark results found."


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
    """Return the PR number for the current branch, or None."""
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


if __name__ == "__main__":
    post_pr_comment()
