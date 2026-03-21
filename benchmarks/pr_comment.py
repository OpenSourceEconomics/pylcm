"""Post benchmark results as a PR comment via ``gh``.

Usage: pixi run asv-pr-comment

Compares HEAD against the merge-base with main (if local results exist for it)
and posts a formatted markdown comment on the current pull request.  If no
merge-base results are available, posts raw benchmark numbers instead.

The comment uses the ``<!-- benchmark-check -->`` marker so the CI workflow
can verify that benchmarks have been run.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_MARKER = "<!-- benchmark-check -->"
_RESULTS_DIR = Path(".asv/results")


def post_pr_comment() -> None:
    """Post benchmark comparison (or raw results) as a PR comment."""
    head_sha = _get_short_hash("HEAD")
    head_sha_full = _get_full_hash("HEAD")

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


def _try_comparison(machine_dir: Path, head_sha_full: str) -> str | None:
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


def _format_comparison_comment(head_sha: str, comparison_md: str) -> str:
    """Format the full PR comment body for a comparison."""
    return "\n".join(
        [
            _MARKER,
            f"<!-- head-sha:{head_sha} -->",
            "",
            "### Benchmark comparison (main → HEAD)",
            "",
            "```",
            comparison_md,
            "```",
        ]
    )


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
            # Parametrized benchmark — list each variant
            param_combos = _expand_params(params)
            for idx, combo_str in enumerate(param_combos):
                if idx < len(raw_values) and raw_values[idx] is not None:
                    lines.append(
                        f"| {short_name}({combo_str}) | "
                        f"{_format_value(short_name, raw_values[idx])} |"
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
    # Cartesian product for multi-dimensional params
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
    if "warmup" in bench_name or "track" in bench_name:
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
            "### Benchmark results (HEAD only — no baseline comparison available)",
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
        subprocess.run(
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
        subprocess.run(
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


def _get_current_pr_number() -> int | None:
    """Return the PR number for the current branch, or None."""
    result = subprocess.run(
        ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
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
    result = subprocess.run(
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


def _get_short_hash(ref: str) -> str:
    """Return the short (8-char) hash for a git ref."""
    return subprocess.run(
        ["git", "rev-parse", "--short=8", ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


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


def _find_result_file(machine_dir: Path, short_hash: str) -> Path | None:
    """Return the first result file matching the short hash, or None."""
    matches = [
        p for p in machine_dir.glob(f"{short_hash}*.json") if "-compare" not in p.name
    ]
    return matches[0] if matches else None


if __name__ == "__main__":
    post_pr_comment()
