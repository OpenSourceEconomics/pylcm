"""Publish ASV benchmark results to the OpenSourceEconomics.github.io repo.

Usage: pixi run asv-run-and-publish-main

Downloads previous results from the org site, merges them with the new run,
generates the HTML dashboard via ``asv publish``, then pushes everything back.
This is intended for the main branch only — PR branches should use
``asv-run-and-pr-comment`` instead.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_ORG_REPO = "git@github.com:OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_BRANCH = "main"
_SITE_DIR = Path(".benchmark-site")
_SUBDIR = "pylcm-benchmarks"


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

    subprocess.run(["asv", "publish"], check=True)
    _patch_html_title(html_dir / "index.html")

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
