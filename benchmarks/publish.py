"""Publish ASV benchmark results to the OpenSourceEconomics.github.io repo.

Usage: pixi run asv-publish

Runs `asv publish` to generate the HTML dashboard, then copies the results
and HTML to the org site repo under pylcm-benchmarks/.
"""

import shutil
import subprocess
from pathlib import Path

_ORG_REPO = "git@github.com:OpenSourceEconomics/OpenSourceEconomics.github.io.git"
_BRANCH = "main"
_SITE_DIR = Path(".benchmark-site")
_SUBDIR = "pylcm-benchmarks"


def publish() -> None:
    """Publish benchmark results and dashboard to the org site."""
    html_dir = Path(".asv/html")
    results_dir = Path(".asv/results")

    if not html_dir.exists():
        msg = "No ASV HTML output. Run `asv publish` first."
        raise FileNotFoundError(msg)

    commit_sha_short = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    print(f"Publishing benchmarks for {commit_sha_short}")

    _patch_html_title(html_dir / "index.html")

    _ensure_site_clone()
    root = _SITE_DIR / _SUBDIR

    # Copy ASV HTML dashboard
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(html_dir, root)

    # Copy ASV results (used by CI check)
    results_dest = root / "results"
    if results_dir.exists():
        shutil.copytree(results_dir, results_dest)

    _commit_and_push(commit_sha_short)
    print("Done.")


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
