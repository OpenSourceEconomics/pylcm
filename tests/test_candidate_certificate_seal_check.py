"""The candidate certificate's two seal files are checked, and repaired, in one step.

The certificate pins every certified source by digest twice: in the generated
`sources.json` inventory and in the hand-maintained seal map of `direct_flow.py`.
An edit to a certified source leaves both stale, and the drift is otherwise visible
only as a test failure. `check_seals.py` reports the drift in well under a second,
names the file and both digests, and `--fix` regenerates the inventory and rewrites
the seal map so the certificate tests pass again.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.candidate_certificate.generate_sources import sha256_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "tests" / "candidate_certificate" / "check_seals.py"
_DRIFTED_SOURCE = "src/_lcm/logsum.py"


def _run(*args: str, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(_SCRIPT), "--repo-root", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def certified_tree(tmp_path: Path) -> Path:
    """A copy of everything the certificate reads, with one certified source edited."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(_REPO_ROOT / "src", tmp_path / "src", ignore=ignore)
    shutil.copytree(
        _REPO_ROOT / "tests" / "candidate_certificate",
        tmp_path / "tests" / "candidate_certificate",
        ignore=ignore,
    )
    certificate = "tests/test_grid_search_candidate_certificate.py"
    shutil.copy(_REPO_ROOT / certificate, tmp_path / certificate)
    drifted = tmp_path / _DRIFTED_SOURCE
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n# A comment moves the seal.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_the_committed_tree_passes_the_seal_check():
    """Both seal files agree with the checked-in sources."""
    result = _run(repo_root=_REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_drifted_source_is_named_with_both_digests(certified_tree: Path):
    """An edited certified source fails the check, naming the file and digests."""
    result = _run(repo_root=certified_tree)

    assert result.returncode == 1
    assert _DRIFTED_SOURCE in result.stdout
    assert sha256_file(certified_tree / _DRIFTED_SOURCE) in result.stdout
    assert "--fix" in result.stdout


def test_fix_reseals_both_files_so_the_check_passes(certified_tree: Path):
    """`--fix` rewrites the inventory and the seal map to the edited source's digest."""
    digest = sha256_file(certified_tree / _DRIFTED_SOURCE)

    fixed = _run("--fix", repo_root=certified_tree)

    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    assert _run(repo_root=certified_tree).returncode == 0
    inventory = json.loads(
        (certified_tree / "tests/candidate_certificate/sources.json").read_text()
    )
    assert {
        item["sha256"]
        for item in inventory["sources"]
        if item["path"] == _DRIFTED_SOURCE
    } == {digest}
    seal_map = (
        certified_tree / "tests/candidate_certificate/direct_flow.py"
    ).read_text()
    assert f'LOGSUM_SOURCE: "{digest}",' in seal_map


def test_fix_leaves_a_clean_tree_untouched():
    """On a tree that already passes, `--fix` rewrites nothing."""
    before = {
        path: path.read_bytes()
        for path in (
            _REPO_ROOT / "tests/candidate_certificate/sources.json",
            _REPO_ROOT / "tests/candidate_certificate/direct_flow.py",
        )
    }

    result = _run("--fix", repo_root=_REPO_ROOT)

    assert result.returncode == 0, result.stdout + result.stderr
    assert {path: path.read_bytes() for path in before} == before
