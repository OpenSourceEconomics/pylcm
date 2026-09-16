"""Portability controls for the source-only certificate corridor.

The two exhaustive source-certificate campaigns
(`tests/test_simulation_candidate_program_certificate.py` and
`tests/test_uniform_process_grid_certificate.py`) verify Python AST, string,
path and hash properties of the certified corridor. Nothing in them executes
JAX or depends on the working floating-point precision, so the full mutation
population runs once, on the Linux/Python-3.14 source-contract lane, instead of
four times across the numerical matrix.

That consolidation is only sound if the *source-I/O and path* layer really is
platform-independent, which is what this module checks. It stays on every
general lane -- Windows and macOS included -- and is deliberately tiny: four
controls, about five seconds, no mutation population. Removing the two
campaigns from those lanes therefore loses no platform signal, because the
platform-sensitive part of what they exercise is asserted here:

1. line endings do not change a source hash (`sha256_file` normalises),
2. a corridor nested inside another directory resolves through `repo_root /
   relative` with the host's own path separator,
3. the actual corridor is accepted clean, and
4. one independently resealed actual-corridor mutation is still rejected, and
   names its own offending path.

Controls 3 and 4 are the smallest possible acceptance/rejection pair. The
exhaustive per-mutation population is *not* duplicated here; it lives on the
source-contract lane, whose authority is the source bytes and the Python
version rather than the runner's OS.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file

_REPO_ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def clean_corridor_sources() -> tuple[str, ...]:
    """Check the frozen checkout once and retain its corridor path tuple."""
    clean = direct_flow.verify_direct_candidate_flow(repo_root=_REPO_ROOT)
    assert clean["ok"], clean["errors"]
    return tuple(clean["certified_corridor_sources"])


def _copy_corridor(*, sources: tuple[str, ...], destination_root: Path) -> None:
    """Copy every corridor source into a private tree under `destination_root`."""
    for relative in sources:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(_REPO_ROOT / relative, destination)


def test_source_hashing_is_canonical_across_line_endings(*, tmp_path: Path) -> None:
    """The same logical source hashes identically whether stored LF or CRLF.

    `sha256_file` reads with `Path.read_text`, whose universal-newline handling
    turns CRLF into LF before hashing. This is what lets a seal recorded on a
    Linux checkout match the same file checked out on Windows with
    `core.autocrlf` translating it -- the property the source-contract lane
    relies on when it verifies bytes on one OS for all of them.
    """
    body = "def f():\n    return 1\n"
    lf_path = tmp_path / "lf.py"
    crlf_path = tmp_path / "crlf.py"
    lf_path.write_bytes(body.encode("utf-8"))
    crlf_path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))

    assert lf_path.read_bytes() != crlf_path.read_bytes()
    assert sha256_file(lf_path) == sha256_file(crlf_path)


def test_corridor_resolves_inside_a_nested_repository_checkout(
    *,
    clean_corridor_sources: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corridor nested below other directories resolves and verifies clean.

    The seal map is keyed by POSIX-relative paths, while the lookup is
    `repo_root / relative` on the host's own separator. Nesting the checkout
    two levels deep is what would expose a separator or drive-relative
    assumption on Windows.
    """
    nested_root = tmp_path / "outer" / "inner_checkout"
    _copy_corridor(sources=clean_corridor_sources, destination_root=nested_root)
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {
            relative: sha256_file(nested_root / relative)
            for relative in clean_corridor_sources
        },
    )

    result = direct_flow.verify_direct_candidate_flow(repo_root=nested_root)

    assert result["ok"], result["errors"]
    assert result["offending_paths"] == []
    assert tuple(result["certified_corridor_sources"]) == clean_corridor_sources


def test_actual_corridor_is_accepted_clean() -> None:
    """The real, unmodified corridor passes the full semantic verifier here.

    This is the platform-local acceptance half of the control pair: it proves
    the checkout this runner has really does satisfy the corridor, rather than
    trusting the Linux lane's verdict about a different set of bytes on disk.
    """
    result = direct_flow.verify_direct_candidate_flow(repo_root=_REPO_ROOT)

    assert result["ok"], result["errors"]
    assert result["offending_paths"] == []
    assert len(result["certified_corridor_sources"]) > 0


def test_one_resealed_actual_corridor_mutation_is_still_rejected(
    *,
    clean_corridor_sources: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One mutated, independently resealed corridor is rejected on this platform.

    Resealing means the mutation cannot be caught by a hash mismatch: only the
    semantic verifier can reject it. Running exactly one such case here shows
    the rejection path itself works on this OS, while the exhaustive population
    stays on the source-contract lane.
    """
    _copy_corridor(sources=clean_corridor_sources, destination_root=tmp_path)
    specs = direct_flow.uniform_process_mutation_specs(repo_root=_REPO_ROOT)
    name = min(specs)
    spec = specs[name]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8", newline="")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {
            relative: sha256_file(tmp_path / relative)
            for relative in clean_corridor_sources
        },
    )

    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)

    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
