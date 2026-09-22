"""The corridor re-pin tool rewrites named drift and refuses everything else.

Each case runs against a throwaway copy of the certified sources and the
verifier, so a run that rewrites a pin cannot touch the checkout.
"""

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_DIRECT_FLOW = "tests/candidate_certificate/direct_flow.py"
_CERTIFICATE = "tests/test_grid_search_candidate_certificate.py"
_DIGIT_NAMED_SOURCE = "src/_lcm/processes/ar1.py"


def _load_by_path(*, name: str, path: Path) -> ModuleType:
    """Load a module by path; `tests/candidate_certificate` is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tool() -> ModuleType:
    """Load the corridor re-pin tool under test."""
    return _load_by_path(
        name="_repin_corridors_under_test",
        path=_REPO_ROOT / "tests/candidate_certificate/repin_corridors.py",
    )


repin_corridors = _load_tool()
check_seals = _load_by_path(
    name="_check_seals_under_test",
    path=_REPO_ROOT / "tests/candidate_certificate/check_seals.py",
)


@pytest.fixture
def certificate_root(tmp_path: Path) -> Path:
    """Return a throwaway tree holding the certified sources and the verifier."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(_REPO_ROOT / "src", tmp_path / "src")
    shutil.copytree(
        _REPO_ROOT / "tests/candidate_certificate",
        tmp_path / "tests/candidate_certificate",
        ignore=ignore,
    )
    (tmp_path / "tests" / _CERTIFICATE.rsplit("/", maxsplit=1)[-1]).write_bytes(
        (_REPO_ROOT / _CERTIFICATE).read_bytes()
    )
    return tmp_path


def _repeated_pin(*, repo_root: Path) -> Any:
    """Return a pin whose digest literal stands more than once in the file."""
    counts: dict[str, int] = {}
    pins = repin_corridors.collect_pins(repo_root=repo_root)
    for pin in pins:
        counts[pin.pinned] = counts.get(pin.pinned, 0) + 1
    for pin in pins:
        if counts[pin.pinned] > 1:
            return pin
    raise AssertionError("no digest literal is written more than once")


def _other_digest(digest: str) -> str:
    return "b" * 64 if digest != "b" * 64 else "c" * 64


def test_a_name_pinned_to_two_different_digests_is_refused(
    certificate_root: Path,
) -> None:
    """One name denoting two digests is re-pinned by hand, never by this tool."""
    pin = _repeated_pin(repo_root=certificate_root)
    path = certificate_root / _DIRECT_FLOW
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(pin.pinned, _other_digest(pin.pinned), 1), encoding="utf-8"
    )

    outcome = repin_corridors.evaluate(
        repo_root=certificate_root, changed_sources=frozenset({pin.source})
    )

    assert any(pin.name in message for message in outcome.ambiguous)


def test_drift_in_a_source_that_was_not_named_is_reported_as_foreign(
    certificate_root: Path,
) -> None:
    """A pin the caller did not claim must still match the tree."""
    pin = repin_corridors.collect_pins(repo_root=certificate_root)[0]
    path = certificate_root / _DIRECT_FLOW
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(pin.pinned, _other_digest(pin.pinned)), encoding="utf-8"
    )

    outcome = repin_corridors.evaluate(
        repo_root=certificate_root, changed_sources=frozenset()
    )

    assert [drifted.name for drifted, _ in outcome.foreign] == [pin.name]


def test_an_unnamed_drifted_pin_blocks_the_rewrite(certificate_root: Path) -> None:
    """The run exits non-zero and leaves the stale digest standing."""
    pin = repin_corridors.collect_pins(repo_root=certificate_root)[0]
    path = certificate_root / _DIRECT_FLOW
    stale = _other_digest(pin.pinned)
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace(pin.pinned, stale), encoding="utf-8")

    repin_corridors.main(["--repo-root", str(certificate_root)])

    assert stale in path.read_text(encoding="utf-8")


def test_one_intentional_drift_is_rewritten_to_the_recomputed_digest(
    certificate_root: Path,
) -> None:
    """Naming the source the pin belongs to restores the digest the tree implies."""
    pin = repin_corridors.collect_pins(repo_root=certificate_root)[0]
    path = certificate_root / _DIRECT_FLOW
    original = path.read_text(encoding="utf-8")
    path.write_text(
        original.replace(pin.pinned, _other_digest(pin.pinned)), encoding="utf-8"
    )

    repin_corridors.main(
        [
            "--repo-root",
            str(certificate_root),
            "--changed-source",
            pin.source,
        ]
    )

    assert path.read_text(encoding="utf-8") == original


def test_check_mode_leaves_the_pin_file_untouched(certificate_root: Path) -> None:
    """`--check` reports drift without writing, so it is safe in a hook."""
    pin = repin_corridors.collect_pins(repo_root=certificate_root)[0]
    path = certificate_root / _DIRECT_FLOW
    drifted_text = path.read_text(encoding="utf-8").replace(
        pin.pinned, _other_digest(pin.pinned)
    )
    path.write_text(drifted_text, encoding="utf-8")

    repin_corridors.main(
        [
            "--repo-root",
            str(certificate_root),
            "--changed-source",
            pin.source,
            "--check",
        ]
    )

    assert path.read_text(encoding="utf-8") == drifted_text


def test_check_mode_exits_non_zero_on_named_drift(certificate_root: Path) -> None:
    """A stale corridor pin is a failure, not a silent repair opportunity."""
    pin = repin_corridors.collect_pins(repo_root=certificate_root)[0]
    path = certificate_root / _DIRECT_FLOW
    path.write_text(
        path.read_text(encoding="utf-8").replace(pin.pinned, _other_digest(pin.pinned)),
        encoding="utf-8",
    )

    status = repin_corridors.main(
        [
            "--repo-root",
            str(certificate_root),
            "--changed-source",
            pin.source,
            "--check",
        ]
    )

    assert status == 1


def test_an_undrifted_tree_reports_nothing_to_re_pin(certificate_root: Path) -> None:
    """Every attributed pin in the committed certificate matches its source."""
    status = repin_corridors.main(["--repo-root", str(certificate_root), "--check"])

    assert status == 0


def test_a_seal_name_containing_a_digit_is_resealed_by_fix(
    certificate_root: Path,
) -> None:
    """`--fix` reseals every certified source, including `PROCESS_AR1_SOURCE`.

    A seal map entry whose constant name carries a digit is an ordinary byte
    seal. Skipping it makes `--fix` report success while leaving that one entry
    stale, and the next check calls the leftover an error a reseal cannot
    repair -- a stale seal wearing a corridor violation's clothes.
    """
    drifted = certificate_root / _DIGIT_NAMED_SOURCE
    drifted.write_text(
        drifted.read_text(encoding="utf-8") + "\n# A comment moves the seal.\n",
        encoding="utf-8",
    )

    check_seals.fix(repo_root=certificate_root)

    assert check_seals.check(repo_root=certificate_root) == ([], [])
