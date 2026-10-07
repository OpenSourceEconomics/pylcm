"""The corridor re-pin tool rewrites named drift and refuses everything else.

Each case runs against a throwaway copy of the certified sources and the
verifier, so a run that rewrites a pin cannot touch the checkout.
"""

import ast
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
direct_flow = _load_by_path(
    name="_direct_flow_under_test", path=_REPO_ROOT / _DIRECT_FLOW
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


def _first_callable_pin(*, repo_root: Path) -> Any:
    """Return the first callable pin in the certificate's pin store."""
    return next(
        pin
        for pin in repin_corridors.collect_pins(repo_root=repo_root)
        if pin.kind == "callable"
    )


def _other_digest(digest: str) -> str:
    return "b" * 64 if digest != "b" * 64 else "c" * 64


def test_each_corridor_pin_stands_once_in_the_certificate() -> None:
    """No source, kind and name is pinned at two places in `direct_flow.py`."""
    pins = repin_corridors.collect_pins(repo_root=_REPO_ROOT)

    assert len(pins) == len({(pin.source, pin.kind, pin.name) for pin in pins})


def test_every_stored_pin_is_selected_by_a_certificate_family() -> None:
    """A pin no family selects would be checked by nothing, so none exists."""
    assert direct_flow._unselected_pins() == ()


def test_a_family_selection_returns_the_stored_digests() -> None:
    """Selecting one callable yields the stored module surface and callable digest."""
    source, (surface, callables) = next(iter(direct_flow._CORRIDOR_PINS.items()))
    name, digest = next(iter(callables.items()))

    assert direct_flow._contracts({source: (name,)}) == {
        source: (surface, {name: digest})
    }


def test_a_family_selecting_an_unpinned_callable_is_refused() -> None:
    """A selection naming a callable the store does not pin raises, naming it."""
    source = next(iter(direct_flow._CORRIDOR_PINS))

    with pytest.raises(ValueError, match="no_such_callable"):
        direct_flow._callable_pins(source=source, names=("no_such_callable",))


def test_a_name_pinned_twice_in_the_store_is_refused(certificate_root: Path) -> None:
    """One name denoting two digests is re-pinned by hand, never by this tool."""
    pin = _first_callable_pin(repo_root=certificate_root)
    path = certificate_root / _DIRECT_FLOW
    lines = path.read_text(encoding="utf-8").split("\n")
    duplicate = lines[pin.line - 1].replace(pin.pinned, _other_digest(pin.pinned))
    lines.insert(pin.line, duplicate)
    path.write_text("\n".join(lines), encoding="utf-8")

    outcome = repin_corridors.evaluate(
        repo_root=certificate_root, changed_sources=frozenset({pin.source})
    )

    assert any(pin.name in message for message in outcome.ambiguous)


def test_a_digest_shared_by_two_sources_is_rewritten_for_the_named_one_only(
    certificate_root: Path,
) -> None:
    """Re-pinning one source leaves an identical digest owned by another in place.

    Two certified sources can hold callables with identical bodies, so their
    digests coincide. Editing one of them and naming it re-pins that entry; the
    other source's entry still matches its own, unchanged tree.
    """
    pins = repin_corridors.collect_pins(repo_root=certificate_root)
    by_digest: dict[str, list[Any]] = {}
    for pin in pins:
        by_digest.setdefault(pin.pinned, []).append(pin)
    edited = next(
        group[0]
        for group in by_digest.values()
        if len({pin.source for pin in group}) > 1 and "." not in group[0].name
    )
    source_path = certificate_root / edited.source
    source_lines = source_path.read_text(encoding="utf-8").split("\n")
    function = next(
        node
        for node in ast.parse("\n".join(source_lines)).body
        if isinstance(node, ast.FunctionDef) and node.name == edited.name
    )
    indent = " " * function.body[-1].col_offset
    source_lines.insert(function.end_lineno or 0, f"{indent}pass")
    source_path.write_text("\n".join(source_lines), encoding="utf-8")

    repin_corridors.main(
        [
            "--repo-root",
            str(certificate_root),
            "--changed-source",
            edited.source,
        ]
    )

    assert (
        repin_corridors.evaluate(
            repo_root=certificate_root, changed_sources=frozenset()
        ).foreign
        == ()
    )


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


def test_fix_rewrites_seal_lines_inside_the_seal_map_only(
    certificate_root: Path,
) -> None:
    """A `<NAME>_SOURCE: "<digest>",` line outside `_SOURCE_SEALS` is left alone."""
    decoy = f'    LOGSUM_SOURCE: "{"0" * 64}",'
    path = certificate_root / _DIRECT_FLOW
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n_DECOY = {{\n{decoy}\n}}\n",
        encoding="utf-8",
    )

    check_seals.fix(repo_root=certificate_root)

    assert decoy in path.read_text(encoding="utf-8").split("\n")
