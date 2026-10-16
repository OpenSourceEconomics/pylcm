"""Canonical loaders preserve callback identity and reseal every certified owner."""

import ast
import hashlib
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from tests.candidate_certificate import check_seals, repin_corridors

_ROOT = Path(__file__).parents[2]
_CHECKER = "tests/candidate_certificate/direct_flow.py"
_CERTIFICATE = "tests/test_grid_search_candidate_certificate.py"
_LITERAL_OWNER = "src/_lcm/egm/asset_row.py"


@pytest.mark.parametrize("tool", [check_seals, repin_corridors])
def test_loaded_verifier_callbacks_resolve_the_requested_module_identity(
    *, tool: ModuleType, tmp_path: Path
) -> None:
    """A checker loaded from a requested tree can pass its own module to callbacks."""
    path = tmp_path / _CHECKER
    path.parent.mkdir(parents=True)
    path.write_text(
        "import sys\n\ndef current_module():\n    return sys.modules[__name__]\n"
    )
    loaded = tool._load_direct_flow(root=tmp_path)
    assert Path(loaded.__file__).resolve() == path.resolve()
    assert loaded.current_module() is loaded


@pytest.fixture
def certificate_root(tmp_path: Path) -> Path:
    """Copy the actual canonical certificate and source owners into a private tree."""
    shutil.copytree(_ROOT / "src", tmp_path / "src")
    shutil.copytree(
        _ROOT / "tests/candidate_certificate",
        tmp_path / "tests/candidate_certificate",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (tmp_path / _CERTIFICATE).write_bytes((_ROOT / _CERTIFICATE).read_bytes())
    return tmp_path


def test_fix_refreshes_a_literal_path_source_seal(*, certificate_root: Path) -> None:
    """A source whose seal uses a literal path receives its actual current digest."""
    source = certificate_root / _LITERAL_OWNER
    source.write_text(source.read_text() + "\n# Seal-only source probe.\n")
    expected = hashlib.sha256(source.read_text().encode()).hexdigest()
    check_seals.fix(repo_root=certificate_root)
    actual = check_seals._load_direct_flow(root=certificate_root)
    assert actual._SOURCE_SEALS[_LITERAL_OWNER] == expected


def _source_key(*, key: ast.expr | None, module: ModuleType) -> str:
    """Resolve only an explicit string path or its existing declared source name."""
    if isinstance(key, ast.Constant):
        assert isinstance(key.value, str)
        return key.value
    assert isinstance(key, ast.Name)
    value = getattr(module, key.id)
    assert isinstance(value, str)
    return value


@pytest.mark.parametrize("owner", [_LITERAL_OWNER, "src/_lcm/processes/ar1.py"])
def test_fix_creates_a_missing_seal_for_an_already_certified_owner(
    *, certificate_root: Path, owner: str
) -> None:
    """Every explicitly certified source obtains exactly one truthful byte seal."""
    module = check_seals._load_direct_flow(root=certificate_root)
    assert owner in module._CERTIFIED_CORRIDOR_SOURCES
    assert owner in module._SOURCE_SEALS
    initially_missing = set(module._CERTIFIED_CORRIDOR_SOURCES) - set(
        module._SOURCE_SEALS
    )
    assert not set(module._SOURCE_SEALS) - set(module._CERTIFIED_CORRIDOR_SOURCES)
    checker = certificate_root / _CHECKER
    original = checker.read_text()
    tree = ast.parse(original)
    store = next(
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_SOURCE_SEALS"
    )
    assert isinstance(store, ast.Dict)
    matches = [
        (key, value)
        for key, value in zip(store.keys, store.values, strict=True)
        if _source_key(key=key, module=module) == owner
    ]
    assert len(matches) == 1
    key, value = matches[0]
    assert key is not None
    assert key.lineno == value.end_lineno
    lines = original.splitlines(keepends=True)
    del lines[key.lineno - 1]
    checker.write_text("".join(lines))
    expected = hashlib.sha256(
        (certificate_root / owner).read_text().encode()
    ).hexdigest()
    check_seals.fix(repo_root=certificate_root)
    actual = check_seals._load_direct_flow(root=certificate_root)
    assert actual._SOURCE_SEALS[owner] == expected
    assert len(actual._SOURCE_SEALS) == len(set(actual._SOURCE_SEALS))
    # Existing union holes are separate from this test's one removed owner.
    assert set(actual._SOURCE_SEALS) | initially_missing == set(
        actual._CERTIFIED_CORRIDOR_SOURCES
    )
