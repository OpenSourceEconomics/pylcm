"""Every certificate profile resolves its sources from the one canonical inventory.

`sources.json` lists the certified sources once. Each profile references that list
and carries an explicit override slot, `exclude_sources`, naming the inventory paths
the profile drops. Regenerating the inventory keeps the slot, and an inventory in
the older schema, where each profile repeated the whole list, upgrades to the
reference form.
"""

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.candidate_certificate.generate_sources import (
    INVENTORY_PATH,
    REQUIRED_PROFILES,
    build_inventory,
    canonical_json,
    profile_sources,
    upgrade_inventory,
)

_REPO_ROOT = Path(__file__).parents[2]
_CERTIFICATE = "tests/test_grid_search_candidate_certificate.py"
_SOURCES = [
    {"path": "src/a.py", "sha256": "a" * 64},
    {"path": "src/b.py", "sha256": "b" * 64},
    {"path": "src/c.py", "sha256": "c" * 64},
]


def _inventory(**exclusions: list[str]) -> dict[str, Any]:
    """Return a current-schema inventory over `_SOURCES` with the given exclusions."""
    return {
        "schema_version": "2",
        "sources": copy.deepcopy(_SOURCES),
        "derived_policy": {
            "profiles": {
                profile: {
                    "inventory": "sources",
                    "exclude_sources": exclusions.get(profile, []),
                }
                for profile in REQUIRED_PROFILES
            }
        },
    }


def _schema_one(**listed: list[dict[str, str]]) -> dict[str, Any]:
    """Return an older-schema inventory whose profiles repeat their source lists."""
    return {
        "schema_version": "1",
        "sources": copy.deepcopy(_SOURCES),
        "derived_policy": {
            "profiles": {
                profile: {
                    "candidate_sources": listed.get(profile, copy.deepcopy(_SOURCES)),
                    "source_count": len(listed.get(profile, _SOURCES)),
                }
                for profile in REQUIRED_PROFILES
            }
        },
    }


@pytest.fixture
def certificate_root(tmp_path: Path) -> Path:
    """Return a throwaway tree holding the certificate and its certified sources."""
    shutil.copytree(
        _REPO_ROOT / "src",
        tmp_path / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (tmp_path / "tests/candidate_certificate").mkdir(parents=True)
    shutil.copy(_REPO_ROOT / _CERTIFICATE, tmp_path / _CERTIFICATE)
    shutil.copy(_REPO_ROOT / INVENTORY_PATH, tmp_path / INVENTORY_PATH)
    return tmp_path


def test_build_inventory_writes_the_current_schema_version() -> None:
    """The generated inventory declares schema version 2."""
    assert build_inventory(_REPO_ROOT)["schema_version"] == "2"


def test_build_inventory_has_each_profile_reference_the_canonical_inventory() -> None:
    """Every profile names the inventory and lists no sources of its own."""
    profiles = build_inventory(_REPO_ROOT)["derived_policy"]["profiles"]

    assert profiles == {
        profile: {"inventory": "sources", "exclude_sources": []}
        for profile in REQUIRED_PROFILES
    }


def test_the_committed_inventory_is_the_generated_inventory() -> None:
    """`sources.json` in the checkout is byte-identical to a fresh generation."""
    committed = (_REPO_ROOT / INVENTORY_PATH).read_text(encoding="utf-8")

    assert committed == canonical_json(build_inventory(_REPO_ROOT))


@pytest.mark.parametrize("profile", REQUIRED_PROFILES)
def test_profile_sources_without_override_are_the_whole_inventory(
    profile: str,
) -> None:
    """A profile with an empty override slot certifies every inventory source."""
    assert profile_sources(_inventory())[profile] == _SOURCES


@pytest.mark.parametrize(
    ("profile", "expected_paths"),
    [
        ("fast", ["src/a.py", "src/c.py"]),
        ("certified", ["src/a.py", "src/b.py", "src/c.py"]),
    ],
)
def test_an_excluded_source_is_dropped_from_its_own_profile_only(
    *, profile: str, expected_paths: list[str]
) -> None:
    """Excluding `src/b.py` from `fast` leaves `certified` untouched."""
    resolved = profile_sources(_inventory(fast=["src/b.py"]))

    assert [item["path"] for item in resolved[profile]] == expected_paths


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        (_inventory(fast=["src/missing.py"]), "src/missing.py"),
        (_inventory(fast=["src/a.py", "src/a.py"]), "src/a.py"),
        ({**_inventory(), "schema_version": "1"}, "schema"),
    ],
    ids=["unknown-path", "duplicate-path", "older-schema"],
)
def test_profile_sources_refuses_a_malformed_override(
    *, inventory: dict[str, Any], message: str
) -> None:
    """An override naming no inventory source, twice, or under another schema fails."""
    with pytest.raises(ValueError, match=re.escape(message)):
        profile_sources(inventory)


def test_a_profile_referencing_another_list_is_refused() -> None:
    """A profile must reference the canonical `sources` list by name."""
    inventory = _inventory()
    inventory["derived_policy"]["profiles"]["fast"]["inventory"] = "other_sources"

    with pytest.raises(ValueError, match="other_sources"):
        profile_sources(inventory)


def test_a_missing_profile_is_refused() -> None:
    """Every required profile carries its reference and override slot."""
    inventory = _inventory()
    del inventory["derived_policy"]["profiles"]["fast"]

    with pytest.raises(ValueError, match="fast"):
        profile_sources(inventory)


def test_a_schema_one_inventory_upgrades_to_references() -> None:
    """Profiles repeating the whole list become references with empty overrides."""
    upgraded = upgrade_inventory(_schema_one())

    assert (upgraded["schema_version"], upgraded["derived_policy"]) == (
        "2",
        _inventory()["derived_policy"],
    )


def test_a_schema_one_profile_listing_a_subset_upgrades_to_an_exclusion() -> None:
    """A profile that listed only `a` and `c` excludes `b` after the upgrade."""
    upgraded = upgrade_inventory(_schema_one(fast=[_SOURCES[0], _SOURCES[2]]))

    assert profile_sources(upgraded)["fast"] == [_SOURCES[0], _SOURCES[2]]


def test_a_schema_one_profile_with_an_unknown_source_is_refused() -> None:
    """A profile entry absent from the inventory cannot become an exclusion."""
    stray = {"path": "src/stray.py", "sha256": "d" * 64}

    with pytest.raises(ValueError, match=re.escape("src/stray.py")):
        upgrade_inventory(_schema_one(fast=[*_SOURCES, stray]))


def test_regeneration_keeps_a_committed_override(certificate_root: Path) -> None:
    """A hand-set exclusion survives `build_inventory`."""
    path = certificate_root / INVENTORY_PATH
    committed = json.loads(path.read_text(encoding="utf-8"))
    excluded = committed["sources"][0]["path"]
    committed["derived_policy"]["profiles"]["fast"]["exclude_sources"] = [excluded]
    path.write_text(canonical_json(committed), encoding="utf-8")

    profiles = build_inventory(certificate_root)["derived_policy"]["profiles"]

    assert profiles["fast"]["exclude_sources"] == [excluded]


def test_regeneration_upgrades_a_schema_one_committed_inventory(
    certificate_root: Path,
) -> None:
    """A committed profile that listed fewer sources regenerates as an exclusion."""
    generated = build_inventory(certificate_root)
    sources = generated["sources"]
    schema_one = {
        "schema_version": "1",
        "sources": sources,
        "derived_policy": {
            "profiles": {
                "fast": {"candidate_sources": sources[1:]},
                "certified": {"candidate_sources": sources},
            }
        },
    }
    (certificate_root / INVENTORY_PATH).write_text(
        canonical_json(schema_one), encoding="utf-8"
    )

    profiles = build_inventory(certificate_root)["derived_policy"]["profiles"]

    assert profiles["fast"]["exclude_sources"] == [sources[0]["path"]]
