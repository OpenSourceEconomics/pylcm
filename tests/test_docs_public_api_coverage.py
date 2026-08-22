"""The curated API index accounts for every supported public export."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import lcm
import lcm.koopmans_aggregation
import lcm.params
import lcm.solvers

_API_INDEX = Path(__file__).parents[1] / "docs" / "reference" / "public_api.md"

_LCM_ENTRY = re.compile(r"\[`lcm\.([A-Za-z_][A-Za-z0-9_]*)`\]")
_SOLVER_ENTRY = re.compile(r"\[`lcm\.solvers\.([A-Za-z_][A-Za-z0-9_]*)`\]")
_ALL_ENTRY = re.compile(r"\[`(lcm(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`\]")
_LOCAL_DESTINATION = re.compile(r"\[`lcm(?:\.solvers)?\.[^`]+`\]\(([^)]+)\)")
_SUBMODULE_ENTRY = re.compile(
    r"\[`(lcm\.(?:params|koopmans_aggregation))\.([A-Za-z_][A-Za-z0-9_]*)`\]"
)


def test_public_api_index_covers_exactly_the_lcm_exports():
    """Every top-level public name has one entry in the curated API index."""
    text = _API_INDEX.read_text()
    documented = set(_LCM_ENTRY.findall(text))

    assert documented == set(lcm.__all__)


def test_public_api_index_covers_exactly_the_solver_exports():
    """Every solver-related public name has one entry in the curated API index."""
    text = _API_INDEX.read_text()
    documented = set(_SOLVER_ENTRY.findall(text))

    assert documented == set(lcm.solvers.__all__)


def test_public_api_index_lists_each_symbol_once():
    """Every indexed public symbol has exactly one canonical destination."""
    text = _API_INDEX.read_text()
    entries = _ALL_ENTRY.findall(text)

    assert len(entries) == len(set(entries))


def test_public_api_index_local_destinations_exist():
    """Every local API-index link points to an existing documentation file."""
    text = _API_INDEX.read_text()

    missing = []
    for destination in _LOCAL_DESTINATION.findall(text):
        path = urlsplit(destination).path
        if path and not (_API_INDEX.parent / path).exists():
            missing.append(path)

    assert missing == []


def test_public_api_index_covers_allowlisted_submodule_surfaces():
    """The deliberate public submodule exports have canonical destinations."""
    text = _API_INDEX.read_text()
    documented = set(_SUBMODULE_ENTRY.findall(text))
    expected = {
        *{("lcm.params", name) for name in lcm.params.__all__},
        (
            "lcm.koopmans_aggregation",
            "KoopmansAggregator",
        ),
    }

    assert documented == expected
