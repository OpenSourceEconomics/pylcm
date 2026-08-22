"""The curated API index accounts for every supported public export."""

import re
from pathlib import Path

import lcm
import lcm.solvers

_API_INDEX = Path(__file__).parents[1] / "docs" / "reference" / "public_api.md"

_LCM_ENTRY = re.compile(r"\[`lcm\.([A-Za-z_][A-Za-z0-9_]*)`\]")
_SOLVER_ENTRY = re.compile(r"\[`lcm\.solvers\.([A-Za-z_][A-Za-z0-9_]*)`\]")


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
