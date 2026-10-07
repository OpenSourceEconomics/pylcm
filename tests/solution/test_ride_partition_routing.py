"""The four NBEGM partition sites use the shared EGM batching contract."""

import ast
from collections import Counter
from pathlib import Path

from _lcm.solution import nbegm


def test_all_four_production_sites_call_the_shared_dispatcher() -> None:
    """The source routes two ride and two branch axes through one dispatcher.

    Ride: the tile-local core's cell loop and the per-interval continuation
    read. Branch: the continuation read's and the envelope solve's branch loops.
    """
    tree = ast.parse(Path(nbegm.__file__).read_text(encoding="utf-8"))
    names = Counter(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )

    assert names["map_over_leading_axis"] == 4
