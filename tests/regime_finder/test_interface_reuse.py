"""Projection alternatives must not rebuild unchanged cell interfaces."""

from copy import deepcopy

from _lcm_regime_finder.production import MetadataAnalysis
from .tiny_metadata_family import original


def test_repeated_interface_returns_an_isolated_plan() -> None:
    analysis = MetadataAnalysis(original())
    analysis.dependencies()
    cell = next(iter(analysis.cells.values()))
    projection = {name: {"kind": "identity", "target": name} for name in cell["states"]}
    first = analysis.function_interfaces(cell, projection, {})
    expected = deepcopy(first)
    first["solve"].clear()
    again = analysis.function_interfaces(cell, projection, {})
    assert (again) == (expected)


def test_changed_projection_constants_and_cell_do_not_share_interfaces() -> None:
    analysis = MetadataAnalysis(original())
    analysis.dependencies()
    cell = next(iter(analysis.cells.values()))
    projection = {name: {"kind": "identity", "target": name} for name in cell["states"]}
    first = analysis.function_interfaces(cell, projection, {})
    bound = analysis.function_interfaces(cell, projection, {"family": 0})
    assert (first) != (bound)
    assert any(
        record.get("bound_arguments", {}).get("family") == 0
        for record in bound["solve"].values()
    )
    changed = deepcopy(projection)
    changed["family"] = {"kind": "drop"}
    dropped = analysis.function_interfaces(cell, changed, {})
    assert (first) != (dropped)
    other = list(analysis.cells.values())[-1]
    late = analysis.function_interfaces(other, projection, {})
    assert (first) != (late)
