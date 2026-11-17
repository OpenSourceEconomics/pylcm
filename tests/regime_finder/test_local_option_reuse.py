"""Returned interface options are independent of the function-interface cache."""

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.records import JSONValue
from _lcm_regime_finder.semantic_check import check_semantics
from .tiny_metadata_family import original


def test_returned_options_cannot_mutate_the_interface_cache() -> None:
    source = original()
    declaration = normalize_declarations(source)
    metadata = MetadataAnalysis(declaration, facts_backend="metadata")
    analysis = metadata.run()
    assert (check_semantics(source, declaration, analysis)["status"]) == ("PASS")
    # Returned plans still cannot mutate the separate function-interface cache.
    cell = next(iter(metadata.cells.values()))
    base = {name: {"kind": "identity", "target": name} for name in cell["states"]}
    before = metadata.function_interfaces(cell, base, {})
    analysis["interfaces"][cell["id"]]["options"][0]["function_interfaces"].clear()
    assert (metadata.function_interfaces(cell, base, {})) == (before)
