"""Repeated policy incidences share only the checker's own local dependency proof."""

from copy import deepcopy
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from .tiny_metadata_family import original


def test_duplicate_relations_and_policies_preserve_source_dependencies() -> None:
    source = original()
    source["callbacks"]["stable"] = {
        "arguments": [],
        "reads": [],
        "status": "proved",
        "expression": "1",
        "identity": "stable-constant",
        "original": {
            "kind": "callable",
            "arguments": [],
            "bindings": {},
            "source": "def stable():\n    return 1\n",
        },
    }
    for item in source["relations"]:
        item["callbacks"] = ["stable"]
    relation = deepcopy(source["relations"][0])
    relation["id"] = "duplicate-incidence"
    source["relations"].append(relation)
    declaration = normalize_declarations(source)
    analysis = MetadataAnalysis(declaration).run()
    result = check_semantics(source, declaration, analysis)
    assert result["status"] == "PASS", result
    assert {row["id"] for row in declaration["relations"]} == {
        row["id"] for row in source["relations"]
    }
    # Repeated relation incidences must not hide a changed original dependency.
    changed = deepcopy(source)
    changed["callbacks"]["stable"]["original"]["source"] = (
        "def stable(assets):\n    return assets\n"
    )
    changed["callbacks"]["stable"]["original"]["arguments"] = ["assets"]
    changed["callbacks"]["stable"]["arguments"] = ["assets"]
    changed["callbacks"]["stable"]["reads"] = ["assets"]
    assert check_semantics(changed, declaration, analysis)["status"] == "FAIL"
