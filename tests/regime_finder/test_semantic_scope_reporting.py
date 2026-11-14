"""Local source safety and finite-family coverage have distinct receipt scopes."""

from copy import deepcopy

from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics


def original_declarations():
    """A literal one-cell finite problem; no capture or numerical model call."""
    return {
        "schema": "regime-declarations/v1",
        "source_identity": "literal-one-cell-scope-reporting-fixture",
        "domains": {
            "binary": {
                "kind": "categorical",
                "size": 2,
                "values": [0, 1],
                "codes": [0, 1],
            }
        },
        "callbacks": {
            "utility": {
                "arguments": ["x"],
                "reads": ["x"],
                "bindings": {},
                "source": "def utility(x):\n    return x\n",
                "expression": "x",
                "status": "proved",
                "identity": "literal-utility",
            }
        },
        "cells": [
            {
                "id": "c0",
                "regime": "life",
                "period": 0,
                "terminal": True,
                "states": {"x": "binary"},
                "actions": {},
                "functions": {"utility": "utility"},
                "consumers": [
                    {"kind": "reward", "callback": "utility", "phase": "both"}
                ],
            }
        ],
        "relations": [],
        "characteristics": {},
        "obligations": [],
        "allocation": {"kind": "per_block", "status": "declared", "boundary_unit": 1},
    }


def test_local_replay_does_not_borrow_complete_family_wording() -> None:
    original = original_declarations()
    declarations = deepcopy(original)
    analysis = MetadataAnalysis(declarations).run()
    complete = check_semantics(original, declarations, analysis)
    local = check_semantics(original, declarations, analysis, check_coverage=False)
    assert (complete["status"]) == ("PASS"), complete
    assert complete["projection_family_complete_verified"]
    assert ("independently checked finite family") in (complete["guarantee"])
    assert (local["errors"]) == ([])
    assert (local["status"]) == ("UNRESOLVED"), local
    assert not (local["projection_family_complete_verified"])
    assert all(item["scope"] == "family" for item in local["unresolved"])
    assert ("supplied coherent projection policies only") in (local["guarantee"])
    assert ("full-family coverage is not established") in (local["guarantee"])
    assert ("every coherent") not in (local["guarantee"])


def test_requested_coverage_is_not_equivalent_to_verified_coverage() -> None:
    original = original_declarations()
    declarations = deepcopy(original)
    analysis = MetadataAnalysis(declarations).run()
    incomplete = deepcopy(analysis)
    incomplete["discovery_complete"] = False
    checked = check_semantics(original, declarations, incomplete, check_coverage=True)
    assert (checked["errors"]) == ([])
    assert (checked["status"]) == ("UNRESOLVED"), checked
    assert not (checked["projection_family_complete_verified"])
    assert ("full-family coverage is not established") in (checked["guarantee"])
    assert ("every coherent") not in (checked["guarantee"])
