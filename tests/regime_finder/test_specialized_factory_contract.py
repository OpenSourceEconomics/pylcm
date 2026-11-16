"""Generated callback inputs stay source-bound through analysis and rechecking.

Uses a synthetic captured graph and independent mutation controls.
"""

import pytest
from copy import deepcopy
from itertools import product

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_declaration_ingestion import json_member, json_rows
from tests.regime_finder.test_model_adapter import snapshot, function, grid
from .specialized_factory_oracle import (
    exact_policy,
    constant_representation_is_policy_sufficient,
)


def specialization(
    *,
    clock: str = "period",
    resolved: bool = False,
    periods: int = 6,
    scale: int = 1,
    shift: int = 0,
) -> dict[str, JSONValue]:
    # The current native source documents AgeSpecializedFunction. The clock
    # mutation changes the captured graph's coordinate convention, not the
    # supported public declaration type or the factory's construction input.
    date = "age"
    marker = "AgeSpecializedFunction"
    body = f"{shift} - {scale} * (choice - history) * (choice - history)"
    build_source = (
        f"def build({date}):\n"
        "    def utility(choice, history):\n"
        f"        return {body}\n"
        "    return utility\n"
    )
    wrapper = {
        "kind": "dataclass",
        "type": "lcm." + marker,
        "fields": {
            "build": function(build_source, [date]),
            "signature": function(
                f"def signature({date}):\n    return {date}\n", [date]
            ),
        },
    }
    if resolved:
        wrapper["resolved_by_period"] = [
            {
                "period": period,
                "value": function(
                    f"def utility(choice, history):\n    return {body}\n",
                    ["choice", "history"],
                ),
            }
            for period in range(periods)
        ]
    return wrapper


def original_case(
    *,
    clock: str = "period",
    phased: bool = False,
    resolved: bool = False,
    values: tuple[int, ...] = (0, 1),
    scale: int = 1,
    shift: int = 0,
) -> dict[str, JSONValue]:
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {"history": grid(list(values))}
    fields["actions"] = {"choice": grid(list(values))}
    fields["constraints"] = {}
    fields["state_transitions"] = {
        "history": function(
            "def next_history(history):\n    return history\n", ["history"]
        )
    }
    wrapper = specialization(clock=clock, resolved=resolved, scale=scale, shift=shift)
    if phased:
        wrapper = {
            "kind": "dataclass",
            "type": "lcm.phased.Phased",
            "fields": {"solve": deepcopy(wrapper), "simulate": deepcopy(wrapper)},
        }
    fields["functions"] = {"utility": wrapper}
    if clock == "age":
        source["ages"] = {
            "kind": "object",
            "fields": {"exact_values": [20 + i for i in range(6)]},
        }
        source["reachability"]["fields"]["nodes"] = [
            [20 + i, "original"] for i in range(6)
        ]
        source["graph"]["fields"]["initial_nodes"] = [[20, "original"]]
        source["graph"]["fields"]["coordinate_kind"] = "age"
    return source


def _assert_no_unproved_removal(
    result: dict[str, JSONValue], *, resolved: bool = False
) -> None:
    assert (result.get("status")) == ("COMPLETE" if resolved else "UNRESOLVED"), (
        "An unsupported specialization must preserve the valid original Model and remain explicit"
    )
    analysis = json_member(result, "analysis")
    cells = json_rows(analysis, "declarations", "cells")
    assert ({(cell["period"], cell["regime"]) for cell in cells}) == (
        {(period, "original") for period in range(6)}
    )
    assert (len(cells)) == (6)
    closure = json_member(analysis, "dependency_closure")
    assert closure
    assert (set(closure)) == ({cell["id"] for cell in cells})
    assert all(
        isinstance(axes, list) and "history" in axes for axes in closure.values()
    )
    if not resolved:
        assert any(
            item["kind"] == "specialized_effective_signature"
            for item in json_rows(analysis, "obligations")
        ), analysis["obligations"]
    for cell in cells:
        cell_id = cell["id"]
        assert isinstance(cell_id, str)
        options = json_rows(analysis, "interfaces", cell_id, "options")
        assert options, "The original cell needs an admissible representation"
        assert (set(json_member(cell, "states"))) == ({"history"})
        assert (set(json_member(cell, "actions"))) == ({"choice"})
        for option in options:
            assert ("history") in (json_member(option, "states")), (
                "A factory signature is not the signature of its returned economic function"
            )
            assert (json_member(option, "state_projection")["history"]) == (
                {"kind": "identity", "target": "history"}
            )
            assert (option["actions"]) == (cell["actions"])


def test_minimal_binary_public_entrypoint() -> None:
    assert (exact_policy()) == ({0: {0}, 1: {1}})
    assert not (constant_representation_is_policy_sufficient())
    source = original_case()
    unchanged = deepcopy(source)
    report = find_regimes(
        source, budget={"seconds": 30}, search_mode="exact", facts_backend="metadata"
    )
    _assert_no_unproved_removal(report)
    assert (source) == (unchanged)
    assert (verify_production_result(source, report)["status"]) == ("UNRESOLVED")


def test_age_period_phase_and_domain_mutations(*, subtests: pytest.Subtests) -> None:
    for clock, phased, values, scale, shift in product(
        ("period", "age"), (False, True), ((0, 1), (1, 0), (-3, 2)), (1, 8), (0, 17)
    ):
        with subtests.test(
            clock=clock, phased=phased, values=values, scale=scale, shift=shift
        ):
            source = original_case(
                clock=clock, phased=phased, values=values, scale=scale, shift=shift
            )
            unchanged = deepcopy(source)
            report = find_regimes(
                source,
                budget={"seconds": 30},
                search_mode="exact",
                facts_backend="metadata",
            )
            _assert_no_unproved_removal(report)
            assert (source) == (unchanged)


def test_resolved_effective_functions_retain_history(
    *, subtests: pytest.Subtests
) -> None:
    for clock, phased in product(("period", "age"), (False, True)):
        with subtests.test(clock=clock, phased=phased):
            source = original_case(clock=clock, phased=phased, resolved=True)
            report = find_regimes(
                source,
                budget={"seconds": 30},
                search_mode="exact",
                facts_backend="metadata",
            )
            assert (report["status"]) == ("COMPLETE"), report.get("semantic_check")
            _assert_no_unproved_removal(report, resolved=True)
            assert (verify_production_result(source, report)["status"]) == ("PASS")


def test_checker_rejects_old_false_complete_claim() -> None:
    # This deliberately corrupted receipt is a checker mutation, not an
    # admitted-input witness. Recreate the relevant old producer assertions
    # while preserving the ORIGINAL wrapper and source identity.
    source = original_case()
    report = find_regimes(
        source, budget={"seconds": 30}, search_mode="exact", facts_backend="metadata"
    )
    _assert_no_unproved_removal(report)
    bad = deepcopy(report)
    analysis = bad["analysis"]
    declarations = analysis["declarations"]
    bad["status"] = "COMPLETE"
    bad["semantic_check"] = {"status": "PASS"}
    analysis["obligations"] = []
    declarations["obligations"] = []
    for callback in declarations["callbacks"].values():
        if callback.pop("unresolved_specialization_signature", False):
            callback.update(
                status="structural",
                expression=None,
                dependency_contract="pylcm-declared-inputs/v1",
            )
            callback["reads"] = list(callback["arguments"])
    for cell in declarations["cells"]:
        analysis["dependency_closure"][cell["id"]] = []
        for option in analysis["interfaces"][cell["id"]]["options"]:
            option["states"].pop("history", None)
            option["state_projection"]["history"] = {"kind": "drop"}
    checked = verify_production_result(source, bad)
    assert (checked["status"]) == ("FAIL")
    source_check = checked["source_semantics"]
    assert (source_check["status"]) == ("FAIL")
    assert any(
        error["kind"] == "original future dependency closure omitted"
        and "history" in error.get("missing", [])
        for error in source_check["errors"]
    ), source_check
