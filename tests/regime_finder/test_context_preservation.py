"""Opaque original decision contexts need preservation, not invented equivalence."""

import pytest
from copy import deepcopy
from _lcm_regime_finder import find_regimes as _find_regimes
from functools import partial

# These regressions exercise the optional stronger semantic audit, not the
# Model-graph contract whose primitives retain their declared dependencies.
find_regimes = partial(_find_regimes, analysis_scope="full_semantics")
from _lcm_regime_finder.proposal_decision import build_decisions, verify_decisions
from tests.regime_finder.test_round5_local_obligations import native_fixture


def _create_source():
    source = native_fixture()
    for regime in source["user_regimes"].values():
        regime["fields"].pop("semantic_role")
    return source


def test_local_unused_state_does_not_need_cross_actor_equivalence() -> None:
    source = _create_source()
    report = find_regimes(source)
    changed = [d for d in report["proposal_decisions"][1:] if d["changed_interfaces"]]
    assert changed
    assert all(d["disposition"] == "adopt" for d in changed), changed
    for row in changed:
        proof = [
            d
            for d in row["discharged_unchanged_obligations"]
            if d["obligation"]["kind"] == "role_timing_contract"
        ]
        assert proof
        assert all(
            d["context_preservation"]["original_declaration_sha256"] for d in proof
        )
    assert (
        verify_decisions(
            source,
            report["analysis"],
            report["scenarios"],
            report["proposal_decisions"],
        )["status"]
    ) == ("PASS")


def test_forged_context_path_is_not_discharged() -> None:
    source = _create_source()
    report = find_regimes(source)
    analysis = deepcopy(report["analysis"])
    for obligation in analysis["obligations"]:
        if obligation["kind"] == "role_timing_contract":
            obligation["source_path"] = "/unavailable"
    rows = build_decisions(source, analysis, report["scenarios"])[1:]
    assert all(r["disposition"] != "adopt" for r in rows)


def test_changed_context_identity_is_rejected(*, subtests: pytest.Subtests) -> None:
    source = _create_source()
    report = find_regimes(source)
    for field, value in [
        ("stage", "different-stage"),
        ("role", {"actor": "other"}),
        ("ownership", [{"leg": "other"}]),
        ("period", 1),
    ]:
        with subtests.test(field=field):
            changed = deepcopy(report["analysis"])
            changed["declarations"]["cells"][0][field] = value
            rows = build_decisions(source, changed, report["scenarios"])[1:]
            assert all(r["disposition"] != "adopt" for r in rows)


def test_cross_regime_merge_does_not_get_context_identity_rule() -> None:
    from _lcm_regime_finder.proposal_decision import _preserved_context

    source = _create_source()
    report = find_regimes(source)
    cells = {c["id"]: c for c in report["analysis"]["declarations"]["cells"]}
    obligation = next(
        o
        for o in report["analysis"]["obligations"]
        if o["kind"] == "role_timing_contract"
    )
    bad_plan = {"blocks": [{"cells": list(cells)}]}
    assert (_preserved_context(source, cells, bad_plan, obligation)) is None


def test_saved_context_certificate_mutation_is_rejected() -> None:
    source = _create_source()
    report = find_regimes(source)
    saved = deepcopy(report["proposal_decisions"])
    proof = next(
        item
        for row in saved
        for item in row["discharged_unchanged_obligations"]
        if "context_preservation" in item
    )
    proof["context_preservation"]["original_declaration_sha256"] = "0" * 64
    assert (
        verify_decisions(source, report["analysis"], report["scenarios"], saved)[
            "status"
        ]
    ) == ("FAIL")
