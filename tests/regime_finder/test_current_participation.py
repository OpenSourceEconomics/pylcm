"""Canonical PR630 names retain the same source-owned query/settlement data.

Source authority: pylcm555c2c0ffc979118130341a9210b5922a0e8125b,
lcm/participation.py ParticipationRead/ParticipationCheck and Transition.participation.
"""

import pytest
from copy import deepcopy
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from tests.regime_finder.test_round5_query_admission import spec, record, reference
from tests.regime_finder.test_round4_semantic import captured_gate_graph
from importlib import import_module
from _lcm_regime_finder.records import JSONValue, SemanticCheck
from tests.regime_finder.test_declaration_ingestion import (
    json_entries,
    json_member,
    json_rows,
)
from _lcm_regime_finder.current_api_contract import inspect_authority
from _lcm_regime_finder.declarations import DeclarationError
from _lcm_regime_finder.facts import (
    FactsBackendName,
    current_reads,
    live_states,
    make_facts,
)
from _lcm_regime_finder.graph_scope import apply_scope
from _lcm_regime_finder.model_snapshot import read_model
from lcm.graphs import analyze_regime_structure
from tests.test_transition_roles import EdgeForm, build_model


def bind_types[Value: JSONValue](raw: Value) -> Value:
    if isinstance(raw, dict):
        tag = raw.get("type", "")
        assert isinstance(tag, str)
        label = tag.split(".")[-1]
        if label in {"ParticipationRead", "ParticipationCheck", "Transition"}:
            canonical = (
                "lcm.transition." if label == "Transition" else "lcm.participation."
            ) + label
            raw["type"] = canonical
            module, class_name = canonical.rsplit(".", 1)
            raw["canonical_type_source"] = inspect_authority(
                getattr(import_module(module), class_name)
            )
        for child in list(raw.values()):
            bind_types(child)
    elif isinstance(raw, list):
        for child in raw:
            bind_types(child)
    return raw


def current_spec() -> dict[str, JSONValue]:
    raw = spec()
    regimes = json_member(raw, "regimes")
    for name in regimes:
        json_member(regimes, name, "fields")["same_period_refs"] = {}
    read = reference("Q", "ParticipationRead")
    json_member(read, "fields").update(kind="Raw", field="support", observes=["x"])
    gate = record(
        "ParticipationCheck",
        predicate=record("callable"),
        reads={"policy": read},
        observes=["x"],
        routes={"self": record("MemberRoute", fallback=reference("R"))},
    )
    json_member(raw, "edges")["A"] = record(
        "Transition", law="B", participation={"B": gate}
    )
    return raw


def analyze(
    raw: dict[str, JSONValue],
) -> tuple[dict[str, JSONValue], dict[str, JSONValue], SemanticCheck]:
    raw = bind_types(raw)
    normalized = normalize_declarations(raw)
    analysis = MetadataAnalysis(normalized, facts_backend="metadata").run()
    checked = check_semantics(raw, normalized, analysis)
    return normalized, analysis, checked


def test_canonical_same_period_read_is_query_only() -> None:
    raw = spec(reference_kind="ParticipationRead")
    normalized, _, checked = analyze(raw)
    assert ({c["regime"] for c in json_rows(normalized, "cells")}) == ({"A", "B"})
    assert ({q["regime"] for q in json_entries(normalized, "query_owners")}) == (
        {"Q", "R"}
    )
    assert (checked["status"]) != ("FAIL"), checked


def test_canonical_transition_preserves_settlement_and_query() -> None:
    normalized, _, checked = analyze(current_spec())
    assert ({c["regime"] for c in json_rows(normalized, "cells")}) == ({"A", "B", "R"})
    assert ({q["regime"] for q in json_entries(normalized, "query_owners")}) == ({"Q"})
    assert (checked["status"]) != ("FAIL"), checked
    gates = [
        r
        for r in json_rows(normalized, "relations")
        if r["kind"] in {"arrival", "gate"}
    ]
    assert gates
    assert all(r["observes"] == ["x"] for r in gates)


def test_independent_check_rejects_erased_modern_gate() -> None:
    raw = current_spec()
    normalized, analysis, _ = analyze(raw)
    bad = deepcopy(normalized)
    bad["relations"] = [
        r for r in json_rows(bad, "relations") if r["kind"] not in {"arrival", "gate"}
    ]
    assert (check_semantics(raw, bad, analysis)["status"]) == ("FAIL")


def test_independent_check_rejects_changed_modern_read() -> None:
    raw = current_spec()
    normalized, analysis, _ = analyze(raw)
    bad = deepcopy(normalized)
    relation = next(
        r for r in json_rows(bad, "relations") if r["kind"] == "policy_query"
    )
    relation["read_field"] = "value"
    assert (check_semantics(raw, bad, analysis)["status"]) == ("FAIL")


def test_native_graph_canonical_participation_is_not_omitted() -> None:
    raw = captured_gate_graph()
    law = json_member(raw, "graph", "fields", "laws", "source", "fields")
    law["participation"] = law.pop("gated_edges")
    for gate in json_entries(law, "participation"):
        gate["type"] = "lcm.participation.ParticipationCheck"
        fields = json_member(gate, "fields")
        fields["predicate"] = fields.pop("gate")
        fields["reads"] = fields.pop("gate_refs")
        fields["routes"] = fields.pop("legs")
    normalized, analysis, checked = analyze(raw)
    assert any(
        r["kind"] in {"arrival", "gate"} for r in json_rows(normalized, "relations")
    )
    assert (checked["status"]) != ("FAIL"), checked
    bad = deepcopy(normalized)
    bad["relations"] = [
        r for r in json_rows(bad, "relations") if r["kind"] not in {"arrival", "gate"}
    ]
    assert (check_semantics(raw, bad, analysis)["status"]) == ("FAIL")


def test_modern_spelling_without_source_authority_is_refused() -> None:
    from _lcm_regime_finder.declarations import DeclarationError

    raw = current_spec()
    with pytest.raises(DeclarationError, match="canonical class source"):
        normalize_declarations(raw)


def test_changed_class_source_is_refused_by_both_sides() -> None:
    from _lcm_regime_finder.declarations import DeclarationError

    raw = bind_types(current_spec())
    normalized, analysis, _ = analyze(raw)
    bad = deepcopy(raw)
    json_member(
        bad, "edges", "A", "fields", "participation", "B", "canonical_type_source"
    )["source_sha256"] = "0" * 64
    with pytest.raises(DeclarationError, match="canonical class source"):
        normalize_declarations(bad)
    result = check_semantics(bad, normalized, analysis)
    assert (result["status"]) == ("FAIL")
    assert any(
        e["kind"] == "original participation class authority changed"
        for e in result["errors"]
    )


def test_duplicate_gate_and_participation_target_is_refused() -> None:
    from _lcm_regime_finder.declarations import DeclarationError

    raw = bind_types(current_spec())
    fields = json_member(raw, "edges", "A", "fields")
    fields["gates"] = deepcopy(fields["participation"])
    with pytest.raises(DeclarationError, match="conflicting"):
        normalize_declarations(raw)


def roles_spec() -> dict[str, JSONValue]:
    raw = spec()
    json_member(raw, "edges")["A"] = record("Transition", law="B", roles={"B": "f"})
    return raw


def test_transition_roles_in_a_declaration_spec_is_refused_by_both_sides() -> None:
    """A spec is read before roles are lowered, so its `roles` is refused."""
    with pytest.raises(DeclarationError, match=r"Transition\.roles.*gates"):
        normalize_declarations(roles_spec())
    normalized, analysis, _ = analyze(spec())
    result = check_semantics(roles_spec(), normalized, analysis)
    assert result["status"] == "FAIL"
    assert any(
        e["kind"] == "original transition roles are not read" for e in result["errors"]
    )


type CellFacts = dict[tuple[str, str], set[str]]


def _model_facts(
    edge_form: EdgeForm, backend: FactsBackendName
) -> tuple[CellFacts, CellFacts, list[tuple[str, str, str]]]:
    """Each cell's current reads and live states, and the relations' kinds."""
    original = read_model(build_model(edge_form=edge_form))
    declarations = apply_scope(original, normalize_declarations(original))
    facts = make_facts(
        backend,
        declarations,
        {"seconds": 900, "metadata_categories": 100_000},
        lambda: None,
    )
    live, _ = live_states(facts, lambda: None)

    def cell_key(cell_id: str) -> tuple[str, str]:
        cell = facts.cells[cell_id]
        return str(cell["period"]), str(cell["regime"])

    reads = {cell_key(c): current_reads(facts, facts.cells[c]) for c in facts.cells}
    relations = sorted(
        (str(r["kind"]), str(r["phase"]), str(r.get("member") or ""))
        for r in json_rows(declarations, "relations")
    )
    return reads, {cell_key(c): states for c, states in live.items()}, relations


@pytest.mark.parametrize("backend", ["metadata", "dense"])
def test_roles_model_has_the_facts_of_its_lowered_gate(
    backend: FactsBackendName,
) -> None:
    """A constructed model reads a role as the always-open gate it is lowered into."""
    roles = _model_facts("roles", backend)
    assert ("gate", "solve", "") in roles[2]
    assert ("fallback", "solve", "f") in roles[2]
    assert roles == _model_facts("exact_gate", backend)


def test_roles_model_analysis_equals_its_lowered_gate() -> None:
    """The public analysis of a roles model is complete and equals the gate's."""
    roles = analyze_regime_structure(
        build_model(edge_form="roles"), facts_backend="dense"
    )
    gate = analyze_regime_structure(
        build_model(edge_form="exact_gate"), facts_backend="dense"
    )
    assert roles.status == gate.status == "COMPLETE"
    assert roles.original.cost == gate.original.cost
    assert [c.cost for c in roles.candidates] == [c.cost for c in gate.candidates]


def test_user_binding_named_type_is_not_a_declaration_tag() -> None:
    from _lcm_regime_finder.current_api_contract import validate_original_types
    from _lcm_regime_finder.semantic_check import _check_current_type_sources

    raw = {
        "bindings": {"type": {"kind": "type", "type": "builtins.int"}},
        "ordinary_mapping": {"type": None, "kind": {"arbitrary": "binding"}},
    }
    validate_original_types(raw)
    failures = []
    _check_current_type_sources(
        raw, lambda *args, **kwargs: failures.append((args, kwargs))
    )
    assert (failures) == ([])
