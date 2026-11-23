"""The current `Transition` declaration: roles are read as their lowered gate.

A declaration spec is read before `Transition.roles` is lowered, so the finder
refuses a spec's `roles`; a constructed model reads each role as the always-open
gate it is lowered into.
"""

import pytest
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from tests.regime_finder.test_round5_query_admission import spec, record
from importlib import import_module
from _lcm_regime_finder.records import JSONValue, SemanticCheck
from tests.regime_finder.test_declaration_ingestion import (
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
        if label == "Transition":
            canonical = "lcm.transition." + label
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


def analyze(
    raw: dict[str, JSONValue],
) -> tuple[dict[str, JSONValue], dict[str, JSONValue], SemanticCheck]:
    raw = bind_types(raw)
    normalized = normalize_declarations(raw)
    analysis = MetadataAnalysis(normalized, facts_backend="metadata").run()
    checked = check_semantics(raw, normalized, analysis)
    return normalized, analysis, checked


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
