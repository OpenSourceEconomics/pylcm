"""Canonical PR630 names retain the same source-owned query/settlement data.

Source authority: pylcm555c2c0ffc979118130341a9210b5922a0e8125b,
lcm/participation.py ParticipationRead/ParticipationCheck and Transition.participation.
"""
from copy import deepcopy
import unittest
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from tests.regime_finder.test_round5_query_admission import spec, record, reference
from tests.regime_finder.test_round4_semantic import captured_gate_graph
from importlib import import_module
from _lcm_regime_finder.current_api_contract import inspect_authority

def bind_types(raw):
    if isinstance(raw, dict):
        tag = raw.get("type", "")
        label = tag.split(".")[-1]
        if label in {"ParticipationRead", "ParticipationCheck", "Transition"}:
            canonical = ("lcm.transition." if label == "Transition" else "lcm.participation.") + label
            raw["type"] = canonical
            module, class_name = canonical.rsplit(".", 1)
            raw["canonical_type_source"] = inspect_authority(getattr(import_module(module), class_name))
        for child in list(raw.values()): bind_types(child)
    elif isinstance(raw, list):
        for child in raw: bind_types(child)
    return raw


def current_spec():
    raw = spec()
    for name in raw['regimes']:
        raw['regimes'][name]['fields']['same_period_refs'] = {}
    read = reference('Q', 'ParticipationRead')
    read['fields'].update(kind='Raw', field='support', observes=['x'])
    gate = record('ParticipationCheck', predicate=record('callable'),
                  reads={'policy': read}, observes=['x'],
                  routes={'self': record('StakeholderRoute', fallback=reference('R'))})
    raw['edges']['A'] = record('Transition', law='B', participation={'B': gate})
    return raw


def analyze(raw):
    raw = bind_types(raw)
    normalized = normalize_declarations(raw)
    analysis = MetadataAnalysis(normalized).run()
    checked = check_semantics(raw, normalized, analysis)
    return normalized, analysis, checked


class CurrentParticipationTests(unittest.TestCase):
    def test_canonical_same_period_read_is_query_only(self):
        raw = spec(reference_kind='ParticipationRead')
        normalized, _, checked = analyze(raw)
        self.assertEqual({c['regime'] for c in normalized['cells']}, {'A', 'B'})
        self.assertEqual({q['regime'] for q in normalized['query_owners'].values()}, {'Q', 'R'})
        self.assertNotEqual(checked['status'], 'FAIL', checked)

    def test_canonical_transition_preserves_settlement_and_query(self):
        normalized, _, checked = analyze(current_spec())
        self.assertEqual({c['regime'] for c in normalized['cells']}, {'A', 'B', 'R'})
        self.assertEqual({q['regime'] for q in normalized['query_owners'].values()}, {'Q'})
        self.assertNotEqual(checked['status'], 'FAIL', checked)
        gates = [r for r in normalized['relations'] if r['kind'] in {'arrival', 'gate'}]
        self.assertTrue(gates)
        self.assertTrue(all(r['observes'] == ['x'] for r in gates))

    def test_independent_check_rejects_erased_modern_gate(self):
        raw = current_spec(); normalized, analysis, _ = analyze(raw)
        bad = deepcopy(normalized)
        bad['relations'] = [r for r in bad['relations'] if r['kind'] not in {'arrival', 'gate'}]
        self.assertEqual(check_semantics(raw, bad, analysis)['status'], 'FAIL')

    def test_independent_check_rejects_changed_modern_read(self):
        raw = current_spec(); normalized, analysis, _ = analyze(raw)
        bad = deepcopy(normalized)
        relation = next(r for r in bad['relations'] if r['kind'] == 'policy_query')
        relation['read_field'] = 'value'
        self.assertEqual(check_semantics(raw, bad, analysis)['status'], 'FAIL')

    def test_native_graph_canonical_participation_is_not_omitted(self):
        raw = captured_gate_graph()
        law = raw['graph']['fields']['laws']['source']['fields']
        law['participation'] = law.pop('gated_edges')
        for gate in law['participation'].values():
            gate['type'] = 'lcm.participation.ParticipationCheck'
            fields = gate['fields']
            fields['predicate'] = fields.pop('gate')
            fields['reads'] = fields.pop('gate_refs')
            fields['routes'] = fields.pop('legs')
        normalized, analysis, checked = analyze(raw)
        self.assertTrue(any(r['kind'] in {'arrival', 'gate'} for r in normalized['relations']))
        self.assertNotEqual(checked['status'], 'FAIL', checked)
        bad = deepcopy(normalized)
        bad['relations'] = [r for r in bad['relations'] if r['kind'] not in {'arrival', 'gate'}]
        self.assertEqual(check_semantics(raw, bad, analysis)['status'], 'FAIL')

    def test_modern_spelling_without_source_authority_is_refused(self):
        from _lcm_regime_finder.declarations import DeclarationError
        raw = current_spec()
        with self.assertRaisesRegex(DeclarationError, 'canonical class source'):
            normalize_declarations(raw)

    def test_changed_class_source_is_refused_by_both_sides(self):
        from _lcm_regime_finder.declarations import DeclarationError
        raw = bind_types(current_spec())
        normalized, analysis, _ = analyze(raw)
        bad = deepcopy(raw)
        bad['edges']['A']['fields']['participation']['B']['canonical_type_source']['source_sha256'] = '0' * 64
        with self.assertRaisesRegex(DeclarationError, 'canonical class source'):
            normalize_declarations(bad)
        result = check_semantics(bad, normalized, analysis)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any(e['kind']=='original participation class authority changed' for e in result['errors']))

    def test_duplicate_gate_and_participation_target_is_refused(self):
        from _lcm_regime_finder.declarations import DeclarationError
        raw = bind_types(current_spec())
        raw['edges']['A']['fields']['gates'] = deepcopy(raw['edges']['A']['fields']['participation'])
        with self.assertRaisesRegex(DeclarationError, 'conflicting'):
            normalize_declarations(raw)

    def test_user_binding_named_type_is_not_a_declaration_tag(self):
        from _lcm_regime_finder.current_api_contract import validate_original_types
        from _lcm_regime_finder.semantic_check import _check_current_type_sources
        raw = {'bindings': {'type': {'kind':'type', 'type':'builtins.int'}},
               'ordinary_mapping': {'type': None, 'kind': {'arbitrary': 'binding'}}}
        validate_original_types(raw)
        failures=[]
        _check_current_type_sources(raw, lambda *args, **kwargs:failures.append((args,kwargs)))
        self.assertEqual(failures, [])


if __name__ == '__main__': unittest.main()
