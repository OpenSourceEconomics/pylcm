from copy import deepcopy
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder import find_cost_frontier, check_cost_frontier, find_regimes, verify_certificate
from _lcm_regime_finder.cost_family import encode_analysis, chain_cover
from tests.regime_finder.test_forest_frontier import example, literal
from tests.regime_finder.test_aca_separator_reference import fixture
from tests.regime_finder.test_borella_symbolic import toy


def forest_analysis():
    original = example([{'x', 'y'}, {'x', 'z'}, {'y', 'z'}])
    cells, interfaces, policy = [], {}, {}
    for c in original['cells']:
        cell = c | {'regime': c['label'], 'role': None, 'ownership': None, 'declaration_type': 'native'}
        cells.append(cell)
        option = {'id': c['id'] + '/identity', 'states': c['states'], 'actions': c['actions'],
            'state_projection': {axis: {'kind': 'identity', 'target': axis} for axis in c['states']},
            'action_reconstruction': {}, 'constants': {}}
        interfaces[c['id']] = {'options': [option]}
        policy[c['id']] = option['id']
    return {'declarations': {'analysis_scope': 'pylcm-structural-graph/v1',
        'allocation': {'kind': 'per_block', 'boundary_unit': 1}, 'cells': cells,
        'domains': original['domains'], 'relations': []},
        'interfaces': interfaces, 'discovery_complete': True,
        'projection_policies': [{'id': 'original', 'option_by_cell': policy}]}, original


class CostFamilyTests(unittest.TestCase):
    def test_no_time_or_count_limits_across_all_cost_backends(self):
        from _lcm_regime_finder.cost_limits import DEFAULTS
        unlimited = dict.fromkeys(DEFAULTS)
        legacy, _ = forest_analysis()
        for cell in legacy['declarations']['cells']:
            cell['regime'] = 'one-dated-label'
        for analysis in (forest_analysis()[0], legacy, fixture(), toy()[0]):
            with self.subTest(symbolic=bool(analysis.get('characteristic_candidates'))):
                result = find_cost_frontier(analysis, seconds=None, limits=unlimited)
                self.assertEqual(result['status'], 'COMPLETE_COST_FAMILY')
                self.assertEqual(result['requested_bookkeeping_limits'], unlimited)
                json.dumps(result, allow_nan=False)
                self.assertTrue(check_cost_frontier(analysis, result, seconds=None,
                                                   limits=unlimited)['cost_family_complete_verified'])

    def test_chain_coordinates_keep_layout_continuity_without_dropping_policies(self):
        candidate = {'cells': ['v0', 'z0', 'z1', 'a0', 'a1'],
            'requires': {'v0': ['a0', 'z0'], 'z0': ['a1', 'z1'], 'z1': [], 'a0': ['a1'], 'a1': []}}
        cells = [{'id': node, 'regime': 'source' if node == 'v0' else 'left' if node.startswith('z') else 'right',
                  'states': {'claim': 'claim', 'x' if node in ('v0', 'z0', 'z1') else 'y': 'other'},
                  'actions': {}, 'period': i} for i, node in enumerate(candidate['cells'])]
        cover = chain_cover(candidate, cells)
        self.assertEqual({tuple(c) for c in cover}, {('v0', 'z0', 'z1'), ('a0', 'a1')})
        nodes = candidate['cells']
        closed = lambda selected: all(target in selected for source in selected
                                     for target in candidate['requires'][source])
        literal = {frozenset(s) for count in range(len(nodes) + 1) for s in combinations(nodes, count) if closed(s)}
        for chains in (cover, chain_cover(candidate)):
            represented = {frozenset(c for chain, threshold in zip(chains, thresholds) for c in chain[threshold:])
                for thresholds in product(*(range(len(chain) + 1) for chain in chains))}
            self.assertEqual({s for s in represented if closed(s)}, literal)

    def test_bookkeeping_caps_do_not_restrict_the_certified_family(self):
        analysis, _ = forest_analysis()
        partial = find_cost_frontier(analysis, limits={'forest_points': 1})
        self.assertEqual(partial['status'], 'INCOMPLETE')
        self.assertTrue(check_cost_frontier(analysis, partial)['original_representation_verified'])
        result = find_cost_frontier(analysis, limits={'forest_points': 10000})
        self.assertEqual(check_cost_frontier(analysis, result, limits={'frontier_points': 10000})['status'], 'PASS')
        self.assertEqual(result['analysis_sha256'], partial['analysis_sha256'])
        for bad in ({'forest_points': -1}, {'segments': True}, {'unknown': 1}, ['segments']):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                find_cost_frontier(analysis, limits=bad)

    def test_cost_parts_use_the_same_registered_reader_closure(self):
        from _lcm_regime_finder.cli import load_problem
        from _lcm_regime_finder.input_reads import discover_inputs
        from _lcm_regime_finder.frontier import digest
        analysis, _ = forest_analysis()
        packet = encode_analysis(analysis)
        records = []
        def intern(value):
            if not isinstance(value, (dict, list)):
                return value
            row = ({'mapping': [[k, intern(v)] for k, v in sorted(value.items())]} if isinstance(value, dict)
                   else {'sequence': [intern(v) for v in value]})
            records.append(row)
            return {'ref': len(records) - 1}
        root = intern(packet)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            part = directory / 'part.json'
            part.write_text(json.dumps({'records': records}))
            source = directory / 'input.json'
            source.write_text(json.dumps({'schema': 'interned-json-cost-instance/v1', 'root': root,
                'decoded_sha256': digest(packet), 'record_parts': [{'path': part.name,
                    'records': len(records), 'sha256': hashlib.sha256(part.read_bytes()).hexdigest()}]}))
            reads = discover_inputs([source])
            self.assertEqual(reads.bind(part), part.resolve())
            self.assertEqual(load_problem(source, reads=reads), packet)
            part.write_text('{}')
            with self.assertRaises(ValueError):
                load_problem(source, reads=reads)

    def test_cli_cost_delivery_and_atomic_incomplete_checkpoints(self):
        from _lcm_regime_finder.cli import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis, _ = forest_analysis()
            packet = encode_analysis(analysis)
            source = root / 'source.json'
            source.write_text(json.dumps(packet))
            output = root / 'result'
            code = main(['find-regimes', str(source), '--output', str(output), '--worker', '--seconds', '20'])
            self.assertEqual(code, 0)
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(report['independent_cost_check']['cost_family_complete_verified'])
            checkpoints = list((output / 'cost-checkpoints').rglob('ROOT.json'))
            self.assertTrue(checkpoints)
            for path in checkpoints:
                record = json.loads(path.read_text())
                self.assertFalse(record['complete_cost_family'])
                self.assertEqual(record['sha256'], hashlib.sha256((path.parent / record['table']).read_bytes()).hexdigest())
            checked = root / 'check.json'
            self.assertEqual(main(['check-result', str(source), str(output / 'report.json'),
                                  '--output', str(checked), '--worker']), 0)
            self.assertTrue(json.loads(checked.read_text())['cost_family_complete_verified'])

    def test_partial_cost_incumbent_is_checked_and_cannot_be_forged(self):
        analysis, _ = forest_analysis()
        result = find_cost_frontier(analysis, seconds=0)
        self.assertEqual(result['status'], 'INCOMPLETE')
        checked = check_cost_frontier(analysis, result)
        self.assertTrue(checked['original_representation_verified'])
        self.assertFalse(checked['cost_family_complete_verified'])
        result['original_representation']['cost']['work'] -= 1
        self.assertEqual(check_cost_frontier(analysis, result)['status'], 'FAIL')
    def test_forest_public_route_covers_unsupported_points(self):
        analysis, original = forest_analysis()
        result = find_cost_frontier(analysis, seconds=30)
        self.assertEqual(result['status'], 'COMPLETE_COST_FAMILY')
        self.assertEqual(result['frontier'], sorted(literal(original)))
        self.assertEqual(check_cost_frontier(analysis, result)['status'], 'PASS')
        self.assertIn((20, 20, 9), result['frontier'])
        self.assertEqual(result['legacy_endpoint_preference'], 'NOT_APPLIED')

    def test_exact_component_omission_and_source_promotion_fail(self):
        analysis, _ = forest_analysis()
        result = find_cost_frontier(analysis)
        for field, value in [('components', []), ('frontier', []), ('source_semantics', 'PASS')]:
            changed = deepcopy(result)
            changed[field] = value
            self.assertEqual(check_cost_frontier(analysis, changed)['status'], 'FAIL')
        original_changed = deepcopy(analysis)
        original_changed['declarations']['domains']['x']['size'] += 1
        self.assertEqual(check_cost_frontier(original_changed, result)['status'], 'FAIL')

    def test_separator_discovered_without_axis_or_label_names(self):
        analysis = fixture(2)
        for cell in analysis['declarations']['cells']:
            cell['regime'] = 'renamed-' + cell['regime']
        for record in analysis['interfaces'].values():
            for option in record['options']:
                if 'health' in option['states']:
                    option['states']['different-axis'] = option['states'].pop('health')
                    option['state_projection']['different-axis'] = option['state_projection'].pop('health')
        result = find_cost_frontier(analysis, seconds=30)
        self.assertEqual(result['backend'], 'ordered-domain-separator')
        self.assertEqual(check_cost_frontier(analysis, result)['status'], 'PASS')

    def test_symbolic_complete_closure_and_source_boundary(self):
        packet, chains = toy()
        result = find_regimes(packet, budget={'seconds': 30})
        self.assertEqual(result['backend'], 'symbolic-policy')
        self.assertEqual(result['status'], 'COMPLETE_COST_FAMILY')
        self.assertEqual(verify_certificate(packet, result)['status'], 'PASS')
        cover = chain_cover(packet['characteristic_candidates'][0])
        self.assertEqual(set(c for row in cover for c in row), set(c for row in chains for c in row))
        changed = deepcopy(result)
        changed['proof']['frontier'] = []
        self.assertEqual(check_cost_frontier(packet, changed)['status'], 'FAIL')

    def test_cost_packet_roundtrip_retains_original_policy_choices(self):
        analysis, _ = forest_analysis()
        packet = encode_analysis(analysis)
        result = find_regimes(packet)
        self.assertEqual(verify_certificate(packet, result)['status'], 'PASS')
        forged = deepcopy(packet)
        forged['policy_encoding']['basis'][0] = -1
        with self.assertRaises(ValueError):
            find_regimes(forged)


if __name__ == '__main__':
    unittest.main()
