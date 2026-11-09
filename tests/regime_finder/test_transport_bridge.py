"""Actual inherited-checker regressions on automatically generated quotients."""
from copy import deepcopy
from fractions import Fraction
import json
import unittest

from _lcm_regime_finder.core import find_regimes
from _lcm_regime_finder.transport_bridge import validate_generated_transport
from tests.regime_finder.fixtures import external_read, fertility, problem, rare_incoming, retirement


def interface_at(report, age, status=None):
    atoms = report['analysis']['atoms']
    atom = next(a for a in atoms if a['age'] == age and (status is None or a['status'] == status))
    block = next(r for r in report['recommended']['regimes'] if atom['id'] in r['atoms'])
    return atom, block['semantic_interfaces'][atom['id']]


class GeneratedTransportTests(unittest.TestCase):
    def run_problem(self, p):
        report = find_regimes(p)
        self.assertIn(report['status'], ('OPTIMAL', 'INCOMPLETE'), report.get('error'))
        return report, validate_generated_transport(p, report)

    def assert_real_pass(self, result):
        self.assertEqual(result['status'], 'PASS', result['diagnostics'])
        self.assertTrue(result['checker_executed'])
        self.assertEqual(result['checker_result']['status'], 'PASS')
        self.assertGreater(result['checker_result']['layers']['finite_model']['checks'], 0)
        self.assertEqual(result['checker_result']['cases'][0]['backward_evaluation']['status'], 'passed')
        json.dumps(result)  # The complete generated evidence is exportable JSON.

    def test_fertility_actions_and_last_child_consumer(self):
        _, result = self.run_problem(fertility())
        self.assert_real_pass(result)
        self.assertEqual(result['generated']['original_cells'], 17)
        self.assertEqual(result['generated']['reduced_cells'], 14)
        self.assertEqual(result['generated']['original_feasible_action_rows'], 20)
        self.assertEqual(result['generated']['reduced_feasible_action_rows'], 17)
        data = result['checker_input']['cases'][0]
        for old_id, cell in data['original']['cells'].items():
            if 2 <= cell['stage'] < 8:
                self.assertEqual(data['certificate']['fixed_action_attributes'][old_id], {'birth': 0})
                reduced = data['reduced']['cells'][data['certificate']['state_map'][old_id]]
                self.assertTrue(all('birth' not in action['decoded'] for action in reduced['actions'].values()))
        self.assertEqual(len(data['starts']), len(data['original']['cells']))

    def test_retirement_keeps_lagged_bills_until_last_consumer(self):
        _, result = self.run_problem(retirement())
        self.assert_real_pass(result)
        self.assertEqual(result['generated']['original_cells'], 33)
        self.assertEqual(result['generated']['reduced_cells'], 18)
        self.assertEqual(result['generated']['original_feasible_action_rows'], 40)
        self.assertEqual(result['generated']['reduced_feasible_action_rows'], 25)
        data = result['checker_input']['cases'][0]
        for cell in data['reduced']['cells'].values():
            if cell['stage'] in (2, 3, 4):
                self.assertEqual(set(cell['state']), {'coverage_lag'})
            if cell['stage'] in (5, 6, 7):
                self.assertEqual(cell['state'], {})

    def test_positive_rare_simulation_edges_are_not_tolerance_zeroed(self):
        _, result = self.run_problem(rare_incoming(True, 'simulation'))
        self.assert_real_pass(result)
        data = result['checker_input']['cases'][0]
        tiny = Fraction(1, 10**50)
        for side in ('original', 'reduced'):
            actions = [a for c in data[side]['cells'].values() for a in c['actions'].values()]
            simulation = [Fraction(w) for a in actions for w in a['kernels']['simulation'].values()]
            solve = [Fraction(w) for a in actions for w in a['kernels']['solve'].values()]
            self.assertIn(tiny, simulation)
            self.assertNotIn(tiny, solve)
            self.assertTrue(all(w > 0 for w in simulation + solve))

    def test_corrupted_fertility_projection_fails_actual_reward_check(self):
        p = fertility()
        report = find_regimes(p)
        _, interface = interface_at(report, 0)
        interface['states'] = []
        result = validate_generated_transport(p, report)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['checker_executed'])
        witnesses = result['checker_result']['layers']['finite_model']['witnesses']
        self.assertTrue(any(w['condition'] == 'reward_transport' for w in witnesses), witnesses)

    def test_corrupted_rare_history_projection_fails_simulation_pushforward(self):
        p = rare_incoming(True, 'simulation')
        report = find_regimes(p)
        _, interface = interface_at(report, 2, {'status': 'active'})
        self.assertIn('history', interface['states'])
        interface['states'] = []
        result = validate_generated_transport(p, report)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['checker_executed'])
        witnesses = result['checker_result']['layers']['finite_model']['witnesses']
        self.assertTrue(any(w['condition'] == 'kernel_pushforward' and w.get('phase') == 'simulation' for w in witnesses), witnesses)

    def test_wrong_fixed_action_is_not_repaired_by_rederiving_interface(self):
        p = fertility()
        report = find_regimes(p)
        _, interface = interface_at(report, 2)
        interface['forced_actions']['birth'] = 1
        result = validate_generated_transport(p, report)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['checker_executed'])
        witnesses = result['checker_result']['layers']['finite_model']['witnesses']
        self.assertTrue(any(w['condition'] == 'feasible_action_surjection' for w in witnesses), witnesses)

    def test_changed_actor_role_is_rejected(self):
        p = retirement()
        report = find_regimes(p)
        report['recommended']['regimes'][0]['actor_information_role'] = {'actor': 'different'}
        result = validate_generated_transport(p, report)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['checker_executed'])
        self.assertEqual(result['diagnostics'][0]['code'], 'economic_role_changed')

    def test_budget_does_not_substitute_reachability_or_sampling(self):
        p = fertility()
        result = validate_generated_transport(p, find_regimes(p), max_cells=1)
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertFalse(result['checker_executed'])
        self.assertIsNone(result['checker_input'])
        self.assertEqual(result['generated']['original_cells'], 0)
        self.assertEqual(result['diagnostics'][0]['code'], 'finite_generation_budget')

    def test_symbolic_size_grid_is_not_executed(self):
        p = fertility()
        p['states']['child'] = {'size': 2, 'grid_id': 'child-grid-without-values'}
        _, result = self.run_problem(p)
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertFalse(result['checker_executed'])
        self.assertEqual(result['diagnostics'][0]['code'], 'nonenumerable_grid')

    def test_opaque_primitive_is_not_replaced_by_a_stub(self):
        p = fertility()
        p['programs'][0]['reward'] = {'opaque': 'actual_native_payoff', 'reads': ['child']}
        _, result = self.run_problem(p)
        self.assertEqual(result['status'], 'NOT_RUN')
        self.assertFalse(result['checker_executed'])
        self.assertEqual(result['diagnostics'][0]['code'], 'opaque_primitive')

    def test_ranked_same_period_extension_is_explicitly_unresolved(self):
        _, result = self.run_problem(external_read())
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertFalse(result['checker_executed'])
        self.assertEqual(result['layers']['finite_transport']['status'], 'NOT_RUN')
        self.assertEqual(result['diagnostics'][0]['code'], 'same_period_or_external_read')

    def test_nonunit_rows_are_preserved_and_not_called_exact_markov(self):
        p = problem('rounded-row', 1)
        p['programs'][0]['transitions'][0]['probability'] = '0.9999999999999'
        _, result = self.run_problem(p)
        self.assert_real_pass(result)
        self.assertFalse(result['exact_markov_normalization'])
        self.assertEqual(result['guarantee'], 'exact_raw_operator_transport_with_mass_error')
        self.assertFalse(result['checker_result']['probability_normalization']['weights_changed'])
        self.assertEqual(result['checker_result']['probability_normalization']['max_absolute_mass_error'], '1/10000000000000')
        data = result['checker_input']['cases'][0]
        for side in ('original', 'reduced'):
            weights = [Fraction(w) for c in data[side]['cells'].values() for a in c['actions'].values() for w in a['kernels']['solve'].values()]
            self.assertEqual(weights, [Fraction(9999999999999, 10000000000000)])

    def test_declared_action_dependent_outputs_use_full_decoded_actions(self):
        p = fertility()
        p['programs'][0]['outputs'] = {'chosen_birth': 'birth', 'child_plus_birth': 'child + birth'}
        _, result = self.run_problem(p)
        self.assert_real_pass(result)
        data = result['checker_input']['cases'][0]
        self.assertTrue(all('outputs_by_decoded_action' in c['observables'] for c in data['original']['cells'].values() if c['terminal'] is None))

    def test_state_and_action_grid_decoding_is_not_integer_index_substitution(self):
        p = problem('decoded-grids', 2, states={'wealth': {'values': [10, 20]}},
                    actions={'consume': {'values': [5, 7]}}, reward='wealth + consume', updates={'wealth': 'wealth'})
        _, result = self.run_problem(p)
        self.assert_real_pass(result)
        data = result['checker_input']['cases'][0]
        rewards = {Fraction(a['reward']['value']) for c in data['original']['cells'].values() for a in c['actions'].values()}
        self.assertEqual(rewards, {Fraction(15), Fraction(17), Fraction(25), Fraction(27)})

    def test_inputs_and_report_are_immutable(self):
        p = retirement()
        report = find_regimes(p)
        old_problem, old_report = deepcopy(p), deepcopy(report)
        self.assert_real_pass(validate_generated_transport(p, report))
        self.assertEqual(p, old_problem)
        self.assertEqual(report, old_report)


if __name__ == '__main__':
    unittest.main()
