"""Unknown effective input sets cannot authorize graph interface reduction."""
from copy import deepcopy
import inspect
import operator
import unittest

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.model_snapshot import source_function
from tests.regime_finder.test_declaration_ingestion import decorated_callback
from tests.regime_finder.test_metadata_independent import captured_snapshot
from tests.regime_finder.test_round4_semantic import one_cell


def keyword_inputs(**kwargs):
    return kwargs['history']


def positional_inputs(*args):
    return args[0]


def no_inputs():
    return 11


class AdvertisedInputs:
    @property
    def __signature__(self):
        return inspect.Signature([inspect.Parameter('history', inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    def __call__(self, *args, **kwargs):
        return kwargs['history'] if kwargs else args[0]


class UninspectableInputs:
    @property
    def __signature__(self):
        return 'unavailable'

    def __call__(self, *args, **kwargs):
        return kwargs['history'] if kwargs else args[0]


class EffectiveInputSignatureTests(unittest.TestCase):
    @staticmethod
    def utility_record(result):
        declarations = result['analysis']['declarations']
        callback = declarations['cells'][0]['functions']['utility']
        return declarations['callbacks'][callback]

    def test_variadic_and_uninspectable_inputs_keep_the_original_domain(self):
        # Direct source truth is independent of capture, normalization and replay:
        # an empty history interface would identify values 11 and 12.
        self.assertEqual([decorated_callback(history=h) for h in (0, 1)], [11, 12])
        self.assertEqual(list(inspect.signature(decorated_callback).parameters), ['history'])
        for callback in (decorated_callback, keyword_inputs, positional_inputs, UninspectableInputs()):
            with self.subTest(callback=type(callback).__name__ + ':' + getattr(callback, '__name__', 'itemgetter')):
                captured = source_function(callback)
                self.assertTrue(captured['variadic'])
                source = captured_snapshot(captured)
                result = find_regimes(source, budget={'seconds': 10}, search_mode="exact")
                self.assertEqual(result['status'], 'UNRESOLVED')
                self.assertTrue(any(item['kind'] == 'effective_input_signature'
                                    for item in result['analysis']['obligations']))
                self.assertTrue(all('history' in option['states']
                                    for interface in result['analysis']['interfaces'].values()
                                    for option in interface['options']))
                record = self.utility_record(result)
                self.assertEqual(record['status'], 'unresolved')
                self.assertNotIn('dependency_contract', record)
                self.assertNotEqual(verify_production_result(source, result)['status'], 'PASS')

    def test_independent_check_rejects_authority_with_flags_and_obligations_cleared(self):
        source = captured_snapshot(source_function(decorated_callback))
        forged = deepcopy(find_regimes(source, budget={'seconds': 10}, search_mode="exact"))
        analysis = forged['analysis']
        analysis['obligations'] = []
        analysis['declarations']['obligations'] = []
        record = self.utility_record(forged)
        record.pop('unresolved_input_signature', None)
        record['status'] = 'structural'
        record['dependency_contract'] = 'pylcm-declared-inputs/v1'
        record['expression'] = None
        checked = verify_production_result(source, forged)
        self.assertNotEqual(checked['status'], 'PASS')
        self.assertTrue(any(item['kind'] == 'finite effective callback input signature was not captured'
                            for item in checked['source_semantics']['unresolved']))

    def test_finite_advertised_inputs_survive_an_opaque_variadic_implementation(self):
        captured = source_function(AdvertisedInputs())
        self.assertTrue(captured['variadic'])
        self.assertEqual(captured['arguments'], ['history'])
        self.assertEqual(captured['parameter_kinds'], {'history': 'POSITIONAL_OR_KEYWORD'})
        result = find_regimes(captured_snapshot(captured), budget={'seconds': 10}, search_mode="exact")
        record = self.utility_record(result)
        self.assertEqual(record['dependency_contract'], 'pylcm-declared-inputs/v1')
        self.assertFalse(record.get('unresolved_input_signature', False))
        self.assertTrue(all('history' in option['states']
                            for interface in result['analysis']['interfaces'].values()
                            for option in interface['options']))

    def test_captured_empty_signature_is_still_a_finite_signature(self):
        captured = source_function(no_inputs)
        self.assertEqual(captured['parameter_kinds'], {})
        result = find_regimes(captured_snapshot(captured), budget={'seconds': 10}, search_mode="exact")
        self.assertFalse(self.utility_record(result).get('unresolved_input_signature', False))
        self.assertTrue(all('history' not in option['states']
                            for interface in result['analysis']['interfaces'].values()
                            for option in interface['options']))

    def test_native_itemgetter_uses_its_available_effective_signature(self):
        callback = operator.itemgetter(0)
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            self.skipTest('This Python version does not expose the native signature')
        captured = source_function(callback)
        self.assertEqual(captured['arguments'], list(signature.parameters))
        from _lcm_regime_finder.declarations import _unresolved_input_signature
        self.assertFalse(_unresolved_input_signature(captured, captured['callable_records']))

    def test_wrapped_original_graph_gate_and_fallback_are_contained(self):
        from tests.regime_finder.test_round4_semantic import captured_gate_graph
        for slot in ('gate', 'fallback'):
            with self.subTest(slot=slot):
                source = captured_gate_graph()
                gate = source['graph']['fields']['laws']['source']['fields']['gated_edges']['landing']['fields']
                fallback = gate['legs']['person']['fields']['fallback']['fields']
                if slot == 'gate':
                    gate['gate'] = source_function(decorated_callback)
                else:
                    fallback['projection']['history'] = source_function(decorated_callback)
                result = find_regimes(source, budget={'seconds': 10}, search_mode="exact")
                self.assertEqual(result['status'], 'UNRESOLVED')
                self.assertEqual(verify_production_result(source, result)['status'], 'UNRESOLVED')
                analysis = result['analysis']
                for cell in analysis['declarations']['cells']:
                    if cell['regime'] not in {'source', 'landing'}:
                        continue
                    for option in analysis['interfaces'][cell['id']]['options']:
                        self.assertEqual(option['states'], cell['states'])
                        self.assertEqual(option['actions'], cell['actions'])

    def test_untagged_declarative_callback_does_not_inspect_unused_helper_bodies(self):
        callback = {'arguments': ['history'], 'reads': ['history'],
                    'source': 'def utility(history):\n    return history\n',
                    'bindings': {'unused_helper': source_function(positional_inputs)},
                    'expression': 'history', 'status': 'proved',
                    'identity': 'explicit untagged declarative utility'}
        source = one_cell({'utility': callback}, {'utility': 'utility'},
                          [{'kind': 'reward', 'callback': 'utility', 'phase': 'both'}],
                          states={'history': 'h'},
                          domains={'h': {'kind': 'categorical', 'size': 2,
                                         'values': [0, 1], 'identity': 'original history'}})
        result = find_regimes(source, budget={'seconds': 10}, search_mode="exact")
        self.assertEqual(result['semantic_check']['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
