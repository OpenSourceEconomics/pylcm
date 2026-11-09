"""Known generated type-check wrappers retain a verified finite interface."""
from copy import deepcopy
import functools
from pathlib import Path
import tempfile
import unittest

from beartype import beartype, BeartypeConf, BeartypeStrategy
from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.callable_identity import _beartype_interface
from _lcm_regime_finder.model_snapshot import source_function
from tests.regime_finder.test_metadata_independent import captured_snapshot


def typed_history(history: int) -> int:
    raise AssertionError('Export must never evaluate an economic callback')


CONF = BeartypeConf(strategy=BeartypeStrategy.On, is_pep484_tower=True)


class BeartypeInterfaceTests(unittest.TestCase):
    def test_native_model_export_keeps_beartyped_utility_and_default_aggregator(self):
        from lcm import Model, Regime, DiscreteGrid, AgeGrid, categorical
        from lcm.typing import ScalarInt
        from _lcm_regime_finder.pytask_api import export_model
        from _lcm_regime_finder.cli import load_problem

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class Regimes:
            alive: ScalarInt
            final: ScalarInt

        def terminal() -> float:
            return 0.0

        def next_history(history):
            return history

        model = Model(regimes={
            'alive': Regime(states={'history': DiscreteGrid(Binary)},
                           functions={'utility': beartype(conf=CONF)(typed_history)},
                           state_transitions={'history': next_history}),
            'final': Regime(functions={'utility': beartype(conf=CONF)(terminal)})},
            regime_id_class=Regimes, ages=AgeGrid(start=20, inclusive_stop=21, step='Y'),
            edges={'alive': {'final': (20,)}}, initial_nodes={20: 'alive'}, enable_jit=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.json'
            export_model(model, path, production_metadata={'status': 'regression'},
                seconds=None, host_gib=None,
                source_transaction={'kind': 'exclusive-owner', 'authority': 'This test owns the newly constructed Model'})
            original = load_problem(path)
            result = find_regimes(original, budget={'seconds': 30}, search_mode="exact")
            self.assertEqual(result['status'], 'COMPLETE', result['analysis']['obligations'])
            self.assertEqual(verify_production_result(original, result)['status'], 'PASS')

    def test_generated_wrapper_is_reconciled_without_mutating_or_calling_target(self):
        wrapper = beartype(conf=CONF)(typed_history)
        before = dict(vars(typed_history))
        actual = source_function(wrapper)
        self.assertEqual(vars(typed_history), before)
        self.assertIsNotNone(_beartype_interface(wrapper))
        self.assertTrue(actual['variadic'])
        self.assertIn('kwargs', actual['raw_arguments'])
        self.assertEqual(actual['arguments'], ['history'])
        self.assertEqual(actual['parameter_kinds'], {'history': 'POSITIONAL_OR_KEYWORD'})
        self.assertEqual(actual['advertised_signature_origin'], 'regenerated-beartype-wrapper/v1')
        self.assertIn('__beartype_func', actual['defaults'])
        source = captured_snapshot(actual)
        result = find_regimes(source, budget={'seconds': 20}, search_mode="exact")
        record = result['analysis']['declarations']['callbacks'][result['analysis']['declarations']['cells'][0]['functions']['utility']]
        self.assertFalse(record.get('unresolved_input_signature', False))
        self.assertTrue(all('history' in option['states'] for item in result['analysis']['interfaces'].values() for option in item['options']))
        self.assertEqual(verify_production_result(source, result)['status'], 'PASS')

    def test_marker_alone_and_changed_wrapper_cannot_claim_a_finite_interface(self):
        @functools.wraps(typed_history)
        def forged(*args, **kwargs):
            return kwargs['other_history']
        setattr(forged, "__beartype_wrapper", True)
        self.assertIsNone(_beartype_interface(forged))
        original = beartype(conf=CONF)(typed_history)
        changed = functools.wraps(original)(lambda *args, **kwargs: kwargs['other_history'])
        changed.__wrapped__ = typed_history
        setattr(changed, "__kwdefaults__", dict(getattr(original, "__kwdefaults__")))
        self.assertIsNone(_beartype_interface(changed))
        altered = beartype(conf=CONF)(typed_history)
        defaults = dict(getattr(altered, '__kwdefaults__') or {})
        defaults['__beartype_func'] = lambda history: 42
        setattr(altered, '__kwdefaults__', defaults)
        self.assertIsNone(_beartype_interface(altered))

    def test_original_payload_still_controls_independent_signature_authority(self):
        from tests.regime_finder.test_declaration_ingestion import decorated_callback
        source = captured_snapshot(source_function(decorated_callback))
        result = find_regimes(source, budget={'seconds': 20}, search_mode="exact")
        forged = deepcopy(result)
        for record in forged['analysis']['declarations']['callbacks'].values():
            record['arguments'] = ['history']
            record['parameter_kinds'] = {'history': 'POSITIONAL_OR_KEYWORD'}
            record.pop('unresolved_input_signature', None)
            record['status'] = 'structural'
        self.assertNotEqual(verify_production_result(source, forged)['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
