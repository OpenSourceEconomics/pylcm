"""Native declarations supply finite DAG interfaces without wrapper regeneration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.pytask_api import export_model
from _lcm_regime_finder.semantic_check import _CallableRecords, _original_inputs_unresolved
from _lcm_regime_finder.structural_store import final_input_check


class ModelDeclarationInterfaceTests(unittest.TestCase):
    def test_native_public_interfaces_cover_transition_wrappers(self):
        from beartype import beartype
        from lcm import Model, Regime, DiscreteGrid, InitialNodes, Periods, StochasticTransition, categorical
        from lcm.typing import ScalarInt, FloatND

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class Regimes:
            alive: ScalarInt
            final: ScalarInt

        @beartype
        def utility(*, history: ScalarInt) -> float:
            raise AssertionError('Inspection must not evaluate economic functions')

        @beartype
        def probabilities(*, history: ScalarInt) -> FloatND:
            raise AssertionError('Inspection must not evaluate transition laws')

        model = Model(n_periods=2, regime_id_class=Regimes, regimes={
            'alive': Regime(states={'history': DiscreteGrid(Binary)},
                           functions={'utility': utility},
                           state_transitions={'history': StochasticTransition(func=probabilities)}),
            'final': Regime(states={'history': DiscreteGrid(Binary)},
                           functions={'utility': utility})},
            initial_nodes=InitialNodes(by_period={0:'alive'}),
            edges={'alive': {'final': Periods(values=(0,))}}, enable_jit=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.json'
            with patch('_lcm_regime_finder.callable_identity._beartype_interface',
                       side_effect=AssertionError('Native export must use public declaration inputs')):
                receipt = export_model(model, path, production_metadata={'fixture': True},
                    seconds=None, host_gib=None,
                    source_transaction={'kind':'exclusive-owner','authority':'test owns this Model'})
            self.assertEqual(receipt['status'], 'COMPLETE', receipt)
            source = load_problem(path)
            self.assertEqual(source['declaration_api']['schema_version'], 1)
            self.assertFalse(_original_inputs_unresolved(source['user_regimes'],
                _CallableRecords(source['callable_records'])))
            self.assertTrue(any(record.get('advertised_signature_origin') ==
                'pylcm-model-declarations/v1' for record in source['callable_records'].values()))
            final_input_check(source)


if __name__ == '__main__':
    unittest.main()
