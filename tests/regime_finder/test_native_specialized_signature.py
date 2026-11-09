"""Actual native constructor/export containment; no solve or simulation."""
from importlib.util import find_spec
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder import find_regimes, verify_production_result
from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.pytask_api import export_model


def economic_utility(choice, history):
    return -(choice - history) ** 2


def next_history(history):
    return history


def terminal_value():
    return 0.0


def finite_aggregator(utility, CE, discount_factor):
    return utility + discount_factor * CE


BUILDS = []


def build_utility(age):
    BUILDS.append(age)
    return economic_utility


def same_signature(age):
    return 'history-dependent-utility'


@unittest.skipUnless(find_spec('lcm'), 'Native pylcm dependency required')
class NativeSpecializedSignatureTests(unittest.TestCase):
    def test_constructor_export_and_public_analysis_retain_original_domain(self):
        from lcm import AgeGrid, AgeSpecializedFunction, DiscreteGrid, Model, Regime, Phased, categorical
        from lcm.typing import ScalarInt

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class RegimeId:
            original: ScalarInt
            final: ScalarInt

        for phased in (False, True):
            with self.subTest(phased=phased), tempfile.TemporaryDirectory() as temp:
                wrapper = AgeSpecializedFunction(build=build_utility, signature=same_signature)
                function = Phased(solve=wrapper, simulate=wrapper) if phased else wrapper
                model = Model(
                    regimes={'original': Regime(
                        states={'history': DiscreteGrid(Binary)},
                        actions={'choice': DiscreteGrid(Binary)},
                        functions={'utility': function},
                        state_transitions={'history': next_history}),
                        'final': Regime(functions={'utility': terminal_value})},
                    regime_id_class=RegimeId,
                    ages=AgeGrid(start=20, inclusive_stop=26, step='Y'),
                    edges={'original': {'original': tuple(range(20, 25)), 'final': (25,)}},
                    initial_nodes={20: 'original'}, enable_jit=False,
                    # Isolate specialization from generated type-check wrappers
                    # whose physical variadic signatures remain unresolved.
                    koopmans_aggregator=finite_aggregator,
                )
                after_construction = list(BUILDS)
                destination = Path(temp)/'constructed-model.json'
                export_model(model, destination,
                             production_metadata={'status': 'regression', 'no_production_claim': True},
                             source_transaction={'kind': 'exclusive-owner',
                                                 'authority': 'this serial test owns the newly constructed Model'},
                             seconds=30, host_gib=8)
                self.assertEqual(BUILDS, after_construction)
                source = load_problem(destination)
                self.assertEqual(source['capture_report']['status'], 'COMPLETE')
                resolved = find_regimes(source, budget={'seconds': 30}, search_mode="exact")
                self.assertEqual(resolved['status'], 'COMPLETE', resolved['analysis']['obligations'])
                self.assertEqual(verify_production_result(source, resolved)['status'], 'PASS')
                self.assertEqual(BUILDS, after_construction)

                # Old captures lack construction products: the safety guard
                # must still retain their whole economic domain.
                def without_resolutions(value):
                    if isinstance(value, dict):
                        value.pop('resolved_by_period', None)
                        for child in value.values(): without_resolutions(child)
                    elif isinstance(value, list):
                        for child in value: without_resolutions(child)
                source = deepcopy(source)
                without_resolutions(source)
                report = find_regimes(source, budget={'seconds': 30}, search_mode="exact")
                self.assertEqual(report['status'], 'UNRESOLVED')
                analysis = report['analysis']
                self.assertEqual({cell['period'] for cell in analysis['declarations']['cells']}, set(range(7)))
                self.assertTrue(any(item['kind']=='specialized_effective_signature'
                                    for item in analysis['obligations']))
                for cell in analysis['declarations']['cells']:
                    if cell['regime']=='original':
                        self.assertIn('history', analysis['dependency_closure'][cell['id']])
                    for option in analysis['interfaces'][cell['id']]['options']:
                        self.assertEqual(option['states'], cell['states'])
                        self.assertEqual(option['actions'], cell['actions'])
                self.assertEqual(verify_production_result(source, report)['status'], 'UNRESOLVED')
                self.assertEqual(BUILDS, after_construction)


if __name__ == '__main__':
    unittest.main()
