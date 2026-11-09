"""Endpoint-only layout preference does not restrict the admissible family."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.frontier_check import check_frontier
from _lcm_regime_finder.production_render import render_markdown, render_html
from tests.regime_finder.test_round6_frontier import fixture


WORK_FIRST = {'weights': {'work': 1000, 'storage': 1, 'clarity': 1}}


def endpoint_fixture(kind: str = 'state', dates: int = 2):
    analysis = fixture(dates)
    for cell in analysis['declarations']['cells']:
        cell['states']['history'] = 'history'
        option = analysis['interfaces'][cell['id']]['options'][0]
        option['states']['history'] = 'history'
        option['state_projection']['history'] = {'kind': 'identity'}
    analysis['declarations']['domains']['history'] = {'size': 10}
    option = analysis['interfaces'][f'a:{dates - 1}']['options'][0]
    if kind == 'state':
        del option['states']['history']
        option['state_projection']['history'] = {'kind': 'drop'}
    else:
        option['actions'] = {}
        option['action_reconstruction'] = {'choice': {'kind': 'constant', 'value': 0}}
    return analysis


def selected_plan(result):
    graph = result['recommendation']['graph_id']
    return next(point for point in result['frontier'] if point['id'] == graph)


class EndpointPreferenceTests(unittest.TestCase):
    def test_native_last_decision_keeps_topology_without_forcing_a_regime_split(self):
        from _lcm_regime_finder.production import analyze_production
        from tests.regime_finder.test_model_adapter import snapshot,function
        source=snapshot()
        source['user_regimes']['original']['fields']['functions']['utility']=function(
            'def utility(x,history,period):\n    return x+(history if period<5 else 0)\n',['x','history','period'])
        storage_first={'weights':{'work':1,'storage':1000,'clarity':1}}
        default=analyze_production(source,recommendation=storage_first, search_mode="exact")
        override=analyze_production(source,recommendation=storage_first|{'allow_final_only_split':True}, search_mode="exact")
        self.assertEqual(default['status'],'COMPLETE',default.get('semantic_check'))
        self.assertEqual(default['scenarios'][0]['frontier'],override['scenarios'][0]['frontier'])
        self.assertEqual(len(selected_plan(default['scenarios'][0])['blocks']),1)
        self.assertGreater(len(selected_plan(override['scenarios'][0])['blocks']),1)
    def test_default_defers_only_final_date_state_cleanup(self) -> None:
        analysis = endpoint_fixture()
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(len(selected_plan(result)['blocks']), 1)
        self.assertTrue(any(len(point['blocks']) == 2 for point in result['frontier']))
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_default_defers_only_final_date_forced_action(self) -> None:
        analysis = endpoint_fixture('action')
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(len(selected_plan(result)['blocks']), 1)
        self.assertTrue(any(len(point['blocks']) == 2 for point in result['frontier']))
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_override_changes_recommendation_not_frontier_or_certificate(self) -> None:
        for kind in ('state', 'action'):
            with self.subTest(kind=kind):
                analysis = endpoint_fixture(kind)
                default = search_frontier(analysis, recommendation=WORK_FIRST)
                override = search_frontier(analysis, recommendation=WORK_FIRST | {'allow_final_only_split': True})
                self.assertEqual(default['frontier'], override['frontier'])
                self.assertEqual(default['certificate'], override['certificate'])
                self.assertEqual(len(selected_plan(override)['blocks']), 2)
                self.assertTrue(default['recommendation']['endpoint_preference']['deferred_graph_ids'])
                self.assertEqual(check_frontier(analysis, override)['status'], 'PASS')

    def test_existing_last_date_node_remains_eligible(self) -> None:
        analysis = endpoint_fixture()
        analysis['declarations']['cells'][-1]['regime'] = 'existing_retired'
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(len(selected_plan(result)['blocks']), 2)
        self.assertEqual(result['recommendation']['endpoint_preference']['deferred_graph_ids'], [])
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_all_age_unused_coordinate_cleanup_remains_eligible(self) -> None:
        analysis = endpoint_fixture()
        analysis['declarations']['domains']['history']['size'] = 1
        option = analysis['interfaces']['a:0']['options'][0]
        del option['states']['history']
        option['state_projection']['history'] = {'kind': 'drop'}
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(len(selected_plan(result)['blocks']), 1)
        self.assertNotIn('history', selected_plan(result)['blocks'][0]['state_axes'])
        self.assertEqual(result['recommendation']['endpoint_preference']['deferred_graph_ids'], [])
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_earlier_multi_date_boundary_remains_eligible(self) -> None:
        analysis = endpoint_fixture(dates=3)
        option = analysis['interfaces']['a:1']['options'][0]
        del option['states']['history']
        option['state_projection']['history'] = {'kind': 'drop'}
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual([block['cells'] for block in selected_plan(result)['blocks']], [['a:0'], ['a:1', 'a:2']])
        self.assertEqual(result['recommendation']['endpoint_preference']['deferred_graph_ids'], [])
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_different_terminal_context_is_not_an_optional_cleanup(self) -> None:
        analysis = endpoint_fixture()
        analysis['declarations']['cells'][-1]['terminal'] = True
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(result['recommendation']['endpoint_preference']['deferred_graph_ids'], [])
        self.assertEqual(len(selected_plan(result)['blocks']), 2)
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_independent_check_rejects_tampered_selection_and_deferral(self) -> None:
        analysis = endpoint_fixture()
        default = search_frontier(analysis, recommendation=WORK_FIRST)
        override = search_frontier(analysis, recommendation=WORK_FIRST | {'allow_final_only_split': True})
        bad = deepcopy(default)
        for field in ('graph_id', 'score'):
            bad['recommendation'][field] = override['recommendation'][field]
        self.assertEqual(check_frontier(analysis, bad)['status'], 'FAIL')
        bad = deepcopy(default)
        bad['recommendation']['endpoint_preference']['deferred_graph_ids'] = []
        self.assertEqual(check_frontier(analysis, bad)['status'], 'FAIL')

    def test_invalid_override_is_not_interpreted_as_a_boolean(self) -> None:
        analysis = endpoint_fixture()
        for value in ('false', 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be a boolean'):
                search_frontier(analysis, recommendation={'allow_final_only_split': value})
        result = search_frontier(analysis)
        result['recommendation']['endpoint_preference']['allow_final_only_split'] = 'false'
        self.assertEqual(check_frontier(analysis, result)['status'], 'FAIL')

    def test_independent_policy_does_not_call_producer_classifier(self) -> None:
        from unittest.mock import patch
        analysis = endpoint_fixture()
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        with patch('_lcm_regime_finder.frontier._final_only_cleanup', side_effect=AssertionError('producer was called')):
            self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_changed_retained_interface_is_not_only_an_omission(self) -> None:
        analysis = endpoint_fixture()
        option = analysis['interfaces']['a:1']['options'][0]
        option['state_projection']['wealth'] = {'kind': 'characteristic', 'id': 'different'}
        result = search_frontier(analysis, recommendation=WORK_FIRST)
        self.assertEqual(result['recommendation']['endpoint_preference']['deferred_graph_ids'], [])
        self.assertEqual(check_frontier(analysis, result)['status'], 'PASS')

    def test_public_api_and_reports_expose_override(self) -> None:
        from _lcm_regime_finder.pytask_api import analyse_snapshot
        source = Path(__file__).resolve().parents[2] / 'tests/regime_finder/examples/declarations/tiny-huge-grid.json'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result'
            summary = analyse_snapshot(source, output, seconds=15, allow_final_only_split=True, search_mode="exact")
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(summary['recommendation_preferences'])
            self.assertTrue(all(row['allow_final_only_split'] is True for row in summary['recommendation_preferences']))
            self.assertIn('Explicit override: True', (output / 'next-actions.html').read_text())
            self.assertIn('--allow-final-only-split', render_markdown(report))
            self.assertIn('allow_final_only_split=True', render_html(report))

    def test_public_api_refuses_invalid_override_before_execution(self) -> None:
        from _lcm_regime_finder.pytask_api import analyse_snapshot
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'must be a boolean'):
                analyse_snapshot(folder / 'not-read.json', folder / 'not-created', allow_final_only_split='false', search_mode="exact")


if __name__ == '__main__':
    unittest.main()
