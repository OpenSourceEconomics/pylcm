"""Capture both public aggregator slots; never execute them."""
import unittest
from copy import deepcopy
from _lcm_regime_finder.model_snapshot import normalize

class DecoupledCaptureTests(unittest.TestCase):
    def test_public_callable_slots_are_retained(self):
        Decoupled=type('Decoupled',(),{'__slots__':('objective','value'),'__module__':'lcm.decoupled'})
        value=Decoupled()
        def objective(utility,CE,beta):return utility+beta*CE
        def stored(utility,CE,delta):return utility+delta*CE
        setattr(value, "objective", objective);setattr(value, "value", stored)
        captured=normalize(value)
        self.assertEqual(set(captured['fields']),{'objective','value'})
        self.assertEqual(captured['fields']['objective']['arguments'],['utility','CE','beta'])
        self.assertEqual(captured['fields']['value']['arguments'],['utility','CE','delta'])

    def test_wrapper_cannot_hide_a_captured_objective_input(self):
        from _lcm_regime_finder.production import analyze_production, MetadataAnalysis
        from _lcm_regime_finder.semantic_check import check_semantics
        from tests.regime_finder.test_model_adapter import snapshot,function
        source=snapshot()
        source['user_regimes']['original']['fields']['koopmans_aggregator']={
            'kind':'object','type':'lcm.decoupled.Decoupled','fields':{
                'objective':function('def objective(utility,CE,history):\n    return utility+CE+history\n',['utility','CE','history']),
                'value':function('def value(utility,CE):\n    return utility+CE\n',['utility','CE'])}}
        result=analyze_production(source,limits={'seconds':10}, search_mode="exact")
        self.assertEqual(result['status'],'COMPLETE')
        declaration=deepcopy(result['analysis']['declarations'])
        for callback in declaration['callbacks'].values():
            if callback.get('original',{}).get('type')=='lcm.decoupled.Decoupled':
                callback.update(arguments=[],reads=[])
        analysis=MetadataAnalysis(declaration).run()
        checked=check_semantics(source,declaration,analysis)
        self.assertTrue(any(e['kind']=='declared aggregator inputs changed' for e in checked['errors']),checked)
