"""One shape-invariant native grid program remains one regime template."""
from copy import deepcopy
import unittest
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder.semantic_check import check_semantics
from tests.regime_finder.test_model_adapter import snapshot,function

class SpecializedGridIdentityTests(unittest.TestCase):
    def test_same_native_program_is_not_split_merely_by_its_period_argument(self):
        source=snapshot()
        grid={'kind':'object','type':'lcm.transition.PeriodSpecializedGrid',
            'fields':{'build':function('def build(period):\n    return make_grid(period)\n',['period'])},
            'resolved_cardinality':3,
            'resolved_cardinality_source':'native canonical phase grid; native specialization shape-invariance contract'}
        source['user_regimes']['original']['fields']['states']['x']=grid
        result=analyze_production(source,limits={'seconds':10}, search_mode="exact")
        self.assertEqual(result['status'],'COMPLETE',result.get('semantic_check'))
        ids={c['states']['x'] for c in result['analysis']['declarations']['cells']}
        self.assertEqual(len(ids),1)
        selected=result['scenarios'][0]['recommendation']['graph_id']
        point=next(p for p in result['scenarios'][0]['frontier'] if p['id']==selected)
        self.assertEqual(len(point['blocks']),1)
        bad=deepcopy(result['analysis']);bad['declarations']['domains'][next(iter(ids))]['size']=4
        checked=check_semantics(source,bad['declarations'],bad)
        self.assertTrue(any(e['kind']=='specialized native grid cardinality changed' for e in checked['errors']),checked)
