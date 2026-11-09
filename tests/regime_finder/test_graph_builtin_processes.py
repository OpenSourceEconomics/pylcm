"""Intrinsic process dependencies and query identity survive graph-only scope."""
from copy import deepcopy
import unittest
from _lcm_regime_finder.production import analyze_production
from _lcm_regime_finder import verify_production_result
from tests.regime_finder.test_model_adapter import snapshot, function
from tests.regime_finder.test_query_owners import policy_snapshot

def process(name):
    module='ar1' if 'AR1' in name else 'iid'
    return {'kind':'dataclass','type':f'_lcm.processes.{module}.{name}',
            'fields':{'n_points':3,'state_conditioned':None,'sigma':1.0,'mu':0.0,'rho':0.8}}

class GraphBuiltinProcessTests(unittest.TestCase):
    def checked(self,source):
        result=analyze_production(source,limits={'seconds':15}, search_mode="exact")
        self.assertEqual(result['status'],'COMPLETE',str(result.get('semantic_check'))[:2000]+str(result['analysis']['obligations']))
        self.assertEqual(verify_production_result(source,result)['status'],'PASS')
        return result

    def test_persistent_process_keeps_its_current_state_dependency(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        fields['states']['x']=process('RouwenhorstAR1Process');del fields['state_transitions']['x']
        result=self.checked(source)
        bad=deepcopy(result)
        cb=next(c for c in bad['analysis']['declarations']['callbacks'].values() if c.get('original',{}).get('type','').endswith('RouwenhorstAR1Process'))
        cb.update(arguments=[],reads=[])
        self.assertEqual(verify_production_result(source,bad)['status'],'FAIL')

    def test_new_iid_is_drawn_in_arrival_regime_without_a_current_value(self):
        source=snapshot();arrival=deepcopy(source['user_regimes']['original'])
        arrival['fields']['states']['fresh']=process('NormalIIDProcess')
        arrival['fields']['functions']['utility']=function('def utility(x,fresh):\n    return x+fresh\n',['x','fresh'])
        source['user_regimes']['arrival']=arrival
        source['reachability']['fields']['nodes']=[[0,'original']]+[[t,'arrival'] for t in range(1,6)]
        rows=source['reachability']['fields']['solution']['fields']['targets_by_period']
        for t in range(5):rows[t]={'original' if t==0 else 'arrival':['arrival']}
        self.checked(source)

    def test_query_identity_is_preserved_without_numerical_factorization_claim(self):
        result=self.checked(policy_snapshot())
        self.assertTrue(result['analysis']['declarations']['query_owners'])

    def test_unknown_process_class_is_not_dispatched_as_a_library_primitive(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        fields['states']['x']=process('RouwenhorstAR1Process')
        fields['states']['x']['type']='foreign.RouwenhorstAR1Process'
        del fields['state_transitions']['x']
        result=analyze_production(source,limits={'seconds':15}, search_mode="exact")
        self.assertNotEqual(result['status'],'COMPLETE')

    def test_lowered_joint_normal_coordinate_is_an_iid_latent_axis(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        coordinate=process('NormalIIDProcess')
        coordinate['type']='_lcm.processes.joint_normal_lowering._JointNormalCoordinate'
        coordinate['fields']['group']='income'
        coordinate['fields']['joint']={'kind':'dataclass','type':'_lcm.processes.joint_normal.JointNormalIIDProcess',
            'fields':{'states':['income','return'],'cov':[[1,.5],[.5,1]],'mean':[0,0]}}
        fields['states']['x']=coordinate;del fields['state_transitions']['x']
        result=self.checked(source)
        callbacks=result['analysis']['declarations']['callbacks'].values()
        self.assertTrue(all(cb['arguments']==[] for cb in callbacks if cb.get('original',{}).get('type')==coordinate['type']))

    def test_empty_target_map_does_not_create_a_next_state_provider(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        fields['state_transitions']['absent']={}
        result=self.checked(source)
        self.assertTrue(all('next_absent' not in c['functions'] for c in result['analysis']['declarations']['cells']))

    def test_canonical_identity_preserves_a_categorical_characteristic(self):
        from _lcm_regime_finder.semantic_check import _identity_callback
        raw={'arguments':['history'],'bound_self':{'type':'_lcm.identity_transition._IdentityTransition',
            'fields':{'_state_name':'history'}}}
        self.assertTrue(_identity_callback({'original':raw},'history'))
        self.assertFalse(_identity_callback({'original':raw},'other'))
        raw['arguments']=[]
        self.assertFalse(_identity_callback({'original':raw},'history'))

    def test_equal_persistent_process_specs_keep_distinct_state_owners(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        fields['states']={name:process('RouwenhorstAR1Process') for name in ('x','history')}
        fields['state_transitions']={}
        fields['functions']['utility']=function('def utility(history,period):\n    return history if period>=3 else 0\n',['history','period'])
        result=self.checked(source)
        self.assertTrue(all('history' in axes and 'x' not in axes for axes in result['analysis']['dependency_closure'].values()))
        from _lcm_regime_finder.production import MetadataAnalysis
        from _lcm_regime_finder.semantic_check import check_semantics
        declaration=deepcopy(result['analysis']['declarations'])
        relation=next(r for r in declaration['relations'] if r['kind']=='transition')
        relation['state_updates']['history']=relation['state_updates']['x']
        changed=MetadataAnalysis(declaration).run()
        checked=check_semantics(source,declaration,changed)
        self.assertTrue(any(e['kind']=='original process dependency owner changed' for e in checked['errors']),checked)

    def test_phased_process_owner_is_the_axis_not_the_phase_slot(self):
        source=snapshot();fields=source['user_regimes']['original']['fields']
        process_grid={'kind':'dataclass','type':'lcm.phased.Phased','fields':{
            'solve':process('RouwenhorstAR1Process'),'simulate':process('TauchenAR1Process')}}
        fields['states']['x']=process_grid;del fields['state_transitions']['x']
        result=self.checked(source)
        for callback in result['analysis']['declarations']['callbacks'].values():
            if callback.get('original',{}).get('type','').startswith('_lcm.processes.ar1.'):
                self.assertEqual(callback['arguments'],['x'])
