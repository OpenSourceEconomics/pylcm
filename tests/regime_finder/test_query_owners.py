"""Query-only policy ownership must not be confused with physical admission."""
from copy import deepcopy
import unittest

from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from _lcm_regime_finder.graph_view import typed_incidence_graph
from _lcm_regime_finder.model_adapter import adapt_snapshot, ModelAdapterError
from tests.regime_finder.test_model_adapter import snapshot, function, grid


def policy_snapshot():
    raw=snapshot()
    raw['user_regimes']['query_partner']={
        'kind':'dataclass','type':'lcm.Regime','fields':{
            'states':{'partner_history':grid([0,1])},'actions':{},
            'functions':{'utility':function('def utility(partner_history):\n    return partner_history\n',['partner_history'])},
            'constraints':{},'state_transitions':{},
            'semantic_role':{'actor':'partner','information':['own history'],'commitments':[]}}}
    raw['user_regimes']['unused_owner']=deepcopy(raw['user_regimes']['query_partner'])
    raw['user_regimes']['original']['fields']['same_period_refs']={
        'partner_price':{'kind':'dataclass','type':'lcm.ProjectedRegimeValue','fields':{
            'regime':'query_partner','kind':'Raw','field':'value','observes':['history'],
            'projection':{'partner_history':function('def project(history):\n    return history\n',['history'])}}}}
    return raw


class QueryOwnerTests(unittest.TestCase):
    def checked(self, raw):
        normalized=normalize_declarations(raw)
        analysis=MetadataAnalysis(normalized).run()
        return normalized,analysis,check_semantics(raw,normalized,analysis)

    def test_current_read_owner_is_registered_without_physical_admission(self):
        raw=policy_snapshot()
        normalized=normalize_declarations(raw)
        self.assertTrue(normalized.get('query_owners'), 'Declared query-only owner was lost')
        self.assertEqual({cell['regime'] for cell in normalized['cells']},{'original'})
        self.assertTrue(all(item.get('regime')!='unused_owner' for item in normalized['query_owners'].values()))
        analysis=MetadataAnalysis(normalized).run()
        checked=check_semantics(raw,normalized,analysis)
        self.assertFalse(any(item['kind']=='original same-period reference target unavailable' for item in checked['unresolved']))
        self.assertTrue(all('history' in axes for axes in analysis['dependency_closure'].values()))

    def test_query_owner_is_not_a_visit_or_stored_bank(self):
        raw=policy_snapshot()
        normalized,analysis,checked=self.checked(raw)
        physical=normalize_declarations(snapshot())
        self.assertEqual([c['id'] for c in normalized['cells']],[c['id'] for c in physical['cells']])
        self.assertEqual([(r['source'],r['targets']) for r in normalized['relations'] if r['kind']=='transition'],
                         [(r['source'],r['targets']) for r in physical['relations'] if r['kind']=='transition'])
        self.assertTrue(all(q['storage']=='not_asserted' and q['admission']=='policy_query_only' for q in normalized['query_owners'].values()))
        self.assertEqual(normalized['allocation'],{'status':'unknown'})
        view=typed_incidence_graph(normalized)
        query_nodes=[n for n in view['nodes'] if n['node_kind']=='policy_query_owner']
        self.assertEqual(len(query_nodes),len(normalized['query_owners']))
        self.assertTrue(all(e['target'].startswith('query:') for e in view['edges'] if e['incidence']=='policy_query'))
        self.assertTrue(any(o['kind']=='original policy query factorization unproved' for o in checked['unresolved']))

    def test_wrong_registry_or_read_contract_is_rejected_independently(self):
        raw=policy_snapshot()
        normalized,analysis,_=self.checked(raw)
        mutations=[('owner',lambda n:n['query_owners'].clear()),
                   ('date',lambda n:next(iter(n['query_owners'].values())).__setitem__('period',99)),
                   ('phase',lambda n:next(iter(n['query_owners'].values())).__setitem__('phase','simulation')),
                   ('role',lambda n:next(iter(n['query_owners'].values())).__setitem__('role',{'actor':'reader'})),
                   ('storage',lambda n:next(iter(n['query_owners'].values())).__setitem__('storage','Full')),
                   ('admission',lambda n:next(iter(n['query_owners'].values())).__setitem__('admission','physical')),
                   ('read_kind',lambda n:next(r for r in n['relations'] if r['kind']=='policy_query').__setitem__('read_type','Participating')),
                   ('information',lambda n:next(r for r in n['relations'] if r['kind']=='policy_query').__setitem__('observes',[])),
                   ('query_target',lambda n:next(r for r in n['relations'] if r['kind']=='policy_query').__setitem__('query_target','not-an-owner')),
                   ('phase_incidence',lambda n:next(r for r in n['relations'] if r['kind']=='policy_query').__setitem__('phase','other'))]
        for name,mutate in mutations:
            with self.subTest(name=name):
                changed=deepcopy(normalized);mutate(changed)
                self.assertEqual(check_semantics(raw,changed,analysis)['status'],'FAIL')

    def test_source_bound_projection_and_genuine_dependency_controls(self):
        raw=policy_snapshot();normalized,analysis,_=self.checked(raw)
        changed=deepcopy(normalized)
        relation=next(r for r in changed['relations'] if r['kind']=='policy_query')
        callback=changed['callbacks'][relation['callbacks'][0]]
        callback['original']['fields']['projection']['partner_history']['source']='def project(history):\n    return 0\n'
        self.assertEqual(check_semantics(raw,changed,analysis)['status'],'FAIL')
        changed=deepcopy(analysis)
        for axes in changed['dependency_closure'].values():
            if 'history' in axes:axes.remove('history')
        self.assertEqual(check_semantics(raw,normalized,changed)['status'],'FAIL')

    def test_missing_target_stays_an_explicit_obligation(self):
        raw=policy_snapshot();del raw['user_regimes']['query_partner']
        normalized,analysis,checked=self.checked(raw)
        self.assertEqual(normalized['query_owners'],{})
        self.assertTrue(any(o['kind']=='original same-period reference target unavailable' for o in checked['unresolved']))

    def test_nested_query_owners_and_phases_are_separate(self):
        raw=policy_snapshot()
        partner=raw['user_regimes']['query_partner']['fields']
        nested=deepcopy(raw['user_regimes']['original']['fields']['same_period_refs']['partner_price'])
        nested['fields']['regime']='unused_owner'
        partner['same_period_refs']={'nested':nested}
        normalized,analysis,checked=self.checked(raw)
        self.assertEqual({q['regime'] for q in normalized['query_owners'].values()},{'query_partner','unused_owner'})
        self.assertEqual({q['phase'] for q in normalized['query_owners'].values()},{'solve','simulation'})
        self.assertTrue(all(q['dependencies'] for q in normalized['query_owners'].values() if q['regime']=='query_partner'))
        self.assertNotEqual(checked['status'],'FAIL',checked)

    def test_legacy_scalar_adapter_refuses_to_erase_query_owner(self):
        with self.assertRaisesRegex(ModelAdapterError,'policy-query owners'):
            adapt_snapshot(policy_snapshot())


if __name__=='__main__':unittest.main()
