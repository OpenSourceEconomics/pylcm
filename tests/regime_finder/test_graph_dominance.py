"""Homogeneous components have a dominance proof, not a smaller search family."""
from copy import deepcopy
import unittest
from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.frontier_check import check_frontier
from .test_round6_frontier import fixture


class GraphDominanceTests(unittest.TestCase):
    def test_domain_incompatibility_factorization_matches_full_family(self):
        for labels in (('a','b'),('a','b','c'),('a','b','c','d')):
            original=fixture(3,labels=labels,edges=[('a:0','b:1','positive')])
            for cell in original['declarations']['cells']:
                domain=cell['regime']+'_wealth'
                original['declarations']['domains'][domain]={'size':5}
                cell['states']['wealth']=domain
                original['interfaces'][cell['id']]['options'][0]['states']['wealth']=domain
            exhaustive=search_frontier(original)
            scoped=deepcopy(original);scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
            result=search_frontier(scoped)
            self.assertEqual([p['cost'] for p in exhaustive['frontier']],[p['cost'] for p in result['frontier']])
            self.assertEqual(check_frontier(scoped,result)['status'],'PASS')
            self.assertEqual(result['certificate']['component_reductions'][0]['rule'],'incompatible-domain-components/v1')
            forged=deepcopy(result);forged['certificate']['component_reductions'][0]['components']=[]
            self.assertEqual(check_frontier(scoped,forged)['status'],'FAIL')

    def test_many_incompatible_regimes_do_not_enumerate_impossible_groupings(self):
        scoped=fixture(3,labels=tuple(f'r{i}' for i in range(20)))
        for cell in scoped['declarations']['cells']:
            domain=cell['regime']+'_wealth'
            scoped['declarations']['domains'][domain]={'size':5}
            cell['states']['wealth']=domain
            scoped['interfaces'][cell['id']]['options'][0]['states']['wealth']=domain
        scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
        result=search_frontier(scoped,limits={'groupings':40,'seconds':10})
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(check_frontier(scoped,result,limits={'groupings':40,'seconds':10})['status'],'PASS')

    def test_a_compatible_bridge_keeps_cross_regime_search(self):
        original=fixture(2,labels=('a','b','bridge'))
        original['declarations']['domains'].update(a_wealth={'size':5},b_wealth={'size':5})
        for cell in original['declarations']['cells']:
            axes={} if cell['regime']=='bridge' else {'wealth':cell['regime']+'_wealth'}
            cell['states']=axes
            option=original['interfaces'][cell['id']]['options'][0]
            option['states']=axes.copy()
            if not axes:option['state_projection']={}
        exhaustive=search_frontier(original)
        original['declarations']['analysis_scope']='pylcm-structural-graph/v1'
        result=search_frontier(original)
        self.assertEqual([p['cost'] for p in exhaustive['frontier']],[p['cost'] for p in result['frontier']])
        self.assertFalse(any(r['rule']=='incompatible-domain-components/v1' for r in result['certificate']['component_reductions']))
        self.assertEqual(check_frontier(original,result)['status'],'PASS')

    def test_matches_exhaustive_family_and_rejects_forged_reduction(self):
        for labels in (('a',),('a','b'),('a','b','c')):
            original=fixture(3,labels=labels,edges=[('a:0','a:1','positive')])
            exhaustive=search_frontier(original)
            scoped=deepcopy(original);scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
            reduced=search_frontier(scoped)
            self.assertEqual([p['cost'] for p in reduced['frontier']],[p['cost'] for p in exhaustive['frontier']])
            self.assertEqual(check_frontier(scoped,reduced)['status'],'PASS')
            self.assertTrue(reduced['certificate']['component_reductions'])
            corrupt=deepcopy(reduced);corrupt['certificate']['component_reductions']=[]
            self.assertEqual(check_frontier(scoped,corrupt)['status'],'FAIL')

    def test_large_component_does_not_enumerate_set_partitions(self):
        scoped=fixture(2,labels=tuple(f'r{i}' for i in range(300)))
        scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
        result=search_frontier(scoped,limits={'groupings':2,'segments':10})
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['certificate']['search_counts']['groupings'],1)
        self.assertEqual(check_frontier(scoped,result,limits={'groupings':2,'segments':10})['status'],'PASS')

    def test_expensive_dispatch_keeps_full_tradeoff_search(self):
        scoped=fixture(2,labels=('a','b','c'));scoped['clarity_weights']={'block':1,'dispatch':10}
        baseline=search_frontier(scoped)
        scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
        result=search_frontier(scoped)
        self.assertEqual(result['certificate']['component_reductions'],[])
        self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in baseline['frontier']])
        self.assertEqual(check_frontier(scoped,result)['status'],'PASS')

    def test_layout_contraction_matches_exhaustive_label_partitions(self):
        for count in (3,4,5):
            original=fixture(1,labels=tuple(chr(97+i) for i in range(count)))
            original['declarations']['domains'].update(history={'size':3},health={'size':2})
            for index,cell in enumerate(original['declarations']['cells']):
                option=original['interfaces'][cell['id']]['options'][0]
                for name in (('history',) if index%3==1 else ('health',) if index%3==2 else ()):
                    cell['states'][name]=name;option['states'][name]=name
                    option['state_projection'][name]={'kind':'identity'}
            exhaustive=search_frontier(original)
            scoped=deepcopy(original);scoped['declarations']['analysis_scope']='pylcm-structural-graph/v1'
            contracted=search_frontier(scoped)
            self.assertEqual([point['cost'] for point in contracted['frontier']],
                             [point['cost'] for point in exhaustive['frontier']])
            self.assertEqual(check_frontier(scoped,contracted)['status'],'PASS')
            self.assertEqual(contracted['certificate']['component_reductions'][0]['rule'],
                             'single-date-layout-contraction/v1')
