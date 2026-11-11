"""Original-family, independent binary-model and exact recurrence integration."""
from copy import deepcopy
import unittest
import json

from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.frontier_check import check_frontier
from _lcm_regime_finder.signature_model import build
from _lcm_regime_finder.signature_proof import binary_model, canonical_model, digest
from .round6_frontier_oracle import brute_family, canonical
from .test_round6_frontier import fixture


def heterogeneous(labels='abc',edges=()):
    analysis=fixture(1,labels=tuple(labels),edges=edges)
    analysis['declarations']['domains'].update(h2={'size':2},h3={'size':3})
    for index,cell in enumerate(analysis['declarations']['cells']):
        if index%3:
            domain='h2' if index%3==1 else 'h3'
            cell['states']['health']=domain
            option=analysis['interfaces'][cell['id']]['options'][0]
            option['states']['health']=domain
            option['state_projection']['health']={'kind':'identity'}
    return analysis



class SignatureIntegrationTests(unittest.TestCase):
    def test_all_directed_three_cell_graphs_match_literal_partition_frontiers(self):
        edges=[(a+':0',b+':0') for a in 'abc' for b in 'abc' if a!=b]
        for mask in range(1<<len(edges)):
            analysis=heterogeneous(edges=[(*edge,'unknown') for i,edge in enumerate(edges) if mask&(1<<i)])
            result=search_frontier(analysis);expected=brute_family(analysis)
            with self.subTest(mask=mask):
                self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in expected])
                self.assertEqual([canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in result['frontier']],
                                 [canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in expected])
                self.assertEqual(check_frontier(analysis,result)['status'],'PASS')
                for record in result['certificate']['signature_records']:
                    self.assertEqual(build(record['instance']),canonical_model(binary_model(record['instance'])))

    def test_equal_input_layouts_may_choose_different_bins_with_internal_edges(self):
        analysis=heterogeneous('abcd',edges=[('a:0','b:0','positive'),('d:0','c:0','positive')])
        result=search_frontier(analysis)
        self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in brute_family(analysis)])
        self.assertEqual(check_frontier(analysis,result)['status'],'PASS')

    def test_saved_complete_signature_certificate_replays_after_json_roundtrip(self):
        analysis=heterogeneous('ab',edges=[('a:0','b:0','positive')])
        result=json.loads(json.dumps(search_frontier(analysis)))
        self.assertEqual(check_frontier(analysis,result)['status'],'PASS')

    def test_large_assignment_cover_stays_incomplete_but_model_is_checked(self):
        analysis=heterogeneous('abcdefgh',edges=[('a:0','b:0','positive')])
        result=search_frontier(analysis,limits={'groupings':5})
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['exhaustion_reason'],'signature assignment cover budget')
        records=result['certificate']['signature_records'];self.assertTrue(records)
        self.assertTrue(all(not record['model']['frontier_complete_verified'] for record in records))
        checked=check_frontier(analysis,result,limits={'groupings':5})
        self.assertEqual(checked['status'],'PASS');self.assertFalse(checked['frontier_complete_verified'])
        for mutation in ('edge','domain','model','false_complete'):
            bad=deepcopy(result);record=bad['certificate']['signature_records'][0]
            if mutation=='edge':record['instance']['pairs']=[]
            elif mutation=='domain':next(iter(record['instance']['domains'].values()))['size']+=1
            elif mutation=='model':record['model']['constraints'].pop();record['model_sha256']=digest(record['model'])
            else:record['model']['frontier_complete_verified']=True
            with self.subTest(mutation=mutation):self.assertEqual(check_frontier(analysis,bad)['status'],'FAIL')



if __name__=='__main__':unittest.main()
