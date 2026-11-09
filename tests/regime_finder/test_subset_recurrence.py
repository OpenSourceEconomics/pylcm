"""Exact label-subset coverage against literal partitions and saved v2 evidence."""
from copy import deepcopy
from pathlib import Path
import json
import random
import unittest

from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.frontier_check import check_frontier
from .round6_frontier_oracle import brute_family, canonical
from .test_round6_frontier import fixture


class SubsetRecurrenceTests(unittest.TestCase):
    def test_shared_subproblems_complete_beyond_partition_count_budget(self):
        analysis=fixture(2, labels=tuple('abcdefgh'))
        result=search_frontier(analysis, limits={'groupings':1500})
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['schema'],'regime-frontier/v5')
        self.assertEqual([p['cost'] for p in result['frontier']],
                         [{'work':48*10**12,'storage':16*10**12,'clarity':11}])
        self.assertEqual(check_frontier(analysis,result,limits={'groupings':1500})['status'],'PASS')

    def test_full_costs_and_layouts_match_literal_group_and_cut_enumeration(self):
        for seed in range(18):
            rng=random.Random(seed)
            edges=[(f'{a}:{i}',f'{b}:{j}',rng.choice(('zero','positive','unknown')))
                   for a in 'abc' for b in 'abc' for i in range(2) for j in range(2)]
            analysis=fixture(2, labels=tuple('abc'), edges=edges)
            if seed%2:
                analysis['declarations']['domains']['health']={'size':2}
                for cell in analysis['declarations']['cells']:
                    if cell['regime']=='b' and cell['period']==1:
                        cell['states']['health']='health'
                        option=analysis['interfaces'][cell['id']]['options'][0]
                        option['states']['health']='health'
                        option['state_projection']['health']={'kind':'identity'}
            if seed%3==0:
                analysis['declarations']['cells'][-1]['role']={'actor':'separate owner'}
            expected=brute_family(analysis)
            result=search_frontier(analysis)
            with self.subTest(seed=seed):
                self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in expected])
                self.assertEqual([canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in result['frontier']],
                                 [canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in expected])
                self.assertEqual(check_frontier(analysis,result)['status'],'PASS')

    def test_independent_recurrence_rejects_missing_branches_and_false_costs(self):
        analysis=fixture(2,labels=tuple('abc'))
        result=search_frontier(analysis)
        self.assertTrue(result['certificate']['grouping_records'])
        for mutation in ('state','cost','policy','count'):
            bad=deepcopy(result)
            record=bad['certificate']['grouping_records'][0]
            if mutation=='state': record['states'].pop()
            elif mutation=='cost': record['states'][-1]['frontier'][0]['work']+=1
            elif mutation=='policy': record['option_by_cell']['a:0']='not-an-option'
            else: bad['certificate']['search_counts']['groupings']-=1
            with self.subTest(mutation=mutation):
                self.assertEqual(check_frontier(analysis,bad)['status'],'FAIL')

    def test_saved_v2_frontier_remains_independently_checkable(self):
        record=json.loads(Path(__file__).with_name('fixtures_metadata').joinpath('frontier-v2-71bd94c.json').read_text())
        self.assertEqual(check_frontier(record['analysis'],record['result'])['status'],'PASS')

    def test_saved_v3_frontier_remains_independently_checkable(self):
        record=json.loads(Path(__file__).with_name('fixtures_metadata').joinpath('frontier-v3-d7620c6.json').read_text())
        self.assertEqual(check_frontier(record['analysis'],record['result'])['status'],'PASS')

    def test_saved_v4_frontier_remains_independently_checkable(self):
        record=json.loads(Path(__file__).with_name('fixtures_metadata').joinpath('frontier-v4-d3f76be.json').read_text())
        self.assertEqual(check_frontier(record['analysis'],record['result'])['status'],'PASS')


if __name__=='__main__':unittest.main()
