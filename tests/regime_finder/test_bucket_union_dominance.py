"""Date buckets can have a constant union without equal individual layouts."""
from copy import deepcopy
import random
import unittest

from _lcm_regime_finder.frontier import Search, _BlockRef, _GrowingInterval, _layout_compare, add, plan_key, search_frontier
from _lcm_regime_finder.frontier_check import IndependentFamily, _IntervalTable, check_frontier
from .round6_frontier_oracle import brute_family, canonical
from .test_round6_frontier import fixture


def mixed(n, edges=()):
    analysis=fixture(n,labels=('a','b'),edges=edges)
    analysis['declarations']['analysis_scope']='pylcm-structural-graph/v1'
    analysis['declarations']['domains']['health']={'size':2}
    for cell in analysis['declarations']['cells']:
        if cell['regime']=='b':
            cell['states']['health']='health'
            option=analysis['interfaces'][cell['id']]['options'][0]
            option['states']['health']='health'
            option['state_projection']['health']={'kind':'identity'}
    return analysis


class BucketUnionDominanceTests(unittest.TestCase):
    def test_unequal_first_blocks_do_not_walk_persistent_plan_chains(self):
        from unittest.mock import patch
        from _lcm_regime_finder.frontier import _JoinedBlocks
        analysis=mixed(4);search=Search(analysis)
        policy=analysis['projection_policies'][0]['option_by_cell']
        def plan(ids):
            block=search.block(ids,policy)
            return {'cost':block['cost'],'blocks':(_BlockRef.from_materialized(search,block,policy),)}
        left=add(plan(['a:0']),plan(['b:0']))
        right=add(plan(['a:1']),plan(['b:1']))
        expected=(plan_key(left)>plan_key(right))-(plan_key(left)<plan_key(right))
        with patch.object(_JoinedBlocks,'__iter__',side_effect=AssertionError('Unnecessary backpointer traversal')):
            self.assertEqual(_layout_compare(left,right),expected)
        # Equal first blocks still compare the complete tail in canonical order.
        right=add(plan(['a:0']),plan(['b:1']))
        self.assertEqual(_layout_compare(left,right),(plan_key(left)>plan_key(right))-(plan_key(left)<plan_key(right)))

    def test_incompatible_domain_extension_is_covered_without_enumerating_suffix(self):
        from unittest.mock import patch
        analysis=mixed(80)
        for cell in analysis['declarations']['cells']:
            domain='age-specific:'+str(cell['period'])
            analysis['declarations']['domains'][domain]={'size':2}
            cell['states']['wealth']=domain
            analysis['interfaces'][cell['id']]['options'][0]['states']['wealth']=domain
        policy=analysis['projection_policies'][0]['option_by_cell'];ids=list(analysis['interfaces'])
        count=[0];original=_GrowingInterval.extend
        def extend(growing,bucket):count[0]+=1;return original(growing,bucket)
        search=Search(analysis)
        with patch.object(_GrowingInterval,'extend',extend):points=search.interval_frontier(ids,policy)
        self.assertLessEqual(count[0],2*80)
        self.assertEqual(search.counts['segments'],80*81//2)
        family=IndependentFamily(analysis)
        count=[0];original_cost=_IntervalTable.cost
        def cost(table,*args):count[0]+=1;return original_cost(table,*args)
        with patch.object(_IntervalTable,'cost',cost):expected=family.interval_points(ids,policy)
        self.assertLessEqual(count[0],2*80)
        self.assertEqual([tuple(p['cost'][k] for k in ('work','storage','clarity')) for p in points],expected)
        self.assertEqual(family.counts['segments'],search.counts['segments'])

    def test_long_heterogeneous_group_completes_without_quadratic_intervals(self):
        analysis=mixed(80)
        result=search_frontier(analysis,limits={'segments':20})
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(check_frontier(analysis,result,limits={'segments':20})['status'],'PASS')
        self.assertTrue(any(r.get('reduction') for r in result['certificate']['interval_records']))

    def test_literal_all_cuts_with_arbitrary_directed_edges(self):
        for seed in range(16):
            rng=random.Random(seed)
            edges=[(f'{a}:{i}',f'{b}:{j}',rng.choice(('zero','positive','unknown')))
                for a in 'ab' for b in 'ab' for i in range(3) for j in range(3)]
            analysis=mixed(3,edges)
            result=search_frontier(analysis)
            expected=brute_family(analysis)
            self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in expected])
            self.assertEqual([canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in result['frontier']],
                [canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in expected])
            self.assertEqual(check_frontier(analysis,result)['status'],'PASS')

    def test_only_final_bucket_differs_needs_all_single_cuts_not_all_partitions(self):
        analysis=mixed(80)
        for label in 'ab':
            option=analysis['interfaces'][f'{label}:79']['options'][0]
            option['actions']={}
            option['action_reconstruction']['choice']={'kind':'constant','value':0}
        result=search_frontier(analysis,limits={'segments':500})
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(check_frontier(analysis,result,limits={'segments':500})['status'],'PASS')

    def test_changed_final_bucket_matches_literal_partition_oracle(self):
        for seed in range(12):
            rng=random.Random(seed)
            edges=[(f'{a}:{i}',f'{b}:{j}',rng.choice(('zero','positive','unknown')))
                for a in 'ab' for b in 'ab' for i in range(3) for j in range(3)]
            analysis=mixed(3,edges)
            for label in 'ab':
                option=analysis['interfaces'][f'{label}:2']['options'][0]
                option['actions']={}
                option['action_reconstruction']['choice']={'kind':'constant','value':0}
            expected=brute_family(analysis)
            result=search_frontier(analysis)
            self.assertEqual([p['cost'] for p in result['frontier']],[p['cost'] for p in expected])
            self.assertEqual([canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in result['frontier']],
                [canonical([[b['cells'],b['option_by_cell']] for b in p['blocks']]) for p in expected])
            self.assertEqual(check_frontier(analysis,result)['status'],'PASS')

    def test_varying_axes_or_nonpositive_coalescence_saving_cannot_use_rule(self):
        for mutation in ('axis','weight','allocation','characteristic','role'):
            analysis=mixed(3)
            if mutation=='axis':
                option=analysis['interfaces']['b:1']['options'][0]
                option['states'].pop('health');option['state_projection']['health']={'kind':'drop'}
            elif mutation=='weight':analysis['clarity_weights']={'block':1,'dispatch':2}
            elif mutation=='allocation':analysis['declarations']['allocation']['kind']='global_union'
            elif mutation=='characteristic':
                analysis['interfaces']['b:1']['options'][0]['state_projection']['health']={'kind':'characteristic','id':'age-dependent'}
            else:analysis['declarations']['cells'][1]['role']={'actor':'other'}
            search=Search(analysis)
            ids=list(search.cells);policy=analysis['projection_policies'][0]['option_by_cell']
            search.interval_frontier(ids,policy)
            self.assertFalse(any(r.get('reduction') for r in search.records),mutation)

    def test_independent_checker_rejects_forged_reduction(self):
        analysis=mixed(3)
        result=search_frontier(analysis)
        record=next(r for r in result['certificate']['interval_records'] if r.get('reduction'))
        for mutation in ('cost','rule','domain','edge'):
            changed=deepcopy(result);source=deepcopy(analysis)
            target=next(r for r in changed['certificate']['interval_records'] if r.get('reduction'))
            if mutation=='cost':target['reduction']['cost']['work']+=1
            elif mutation=='rule':target['reduction']['rule']='trust-the-producer'
            elif mutation=='domain':source['declarations']['domains']['health']['size']=3
            else:source['declarations']['relations'].append({'source':'a:0','targets':['outside'],'support':'unknown'})
            self.assertEqual(check_frontier(source,changed)['status'],'FAIL',mutation)


if __name__=='__main__':unittest.main()
