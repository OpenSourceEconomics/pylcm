"""Exact indexed costs, independent recurrences and family controls."""
from copy import deepcopy
from itertools import product
import json
import random
import unittest

from _lcm_regime_finder.frontier import Search, _BlockRef, _GrowingInterval, _layout_compare, pareto, search_frontier
from _lcm_regime_finder.frontier_check import IndependentFamily, _IntervalTable, _antichain, check_frontier
from .round6_frontier_oracle import block, brute_family, canonical


def fixture(n=4, labels=('a',), *, edges=(), allocation='per_block'):
    cells, interfaces = [], {}
    for label in labels:
        for t in range(n):
            cid = f'{label}:{t}'
            cell = {'id': cid, 'period': t, 'age': 20 + 3*t, 'regime': label,
                    'role': {'actor': 'same'}, 'ownership': 'owner', 'stage': 'decision',
                    'states': {'wealth': 'wealth'}, 'actions': {'choice': 'choice'}}
            cells.append(cell)
            option = {'id': 'keep', 'states': dict(cell['states']), 'actions': dict(cell['actions']),
                'state_projection': {'wealth': {'kind': 'identity'}},
                'action_reconstruction': {'choice': {'kind': 'identity'}}}
            interfaces[cid] = {'options': [option]}
    relations = [{'source': s, 'targets': [t], 'support': support} for s, t, support in edges]
    return {'declarations': {'cells': cells, 'domains': {'wealth': {'size': 10**12}, 'choice': {'size': 3}},
             'relations': relations, 'allocation': {'kind': allocation, 'boundary_unit': 2}},
            'interfaces': interfaces,
            'projection_policies': [{'id': 'P0', 'option_by_cell': {c['id']: 'keep' for c in cells}}],
            'discovery_complete': True}


class Round6FrontierTests(unittest.TestCase):
    def assert_all_intervals(self, analysis, ids=None):
        search, family = Search(analysis), IndependentFamily(analysis)
        policy = analysis['projection_policies'][0]['option_by_cell']
        ids = ids or [c['id'] for c in analysis['declarations']['cells']]
        dates = {}
        for c in ids:
            dates.setdefault(search.cells[c]['period'], []).append(c)
        buckets = [sorted(v) for _, v in sorted(dates.items())]
        table = _IntervalTable(family, buckets, policy)
        maxima = [0] * len(buckets)
        for start in reversed(range(len(buckets))):
            for maximum in table.activate[start]:
                maxima[maximum] += 1
            grow, members, internal = _GrowingInterval(search, policy), [], 0
            for stop in range(start + 1, len(buckets) + 1):
                members += buckets[stop - 1]
                grow.extend(buckets[stop - 1])
                internal += maxima[stop - 1]
                literal = block(analysis, members, policy)
                expected = None if literal is None else literal['cost']
                self.assertEqual(grow.cost(), expected)
                self.assertEqual(table.cost(start, stop, internal), expected)

    def test_all_three_atom_directed_graphs(self):
        pairs = list(product(range(3), repeat=2))
        for mask in range(1 << len(pairs)):
            edges = [(f'a:{s}', f'a:{t}', 'positive') for i, (s,t) in enumerate(pairs) if mask & (1 << i)]
            with self.subTest(mask=mask):
                self.assert_all_intervals(fixture(3, edges=edges))

    def test_repeated_self_external_zero_unknown_and_same_date(self):
        edges = [('a:0', 'a:0', 'positive'), ('a:0', 'a:3', 'unknown'),
                 ('a:0', 'a:3', 'positive'), ('a:1', 'b:1', 'positive'),
                 ('b:1', 'a:1', 'positive'), ('a:2', 'external', 'unknown'),
                 ('a:3', 'a:0', 'zero'), ('a:0', 'a:3', 'zero')]
        analysis = fixture(4, ('a','b'), edges=edges)
        self.assert_all_intervals(analysis)
        self.assert_all_intervals(analysis, ['a:' + str(i) for i in range(4)])
        analysis['declarations']['cells'][-1]['role'] = {'actor': 'other'}
        self.assert_all_intervals(analysis)

    def test_axis_compatibility_and_allocation_contract(self):
        for allocation in ('per_block', 'global_union'):
            analysis = fixture(4, allocation=allocation)
            analysis['declarations']['domains']['other'] = {'size': 13}
            analysis['interfaces']['a:2']['options'][0]['states']['wealth'] = 'other'
            analysis['interfaces']['a:3']['options'][0]['states'] = {'coarse': 'other'}
            analysis['interfaces']['a:3']['options'][0]['state_projection']['wealth'] = {'kind': 'characteristic', 'id': 'wealth_class'}
            self.assert_all_intervals(analysis)

    def test_query_only_owner_does_not_become_a_physical_cost_cell(self):
        analysis = fixture(4, edges=[('a:0', 'a:3', 'unknown')])
        reference = search_frontier(analysis)
        analysis['declarations']['query_owners'] = {'Q': {'id':'Q','period':3,
            'phase':'solve','regime':'prospective_partner','storage':'not_asserted',
            'dependencies':[{'query_target':'R','phase':'solve'}]}}
        analysis['declarations']['relations'].append({'kind':'policy_read','source':'a:0',
            'targets':[], 'query_target':'Q', 'phase':'solve', 'support':'unknown'})
        result = search_frontier(analysis)
        self.assertEqual([p['cost'] for p in reference['frontier']], [p['cost'] for p in result['frontier']])
        self.assertTrue(all(set(c for b in p['blocks'] for c in b['cells']) == {'a:0','a:1','a:2','a:3'}
                            for p in result['frontier']))
        self.assertEqual(check_frontier(analysis,result)['status'],'PASS')
        self.assert_all_intervals(analysis)

    def test_full_family_and_exact_layout_ties(self):
        for seed in range(16):
            rng = random.Random(seed)
            analysis = fixture(3, ('a','b'), edges=[(f'{s}:{i}', f'{t}:{j}', rng.choice(('zero','positive','unknown')))
                for s in ('a','b') for t in ('a','b') for i in range(3) for j in range(3) if rng.random() < .18])
            for cell in analysis['declarations']['cells']:
                if rng.random() < .3:
                    analysis['interfaces'][cell['id']]['options'][0]['actions'] = {}
                    analysis['interfaces'][cell['id']]['options'][0]['action_reconstruction'] = {'choice': {'kind': 'constant', 'value': 0}}
            expected = brute_family(analysis)
            result = search_frontier(analysis)
            layout = lambda p: canonical([[b['cells'], b['option_by_cell']] for b in p['blocks']])
            self.assertEqual([(p['cost'],layout(p)) for p in result['frontier']], [(p['cost'],layout(p)) for p in expected])
            self.assertEqual(check_frontier(analysis,result)['status'], 'PASS')

    def test_cost_skyline_algorithms_against_pairwise_oracle(self):
        for seed in range(30):
            rng = random.Random(seed)
            rows = [tuple(rng.randrange(8) for _ in range(3)) for _ in range(70)]
            expected = sorted(v for v in set(rows) if not any(u != v and all(a <= b for a,b in zip(u,v)) for u in rows))
            self.assertEqual(_antichain(rows), expected)
            plans = [{'cost':dict(zip(('work','storage','clarity'), v)), 'blocks':[]} for v in rows]
            self.assertEqual([tuple(p['cost'].values()) for p in pareto(plans)], expected)

    def test_compact_layout_ties_equal_exact_json_for_escaped_ids(self):
        analysis = fixture(5)
        names = ['a\n', 'a"', 'a\\', 'a\t', '\u00e4']
        old = list(analysis['interfaces'])
        for cell, name in zip(analysis['declarations']['cells'], names):
            cell['id'] = name
        analysis['interfaces'] = {name: analysis['interfaces'][before] for before,name in zip(old,names)}
        policies = [{name:'keep' for name in names}, {name:'other"' for name in names}]
        for options in analysis['interfaces'].values():
            extra=deepcopy(options['options'][0]);extra['id']='other"';options['options'].append(extra)
        analysis['projection_policies'] = [{'id':str(i),'option_by_cell':p} for i,p in enumerate(policies)]
        search = Search(analysis)
        plans=[]
        for mask in range(1,1<<len(names)):
            ids=[name for i,name in enumerate(names) if mask & (1<<i)]
            for policy in policies:
                direct=search.block(ids,policy)
                plans.append(({'blocks':[_BlockRef.from_materialized(search,direct,policy)]},
                    canonical([[direct['cells'],direct['option_by_cell']]])))
        for left,a in plans:
            for right,b in plans:
                self.assertEqual(_layout_compare(left,right), (a>b)-(a<b))

    def test_final_forced_action_is_admissible_and_preference_configurable(self):
        analysis = fixture(2)
        analysis['interfaces']['a:1']['options'][0]['actions'] = {}
        analysis['interfaces']['a:1']['options'][0]['action_reconstruction'] = {'choice': {'kind':'constant','value':0}}
        result = search_frontier(analysis)
        self.assertTrue(any(b['cells'] == ['a:1'] for p in result['frontier'] for b in p['blocks']))
        cost_set = [p['cost'] for p in result['frontier']]
        for weight in (1,100):
            selected = search_frontier(analysis,recommendation={'weights':{'clarity':weight}})
            self.assertEqual([p['cost'] for p in selected['frontier']],cost_set)
            self.assertEqual(check_frontier(analysis,selected)['status'],'PASS')

    def test_compact_certificate_mutations_and_budget_exit(self):
        analysis = fixture(5,('a','b'))
        result = search_frontier(analysis)
        self.assertEqual(check_frontier(analysis,result)['status'],'PASS')
        self.assertEqual(result['certificate']['family_encoding'].count('buckets'),1)
        self.assertTrue(all('group_id' in r and 'buckets' in r for r in result['certificate']['interval_records']))
        for mutation in ('bucket','option','encoding','recurrence'):
            changed = deepcopy(result)
            if mutation == 'bucket': changed['certificate']['interval_records'][0]['buckets'][0].append('not_an_atom')
            if mutation == 'option': changed['certificate']['interval_records'][0]['option_by_cell']['a:0'] = 'wrong'
            if mutation == 'encoding': changed['certificate']['family_encoding'] = 'old_full_memberships'
            if mutation == 'recurrence': changed['certificate']['interval_records'][0]['suffix_frontiers'][0][0]['work'] += 1
            self.assertEqual(check_frontier(analysis,changed)['status'],'FAIL')
        partial = search_frontier(analysis,limits={'segments':0})
        self.assertEqual(partial['status'],'INCOMPLETE')
        self.assertEqual(check_frontier(analysis,partial)['status'],'PASS')
        self.assertFalse(check_frontier(analysis,partial)['frontier_complete_verified'])


if __name__ == '__main__':
    unittest.main()
