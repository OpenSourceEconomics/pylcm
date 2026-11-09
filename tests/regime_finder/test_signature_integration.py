"""Original-family, independent binary-model and exact proof integration."""
from copy import deepcopy
from itertools import product
import unittest
import json

from _lcm_regime_finder.frontier import search_frontier
from _lcm_regime_finder.frontier_check import check_frontier, verify_signature_cover
from _lcm_regime_finder.signature_model import build
from _lcm_regime_finder.signature_proof import binary_model, canonical_model, digest, frontier_cover_clauses
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


def assignment(record,point):
    instance=record['instance'];model=record['model'];owner={}
    for block in point['blocks']:
        members=[c for c in instance['cells'] if c['id'] in block['cells']]
        chars=sorted({value for c in members for value in c['characteristics']})
        bin=next(k for k,sig in enumerate(model['bins']) if dict(sig[0])==block['state_axes']
                 and dict(sig[1])==block['action_axes'] and list(sig[2])==chars)
        owner.update({c['id']:bin for c in members})
    values={}
    for variable in model['variables']:
        kind,*parts=variable.split('_');indexes=list(map(int,parts))
        if kind=='x':value=owner[model['cell_order'][indexes[0]]]==indexes[1]
        elif kind=='y':value=indexes[0] in set(owner.values())
        else:
            source,target=instance['pairs'][indexes[0]]
            value=owner[source]==indexes[1] and owner.get(target)!=indexes[1]
        values[variable]=int(value)
    return values


def literal_unsat(model,clauses):
    """Generate tiny proofs from literal completions, not the interval kernel."""
    variables=model['variables']
    def holds(row,values):
        left=sum(c*values[v] for v,c in row['terms'])
        return {'==':left==row['rhs'],'<=':left<=row['rhs'],'>=':left>=row['rhs']}[row['relation']]
    def visit(partial):
        rest=[v for v in variables if v not in partial]
        possibilities=[partial|dict(zip(rest,values)) for values in product((0,1),repeat=len(rest))]
        for i,row in enumerate(model['constraints']):
            if all(not holds(row,values) for values in possibilities):return {'kind':'constraint','index':i}
        for i,clause in enumerate(clauses):
            if all(not any(holds(row,values) for row in clause) for values in possibilities):return {'kind':'clause','index':i}
        if not rest:raise AssertionError('The proposed frontier has an uncovered feasible vector')
        variable=rest[0]
        return {'kind':'split','variable':variable,'zero':visit(partial|{variable:0}),'one':visit(partial|{variable:1})}
    return visit({})


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

    def test_full_integer_cover_proof_is_bound_to_actual_analysis(self):
        analysis=heterogeneous('ab',edges=[('a:0','b:0','positive'),('b:0','a:0','positive')])
        result=search_frontier(analysis);self.assertGreater(len(result['frontier']),1)
        record=result['certificate']['signature_records'][0]
        witnesses=[assignment(record,point) for point in result['frontier']]
        vectors=[tuple(point['cost'][key] for key in ('work','storage','clarity')) for point in result['frontier']]
        proof=literal_unsat(record['model'],frontier_cover_clauses(record['model'],vectors))
        certificate={'model_sha256':record['model_sha256'],'assignments':witnesses,'coverage_proof':proof}
        checked=verify_signature_cover(analysis,record,certificate)
        self.assertEqual(checked['status'],'PASS');self.assertTrue(checked['component_frontier_complete_verified'])
        self.assertFalse(checked['global_frontier_complete_verified'])
        for mutation in ('hash','witness','branch'):
            bad=deepcopy(certificate)
            if mutation=='hash':bad['model_sha256']='wrong'
            elif mutation=='witness':bad['assignments'][0][next(iter(bad['assignments'][0]))]^=1
            else:bad['coverage_proof']={'kind':'split','variable':record['model']['variables'][0],'zero':proof}
            with self.subTest(mutation=mutation):self.assertEqual(verify_signature_cover(analysis,record,bad)['status'],'FAIL')
        self.assertEqual(verify_signature_cover(analysis,record,certificate,max_visits=0)['status'],'UNRESOLVED')


if __name__=='__main__':unittest.main()
