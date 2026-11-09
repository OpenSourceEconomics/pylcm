"""Metadata/source identity regressions; no native Model or solver required."""
from copy import deepcopy
from dataclasses import dataclass
import functools
import hashlib
import marshal
from pathlib import Path
import unittest

from _lcm_regime_finder.callable_identity import resolve_callable, validate_callable_records
from _lcm_regime_finder.model_snapshot import source_function, normalize, read_declarations
from _lcm_regime_finder.declarations import normalize_declarations, ScalarSource, DeclarationError
from _lcm_regime_finder.model_adapter import adapt_snapshot
from tests.regime_finder.test_model_adapter import snapshot, function, grid


def where(condition,left,right):
    del condition,right
    return left


alias_where=where


def custom(history,age):
    return alias_where(age<2,history,0)


def captured_factory(cutoff):
    def callback(history,age):
        return history if age<cutoff else 0
    return callback


def numeric_bits(history):
    return history & 1


def decorated(function):
    @functools.wraps(function)
    def wrapper(*args,**kwargs):
        return function(*args,**kwargs)+11
    return wrapper


@decorated
def decorated_callback(history):
    return history


def helper_default(value=9):
    return value


def with_helper_default(history):
    del history
    return helper_default()


@dataclass
class BoundHelper:
    shift:int
    def where(self,condition,left,right):
        del condition,right
        return left+self.shift


class DeclarationIngestionTests(unittest.TestCase):
    def test_custom_alias_where_has_actual_helper_identity(self):
        meta=source_function(custom)
        reference=meta['call_targets']['alias_where']
        self.assertEqual(reference['kind'],'callable-reference')
        self.assertTrue(validate_callable_records(meta['callable_records']))
        self.assertEqual(reference,meta['bindings']['alias_where'])
        target=resolve_callable(reference,meta['callable_records'])
        self.assertEqual(target['kind'],'callable')
        self.assertEqual(target['qualname'],where.__qualname__)
        self.assertEqual(target['module'],where.__module__)
        self.assertEqual(target['code_sha256'],hashlib.sha256(marshal.dumps(where.__code__)).hexdigest())
        self.assertEqual(target['source_sha256'],hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.assertIn('return left',target['source'])
        self.assertEqual(ScalarSource(meta).expression(),'history')

    def test_captured_binding_changes_callable_identity(self):
        a,b=source_function(captured_factory(2)),source_function(captured_factory(3))
        self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])
        self.assertNotEqual(a['binding_sha256'],b['binding_sha256'])
        self.assertNotEqual(ScalarSource(a).expression(),ScalarSource(b).expression())

    def test_bound_receiver_and_partial_arguments_are_captured(self):
        a,b=source_function(BoundHelper(2).where),source_function(BoundHelper(3).where)
        self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])
        self.assertEqual(a['bound_self']['fields']['shift'],2)
        p=source_function(functools.partial(where,left=7))
        self.assertEqual(p['partial_bindings']['left'],7)

    def test_actual_numpy_alias_can_be_identified_when_available(self):
        try:import numpy as np
        except ImportError:self.skipTest('Existing NumPy unavailable; no installation attempted')
        choose=np.where
        def callback(history,age):
            return choose(age<2,history,0)
        meta=source_function(callback)
        reference=meta['call_targets']['choose']
        self.assertEqual(reference['kind'],'callable-reference')
        self.assertTrue(validate_callable_records(meta['callable_records']))
        self.assertEqual(reference,meta['bindings']['choose'])
        target=resolve_callable(reference,meta['callable_records'])
        self.assertIs(choose,np.where)
        self.assertEqual(target['kind'],'primitive')
        self.assertEqual(target['canonical'],'numpy.where')
        self.assertEqual(target['resolution'],'actual_object_identity')
        self.assertEqual(target['operation'],'select')
        self.assertEqual(target['library_module_sha256'],hashlib.sha256(Path(np.__file__).read_bytes()).hexdigest())

    def test_unbound_textual_jnp_where_is_not_a_primitive(self):
        x=snapshot();x['user_regimes']['original']['fields']['functions']['utility']=function(
            'def utility(history,age):\n    return jnp.where(age < 2, history, 0)\n',['history','age'])
        p=adapt_snapshot(x)
        from _lcm_regime_finder import find_regimes
        result=find_regimes(p)
        self.assertEqual(result['status'],'OPTIMAL')
        self.assertTrue(all('history' in atom['live_states'] for atom in result['analysis']['atoms']))

    def test_large_metadata_array_is_never_materialized(self):
        class Huge:
            shape=(10**8,10**8)
            dtype='float64'
            def __array__(self,*args,**kwargs):raise AssertionError('Dense conversion forbidden')
        value=normalize(Huge())
        self.assertEqual(value['shape'],[10**8,10**8])
        self.assertEqual(value['identity_status'],'UNRESOLVED_LARGE_BINDING')
        self.assertNotIn('values',value)

    def test_huge_grid_cardinality_is_only_metadata(self):
        x=snapshot();x['user_regimes']['original']['fields']['states']['x']={
            'kind':'dataclass','type':'lcm.LinSpacedGrid','fields':{'n_points':10**18,'start':0,'stop':1}}
        n=normalize_declarations(x)
        d=n['domains'][n['cells'][0]['states']['x']]
        self.assertEqual(d['size'],10**18)
        self.assertNotIn('values',d)
        self.assertEqual(n['execution']['numerical_model_rows'],0)

    def test_stateful_phase_imputation_is_a_real_relation(self):
        x=snapshot();r=x['user_regimes']['original']['fields']
        r['states']['belief']={'kind':'dataclass','type':'lcm.Phased','fields':{
            'solve':function('def impute(history):\n    return history\n',['history']), 'simulate':grid([0,1])}}
        r['functions']['utility']=function('def utility(belief):\n    return belief\n',['belief'])
        r['state_transitions']['belief']=function('def next_belief(belief):\n    return belief\n',['belief'])
        n=normalize_declarations(x);impute=next(r for r in n['relations'] if r['kind']=='imputation')
        self.assertEqual(impute['phase'],'solve');self.assertEqual(impute['callbacks'],[])
        self.assertEqual(n['callbacks'][impute['state_updates']['belief']]['reads'],['history'])
        self.assertFalse(n['domains'][n['cells'][0]['states']['belief']]['solve_axis'])

    def test_joint_kernel_remains_single_law_with_decoded_outputs(self):
        x=snapshot();r=x['user_regimes']['original']['fields'];r['state_transitions']={}
        r['joint_transitions']={'original':{'joint':{'kind':'dataclass','type':'lcm.JointTransition','fields':{
            'support_size':2,'support':[[0,0],[1,1]],
            'probabilities':function('def probabilities(choice):\n    return choice\n',['choice']),
            'outputs':{'x':function('def x(joint):\n    return joint[0]\n',['joint']),
                       'history':function('def history(joint):\n    return joint[1]\n',['joint'])}}}}}
        n=normalize_declarations(x);j=next(r for r in n['relations'] if r['kind']=='joint')
        self.assertEqual(j['support'],[[0,0],[1,1]])
        self.assertEqual(set(j['joint_outputs']),{'x','history'})
        self.assertEqual(j['produces'],['history','x'])
        self.assertEqual(len(j['callbacks']),1)

    def test_spec_export_does_not_call_arbitrary_factories(self):
        def never():raise AssertionError('Factory executed')
        captured=read_declarations({'regimes':{},'factory':never})
        self.assertEqual(captured['execution_contract']['specialization_factories_executed'],0)
        self.assertEqual(captured['factory']['kind'],'callable')

    def raw_spec(self):
        r=deepcopy(snapshot()['user_regimes']['original'])
        selector={'kind':'dataclass','type':'lcm.AgeRange','fields':{'exclusive_stop':21}}
        schedule={'kind':'object','type':'lcm.ByAge','fields':{
            'cases':[[selector,'working'],[21,'retired']],'default':None,'until':None}}
        return {'regimes':{'working':r,'retired':deepcopy(r)},
                'ages':{'kind':'object','fields':{'exact_values':[20,20.5,21,21.5]}},
                'initial_nodes':{'kind':'dataclass','type':'lcm.InitialNodes','fields':{
                    'by_age':{'kind':'mapping','items':[[20,['working']]]}}},
                'edges':{'working':{'kind':'dataclass','type':'lcm.Transition','fields':{'law':schedule}}}}

    def test_preprogram_spec_resolves_original_schedule_and_exact_dates(self):
        x=self.raw_spec();n=normalize_declarations(x)
        self.assertEqual([(c['age'],c['regime']) for c in n['cells']],
                         [(20,'working'),(20.5,'working'),(21,'working'),(21.5,'retired')])
        self.assertEqual(n['graph_derivation']['kind'],'resolved_original_declarations')
        self.assertEqual(len(n['relations']),6)
        self.assertTrue(all(c['stage'] is None for c in n['cells']))

    def test_until_uses_actual_predecessor_and_later_initial_starts(self):
        x=self.raw_spec();x['edges']['working']['fields']['law']['fields']={
            'until':{'kind':'dataclass','type':'lcm._Until','fields':{
                'stop_age_exclusive':21.5,'start_age_inclusive':20,'law':'working','then':'retired'}}}
        x['initial_nodes']['fields']['by_age']['items'].append([21.5,['working']])
        n=normalize_declarations(x)
        starts=[(c['age'],c['regime']) for c in n['cells'] if c['admissible_start']]
        self.assertEqual(starts,[(20,'working'),(21.5,'working')])
        self.assertIn((21.5,'retired'),[(c['age'],c['regime']) for c in n['cells']])

    def test_old_empty_schedule_requires_real_metadata(self):
        x=self.raw_spec();x['edges']['working']['fields']['law']['fields']={}
        with self.assertRaisesRegex(DeclarationError,'omitted its cases'):
            normalize_declarations(x)

    def test_separately_supplied_semantics_contract_is_bound(self):
        from typing import TypedDict
        class Contract(TypedDict):
            roles: dict[str, dict[str, str]]
            stages: dict[str, str]
            allocation: dict[str, str]
        contract: Contract = {'roles':{'original':{'actor':'one'}},'stages':{'original':'decision'},
                  'allocation':{'kind':'per_block'}}
        n=normalize_declarations(snapshot(),contract)
        self.assertEqual(n['original_contract'],contract)
        self.assertTrue(all(c['stage']=='decision' for c in n['cells']))
        changed=deepcopy(contract);changed['roles']['original']['actor']='two'
        self.assertNotEqual(n['source_identity'],normalize_declarations(snapshot(),changed)['source_identity'])

    def test_integer_bitwise_operation_cannot_prove_wrong_characteristic(self):
        with self.assertRaisesRegex(DeclarationError,'bitwise'):
            ScalarSource(source_function(numeric_bits)).expression()

    def test_decorator_body_is_not_unwrapped_into_another_function(self):
        meta=source_function(decorated_callback)
        self.assertTrue(meta['variadic'])
        self.assertIn('+11',meta['source'])
        self.assertIn('wrapper',meta['source'])

    def test_actual_helper_defaults_are_bound(self):
        self.assertEqual(ScalarSource(source_function(with_helper_default)).expression(),'9')

    def test_old_unwrapped_decorator_signature_cannot_drop_period(self):
        x=snapshot();r=x['user_regimes']['original']['fields']
        r['functions']['utility']=function(
            '@time_varying_params("table")\ndef utility(history,table):\n    return table[history]\n',
            ['history','table','period'])
        n=normalize_declarations(x);cid=n['cells'][0]['functions']['utility'];record=n['callbacks'][cid]
        self.assertEqual(record['status'],'opaque')
        self.assertIsNone(record['expression'])
        self.assertEqual(record['reads'],['history','period','table'])

    def test_tuple_subscript_is_not_rewritten_as_list_index(self):
        meta=function('def lookup(table,x,y):\n    return table[x,y]\n',['table','x','y'])
        self.assertEqual(ScalarSource(meta).expression(),'(table)[(x, y)]')

    def test_legacy_decorator_with_unchanged_signature_stays_opaque(self):
        meta=function('@custom_decorator\ndef utility(history):\n    return 0\n',['history'])
        with self.assertRaisesRegex(DeclarationError,'decorated source'):
            ScalarSource(meta).expression()

    def test_current_payoff_uses_truth_variant_and_next_state_provider(self):
        x=snapshot();r=x['user_regimes']['original']['fields']
        r['functions']['utility']={'kind':'object','type':'lcm.Phased','fields':{
            'solve':function('def utility(history):\n    return history\n',['history']),
            'simulate':function('def utility(next_x):\n    return next_x\n',['next_x'])}}
        n=normalize_declarations(x)
        for c in n['cells']:
            roots=[v for v in c['consumers'] if v['kind']=='reward']
            self.assertEqual([v['phase'] for v in roots],['simulation'])
            self.assertEqual(n['callbacks'][roots[0]['callback']]['reads'],['next_x'])
            self.assertIn('next_x',c['phase_functions']['simulation'])

    def gated_spec(self):
        x=self.raw_spec();x['ages']['fields']['exact_values']=[20,20.5]
        x['regimes']['outside']=deepcopy(x['regimes']['retired'])
        projection={'x':function('def projected(history):\n    return history\n',['history']),
                    'history':function('def projected(history):\n    return history\n',['history'])}
        fallback={'kind':'dataclass','type':'lcm.ProjectedRegimeValue','fields':{'regime':'outside','projection':projection}}
        route={'kind':'dataclass','type':'lcm.StakeholderRoute','fields':{'fallback':fallback}}
        gate={'kind':'dataclass','type':'lcm.Gate','fields':{
            'predicate':function('def gate(history):\n    return history > 0\n',['history']),
            'routes':{'self':route}}}
        x['edges']['working']['fields']={'law':'retired','gates':{'retired':gate}}
        return x

    def test_gate_and_fallback_use_landing_context_and_incoming_owner(self):
        n=normalize_declarations(self.gated_spec());cells={c['id']:c for c in n['cells']}
        gate=next(r for r in n['relations'] if r['kind']=='gate')
        self.assertEqual((cells[gate['source']]['regime'],cells[gate['source']]['age']),('retired',20.5))
        self.assertEqual(cells[gate['incoming_source']]['regime'],'working')
        for fallback in [r for r in n['relations'] if r['kind']=='fallback']:
            self.assertEqual(fallback['source'],gate['source'])
            self.assertEqual(fallback['incoming_source'],gate['incoming_source'])
            self.assertEqual(fallback['ownership']['leg'],'self')
            self.assertEqual(cells[fallback['targets'][0]]['regime'],'outside')


if __name__=='__main__':unittest.main()

class DescriptorClassCaptureTests(unittest.TestCase):
    def test_array_class_metadata_does_not_access_shape_descriptor(self):
        class ArrayType:
            @property
            def shape(self):
                raise AssertionError('A class must not be instantiated')
            @property
            def dtype(self):
                raise AssertionError('A class must not be instantiated')
        value = normalize(ArrayType)
        self.assertEqual(value['kind'], 'type')
        self.assertTrue(value['type'].endswith('.ArrayType'))

    def test_binding_named_type_is_not_a_declaration_tag(self):
        from _lcm_regime_finder.declarations import short
        self.assertEqual(short({'type': {'kind': 'type', 'type': 'builtins.float'}}), '')

    def test_class_call_target_is_not_a_bound_instance(self):
        meta = source_function(float)
        self.assertNotIn('bound_self', meta)
        self.assertTrue(meta['source_unavailable'])
        self.assertEqual(meta['qualname'], 'float')

    def test_opaque_native_callback_retains_domain_in_independent_family(self):
        from _lcm_regime_finder.production import analyze_production
        original = snapshot()
        regime = original['user_regimes']['original']['fields']
        regime['states']['other_history'] = grid([0, 1])
        regime['functions']['utility'] = {
            'kind': 'dataclass', 'type': 'native.UnknownPayoff', 'fields': {}}
        result = analyze_production(original, limits={'seconds': 15})
        errors = result.get('semantic_check', {}).get('errors', [])
        self.assertFalse(any('projection policy is outside' in e.get('kind', '') for e in errors), errors)
        self.assertNotEqual(result.get('semantic_check', {}).get('status'), 'PASS')

    def test_cached_original_identities_match_canonical_json(self):
        from _lcm_regime_finder.declarations import Normalizer, digest
        values = [None, True, 5, 'alpha', {'name': 'ä/β', 'nested': [1, None, {'size': 81}]}]
        normal = Normalizer({}, None)
        for value in values:
            self.assertEqual(normal.value_digest(value), digest(value))
            for field, setting in [('period', None), ('period', 41), ('selected_phase', 'simulation')]:
                expected = digest({'declaration': value, field: setting})
                self.assertEqual(normal.declaration_key(value, field, setting), expected)
                self.assertEqual(normal.declaration_key(value, field, setting), expected)

    def test_native_linear_expectation_does_not_read_unrelated_current_state(self):
        from _lcm_regime_finder.production import analyze_production
        original = snapshot()
        regime = original['user_regimes']['original']['fields']
        regime['states']['unread'] = grid([0, 1])
        regime['certainty_equivalent'] = {'kind': 'dataclass', 'type': '_lcm.certainty_equivalent.LinearExpectation', 'fields': {}}
        result = analyze_production(original, limits={'seconds': 15})
        errors = result.get('semantic_check', {}).get('errors', [])
        self.assertFalse(any('dependency closure omitted' in e.get('kind', '') or 'projection policy is outside' in e.get('kind', '') for e in errors), errors)
