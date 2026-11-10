"""Source parsing is shared; proof environments and changed source stay distinct."""
from copy import deepcopy
import ast
import unittest
from typing import TypedDict
from unittest.mock import patch


type PythonSource = str


class ConstantObject(TypedDict):
    type: str
    fields: dict[str, int]


class ConstantCallable(TypedDict):
    bound_self: ConstantObject
    capture_contract: str
    arguments: list[str]
    raw_arguments: list[str]
    variadic: bool
    bindings: dict[str, dict[str, str]]
    source: PythonSource


class ConstantCapture(ConstantObject):
    callable: ConstantCallable

from _lcm_regime_finder.semantic_check import SourceProof, _CallableRecords


def callback(source):
    return {'arguments':['history','age'], 'original':{
        'kind':'callable','arguments':['history','age'],'source':source}}


class SourceParseReuseTests(unittest.TestCase):
    def test_repeated_source_is_parsed_once_but_dates_are_independent(self):
        callbacks={'utility':callback('def utility(history, age):\n    return history if age < 30 else 0\n')}
        records=_CallableRecords()
        with patch('_lcm_regime_finder.semantic_check.ast.parse',wraps=ast.parse) as parse:
            for age in (20,40,20,40):
                answer=SourceProof(callbacks,static={'age':age},callable_records=records).callback('utility')
                self.assertTrue(answer.proved)
                self.assertEqual(answer.reads,frozenset({'history'}) if age<30 else frozenset())
            self.assertEqual(parse.call_count,1)

    def test_changed_source_with_same_name_cannot_reuse_old_tree(self):
        callbacks={'utility':callback('def utility(history, age):\n    return history\n')}
        records=_CallableRecords()
        first=SourceProof(callbacks,static={'age':40},callable_records=records).callback('utility')
        callbacks=deepcopy(callbacks)
        callbacks['utility']['original']['source']='def utility(history, age):\n    return 0\n'
        second=SourceProof(callbacks,static={'age':40},callable_records=records).callback('utility')
        self.assertEqual(first.reads,frozenset({'history'}))
        self.assertEqual(second.reads,frozenset())

    def test_unparseable_source_does_not_acquire_a_proof_from_cache(self):
        callbacks={'utility':callback('def utility(history, age): ???')}
        records=_CallableRecords()
        with patch('_lcm_regime_finder.semantic_check.ast.parse',wraps=ast.parse) as parse:
            for age in (20,40):
                proof=SourceProof(callbacks,static={'age':age},callable_records=records)
                self.assertFalse(proof.callback('utility').proved)
                self.assertTrue(proof.notes)
            self.assertEqual(parse.call_count,1)



class IdentityParseReuseTests(unittest.TestCase):
    def test_identity_reuses_syntax_but_checks_the_requested_state_each_time(self):
        from _lcm_regime_finder.semantic_check import _identity_callback
        cb={'original':{'source':'def carry(history):\n    return history\n'}}
        records=_CallableRecords()
        with patch('_lcm_regime_finder.semantic_check.ast.parse',wraps=ast.parse) as parse:
            self.assertTrue(_identity_callback(cb,'history',records))
            self.assertFalse(_identity_callback(cb,'assets',records))
            self.assertTrue(_identity_callback(cb,'history',records))
            self.assertEqual(parse.call_count,1)


class ConstantParseReuseTests(unittest.TestCase):
    def test_constant_reuses_syntax_but_rechecks_its_bound_receiver(self):
        from _lcm_regime_finder.semantic_check import _closed_native_constant
        tag='_lcm.regime_building.schedules._RegimeCodeConstant'
        original: ConstantCapture = {'type':tag,'fields':{'value':0},'callable':{
            'bound_self':{'type':tag,'fields':{'value':0}},
            'capture_contract':'inspection-only-frozen-bindings-2',
            'arguments':[],'raw_arguments':['self'],'variadic':False,
            'bindings':{'jnp':{'kind':'module','name':'jax.numpy'}},
            'source':'def __call__(self):\n    return jnp.asarray(self.value)\n'}}
        records=_CallableRecords()
        with patch('_lcm_regime_finder.semantic_check.ast.parse',wraps=ast.parse) as parse:
            self.assertTrue(_closed_native_constant(original,records))
            self.assertTrue(_closed_native_constant(original,records))
            changed=deepcopy(original)
            changed['callable']['bound_self']['fields']['value']=1
            self.assertFalse(_closed_native_constant(changed,records))
            self.assertEqual(parse.call_count,1)


if __name__=='__main__':unittest.main()
