"""Class regressions for bounded callable capture; no native Model or DP solve."""
from dataclasses import dataclass
import functools
import json
import unittest
from _lcm_regime_finder.callable_identity import capture_callable, capture_session, primitive_operation
from _lcm_regime_finder.model_snapshot import normalize, source_function

_REPR_DIRECTION = 1

@dataclass(frozen=True)
class ReprDependentCategory:
    value: int
    def __repr__(self):
        return str(_REPR_DIRECTION * self.value)

def independent_helper(history):
    return history + 1

def custom_where(condition, left, right):
    return left

def local_callback(history):
    return independent_helper(history)

def factory(cutoff):
    def callback(history, age):
        return history if age < cutoff else 0
    return callback

class Round4CaptureTests(unittest.TestCase):
    def numpy(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('NumPy is optional; no environment change requested')
        return numpy

    def test_known_primitive_stops_before_library_implementation_graph(self):
        meta = source_function(self.numpy().where)
        self.assertEqual(meta['kind'], 'primitive')
        self.assertEqual(meta['canonical'], 'numpy.where')
        self.assertEqual(meta['resolution'], 'actual_object_identity')
        self.assertEqual(meta['call_targets'], {})
        self.assertEqual(meta['referenced_functions'], {})
        self.assertNotIn('bound_self', meta)
        self.assertNotIn('source', meta)

    def test_naked_primitive_has_no_invented_named_input_contract(self):
        meta = source_function(self.numpy().logical_and)
        self.assertTrue(meta['variadic'])
        self.assertEqual(meta['arguments'], [])

    def test_callsite_rule_survives_terminal_capture(self):
        import ast
        choose = self.numpy().where
        def callback(history, age):
            return choose(age < 2, history, 0)
        meta = source_function(callback)
        self.assertEqual(primitive_operation(meta, ast.Name(id='choose')), 'select')

    def test_identical_name_does_not_authenticate_an_unrelated_callable(self):
        meta = source_function(custom_where)
        self.assertEqual(meta['kind'], 'callable')
        self.assertNotIn('canonical', meta)

    def test_repeated_capture_shares_record_within_one_frozen_transaction(self):
        count = 10000
        record = normalize({'callbacks': [local_callback] * count,
                            'production_domain': {'n_points': 10**18}})
        callbacks = record['callbacks']
        self.assertEqual(len(callbacks), count)
        self.assertTrue(all(value is callbacks[0] for value in callbacks))
        self.assertEqual(record['production_domain']['n_points'], 10**18)

    def test_capture_cache_does_not_survive_the_transaction(self):
        fn = functools.partial(independent_helper, history=1)
        first = source_function(fn)
        fn.keywords['history'] = 2
        second = source_function(fn)
        self.assertNotEqual(first['binding_sha256'], second['binding_sha256'])
        self.assertEqual(first['partial_bindings']['history'], 1)
        self.assertEqual(second['partial_bindings']['history'], 2)

    def test_normalizer_is_part_of_the_capture_cache_key(self):
        fn = functools.partial(independent_helper, history=1)
        def labelled(value, **kwargs):
            return {"normalizer": "labelled", "value": normalize(value, **kwargs)}
        with capture_session():
            first = capture_callable(fn, normalize)
            second = capture_callable(fn, labelled)
        self.assertNotEqual(first['binding_sha256'], second['binding_sha256'])
        self.assertNotEqual(first['partial_bindings'], second['partial_bindings'])

    def test_depth_does_not_truncate_or_duplicate_callable_identity(self):
        fn = factory(2)
        with capture_session():
            deep = source_function(fn, depth=800)
            full = source_function(fn, depth=0)
        self.assertIs(deep, full)
        self.assertIsNotNone(full['binding_sha256'])
        self.assertIn('source', full)
        self.assertEqual(full['capture_status'], 'COMPLETE')

    def test_set_identity_uses_captured_values_not_process_representation(self):
        global _REPR_DIRECTION
        values = {ReprDependentCategory(1), ReprDependentCategory(2)}
        try:
            _REPR_DIRECTION = 1
            first = normalize(values)
            _REPR_DIRECTION = -1
            second = normalize(values)
        finally:
            _REPR_DIRECTION = 1
        self.assertEqual(first, second)
        self.assertEqual([x['fields']['value'] for x in first], [1, 2])
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

if __name__ == '__main__':
    unittest.main()
