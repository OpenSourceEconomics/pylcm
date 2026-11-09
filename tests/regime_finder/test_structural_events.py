"""Original-source commitments and checked wire views use separate traversals."""
import importlib.util
import unittest

from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_events import source_anchor, compare_source
from _lcm_regime_finder.structural_transport import reseal


class StructuralEvents(unittest.TestCase):
    def capture(self, source):
        anchor = source_anchor(source, source_context={'file': 'test-input'},
                               transaction={'kind': 'test-owned-quiescent-roots'})
        bundle = NativeWriter().finish(source, anchor)
        reader = Reader(bundle)
        self.assertEqual(compare_source(reader, anchor), anchor['logical_sha256'])
        return bundle, anchor

    def test_order_types_aliases_and_large_distinct_equal_rows(self):
        first = tuple([9, 8]); equal = tuple([9, 8])
        self.capture({'rows': [first, equal, first], 'node': frozenset({(2, 3), (0, 1)}),
                      'objects': [None, True, 1, -0.0, b'bytes']})
        self.capture([str(v) for v in range(9000)])

    def test_rehashed_correctly_typed_substitution_still_fails_source_binding(self):
        bundle, anchor = self.capture([1, 2, 3])
        bundle['manifest']['records'][0][1][0] = ['int', '4']
        reseal(bundle=bundle)
        with self.assertRaisesRegex(ValueError, 'differ'):
            compare_source(Reader(bundle), anchor)

    def test_equal_content_alias_change_is_not_source_equivalence(self):
        first = [1, 2]
        bundle, anchor = self.capture([first, first, list(first)])
        record = bundle['manifest']['records'][0]
        record[1][1], record[1][2] = record[1][2], record[1][1]
        reseal(bundle=bundle)
        with self.assertRaisesRegex(ValueError, 'differ'):
            compare_source(Reader(bundle), anchor)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'native NumPy required')
    def test_numpy_source_buffers_views_and_exact_data_commitment(self):
        import numpy as np
        source = np.arange(20000, dtype='<i4')
        self.capture({'base': source, 'same': source, 'slice': source[::-2], 'equal': source.copy()})

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'actual JAX required')
    def test_actual_jax_logical_value_contract_is_explicit(self):
        import jax.numpy as jnp
        source = jnp.arange(12000, dtype=jnp.int32)
        bundle, _ = self.capture([source, source])
        self.assertTrue(bundle['manifest']['storage_obligations'])
