"""Exact native bytes, shared buffers, view bounds and mutation observations."""
import importlib.util
import unittest

from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import decode
from _lcm_regime_finder.structural_transport import reseal


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'native NumPy environment required')
class NativeArraysTests(unittest.TestCase):
    def test_shared_contiguous_and_negative_stride_views(self):
        import numpy as np
        base = np.arange(12, dtype='<i4')
        source = [base, base[2:10:2], base[::-1], base]
        bundle = NativeWriter().finish(source, {})
        result = decode(bundle=bundle)
        self.assertIs(result[0], result[3])
        self.assertIs(result[0].buffer, result[1].buffer)
        self.assertIs(result[0].buffer, result[2].buffer)
        self.assertEqual(result[1].offset, 8)
        self.assertEqual(result[2].strides, (-4,))
        self.assertEqual(b''.join(result[0].buffer.iter_bytes()), base.tobytes())

    def test_nan_signed_zero_and_endianness_are_not_json_numbers(self):
        import numpy as np
        data = bytes.fromhex('00000000000000800100c07f')
        source = np.frombuffer(bytearray(data), dtype='<f4').copy()
        bundle = NativeWriter().finish(source, {})
        result = decode(bundle=bundle)
        self.assertEqual(b''.join(result.buffer.iter_bytes()), data)
        self.assertEqual(result.dtype, '<f4')

    def test_real_buffer_mutation_after_read_is_rejected(self):
        import numpy as np
        source = np.arange(20000, dtype='<i4')
        def mutate(): source[-1] = -1
        with self.assertRaisesRegex(ValueError, 'changed'):
            NativeWriter().finish(source, {}, mutate)

    def test_rehashed_layout_cannot_escape_buffer(self):
        import numpy as np
        bundle = NativeWriter().finish(np.arange(3, dtype='<i4'), {})
        bundle['manifest']['records'][0][5] = 10**9
        reseal(bundle=bundle)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            decode(bundle=bundle)

    def test_equal_distinct_arrays_do_not_acquire_shared_owners(self):
        import numpy as np
        first = np.arange(4, dtype='<i4')
        result = decode(bundle=NativeWriter().finish([first, first.copy()], {}))
        self.assertIsNot(result[0].buffer, result[1].buffer)

    def test_object_dtype_is_explicitly_unsupported(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError, 'dtype'):
            NativeWriter().finish(np.array([object()]), {})

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'actual JAX runtime required')
    def test_actual_jax_array_retains_exact_value_and_object_alias(self):
        import numpy as np
        import jax.numpy as jnp
        source = jnp.arange(12000, dtype=jnp.int32)
        bundle = NativeWriter().finish([source, source], {})
        result = decode(bundle=bundle)
        self.assertIs(result[0], result[1])
        self.assertEqual(b''.join(result[0].buffer.iter_bytes()), np.asarray(source).tobytes())
        self.assertTrue(bundle['manifest']['storage_obligations'])
