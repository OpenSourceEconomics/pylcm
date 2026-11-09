"""Small direct oracles supplement the unmodified external acceptance harness."""
from copy import deepcopy
import hashlib
import unittest

from _lcm_regime_finder.structural_transport import capture_structure, reseal, canonical
from _lcm_regime_finder.structural_check import decode, validate, logical_observation, encoded


def anchored(source):
    anchor = {'logical_sha256': hashlib.sha256(encoded(logical_observation(source))).hexdigest()}
    bundle = capture_structure(source=source, source_anchor=anchor)
    validate(bundle=bundle, source_anchor=anchor)
    return bundle, anchor


class StructuralTransportTests(unittest.TestCase):
    def test_equal_rows_keep_distinct_identity_and_repeated_alias(self):
        first = tuple([1, -2])
        second = tuple([1, -2])
        bundle, _ = anchored([first, second, first])
        result = decode(bundle=bundle)
        self.assertEqual(result, [(1, -2), (1, -2), (1, -2)])
        self.assertIs(result[0], result[2])
        self.assertIsNot(result[0], result[1])
        self.assertEqual(bundle['manifest']['counts']['row_values'], 1)
        self.assertEqual(bundle['manifest']['counts']['row_objects'], 2)

    def test_bool_bigint_binary64_and_present_none_are_exact(self):
        bundle, _ = anchored({'x': [True, 1, 2**100, -(2**100), -0.0], 'none': None})
        result = decode(bundle=bundle)
        self.assertIs(type(result['x'][0]), bool)
        self.assertIs(type(result['x'][1]), int)
        self.assertEqual(result['x'][2:4], [2**100, -(2**100)])
        self.assertNotIn('missing', result)
        self.assertEqual(result['none'], None)
        self.assertEqual(result['x'][4].hex(), '-0x0.0p+0')

    def test_empty_and_ordered_containers_and_set_types(self):
        source = [[], (), set(), frozenset(), [3, 1, 3], (3, 1, 3), {(4, 2), (0, 1)}]
        bundle, _ = anchored(source)
        result = decode(bundle=bundle)
        self.assertEqual(result, source)
        self.assertEqual([type(v) for v in result], [type(v) for v in source])

    def test_cycle_is_explicit_instead_of_false_complete(self):
        source = []; source.append(source)
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            capture_structure(source=source, source_anchor={})

    def test_rehashed_counts_and_undeclared_payload_do_not_pass(self):
        bundle, anchor = anchored([tuple([1, 2])])
        for change in ('count', 'boolean_count', 'extra'):
            with self.subTest(change=change):
                bad = deepcopy(bundle)
                if change == 'count': bad['manifest']['counts']['objects'] += 1
                elif change == 'boolean_count': bad['manifest']['counts']['objects'] = True
                else: bad['payloads']['extra'] = b'x'
                if change != 'extra': reseal(bundle=bad)
                with self.assertRaises(ValueError): validate(bundle=bad, source_anchor=anchor)

    def test_rehashed_out_of_range_row_reference_rejected(self):
        bundle, anchor = anchored([tuple([7])])
        bundle['manifest']['roots'] = ['row', '1', 10**9]
        reseal(bundle=bundle)
        with self.assertRaises(ValueError): validate(bundle=bundle, source_anchor=anchor)

    def test_rehashed_orphan_interned_value_is_not_a_complete_inventory(self):
        bundle, anchor = anchored([tuple([1])])
        table = bundle['manifest']['row_tables']['1']
        key = table['values']
        bundle['payloads'][key] = canonical([['1'], ['999']])
        table['value_count'] = 2
        bundle['manifest']['blobs'][key]['bytes'] = len(bundle['payloads'][key])
        bundle['manifest']['counts']['row_values'] = 2
        bundle['manifest']['counts']['payload_bytes'] = sum(map(len, bundle['payloads'].values()))
        reseal(bundle=bundle)
        with self.assertRaisesRegex(ValueError, 'Unreachable interned'):
            validate(bundle=bundle, source_anchor=anchor)

    def test_final_same_value_object_replacement_is_a_mutation(self):
        source = [[1]]
        def mutate(): source[0] = [1]
        with self.assertRaisesRegex(ValueError, 'changed'):
            capture_structure(source=source, source_anchor={}, before_final_validation=mutate)
