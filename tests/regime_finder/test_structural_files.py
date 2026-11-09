"""File-store trust boundaries and non-materializing structural validation."""
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.structural_transport import Writer, reseal
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_files import FilePayloads, load_store, verify_store_again


class StructuralFiles(unittest.TestCase):
    def test_packed_rows_validate_without_expanding_python_tuple_objects(self):
        source = [tuple([i, i + 1]) for i in range(9000)]
        bundle = Writer().finish(source, {})
        reader = Reader(bundle)
        result = reader.validate_graph()
        self.assertEqual(result['row_objects'], 9000)
        self.assertEqual(reader.row_cache, {})

    def test_file_store_matches_memory_store_and_final_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = Writer()
            writer.payloads = FilePayloads(tmp, writable=True)
            bundle = writer.finish([tuple([2, 4]), bytearray(b'original')], {})
            reader = load_store(bundle['manifest'], tmp)
            self.assertEqual(reader.decode(), [(2, 4), bytearray(b'original')])
            verify_store_again(reader)
            key = next(iter(writer.payloads))
            writer.payloads.path(key).write_bytes(b'corruption')
            with self.assertRaises(ValueError): verify_store_again(reader)

    def test_paths_and_chunk_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FilePayloads(tmp, writable=True)
            for name in ('../p0', '/p0', 'p0/elsewhere', 'p-1'):
                with self.assertRaises(ValueError): store[name] = b'bad'
            store['p0'] = b'first'
            with self.assertRaises(ValueError): store['p0'] = b'second'

    def test_chunk_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'external').write_bytes(b'secret')
            Path(tmp, 'p0.bin').symlink_to(Path(tmp, 'external'))
            with self.assertRaises(ValueError): FilePayloads(tmp, ['p0'])['p0']

    def test_rehashed_duplicate_set_row_does_not_validate(self):
        bundle = Writer().finish({tuple([0]), tuple([1])}, {})
        key = bundle['manifest']['records'][0][2]
        bundle['payloads'][key] = b'\0' * 16
        reseal(bundle=bundle)
        with self.assertRaises(ValueError): Reader(bundle).validate_graph()

    def test_large_ordered_scalar_payload_does_not_expand_header(self):
        values = [str(i) for i in range(9000)]
        bundle = Writer().finish(values, {})
        self.assertEqual(len(bundle['manifest']['records'][0]), 3)
        reader = Reader(bundle)
        reader.validate_graph()
        self.assertEqual(reader.decode(), values)
