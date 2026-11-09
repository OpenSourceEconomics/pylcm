"""Bound capture generations and independent cold-loading of all chunks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.structural_store import CaptureStore, load_capture_structure, reference_info, final_input_check


class StructuralStoreTests(unittest.TestCase):
    def capture(self, root, source):
        store = CaptureStore(root / 'snapshot.json', check=lambda: None,
                             source_context={'source': 'literal unit-test roots'},
                             transaction={'kind': 'exclusive-owner', 'authority': 'this serial test owns its local inputs'})
        ref = store.reference(source)
        header = store.finish()
        snapshot = {'structural_store': header, 'example': ref}
        path = root / 'snapshot.json'
        path.write_text(json.dumps(snapshot))
        return store, snapshot, path

    def test_cold_load_checks_source_and_keeps_lazy_root_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, snapshot, path = self.capture(root, [str(i) for i in range(9000)])
            loaded = load_capture_structure(json.loads(path.read_text()), path)
            self.assertEqual(reference_info(loaded['example']), {'type': 'list', 'length': 9000})
            self.assertIs(deepcopy(loaded).structure, loaded.structure)
            final_input_check(loaded)

    def test_forged_descriptor_does_not_change_actual_cardinality(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot, path = self.capture(Path(tmp), [1, 2, 3])
            snapshot['example']['length'] = 4
            with self.assertRaisesRegex(ValueError, 'descriptor'):
                load_capture_structure(snapshot, path)

    def test_unknown_store_and_unknown_root_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'Unverified'):
            reference_info({'kind': 'structural-reference', 'store_root': 'absent', 'root': 'S0'})
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot, path = self.capture(Path(tmp), [1])
            snapshot['example']['root'] = 'absent'
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                load_capture_structure(snapshot, path)

    def test_store_path_cannot_escape_the_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot, path = self.capture(Path(tmp), [1])
            snapshot['structural_store']['path'] = '../external'
            with self.assertRaisesRegex(ValueError, 'escapes'):
                load_capture_structure(snapshot, path)

    def test_repeated_immutable_generation_reuses_checked_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, snapshot, _ = self.capture(root, [1, 2, 3])
            second, repeated, _ = self.capture(root, [1, 2, 3])
            self.assertEqual(snapshot['structural_store'], repeated['structural_store'])
            self.assertEqual(len(list(first.parent.iterdir())), 1)

    def test_late_chunk_corruption_invalidates_the_whole_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot, path = self.capture(Path(tmp), [tuple([1, 2, 3])])
            loaded = load_capture_structure(snapshot, path)
            reader = loaded.structure
            key = next(iter(reader.payloads))
            reader.payloads.path(key).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                final_input_check(loaded)
