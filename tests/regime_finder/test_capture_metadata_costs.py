"""Preserve source locations and bulk sharing while exporting metadata."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.declarations import Normalizer
from _lcm_regime_finder.structural_store import CaptureStore, LoadedCapture, relocate_captures


class MetadataCostsTests(unittest.TestCase):
    def test_actual_source_object_uses_its_own_location(self):
        first = {'kind': 'object', 'fields': {'value': 1}}
        second = deepcopy(first)
        source = {'first': first, 'second': second}
        normal = Normalizer(source, {})
        self.assertEqual(normal.source_path(second, '/absent'), '/second')
        self.assertEqual(normal.source_path(deepcopy(second), '/absent'), '/first')

    def test_relocation_preserves_read_only_bulk_sharing_and_original_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CaptureStore(root/'snapshot.json', check=lambda: None, source_context={'generated': True})
            ref = store.reference([tuple([1, 2])])
            header = store.finish()
            before = header['path']
            holder = {'structural_store': header, 'data': ref}
            bulk = [{'value': i} for i in range(2000)]
            source = LoadedCapture({'one': holder, 'two': holder, 'bulk': bulk})
            source.structure = store.reader
            result = relocate_captures(source, root/'output')
            self.assertIs(result['bulk'], bulk)
            self.assertIs(result['one'], result['two'])
            self.assertEqual(header['path'], before)
            self.assertEqual(result['one']['structural_store']['path'], before)
            self.assertTrue((root/'output'/before/'manifest.json').is_file())
            self.assertIs(result.structure, source.structure)
