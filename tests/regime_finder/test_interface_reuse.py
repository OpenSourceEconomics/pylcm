"""Projection alternatives must not rebuild unchanged cell interfaces."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from _lcm_regime_finder.production import MetadataAnalysis
from .tiny_metadata_family import original


class InterfaceReuseTests(unittest.TestCase):
    def setUp(self):
        self.analysis = MetadataAnalysis(original())
        self.analysis.dependencies()
        self.cell = next(iter(self.analysis.cells.values()))
        self.projection = {name: {'kind': 'identity', 'target': name}
                           for name in self.cell['states']}

    def test_repeated_interface_retains_plan_without_reinterpreting_callbacks(self):
        with patch.object(self.analysis, 'direct_reads', wraps=self.analysis.direct_reads) as reads:
            first = self.analysis.function_interfaces(self.cell, self.projection, {})
            work = reads.call_count
            self.assertGreater(work, 0)
            expected = deepcopy(first)
            first['solve'].clear()
            again = self.analysis.function_interfaces(self.cell, self.projection, {})
            self.assertEqual(again, expected)
            self.assertEqual(reads.call_count, work)

    def test_changed_projection_constants_and_cell_do_not_share_interfaces(self):
        first = self.analysis.function_interfaces(self.cell, self.projection, {})
        bound = self.analysis.function_interfaces(self.cell, self.projection, {'family': 0})
        self.assertNotEqual(first, bound)
        self.assertTrue(any(record.get('bound_arguments', {}).get('family') == 0
                            for record in bound['solve'].values()))
        changed = deepcopy(self.projection)
        changed['family'] = {'kind': 'drop'}
        dropped = self.analysis.function_interfaces(self.cell, changed, {})
        self.assertNotEqual(first, dropped)
        other = list(self.analysis.cells.values())[-1]
        late = self.analysis.function_interfaces(other, self.projection, {})
        self.assertNotEqual(first, late)


if __name__ == '__main__':
    unittest.main()
