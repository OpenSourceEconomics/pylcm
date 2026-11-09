"""Late metadata exhaustion must report incomplete before the supervisor kills it."""
import unittest
from unittest.mock import patch

from _lcm_regime_finder.production import MetadataAnalysis, analyze_production
from .tiny_metadata_family import original


class LateMetadataBudgetTests(unittest.TestCase):
    def test_interface_expansion_observes_the_metadata_deadline(self):
        clock = [0.0]
        expanded = []
        build = MetadataAnalysis.function_interfaces

        def overrun(analysis, *args):
            expanded.append(args[0]['id'])
            answer = build(analysis, *args)
            clock[0] = 2.0
            return answer

        with patch('_lcm_regime_finder.production.time.perf_counter', side_effect=lambda: clock[0]), \
                patch.object(MetadataAnalysis, 'function_interfaces', overrun):
            result = analyze_production(original(), limits={'seconds': 1})
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['semantic_check']['status'], 'NOT_RUN')
        self.assertIn('Metadata dependency analysis budget exhausted', result['error'])
        self.assertFalse(result['scenarios'])
        self.assertEqual(len(expanded), 1)


if __name__ == '__main__':
    unittest.main()
