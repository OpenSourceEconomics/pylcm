"""ACA's labelled parameter Series retain their values and index metadata."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from _lcm_regime_finder.callable_identity import capture_session
from _lcm_regime_finder.model_snapshot import normalize
from _lcm_regime_finder.pytask_api import export_model
from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.structural_store import final_input_check
from tests.regime_finder.test_round6_capture import FakeModel


@unittest.skipUnless(importlib.util.find_spec('pandas'), 'actual pandas required')
class PandasCapture(unittest.TestCase):
    def series(self):
        import pandas as pd
        return pd.Series([0.7, 0.9], name='survival', index=pd.MultiIndex.from_tuples(
            [(51, 'bad'), (51, 'good')], names=['age', 'health']))

    def test_production_series_retains_labelled_domain_and_values(self):
        model = FakeModel()
        model.fixed_params = {'survival': self.series()}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.json'
            receipt = export_model(model, path, production_metadata={'fixture': True},
                seconds=None, host_gib=None,
                source_transaction={'kind': 'exclusive-owner', 'authority': 'test owns the Model'})
            self.assertEqual(receipt['status'], 'COMPLETE', receipt)
            captured = load_problem(path)
            fields = captured['fixed_params']['survival']['fields']
            self.assertEqual(fields['name'], 'survival')
            self.assertEqual(fields['index_names'], ['age', 'health'])
            self.assertEqual(fields['index'], [[51, 'bad'], [51, 'good']])
            self.assertEqual(fields['values']['values'], [0.7, 0.9])
            final_input_check(captured)

    def test_value_and_index_name_mutation_prevent_complete_capture(self):
        for label_only in (False, True):
            with self.subTest(label_only=label_only):
                series = self.series()
                with capture_session() as session:
                    record = normalize(series)
                    if label_only:
                        series.index.names = ['wrong_age', 'health']
                    else:
                        series.iloc[0] = 0.1
                self.assertNotEqual(session.report()['status'], 'COMPLETE')
                self.assertEqual(record['capture_status'], 'UNRESOLVED_MUTATION')

    def test_unsupported_labels_remain_incomplete_instead_of_losing_them(self):
        import pandas as pd
        model = FakeModel()
        model.fixed_params = {'table': pd.Series([1.0], index=pd.date_range('2020', periods=1))}
        with tempfile.TemporaryDirectory() as temp:
            receipt = export_model(model, Path(temp) / 'model.json',
                production_metadata={'fixture': True}, seconds=None, host_gib=None)
            self.assertEqual(receipt['status'], 'INCOMPLETE')
            self.assertIn('Unsupported labelled Series index',
                          receipt['capture_report']['exhaustion_reason'])

    def test_series_with_external_buffer_captures_owned_logical_values(self):
        import numpy as np
        import pandas as pd
        model = FakeModel()
        model.fixed_params = {'table': pd.Series(np.frombuffer(bytes([1, 2]), dtype='u1'))}
        with tempfile.TemporaryDirectory() as temp:
            receipt = export_model(model, Path(temp) / 'model.json',
                production_metadata={'fixture': True}, seconds=None, host_gib=None)
            self.assertEqual(receipt['status'], 'COMPLETE', receipt)


if __name__ == '__main__':
    unittest.main()
