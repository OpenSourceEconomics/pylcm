"""The topology checkpoint is partial; the final graph must keep its callbacks."""
from dataclasses import dataclass
from collections.abc import Callable
from types import SimpleNamespace
import unittest
from _lcm_regime_finder.model_snapshot import read_model


def gate(divorce_shock):
    raise AssertionError('An exporter must never execute the gate')


@dataclass
class Graph:
    gate: Callable[..., None]


class GraphCallbackCaptureTests(unittest.TestCase):
    def test_final_graph_expands_callbacks_without_changing_topology_checkpoint(self):
        model=SimpleNamespace(description='capture regression',n_periods=1,
            user_regimes={},fixed_params={},initial_nodes={},regime_names_to_ids={},graph=Graph(gate))
        checkpoints=[]
        source=read_model(model,checkpoint=checkpoints.append)
        partial=checkpoints[0]['snapshot']['graph']['fields']['gate']
        self.assertEqual(partial['capture_status'],'INCOMPLETE')
        callback=source['graph']['fields']['gate']
        if callback.get('kind')=='callable-reference':
            callback=source['callable_records'][callback['capture_ref']]
        self.assertEqual(callback['arguments'],['divorce_shock'])
        self.assertEqual(callback['capture_status'],'COMPLETE')
        self.assertEqual(partial['arguments'],[])
        self.assertTrue(checkpoints[0]['topology_complete'])



class LegacyGraphOmissionTests(unittest.TestCase):
    def test_old_checkpoint_signatures_stay_unresolved_and_cannot_be_forged_complete(self):
        from copy import deepcopy
        from _lcm_regime_finder import find_regimes,verify_production_result
        from tests.regime_finder.test_metadata_independent import captured_snapshot
        source=captured_snapshot()
        source['user_regimes']['life']['fields']['functions']['utility']={
            'kind':'callable','arguments':[],'capture_status':'INCOMPLETE',
            'source_unavailable':True,'variadic':True,
            'obligations':['Original graph checkpoint omits optional callback expansion']}
        result=find_regimes(source,budget={'seconds':10}, search_mode="exact")
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertTrue(any(o['kind']=='graph_callback_signature' for o in result['analysis']['obligations']))
        self.assertTrue(all('history' in row['states'] for interface in result['analysis']['interfaces'].values() for row in interface['options']))
        self.assertEqual(verify_production_result(source,result)['status'],'UNRESOLVED')
        forged=deepcopy(result)
        forged['analysis']['obligations']=[]
        for callback in forged['analysis']['declarations']['callbacks'].values():
            callback.pop('unresolved_graph_signature',None)
            callback['dependency_contract']='pylcm-declared-inputs/v1'
        self.assertNotEqual(verify_production_result(source,forged)['status'],'PASS')


if __name__=='__main__':unittest.main()
