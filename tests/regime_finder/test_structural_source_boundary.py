"""Graph-only opaque callbacks use effective inputs, not numerical bodies."""
from unittest.mock import patch
import unittest
from _lcm_regime_finder.semantic_check import SourceProof


class StructuralSourceBoundaryTests(unittest.TestCase):
    def test_opaque_graph_consumer_retains_inputs_without_interpreting_body(self):
        callback={'arguments':['history','age'],'status':'structural','expression':None,
          'dependency_contract':'pylcm-declared-inputs/v1',
          'original':{'kind':'callable','arguments':['history','age'],
            'source':'def utility(history, age):\n    return unsupported_numerical_kernel(history, age)\n'}}
        with patch.object(SourceProof,'_body',side_effect=AssertionError('numerical body entered')):
            proof=SourceProof({'utility':callback},static={'age':20})
            answer=proof.callback('utility')
        self.assertTrue(answer.proved)
        self.assertEqual(answer.reads,frozenset({'history'}))

    def test_scalar_proof_still_uses_its_body(self):
        callback={'arguments':['history','age'],'status':'proved','expression':'history if age < 30 else 0',
          'dependency_contract':'pylcm-declared-inputs/v1',
          'original':{'kind':'callable','arguments':['history','age'],
            'source':'def utility(history, age):\n    return history if age < 30 else 0\n'}}
        self.assertEqual(SourceProof({'utility':callback},static={'age':40}).callback('utility').reads,frozenset())
        self.assertEqual(SourceProof({'utility':callback},static={'age':20}).callback('utility').reads,frozenset({'history'}))

    def test_factory_cannot_gain_input_authority_from_forged_structural_flags(self):
        callback={'arguments':['age'],'status':'structural','expression':None,
          'dependency_contract':'pylcm-declared-inputs/v1',
          'original':{'kind':'dataclass','type':'lcm.AgeSpecializedFunction','fields':{
            'build':{'kind':'callable','arguments':['age'],'source':'def build(age):\n    return economic_function\n'}}}}
        proof=SourceProof({'utility':callback},static={'age':20})
        self.assertFalse(proof.callback('utility').proved)
        self.assertTrue(any('specialized' in n for n in proof.notes))


if __name__=='__main__': unittest.main()
