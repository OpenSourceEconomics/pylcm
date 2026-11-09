"""Repeated policy incidences share only the checker's own local dependency proof."""
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import SourceProof,check_semantics
from .tiny_metadata_family import original


class RelationProofReuseTests(unittest.TestCase):
    def test_duplicate_relations_and_policies_do_not_reprove_local_dependencies(self):
        source=original()
        source['callbacks']['stable']={'arguments':[],'reads':[],'status':'proved',
            'expression':'1','identity':'stable-constant',
            'original':{'kind':'callable','arguments':[],'bindings':{},
                'source':'def stable():\n    return 1\n'}}
        for item in source['relations']: item['callbacks']=['stable']
        relation=deepcopy(source['relations'][0]);relation['id']='duplicate-incidence'
        source['relations'].append(relation)
        declaration=normalize_declarations(source)
        analysis=MetadataAnalysis(declaration).run()
        calls=[]
        original_callback=SourceProof.callback
        def observe(proof,callback,*args,**kwargs):
            current=inspect.currentframe()
            assert current is not None
            frame=current.f_back
            assert frame is not None
            if frame.f_code.co_name=='relation_dependency':
                local=frame.f_locals
                calls.append((local['source_cell']['id'],id(local['source_option']),callback,local['phase']))
            return original_callback(proof,callback,*args,**kwargs)
        with patch.object(SourceProof,'callback',observe):
            result=check_semantics(source,declaration,analysis)
        self.assertEqual(result['status'],'PASS',result)
        self.assertTrue(calls)
        self.assertEqual(len(calls),len(set(calls)))


if __name__=='__main__':unittest.main()
