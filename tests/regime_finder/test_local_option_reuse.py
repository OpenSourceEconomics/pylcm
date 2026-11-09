"""Global policy combinations must not rebuild identical local options."""
from unittest.mock import patch
import unittest
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from _lcm_regime_finder.semantic_check import check_semantics
from .tiny_metadata_family import original


class LocalOptionReuseTests(unittest.TestCase):
    def test_distinct_global_policies_reuse_identical_local_choices(self):
        source=original()
        declaration=normalize_declarations(source)
        metadata=MetadataAnalysis(declaration)
        with patch.object(metadata,'function_interfaces',wraps=metadata.function_interfaces) as build:
            analysis=metadata.run()
        emitted=sum(len(row['options']) for row in analysis['interfaces'].values())
        self.assertLess(emitted,len(analysis['projection_policies'])*len(declaration['cells']))
        self.assertEqual(build.call_count,emitted)
        self.assertEqual(check_semantics(source,declaration,analysis)['status'],'PASS')
        # Returned plans still cannot mutate the separate function-interface cache.
        cell=next(iter(metadata.cells.values()))
        base={name:{'kind':'identity','target':name} for name in cell['states']}
        before=metadata.function_interfaces(cell,base,{})
        analysis['interfaces'][cell['id']]['options'][0]['function_interfaces'].clear()
        self.assertEqual(metadata.function_interfaces(cell,base,{}),before)



class DerivedRegistryOrderTests(unittest.TestCase):
    def test_shared_domain_keeps_original_last_active_cell_registration(self):
        from _lcm_regime_finder.frontier import digest
        declaration=normalize_declarations(original())
        domain=declaration['cells'][0]['states']['family']
        for cell in declaration['cells']:cell['states']['other']=domain
        metadata=MetadataAnalysis(declaration)
        metadata.live={cid:set(cell['states']) for cid,cell in metadata.cells.items()}
        metadata.reasons={cid:{} for cid in metadata.cells}
        base={'domain':domain,'callback':'children','values':[0,1,1,0],
              'target_values':[0,1],'target_domain':'shared-derived-grid','requires':{}}
        left=dict(base,id='left',state='family',target='left_value',cells=['c0','c2'])
        right=dict(base,id='right',state='other',target='right_value',cells=['c1'])
        def regions(candidate):return [dict(candidate,active_cells=candidate['cells'])],True
        with patch.object(metadata,'dependencies'),patch.object(metadata,'characteristic_candidates',return_value=[left,right]),patch.object(metadata,'closed_regions',side_effect=regions),patch.object(metadata,'function_interfaces',return_value={}):
            result=metadata.run()
        self.assertEqual(result['derived_domains']['shared-derived-grid']['identity'],
                         digest(dict(left,active_cells=left['cells'])))


if __name__=='__main__':unittest.main()
