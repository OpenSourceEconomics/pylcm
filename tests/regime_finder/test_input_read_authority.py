"""Reader closure and public writer admission share contextual input identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _lcm_regime_finder.cli import load_problem
from _lcm_regime_finder.input_reads import discover_inputs
from _lcm_regime_finder.owned_mutations import Mutation, OwnedMutationPlan
from . import rf11_alias_provenance_matrix as matrix
from . import rf11_input_provenance_oracle as oracle

PROJECT = Path(__file__).resolve().parents[2]
FIXTURE = PROJECT/'tests/regime_finder/data/generated-fixture'


class InputReadAuthorityTests(unittest.TestCase):
    def test_public_provenance_matrix(self):
        for case in matrix.cases():
            with self.subTest(case=case):
                observed = matrix.one_case(PROJECT, case, oracle, FIXTURE)
                self.assertTrue(observed['passed'], observed.get('violations'))

    def test_every_actual_structural_read_is_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source = root/'input.json'
            source.write_bytes((FIXTURE/'input.json').read_bytes())
            import shutil
            relative = json.loads(source.read_text())['structural_store']['path']
            shutil.copytree(FIXTURE/relative, root/relative)
            alias = root/'B/source.json';alias.parent.mkdir();alias.symlink_to(source)
            reads = discover_inputs([alias])
            actual = set();read_text = Path.read_text;os_open = os.open
            def text(path, *args, **kwargs):
                actual.add(path.resolve());return read_text(path,*args,**kwargs)
            def binary(path, *args, **kwargs):
                actual.add(Path(path).resolve());return os_open(path,*args,**kwargs)
            with patch.object(Path,'read_text',text),patch.object(os,'open',binary):
                load_problem(alias, reads=reads)
            expected = oracle.fixture_dependencies(alias)
            protected = {identity.resolved for identity in reads.identities}
            self.assertTrue(actual.issubset(protected),actual-protected)
            self.assertTrue({path.resolve() for path in expected['paths']}.issubset(protected))
            self.assertIn(expected['actual_manifest'],protected)
            self.assertNotIn(expected['lexical_manifest'],protected)
            with self.assertRaisesRegex(ValueError,'Unregistered input'):
                reads.read_text(root/'undeclared.json')

    def test_same_wrapper_identity_preserves_distinct_supplied_parent_contexts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);real=root/'wrapper.json'
            real.write_text(json.dumps({'adapter':'pylcm_declarations','snapshot':'leaf.json'}))
            aliases=[]
            for name in ('B','C'):
                parent=root/name;parent.mkdir();alias=parent/'link.json';alias.symlink_to(real)
                (parent/'leaf.json').write_text(json.dumps({'name':name}));aliases.append(alias)
            outer=root/'outer.json';outer.write_text(json.dumps({'adapter':'pylcm_declarations',
                'declaration_spec':str(aliases[0]),'snapshot':str(aliases[1])}))
            reads=discover_inputs([outer]);loaded=load_problem(outer,reads=reads)
            self.assertEqual(loaded,{'declaration_spec':{'name':'B'},'graph_snapshot':{'name':'C'}})
            self.assertTrue(set(aliases).issubset({item.supplied for item in reads.identities}))
            with self.assertRaisesRegex(ValueError,'aliases original input'):
                OwnedMutationPlan.admit(inputs=[outer],owned_roots=[root/'result'],
                    mutations=[Mutation('receipt','write_text',root/'C/leaf.json')],reads=reads)

    def test_legacy_fragment_reader_uses_its_supplied_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'A';b=root/'B';a.mkdir();b.mkdir()
            payload=b'{"source":"B"}';(b/'part.json').write_bytes(payload)
            decoded=json.loads(payload);identity=hashlib.sha256(json.dumps(decoded,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            index=a/'index.json';index.write_text(json.dumps({'format':'lossless-json-fragments-1',
                'marker':'fragment','parts':[{'path':'part.json','sha256':hashlib.sha256(payload).hexdigest()}],
                'value':{'fragment':'part.json'},'canonical_value_sha256':identity}))
            alias=b/'index.json';alias.symlink_to(index)
            reads=discover_inputs([alias]);self.assertEqual(load_problem(alias,reads=reads),decoded)
            self.assertIn(b/'part.json',{item.supplied for item in reads.identities})

    def test_ast_adapter_registers_actual_imported_sibling_sources(self):
        from _lcm_regime_finder.source_adapter import Module, register_source_inputs
        from _lcm_regime_finder.input_reads import ReadDiscovery
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'model.py';source.write_text('from constants import VALUE\n')
            sibling=root/'constants.py';sibling.write_text('VALUE = 7\n')
            discovery=ReadDiscovery();register_source_inputs(source,discovery);reads=discovery.freeze()
            module=Module(source,reads=reads)
            self.assertEqual(module.imported('VALUE'),7)
            self.assertIn(sibling,{item.resolved for item in reads.identities})

    def test_final_manifest_check_does_not_reuse_discovery_bytes(self):
        from _lcm_regime_finder.structural_store import check_capture_chunks
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input.json'
            source.write_bytes((FIXTURE/'input.json').read_bytes())
            relative=json.loads(source.read_text())['structural_store']['path']
            shutil.copytree(FIXTURE/relative,root/relative)
            reads=discover_inputs([source])
            manifest=root/relative/'manifest.json';manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                check_capture_chunks(source,reads=reads)
