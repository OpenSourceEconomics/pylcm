"""Original-source adapter tests; expected graphs never enter adapter inputs."""
import ast
from collections.abc import Callable, Mapping
import copy
import json
import operator
from pathlib import Path
import shutil
import tempfile
import unittest

from _lcm_regime_finder.source_adapter import adapt_manifest, adapt_source, SourceAdapterError
from _lcm_regime_finder import find_regimes, verify_certificate


from .source_data import source_fixture_project

PROJECT = source_fixture_project()


def known_scalar(text, env):
    """Small independent test oracle: unknown names remain unknown."""
    def visit(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return env.get(node.id)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if a is None or b is None: return None
            return {ast.Div:operator.truediv,ast.Add:operator.add,ast.Sub:operator.sub,
                    ast.Mult:operator.mul}[type(node.op)](a,b)
        if isinstance(node, ast.Compare):
            vals=[visit(node.left)]+[visit(x) for x in node.comparators]
            if any(v is None for v in vals): return None
            ops: Mapping[type[ast.cmpop], Callable[..., bool]] = {ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,
                 ast.GtE:operator.ge,ast.Eq:operator.eq,ast.NotEq:operator.ne}
            return all(ops[type(op)](a,b) for op,a,b in zip(node.ops,vals,vals[1:]))
        raise AssertionError(ast.dump(node))
    return visit(ast.parse(text,mode="eval").body)


def predicate_value(node, env):
    if isinstance(node, str): return known_scalar(node,env)
    if isinstance(node, list):
        vals=[predicate_value(x,env) for x in node]
        return False if False in vals else (None if None in vals else True)
    if "opaque" in node: return None
    if "not" in node:
        v=predicate_value(node["not"],env)
        return None if v is None else not v
    if "and" in node: return predicate_value(node["and"],env)
    vals=[predicate_value(x,env) for x in node["or"]]
    return True if True in vals else (None if None in vals else False)


def all_reads(node):
    if isinstance(node, dict):
        return set(node.get("reads",[])) | set().union(*(all_reads(v) for v in node.values()))
    if isinstance(node, list): return set().union(*(all_reads(x) for x in node))
    return set()


class SourceAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetter=adapt_manifest(PROJECT/"examples/fetter-source.json")
        cls.adda=adapt_manifest(PROJECT/"examples/adda-source.json")

    def test_fetter_full_decision_structure_and_dead_type(self):
        programs={x["id"]:x for x in self.fetter["programs"]}
        self.assertEqual(programs["working_life"]["declared_actions"],["labor_supply","consumption"])
        self.assertEqual(programs["retirement"]["declared_actions"],["consumption"])
        self.assertIn("disutility",programs["working_life"]["reward"]["reads"])
        retired=programs["retirement"]
        self.assertNotIn("disutility",all_reads([retired["reward"],retired["feasible"]]))
        self.assertEqual(retired["transitions"][0]["updates"]["disutility"]["reads"],["disutility"])
        self.assertNotIn("disutility",retired["transitions"][0]["updates"]["assets"]["reads"])
        self.assertEqual(self.fetter["external_report_tags"],["disutility"])
        self.assertEqual(programs["dead"]["declared_states"],[])
        self.assertEqual(programs["dead"]["declared_actions"],[])

    def test_adda_forced_action_comes_from_original_constraint(self):
        working=next(p for p in self.adda["programs"] if p["id"]=="working")
        self.assertIsNone(predicate_value(working["feasible"],{"age":91,"conceive":1}))
        self.assertFalse(predicate_value(working["feasible"],{"age":92,"conceive":1}))
        self.assertFalse(predicate_value(working["feasible"],{"age":118,"conceive":1}))
        self.assertIsNone(predicate_value(working["feasible"],{"age":92,"conceive":0}))
        self.assertIn("family",working["reward"]["reads"])
        self.assertIn("family",all_reads(working["feasible"]))
        self.assertIn("conceive",working["transitions"][0]["updates"]["family"]["reads"])
        self.assertFalse(self.adda["states"]["fecundity"]["solve_axis"])
        self.assertTrue(self.adda["states"]["fecundity"]["simulation_axis"])

    def test_source_and_filename_renaming_does_not_encode_the_answer(self):
        config=json.loads((PROJECT/"examples/adda-source.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            source=PROJECT/"examples/source_inputs/adda"
            shutil.copyfile(source/"model.py",tmp/"arbitrary_filename.py")
            shutil.copyfile(source/"calibration.py",tmp/"calibration.py")
            actual=adapt_source(tmp/"arbitrary_filename.py",config)
        self.assertEqual(actual["ages"],self.adda["ages"])
        # Strip source anchors only; masks and read sets must remain identical.
        def strip(x):
            if isinstance(x,dict): return {k:strip(v) for k,v in x.items() if k!="source_anchor"}
            if isinstance(x,list): return [strip(v) for v in x]
            return x
        self.assertEqual(strip(actual["programs"]),strip(self.adda["programs"]))

    def test_mutating_original_constant_moves_discovered_restriction(self):
        config=json.loads((PROJECT/"examples/adda-source.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); source=PROJECT/"examples/source_inputs/adda"
            shutil.copyfile(source/"model.py",tmp/"model.py")
            before=(source/"calibration.py").read_text()
            after=before.replace("AGE_INFERTILE = 46.0","AGE_INFERTILE = 48.0")
            self.assertNotEqual(before,after)
            (tmp/"calibration.py").write_text(after)
            actual=adapt_source(tmp/"model.py",config)
        feasible=next(p for p in actual["programs"] if p["id"]=="working")["feasible"]
        self.assertIsNone(predicate_value(feasible,{"age":94,"conceive":1}))
        self.assertFalse(predicate_value(feasible,{"age":96,"conceive":1}))

    def test_new_retired_payoff_consumer_is_retained(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        source=PROJECT/"examples/source_inputs/fetter/model.py"
        tree=ast.parse(source.read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="utility_retired")
        fn.args.args.append(ast.arg(arg="disutility"))
        ret=next(x for x in fn.body if isinstance(x,ast.Return))
        assert ret.value is not None
        ret.value=ast.BinOp(left=ret.value,op=ast.Add(),right=ast.Name(id="disutility",ctx=ast.Load()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            actual=adapt_source(path,config)
        retired=next(x for x in actual["programs"] if x["id"]=="retirement")
        self.assertIn("disutility",retired["reward"]["reads"])

    def test_mutable_coordinate_cannot_be_claimed_immutable_report_tag(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        text=(PROJECT/"examples/source_inputs/fetter/model.py").read_text()
        changed=text.replace("    return disutility\n","    return disutility + 1\n")
        self.assertNotEqual(changed,text)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(changed)
            with self.assertRaises(SourceAdapterError): adapt_source(path,config)

    def test_bound_constructor_inputs_participate_in_grid_identity(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        config["symbols"]["assets_max"]=30.0
        actual=adapt_source(PROJECT/"examples/source_inputs/fetter/model.py",config)
        self.assertNotEqual(actual["states"]["assets"]["grid_id"],self.fetter["states"]["assets"]["grid_id"])
        self.assertEqual(actual["states"]["assets"]["size"],self.fetter["states"]["assets"]["size"])

    def test_reject_answer_bearing_fields(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        config["boundaries"]=[65]
        with self.assertRaises(SourceAdapterError):
            adapt_source(PROJECT/"examples/source_inputs/fetter/model.py",config)

    def test_unsupported_value_consumer_root_cannot_be_silently_omitted(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        tree=ast.parse((PROJECT/"examples/source_inputs/fetter/model.py").read_text())
        tree.body.append(ast.parse("def pending_bill(disutility):\n    return disutility > 0\n").body[0])
        constructor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="create_model")
        assignment=next(n for n in constructor.body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=="retirement" for t in n.targets))
        assert isinstance(assignment.value, ast.Call)
        assignment.value.keywords.append(ast.keyword(arg="value_constraints",value=ast.Dict(
            keys=[ast.Constant("pending_bill")],values=[ast.Name(id="pending_bill",ctx=ast.Load())])))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            with self.assertRaisesRegex(SourceAdapterError,"Unsupported Regime keywords/consumer roots"):
                adapt_source(path,config)

    def test_state_dependent_solve_imputation_is_explicitly_unsupported(self):
        source=PROJECT/"source_fixtures/phased_imputation_reads_x.py"
        with self.assertRaisesRegex(SourceAdapterError,"Phased.solve imputation"):
            adapt_source(source,{"discount_factor":"0.97"})

    def test_conditional_constructor_consumer_mutation_is_not_ignored(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        tree=ast.parse((PROJECT/"examples/source_inputs/fetter/model.py").read_text())
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="utility_retired")
        changed=copy.deepcopy(original);changed.name="conditional_retired_utility"
        changed.args.args.append(ast.arg(arg="disutility"))
        ret=next(n for n in changed.body if isinstance(n,ast.Return))
        assert ret.value is not None
        ret.value=ast.BinOp(left=ret.value,op=ast.Add(),right=ast.Name(id="disutility",ctx=ast.Load()))
        tree.body.append(changed)
        constructor=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="create_model")
        original_declaration=next(n for n in constructor.body if isinstance(n,ast.Assign)
                                 and any(isinstance(t,ast.Name) and t.id=="retirement" for t in n.targets))
        mutation=copy.deepcopy(original_declaration)
        assert isinstance(mutation.value, ast.Call)
        functions=next(k.value for k in mutation.value.keywords if k.arg=="functions")
        assert isinstance(functions, ast.Dict)
        for i,key in enumerate(functions.keys):
            if isinstance(key,ast.Constant) and key.value=="utility":
                functions.values[i]=ast.Name(id="conditional_retired_utility",ctx=ast.Load())
        index=constructor.body.index(original_declaration)+1
        constructor.body.insert(index,ast.If(test=ast.Constant(True),body=[mutation],orelse=[]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            with self.assertRaisesRegex(SourceAdapterError,"constructor control flow/consumer mutation"):
                adapt_source(path,config)

    def test_native_execution_and_unsupported_scope_are_explicit(self):
        for problem in (self.fetter,self.adda):
            self.assertEqual(problem["source_adapter"]["status"],"SUPPORTED_STATIC")
            self.assertEqual(problem["source_adapter"]["native_model_execution"],"NOT_RUN")
            self.assertTrue(problem["source_adapter"]["no_oracle_inputs"])
            self.assertTrue(problem["source_adapter"]["limitations"])
            for state in problem["states"].values():
                if state["kind"]!="DiscreteGrid": self.assertNotIn("values",state)

    def test_external_parameter_declarations_have_original_signature_witnesses(self):
        for problem in (self.fetter,self.adda):
            external=set(problem["external_parameters"])
            self.assertEqual(external,set(problem["source_adapter"]["external_parameter_declarations"]))
            allowed=set(problem["states"])|set(problem["actions"])|external|{"age"}
            self.assertLessEqual(all_reads(problem["programs"]),allowed)
            for parameter,witnesses in problem["source_adapter"]["external_parameter_declarations"].items():
                self.assertTrue(witnesses)
                for witness in witnesses:
                    path=Path(witness["source_anchor"]["path"])
                    functions={n.name:n for n in ast.walk(ast.parse(path.read_text()))
                               if isinstance(n,ast.FunctionDef)}
                    self.assertIn(witness["function"],functions)
                    fn=functions[witness["function"]]
                    self.assertIn(parameter,{a.arg for a in fn.args.args+fn.args.posonlyargs+fn.args.kwonlyargs})


class SourceOptimizerIntegrationTests(unittest.TestCase):
    """Exercise the real source bridge, finder and regenerated certificate together."""
    @classmethod
    def setUpClass(cls):
        cls.problems={n:adapt_manifest(PROJECT/f"examples/{n}-source.json") for n in ("fetter","adda")}
        cls.results={n:find_regimes(p) for n,p in cls.problems.items()}

    def checked(self,problem):
        result=find_regimes(problem)
        self.assertEqual(result["status"],"OPTIMAL",result)
        check=verify_certificate(problem,result)
        self.assertEqual(check["status"],"PASS",check)
        return result

    @staticmethod
    def layout(result):
        return [(s["ages"],s["allocated_state_axes"],s.get("simulation_state_axes",[]),
                 s["allocated_action_axes"],s["cost"]) for s in result["recommended"]["regimes"]]

    def test_original_source_optima_and_certificates(self):
        for name,result in self.results.items():
            self.assertEqual(result["status"],"OPTIMAL")
            self.assertEqual(verify_certificate(self.problems[name],result)["status"],"PASS")
        fetter=self.results["fetter"]["recommended"]
        retired=next(s for s in fetter["regimes"] if s["original_labels"]==["retirement"])
        self.assertNotIn("disutility",retired["allocated_state_axes"])
        self.assertEqual(fetter["external_report_tags"],["disutility"])
        adda=self.results["adda"]["recommended"]["regimes"]
        infertile=next(s for s in adda if s["ages"][0]==92)
        self.assertNotIn("conceive",infertile["allocated_action_axes"])
        self.assertIn("family",infertile["allocated_state_axes"])
        self.assertIn("fecundity",infertile["simulation_state_axes"])
        self.assertNotIn("fecundity",infertile["allocated_state_axes"])

    def test_original_constant_mutation_moves_the_optimal_cut(self):
        config=json.loads((PROJECT/"examples/adda-source.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=PROJECT/"examples/source_inputs/adda"
            shutil.copyfile(source/"model.py",tmp/"model.py")
            before=(source/"calibration.py").read_text()
            after=before.replace("AGE_INFERTILE = 46.0","AGE_INFERTILE = 48.0")
            self.assertNotEqual(before,after)
            (tmp/"calibration.py").write_text(after)
            result=self.checked(adapt_source(tmp/"model.py",config))
        starts=[s["ages"][0] for s in result["recommended"]["regimes"]]
        self.assertIn(96,starts)
        self.assertNotIn(92,starts)
        late=next(s for s in result["recommended"]["regimes"] if s["ages"][0]==96)
        self.assertIn("family",late["allocated_state_axes"])
        self.assertNotIn("conceive",late["allocated_action_axes"])

    def test_original_filename_and_labels_are_not_discovery_keys(self):
        config=json.loads((PROJECT/"examples/adda-source.json").read_text())
        source=PROJECT/"examples/source_inputs/adda"
        class RenameLabels(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value in ("working","retired","terminal"):
                    return ast.copy_location(ast.Constant("renamed_"+node.value),node)
                return node
        tree=RenameLabels().visit(ast.parse((source/"model.py").read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/"unrelated_filename.py").write_text(ast.unparse(ast.fix_missing_locations(tree)))
            shutil.copyfile(source/"calibration.py",tmp/"calibration.py")
            result=self.checked(adapt_source(tmp/"unrelated_filename.py",config))
        self.assertEqual(self.layout(result),self.layout(self.results["adda"]))
        self.assertTrue(all(label.startswith("renamed_") for s in result["recommended"]["regimes"]
                            for label in s["original_labels"]))

    def test_new_original_payoff_consumer_prevents_state_deletion(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        tree=ast.parse((PROJECT/"examples/source_inputs/fetter/model.py").read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="utility_retired")
        fn.args.args.append(ast.arg(arg="disutility"))
        ret=next(x for x in fn.body if isinstance(x,ast.Return))
        assert ret.value is not None
        ret.value=ast.BinOp(left=ret.value,op=ast.Add(),right=ast.Name(id="disutility",ctx=ast.Load()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            result=self.checked(adapt_source(path,config))
        retired=next(s for s in result["recommended"]["regimes"] if s["original_labels"]==["retirement"])
        self.assertIn("disutility",retired["allocated_state_axes"])
        self.assertEqual(result["recommended"]["external_report_tags"],["disutility"])

    def test_original_argument_rebinding_preserves_earlier_alias_dependency(self):
        config=json.loads((PROJECT/"examples/fetter-source.json").read_text())
        tree=ast.parse((PROJECT/"examples/source_inputs/fetter/model.py").read_text())
        fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="utility_retired")
        fn.args.args.append(ast.arg(arg="disutility"))
        ret=next(x for x in fn.body if isinstance(x,ast.Return))
        before=fn.body.index(ret)
        fn.body[before:before]=ast.parse("captured_type = disutility\ndisutility = 0\n").body
        assert ret.value is not None
        ret.value=ast.BinOp(left=ret.value,op=ast.Add(),right=ast.Name(id="captured_type",ctx=ast.Load()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.py";path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
            problem=adapt_source(path,config)
            retired=next(p for p in problem["programs"] if p["id"]=="retirement")
            self.assertIn("disutility",retired["reward"]["reads"])
            result=self.checked(problem)
        retired=next(s for s in result["recommended"]["regimes"] if s["original_labels"]==["retirement"])
        self.assertIn("disutility",retired["allocated_state_axes"])


if __name__=="__main__": unittest.main()
