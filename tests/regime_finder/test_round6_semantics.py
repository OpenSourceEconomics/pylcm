"""Independent source/binding mutation controls; no Model execution."""
from copy import deepcopy
import unittest

from _lcm_regime_finder.semantic_check import SourceProof
from _lcm_regime_finder.model_snapshot import source_function
from _lcm_regime_finder.declarations import ScalarSource, DeclarationError
from _lcm_regime_finder.expressions import partial
from _lcm_regime_finder import find_regimes
from _lcm_regime_finder.declarations import normalize_declarations
from _lcm_regime_finder.production import MetadataAnalysis
from tests.regime_finder.test_round5_local_obligations import native_fixture
from tests.regime_finder.round6_source_oracle import literal_transitive, varies_in_history


def metadata(source, arguments, **extra):
    return {"kind": "callable", "source": source, "arguments": arguments,
            "raw_arguments": arguments, "variadic": False,
            "capture_contract": "inspection-only-frozen-bindings-1", **extra}


def wrapper_metadata():
    # This is the original Adda capture shape: an effective x argument, while
    # the inspected wrapper body receives self/*args/**kwargs. The checker
    # must not decide independence from those raw implementation names.
    return metadata("def __call__(self, *args, **kwargs):\n    return self.fun(*args, **kwargs)\n",
                    ["x"], raw_arguments=["self", "args", "kwargs"],
                    variadic=True,
                    parameter_kinds={"x": "POSITIONAL_OR_KEYWORD"})


def known_helper(history, shift=2):
    return history + shift


def known_caller(history):
    return known_helper(history)


def table_callback(table):
    def lookup(history):
        return history + table['bias'][0]
    return lookup


def shared_table_caller():
    table = {'bias': [2, 3]}
    left, right = table_callback(table), table_callback(table)
    def caller(history):
        return left(history) + right(history)
    return caller


def tuple_default_helper(history, pair=(2, 3)):
    return history if pair == (2, 3) else 0


def tuple_default_caller(history):
    return tuple_default_helper(history)


def binding_callback(payload):
    def callback(history):
        return history + payload[0]
    return callback


class OriginalInputClosureTests(unittest.TestCase):
    def test_unknown_effective_wrapper_retains_actual_argument(self):
        captured = metadata("def consumer(history):\n    return wrapped(history)\n",
                            ["history"], call_targets={"wrapped": wrapper_metadata()})
        self.assertTrue(varies_in_history(literal_transitive))
        checked = SourceProof({}).callback(captured)
        self.assertIn("history", checked.reads)
        self.assertFalse(checked.proved)

    def test_transitive_effective_arguments_are_not_replaced_by_raw_formals(self):
        callbacks = {
            "derived": metadata("def derived(history, shift):\n    return history + shift\n", ["history", "shift"]),
            "consumer": metadata("def consumer(derived):\n    return wrapped(derived)\n", ["derived"],
                                 call_targets={"wrapped": wrapper_metadata()}),
        }
        for shift in (-2, 0, 3):
            with self.subTest(shift=shift):
                checked = SourceProof(callbacks, {"derived": "derived"}, {"shift": shift}).callback("consumer")
                self.assertIn("history", checked.reads)
                self.assertNotIn("shift", checked.reads)
                self.assertFalse(checked.proved)

    def test_named_unknown_helper_also_preserves_original_input(self):
        for source in ("return wrapped(history)", "return wrapped(x=history)"):
            captured = metadata("def consumer(history):\n    " + source + "\n", ["history"],
                                call_targets={"wrapped": wrapper_metadata()})
            self.assertIn("history", SourceProof({}).callback(captured).reads)

    def test_fixed_input_does_not_reappear_as_raw_wrapper_name(self):
        captured = metadata("def consumer(history):\n    return wrapped(history)\n", ["history"],
                            call_targets={"wrapped": wrapper_metadata()})
        checked = SourceProof({}, static={"history": 4}).callback(captured)
        self.assertNotIn("history", checked.reads)
        self.assertFalse(checked.proved)

    def test_unknown_effective_default_does_not_read_an_outer_axis(self):
        target = wrapper_metadata()
        target["defaults"] = {"x": 4}
        captured = metadata("def consumer():\n    return wrapped()\n", [],
                            call_targets={"wrapped": target})
        checked = SourceProof({}).callback(captured)
        self.assertNotIn("x", checked.reads)
        self.assertFalse(checked.proved)

    def test_time_branch_and_transitive_input_mutations(self):
        for period in (0, 1, 2, 3):
            for source, live in (("history if period < 2 else 0", period < 2),
                                 ("0 if period < 2 else history", period >= 2)):
                callbacks = {
                    "derived": metadata("def derived(history, period):\n    return " + source + "\n", ["history", "period"]),
                    "consumer": metadata("def consumer(derived):\n    return wrapped(derived)\n", ["derived"],
                                         call_targets={"wrapped": wrapper_metadata()}),
                }
                checked = SourceProof(callbacks, {"derived": "derived"}, {"period": period}).callback("consumer")
                self.assertEqual("history" in checked.reads, live)
                self.assertNotIn("period", checked.reads)

    def test_compact_helper_table_is_consumed_by_both_interpreters(self):
        captured = source_function(known_caller)
        expression = ScalarSource(captured).expression()
        for history in (-3, 0, 7):
            answer = SourceProof({}, static={"history": history}).callback(captured)
            self.assertTrue(answer.proved)
            self.assertEqual(answer.value, known_caller(history))
            self.assertEqual(partial(expression, {"history": history}).value, answer.value)

    def test_compact_graph_identity_mutation_is_not_certified(self):
        captured = source_function(known_caller)
        self.assertIn("callable_records", captured)
        for field in ("source", "identity_sha256", "capture_status"):
            mutated = deepcopy(captured)
            record = next(record for record in mutated["callable_records"].values()
                          if record.get("qualname", "").endswith("known_helper"))
            record[field] = "changed" if field != "capture_status" else "UNRESOLVED_MUTATION"
            answer = SourceProof({}).callback(mutated)
            self.assertFalse(answer.proved)
            self.assertIn("history", answer.reads)

    def test_compact_root_copy_must_match_its_record(self):
        captured = source_function(known_caller)
        for field, value in (("source", "def known_caller(history):\n    return 0\n"),
                             ("bindings", {"unrecorded": 0})):
            mutated = deepcopy(captured)
            mutated[field] = value
            checked = SourceProof({}).callback(mutated)
            self.assertFalse(checked.proved)
            self.assertIn("history", checked.reads)

    def test_shared_structured_binding_is_interpreted_once_as_original_data(self):
        original = shared_table_caller()
        captured = source_function(original)
        table_refs = [row['bindings']['table']['capture_ref'] for row in captured['callable_records'].values()
                      if row.get('qualname', '').endswith('table_callback.<locals>.lookup')]
        self.assertEqual(len(table_refs), 2)
        self.assertEqual(len(set(table_refs)), 1)
        expression = ScalarSource(captured).expression()
        for history in (-2, 0, 7):
            checked = SourceProof({}, static={'history': history}).callback(captured)
            self.assertTrue(checked.proved)
            self.assertEqual(checked.value, original(history))
            self.assertEqual(partial(expression, {'history': history}).value, checked.value)

    def test_default_binding_keeps_tuple_identity(self):
        captured = source_function(tuple_default_caller)
        expression = ScalarSource(captured).expression()
        for history in (-2, 0, 7):
            checked = SourceProof({}, static={'history': history}).callback(captured)
            self.assertTrue(checked.proved)
            self.assertEqual(checked.value, history)
            self.assertEqual(partial(expression, {'history': history}).value, history)

    def test_shared_binding_payload_identity_and_status_mutations_fail_closed(self):
        captured = source_function(shared_table_caller())
        for field, value in (('value', [0]), ('identity_sha256', 'changed'),
                             ('capture_status', 'UNRESOLVED_MUTATION')):
            mutated = deepcopy(captured)
            binding = next(row for row in mutated['callable_records'].values()
                           if row.get('kind') == 'binding-record')
            binding[field] = value
            checked = SourceProof({}).callback(mutated)
            self.assertFalse(checked.proved)
            self.assertIn('history', checked.reads)

    def test_shared_binding_cycle_is_unresolved_without_expansion(self):
        payload = []
        payload.append(payload)
        captured = source_function(binding_callback(payload))
        checked = SourceProof({}).callback(captured)
        self.assertFalse(checked.proved)
        self.assertIn('history', checked.reads)
        with self.assertRaises(DeclarationError): ScalarSource(captured).expression()

    def test_compact_data_dag_cannot_force_exponential_literal_expansion(self):
        payload = [1]
        for _ in range(40): payload = [payload, payload]
        captured = source_function(binding_callback(payload))
        self.assertLess(len(captured['callable_records']), 50)
        checked = SourceProof({}).callback(captured)
        self.assertFalse(checked.proved)
        self.assertIn('history', checked.reads)
        with self.assertRaises(DeclarationError): ScalarSource(captured).expression()

    def test_callable_binding_is_not_certified_as_an_immutable_value(self):
        captured = source_function(binding_callback(known_helper))
        checked = SourceProof({}).callback(captured)
        self.assertFalse(checked.proved)
        self.assertIn('history', checked.reads)

    def test_bounded_interpretation_keeps_actual_input_at_any_helper_depth(self):
        for depth in (1, 8, 40):
            inner = metadata("def leaf(x):\n    return x\n", ["x"])
            for _ in range(depth):
                inner = metadata("def helper(x):\n    return child(x)\n", ["x"],
                                 call_targets={"child": inner})
            outer = metadata("def consumer(history):\n    return helper(history)\n", ["history"],
                             call_targets={"helper": inner})
            answer = SourceProof({}).callback(outer)
            self.assertIn("history", answer.reads)
            if depth == 40:
                self.assertFalse(answer.proved)


class FamilyBoundaryTests(unittest.TestCase):
    def test_unknown_context_has_concrete_unadopted_merge_question(self):
        source = native_fixture()
        for regime in source["user_regimes"].values():
            regime["fields"].pop("semantic_role")
        report = find_regimes(source)
        boundary = report["analysis"]["family_boundary"]
        self.assertFalse(boundary["unrestricted_model_representation_optimum_claimed"])
        self.assertFalse(boundary["endpoint_preference_is_admissibility_rule"])
        candidates = boundary["conditional_merge_candidates"]
        self.assertEqual([row["regimes"] for row in candidates], [["opaque", "safe"]])
        self.assertEqual(candidates[0]["status"], "CONDITIONAL_NOT_ADMITTED")
        self.assertFalse(candidates[0]["automatic_graph_change"])
        self.assertEqual(len(candidates[0]["source_paths"]), 2)
        self.assertTrue(candidates[0]["required_obligations"])
        self.assertTrue(any(row["disposition"] == "adopt" and row["changed_interfaces"]
                            for row in report["proposal_decisions"]))

    def test_conditional_display_budget_does_not_silently_exhaust_search_family(self):
        source = native_fixture()
        for regime in source["user_regimes"].values():
            regime["fields"].pop("semantic_role")
        analysis = MetadataAnalysis(normalize_declarations(source),
                                    {"conditional_merge_pair_checks": 0}).run()
        self.assertTrue(analysis["discovery_complete"])
        self.assertEqual(analysis["family_boundary"]["conditional_merge_pairs_unexamined"], 1)
        self.assertEqual(analysis["family_boundary"]["conditional_merge_candidates"], [])


if __name__ == "__main__":
    unittest.main()
