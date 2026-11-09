"""Receiver-side regression and mutation family; reviewer execution: NOT_RUN.

Run from the repository root, with this directory on PYTHONPATH:
  python -m unittest discover -s audit-tests -p test_specialized_factory_contract.py -v

The decisive input shape extends the repository's supported original-graph fixture in
tests.regime_finder.test_model_adapter, including its existing specialized-function
regression. No certificate, private search object, or producer result is forged
in the decisive witness. The final test deliberately corrupts a receipt to test
independent checker rejection; it is a supplementary adversarial control.
The native-constructor acceptance requirement is stated in the repair contract;
this regression exercises the supported captured-metadata public entry point.
"""
from copy import deepcopy
from itertools import product
import unittest

from _lcm_regime_finder import find_regimes, verify_production_result
from tests.regime_finder.test_model_adapter import snapshot, function, grid
from .specialized_factory_oracle import exact_policy, constant_representation_is_policy_sufficient


def specialization(*, clock="period", resolved=False, periods=6, scale=1, shift=0):
    # The current native source documents AgeSpecializedFunction. The clock
    # mutation changes the captured graph's coordinate convention, not the
    # supported public declaration type or the factory's construction input.
    date = "age"
    marker = "AgeSpecializedFunction"
    body = f"{shift} - {scale} * (choice - history) * (choice - history)"
    build_source = (f"def build({date}):\n"
                    "    def utility(choice, history):\n"
                    f"        return {body}\n"
                    "    return utility\n")
    wrapper = {"kind": "dataclass", "type": "lcm." + marker, "fields": {
        "build": function(build_source, [date]),
        "signature": function(f"def signature({date}):\n    return {date}\n", [date]),
    }}
    if resolved:
        wrapper["resolved_by_period"] = [
            {"period": period, "value": function(
                f"def utility(choice, history):\n    return {body}\n", ["choice", "history"])}
            for period in range(periods)]
    return wrapper


def original_case(*, clock="period", phased=False, resolved=False, values=(0, 1), scale=1, shift=0):
    source = snapshot()
    fields = source["user_regimes"]["original"]["fields"]
    fields["states"] = {"history": grid(list(values))}
    fields["actions"] = {"choice": grid(list(values))}
    fields["constraints"] = {}
    fields["state_transitions"] = {"history": function(
        "def next_history(history):\n    return history\n", ["history"])}
    wrapper = specialization(clock=clock, resolved=resolved, scale=scale, shift=shift)
    if phased:
        wrapper = {"kind": "dataclass", "type": "lcm.phased.Phased", "fields": {
            "solve": deepcopy(wrapper), "simulate": deepcopy(wrapper)}}
    fields["functions"] = {"utility": wrapper}
    if clock == "age":
        source["ages"] = {"kind": "object", "fields": {"exact_values": [20 + i for i in range(6)]}}
        source["reachability"]["fields"]["nodes"] = [[20 + i, "original"] for i in range(6)]
        source["graph"]["fields"]["initial_nodes"] = [[20, "original"]]
        source["graph"]["fields"]["coordinate_kind"] = "age"
    return source


class SpecializedFactoryContractTests(unittest.TestCase):
    def assert_no_unproved_removal(self, result, *, resolved=False):
        self.assertEqual(result.get("status"), "COMPLETE" if resolved else "UNRESOLVED",
                         "An unsupported specialization must preserve the valid original Model and remain explicit")
        analysis = result["analysis"]
        cells = analysis["declarations"]["cells"]
        self.assertEqual({(cell["period"], cell["regime"]) for cell in cells},
                         {(period, "original") for period in range(6)})
        self.assertEqual(len(cells), 6)
        self.assertTrue(analysis["dependency_closure"])
        self.assertEqual(set(analysis["dependency_closure"]), {cell["id"] for cell in cells})
        self.assertTrue(all("history" in axes for axes in analysis["dependency_closure"].values()))
        if not resolved:
            self.assertTrue(any(item["kind"] == "specialized_effective_signature"
                                for item in analysis["obligations"]), analysis["obligations"])
        for cell in cells:
            options = analysis["interfaces"][cell["id"]]["options"]
            self.assertTrue(options, "The original cell needs an admissible representation")
            self.assertEqual(set(cell["states"]), {"history"})
            self.assertEqual(set(cell["actions"]), {"choice"})
            for option in options:
                self.assertIn("history", option["states"],
                    "A factory signature is not the signature of its returned economic function")
                self.assertEqual(option["state_projection"]["history"],
                                 {"kind": "identity", "target": "history"})
                self.assertEqual(option["actions"], cell["actions"])

    def test_minimal_binary_public_entrypoint(self):
        self.assertEqual(exact_policy(), {0: {0}, 1: {1}})
        self.assertFalse(constant_representation_is_policy_sufficient())
        source = original_case()
        unchanged = deepcopy(source)
        report = find_regimes(source, budget={"seconds": 30}, search_mode="exact")
        self.assert_no_unproved_removal(report)
        self.assertEqual(source, unchanged)
        self.assertEqual(verify_production_result(source, report)["status"], "UNRESOLVED")

    def test_age_period_phase_and_domain_mutations(self):
        for clock, phased, values, scale, shift in product(
                ("period", "age"), (False, True), ((0, 1), (1, 0), (-3, 2)),
                (1, 8), (0, 17)):
            with self.subTest(clock=clock, phased=phased, values=values, scale=scale, shift=shift):
                source = original_case(clock=clock, phased=phased, values=values,
                                       scale=scale, shift=shift)
                unchanged = deepcopy(source)
                report = find_regimes(source, budget={"seconds": 30}, search_mode="exact")
                self.assert_no_unproved_removal(report)
                self.assertEqual(source, unchanged)

    def test_resolved_effective_functions_retain_history(self):
        for clock, phased in product(("period", "age"), (False, True)):
            with self.subTest(clock=clock, phased=phased):
                source = original_case(clock=clock, phased=phased, resolved=True)
                report = find_regimes(source, budget={"seconds": 30}, search_mode="exact")
                self.assertEqual(report["status"], "COMPLETE", report.get("semantic_check"))
                self.assert_no_unproved_removal(report, resolved=True)
                self.assertEqual(verify_production_result(source, report)["status"], "PASS")

    def test_checker_rejects_old_false_complete_claim(self):
        # This deliberately corrupted receipt is a checker mutation, not an
        # admitted-input witness. Recreate the relevant old producer assertions
        # while preserving the ORIGINAL wrapper and source identity.
        source = original_case()
        report = find_regimes(source, budget={"seconds": 30}, search_mode="exact")
        self.assert_no_unproved_removal(report)
        bad = deepcopy(report)
        analysis = bad["analysis"]
        declarations = analysis["declarations"]
        bad["status"] = "COMPLETE"
        bad["semantic_check"] = {"status": "PASS"}
        analysis["obligations"] = []
        declarations["obligations"] = []
        for callback in declarations["callbacks"].values():
            if callback.pop("unresolved_specialization_signature", False):
                callback.update(status="structural", expression=None,
                                dependency_contract="pylcm-declared-inputs/v1")
                callback["reads"] = list(callback["arguments"])
        for cell in declarations["cells"]:
            analysis["dependency_closure"][cell["id"]] = []
            for option in analysis["interfaces"][cell["id"]]["options"]:
                option["states"].pop("history", None)
                option["state_projection"]["history"] = {"kind": "drop"}
        checked = verify_production_result(source, bad)
        self.assertEqual(checked["status"], "FAIL")
        source_check = checked["source_semantics"]
        self.assertEqual(source_check["status"], "FAIL")
        self.assertTrue(any(error["kind"] == "original future dependency closure omitted"
                            and "history" in error.get("missing", [])
                            for error in source_check["errors"]), source_check)


if __name__ == "__main__":
    unittest.main()
